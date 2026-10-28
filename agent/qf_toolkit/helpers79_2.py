"""know79 group 2 (local / stochastic vol, IV approximations, DCC): finite-difference and local-vol glue.

Patterns behind the know78 losses of this group:
- "Dupire on a discrete call-price grid with central differences inside and one-sided ones at the edges":
  the second strike derivative was taken as numpy.gradient applied twice, a 2h-wide stencil that also
  uses one-sided first derivatives next to the boundary. The textbook central second difference is the
  three-point stencil, defined on interior nodes only (which is why such tasks drop the boundary strikes).
  fd_second_derivative gives both (and the non-uniform three-point form), one switch.
- "local-vol Monte Carlo fed by a Dupire grid": the low-level MC got the local-vol TABLE (a DataFrame)
  instead of a sigma(t, S) callable and crashed. local_vol_callable turns a callable, a constant, a 2-D
  grid or a long (expiry/ttm, strike, local_vol) table into the bilinear sigma(t, S) the MC needs.

Self-tests: proven_work/fail72/selftest79_2.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["fd_second_derivative", "local_vol_callable"]


def fd_second_derivative(y, x, axis=0, scheme="three_point"):
    """d2y/dx2 on a 1-D axis of an array: scheme='three_point' (textbook central difference, interior nodes only;
    boundary nodes NaN) | 'gradient2' (numpy.gradient applied twice, edge_order=1) | 'gradient2_e2' (edge_order=2).

    three_point on a possibly non-uniform axis, h- = x_i - x_{i-1}, h+ = x_{i+1} - x_i:
        y''_i = 2 [ (y_{i+1} - y_i)/h+ - (y_i - y_{i-1})/h- ] / (h+ + h-)
    which is (y_{i+1} - 2 y_i + y_{i-1}) / h^2 on a uniform grid (exact for quadratics). The second
    derivative at the first and last node is undefined for this stencil (NaN) - a Dupire table built
    from it is reported on the interior strikes.
    gradient2 = np.gradient(np.gradient(y, x), x): central inside, one-sided at the ends, but its interior
    stencil spans 2h ((y_{i+2} - 2 y_i + y_{i-2})/(4h^2) on a uniform grid) and the nodes next to the edges
    mix in the one-sided first derivatives. Use it only when a task names numpy.gradient twice."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    ax = axis % y.ndim
    if y.shape[ax] != len(x):
        raise ValueError(f"fd_second_derivative: axis {axis} has {y.shape[ax]} points but x has {len(x)}")
    s = str(scheme).lower().replace("-", "_")
    if s in ("gradient2", "gradient", "np_gradient", "gradient_twice"):
        return np.gradient(np.gradient(y, x, axis=ax, edge_order=1), x, axis=ax, edge_order=1)
    if s in ("gradient2_e2", "gradient_e2"):
        eo = 2 if len(x) > 2 else 1
        return np.gradient(np.gradient(y, x, axis=ax, edge_order=eo), x, axis=ax, edge_order=eo)
    if s not in ("three_point", "3pt", "central", "three_point_central"):
        raise ValueError(f"fd_second_derivative: unknown scheme {scheme!r} (three_point | gradient2 | gradient2_e2)")
    ym = np.moveaxis(y, ax, 0)
    out = np.full(ym.shape, np.nan)
    if len(x) >= 3:
        hm = (x[1:-1] - x[:-2]).reshape((-1,) + (1,) * (ym.ndim - 1))
        hp = (x[2:] - x[1:-1]).reshape((-1,) + (1,) * (ym.ndim - 1))
        out[1:-1] = 2.0 * ((ym[2:] - ym[1:-1]) / hp - (ym[1:-1] - ym[:-2]) / hm) / (hp + hm)
    return np.moveaxis(out, 0, ax)


def _pick(cols, *names):
    low = {str(c).lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def local_vol_callable(local_vol, strikes=None, ttms=None, *, t_col=None, k_col=None, v_col=None):
    """sigma(t, S) callable for a local-vol Monte Carlo from a callable | a constant | a 2-D grid | a long table.

    - callable: returned as is; a number: flat sigma.
    - 2-D array [len(ttms), len(strikes)] (rows = maturities ascending): needs strikes= and ttms=.
    - long DataFrame with one row per (maturity, strike) node: the time column is t_col or the first of
      ttm_years / ttm / t / maturity / tau, the strike column k_col or strike / k, the value column v_col or
      local_vol / lv / sigma_loc / sigma. Pivoted to the grid (duplicates averaged).
    Interpolation (local_vol_interpolator): BILINEAR - linear in T between maturity slices and linear in K,
    flat (clamped) outside the grid. This is the textbook reading of a Dupire grid in a path simulation."""
    from .helpers75_3 import local_vol_interpolator
    if callable(local_vol):
        return local_vol
    if np.ndim(local_vol) == 0 and not isinstance(local_vol, pd.DataFrame):
        s0 = float(local_vol)
        return lambda t, S: np.full(np.shape(S), s0)
    if isinstance(local_vol, pd.DataFrame):
        d = local_vol
        tc = t_col or _pick(d.columns, "ttm_years", "ttm", "t", "maturity", "tau", "time_to_maturity")
        kc = k_col or _pick(d.columns, "strike", "k")
        vc = v_col or _pick(d.columns, "local_vol", "lv", "sigma_loc", "local_volatility", "sigma")
        if tc is None or kc is None or vc is None:
            raise ValueError("local_vol_callable: a local-vol table needs a maturity (ttm_years/T), a strike and a "
                             f"local_vol column; got {list(d.columns)} - or pass t_col=, k_col=, v_col=")
        g = d.pivot_table(index=tc, columns=kc, values=vc, aggfunc="mean").sort_index().sort_index(axis=1)
        if g.isna().to_numpy().any():
            g = g.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
        return local_vol_interpolator(g.columns.to_numpy(float), g.index.to_numpy(float), g.to_numpy(float))
    G = np.asarray(local_vol, dtype=float)
    if strikes is None or ttms is None:
        raise ValueError("local_vol_callable: a local-vol GRID needs strikes= and ttms= (rows = maturities)")
    K = np.asarray(strikes, float).ravel()
    t = np.asarray(ttms, float).ravel()
    if G.shape == (len(K), len(t)) and len(K) != len(t):
        G = G.T
    if G.shape != (len(t), len(K)):
        raise ValueError(f"local_vol_callable: grid shape {G.shape} != (len(ttms), len(strikes)) = {(len(t), len(K))}")
    return local_vol_interpolator(K, t, G)
