"""Performance attribution and hedging P&L with the formula variant as a keyword.

Why it exists: re-derived attribution used the wrong Brinson variant (allocation off by ~0.0004
per month: BF's (wp-wb)(rb-Rb) vs BHB's (wp-wb)rb) and hedging P&L came out with the wrong sign.

    from qf_toolkit import attribution
    eff = attribution.brinson_attribution(wp, wb, rp, rb, model="BF")   # sectors x effects
    linked = attribution.link_attribution(frames, rp_list, rb_list, method="carino")
    res = attribution.delta_hedge_pnl(S, V, deltas, r=0.03, dt=1/252, position="short_option")

All returns are simple DECIMAL returns per period (0.012 = 1.2%); weights are BEGINNING-of-period.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .stats import ols

__all__ = ["brinson_attribution", "link_attribution", "returns_based_attribution", "delta_hedge_pnl"]

_EFFECTS = ["allocation", "selection", "interaction"]


def _vec(x, index):
    if isinstance(x, dict):
        x = pd.Series(x)
    if isinstance(x, pd.Series):
        return x.astype(float) if index is None else x.reindex(index).astype(float)
    return pd.Series(np.asarray(x, dtype=float).reshape(-1), index=index)


def brinson_attribution(wp, wb, rp, rb, *, model="BHB", interaction="separate",
                        total_benchmark_return=None, add_total_row=False, atol=1e-10) -> pd.DataFrame:
    """Single-period Brinson attribution per sector -> DataFrame[allocation, selection, interaction, total].

    wp, wb : portfolio / benchmark sector weights (begin of period); rp, rb : sector returns.
        Series/dicts are aligned on the sector labels of wp; arrays by position.
        A sector the portfolio does not hold needs an rp value anyway (any number; wp=0 makes
        allocation/interaction use rb, pass rp=rb to get zero selection/interaction there).
    Rp = sum(wp*rp), Rb = sum(wb*rb) (or total_benchmark_return, used ONLY inside BF allocation).
    model="BHB" (Brinson-Hood-Beebower 1986, default): allocation = (wp-wb)*rb
    model="BF"  (Brinson-Fachler 1985):               allocation = (wp-wb)*(rb-Rb)
        (sector allocations differ; totals agree whenever sum(wp) == sum(wb))
    selection   = wb*(rp-rb) ;  interaction = (wp-wb)*(rp-rb)
    interaction="in_selection": selection = wp*(rp-rb) and interaction = 0 (two-effect version).
    Identity checked (raises ValueError): sum(total) == Rp - sum(wb*rb) within atol. BF needs
    equal weight sums for it (else the unexplained -Rb*sum(wp-wb) is reported in the error).
    df.attrs holds Rp, Rb, excess (= Rp - Rb) and model. add_total_row=True appends a "Total"
    row (do NOT then sum the columns again).
    """
    wp = _vec(wp, None)
    idx = wp.index
    wb, rp, rb = _vec(wb, idx), _vec(rp, idx), _vec(rb, idx)
    if wb.isna().any() or rp.isna().any() or rb.isna().any():
        raise ValueError("wb/rp/rb missing a sector present in wp (pass 0 weights / rb for unheld sectors)")
    Rp = float((wp * rp).sum())
    Rb_sum = float((wb * rb).sum())
    Rb = Rb_sum if total_benchmark_return is None else float(total_benchmark_return)
    dw = wp - wb
    m = model.upper()
    if m == "BHB":
        alloc = dw * rb
    elif m == "BF":
        alloc = dw * (rb - Rb)
    else:
        raise ValueError("model must be 'BHB' or 'BF'")
    if interaction == "separate":
        sel = wb * (rp - rb)
        inter = dw * (rp - rb)
    elif interaction == "in_selection":
        sel = wp * (rp - rb)
        inter = dw * 0.0
    else:
        raise ValueError("interaction must be 'separate' or 'in_selection'")
    df = pd.DataFrame({"allocation": alloc, "selection": sel, "interaction": inter})
    df["total"] = df.sum(axis=1)
    gap = float(df["total"].sum()) - (Rp - Rb_sum)
    if abs(gap) > atol:
        raise ValueError(f"attribution identity failed: sum(effects) - (Rp-Rb) = {gap:.3e}; "
                         f"sum(wp)={wp.sum():.6f} sum(wb)={wb.sum():.6f} (BF needs equal weight sums)")
    df.attrs.update({"Rp": Rp, "Rb": Rb_sum, "excess": Rp - Rb_sum, "model": m})
    if add_total_row:
        df.loc["Total"] = df.sum(axis=0)
    return df


def _linking_factors(rp, rb, method):
    rp, rb = np.asarray(rp, dtype=float), np.asarray(rb, dtype=float)
    T = len(rp)
    RP, RB = float(np.prod(1 + rp) - 1), float(np.prod(1 + rb) - 1)
    if method == "arithmetic":
        return np.ones(T), RP, RB
    if method == "carino":
        def k(a, b):
            return (math.log1p(a) - math.log1p(b)) / (a - b) if abs(a - b) > 1e-14 else 1.0 / (1 + a)
        kt = np.array([k(a, b) for a, b in zip(rp, rb)])
        return kt / k(RP, RB), RP, RB
    if method == "menchero":
        if abs(RP - RB) > 1e-14:
            M = (RP - RB) / (T * ((1 + RP) ** (1 / T) - (1 + RB) ** (1 / T)))
        else:
            M = (1 + RP) ** ((T - 1) / T)
        d = rp - rb
        s2 = float(d @ d)
        alpha = (RP - RB - M * d.sum()) / s2 * d if s2 > 0 else np.zeros(T)
        return M + alpha, RP, RB
    if method in ("geometric", "grap", "frongello"):
        f = np.empty(T)
        for t in range(T):
            f[t] = np.prod(1 + rp[:t]) * np.prod(1 + rb[t + 1:])
        return f, RP, RB
    raise ValueError("method must be 'carino', 'menchero', 'geometric' (GRAP) or 'arithmetic'")


def link_attribution(period_frames, rp_series=None, rb_series=None, *, method="carino",
                     effects=("allocation", "selection", "interaction")) -> pd.DataFrame:
    """Link single-period attribution effects over T periods -> DataFrame (sector x effects + total).

    period_frames : list (or dict date->frame) of brinson_attribution outputs, oldest first; sectors
        are aligned by label (missing sector in a period = 0 effect).
    rp_series, rb_series : per-period TOTAL portfolio / benchmark returns (simple). None -> taken
        from each frame's attrs (Rp, Rb). Each period's effects must sum to rp_t - rb_t.
    method (linked effect = sum_t factor_t * effect_t):
      "carino" (default, Carino 1999): factor_t = k_t / K, k_t = [ln(1+rp_t)-ln(1+rb_t)]/(rp_t-rb_t)
          (1/(1+rp_t) if equal), K the same on the compounded totals RP, RB.
      "menchero" (Menchero 2000): factor_t = M + alpha_t, M = (RP-RB)/(T[(1+RP)^(1/T)-(1+RB)^(1/T)]),
          alpha_t = (RP-RB - M*sum d)/sum d^2 * d_t, d_t = rp_t - rb_t.
      "geometric" (= "grap"/"frongello", GRAP 1997 chain-linking): factor_t =
          prod_{s<t}(1+rp_s) * prod_{s>t}(1+rb_s).
      "arithmetic": factor 1 (plain sum of effects) - does NOT reconcile to RP-RB when compounding.
    The first three reconcile: grand total = RP - RB = prod(1+rp)-1 - (prod(1+rb)-1) (arithmetic
    excess of the COMPOUNDED returns). attrs: RP, RB, excess, factors, method.
    """
    frames = list(period_frames.values()) if isinstance(period_frames, dict) else list(period_frames)
    if rp_series is None:
        rp_series = [f.attrs["Rp"] for f in frames]
    if rb_series is None:
        rb_series = [f.attrs["Rb"] for f in frames]
    rp, rb = np.asarray(rp_series, dtype=float), np.asarray(rb_series, dtype=float)
    if not (len(frames) == len(rp) == len(rb)):
        raise ValueError("period_frames, rp_series, rb_series must have the same length")
    fac, RP, RB = _linking_factors(rp, rb, method)
    effects = list(effects)
    frames = [f.drop(index="Total", errors="ignore")[effects] for f in frames]
    sectors = pd.Index([])
    for f in frames:
        sectors = sectors.union(f.index, sort=False)
    out = sum(f.reindex(sectors).fillna(0.0) * k for f, k in zip(frames, fac))
    out["total"] = out[effects].sum(axis=1)
    out.attrs.update({"RP": RP, "RB": RB, "excess": RP - RB, "factors": fac, "method": method})
    return out


def returns_based_attribution(port_returns, factor_returns, *, method="ols", add_const=True,
                              hac_lags=None) -> dict:
    """Returns-based attribution of a portfolio return series on factor return series.

    port_returns : Series (T); factor_returns : DataFrame (T x K) (same frequency, same units;
        pass EXCESS returns yourself if the model is in excess returns). Rows with NaN dropped.
    method="ols" (default): r_t = alpha + sum_k beta_k f_k,t + e_t via stats.ols (add_const=True,
        hac_lags for Newey-West t-stats).
    method="style": Sharpe (1992) style analysis, min sum e^2 s.t. beta >= 0, sum beta = 1, no
        intercept; alpha = mean(r - F beta) (selection return per period).
    Returns dict:
      alpha (per period), betas (Series), t_stats (Series incl. 'alpha', ols only), r2,
      contribution (DataFrame T x [factors..., alpha, residual]: beta_k*f_k,t, alpha, e_t;
          rows sum to r_t exactly),
      contribution_mean (Series, per-period average), contribution_total (Series, arithmetic sum),
      risk_share (Series: beta_k cov(f_k, r)/var(r) plus residual var(e)/var(r); sums to 1 for OLS
          with intercept).
    """
    r = pd.Series(port_returns, dtype=float) if not isinstance(port_returns, pd.Series) else port_returns.astype(float)
    F = factor_returns if isinstance(factor_returns, pd.DataFrame) else pd.DataFrame(np.asarray(factor_returns, dtype=float))
    if not isinstance(port_returns, pd.Series) or not isinstance(factor_returns, pd.DataFrame):
        r, F = r.reset_index(drop=True), F.reset_index(drop=True)   # align by position
    data = pd.concat([r.rename("__r__"), F.astype(float)], axis=1, join="inner").dropna()
    y, X = data["__r__"], data.drop(columns="__r__")
    names = list(X.columns)
    t_stats = None
    if method == "ols":
        res = ols(y.to_numpy(), X.to_numpy(), add_const=add_const, hac_lags=hac_lags)
        if add_const:
            alpha, betas = float(res["beta"][0]), res["beta"][1:]
            t_stats = pd.Series(res["t"], index=["alpha"] + names)
        else:
            alpha, betas = 0.0, res["beta"]
            t_stats = pd.Series(res["t"], index=names)
        r2 = float(res["r2"])
    elif method == "style":
        from scipy.optimize import minimize
        A, b = X.to_numpy(), y.to_numpy()
        k = A.shape[1]
        sol = minimize(lambda w: float(np.sum((b - A @ w) ** 2)), np.full(k, 1.0 / k), method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        betas = sol.x
        alpha = float(np.mean(b - A @ betas))
        e0 = b - A @ betas
        r2 = float(1 - np.var(e0) / np.var(b)) if np.var(b) > 0 else float("nan")
    else:
        raise ValueError("method must be 'ols' or 'style'")
    betas = pd.Series(np.asarray(betas, dtype=float), index=names)
    contrib = X * betas
    contrib["alpha"] = alpha
    contrib["residual"] = y - contrib.sum(axis=1)
    var_r = float(np.var(y, ddof=1))
    risk = pd.Series({n: betas[n] * float(np.cov(X[n], y, ddof=1)[0, 1]) / var_r for n in names})
    risk["residual"] = float(np.cov(contrib["residual"], y, ddof=1)[0, 1]) / var_r
    return {"alpha": alpha, "betas": betas, "t_stats": t_stats, "r2": r2, "contribution": contrib,
            "contribution_mean": contrib.mean(), "contribution_total": contrib.sum(),
            "risk_share": risk}


def delta_hedge_pnl(S_path, option_values, deltas, *, r=0.0, dt=1 / 252, hedge_freq=1,
                    position="short_option", cost_bps=0.0, contracts=1.0) -> dict:
    """Discrete delta-hedging P&L of ONE option position hedged with the underlying.

    S_path, option_values, deltas : arrays over the time grid t_0..t_n (length n+1), or 2-D
        (n_paths, n+1) for many paths at once. option_values[-1] must be the PAYOFF at expiry;
        deltas are the option's own delta per unit (e.g. BS N(d1)); deltas[-1] is ignored.
    position="short_option" (default): SELL the option (receive V_0) and hold +delta shares.
             "long_option": BUY the option (pay V_0) and hold -delta shares.
    Sign convention: pnl > 0 = money made by the hedger. A short option hedged at implied vol
        earns ~ +0.5*Gamma*S^2*(sigma_implied^2 - sigma_realised^2)*dt per step: positive when
        realised vol < implied; long_option is the exact negative (when cost_bps = 0).
    Loop (theory-card order): t_0 trade to the hedge; cash = premium - shares*S_0 - cost.
        Each step i=1..n: cash *= exp(r*dt) (r continuous, per year; dt in years); if i < n and
        i % hedge_freq == 0 rebalance to deltas[i] (cost |dN|*S*cost_bps/1e4). At expiry no
        rebalance; the book is marked: value_n = cash + shares*S_n -/+ payoff, minus the liquidation
        cost |shares|*S_n*cost_bps/1e4. P&L is terminal (NOT discounted to t_0).
    contracts=1.0 scales the option and hedge quantities.
    Returns dict: total_pnl (float or array per path), value (book value t_0..t_n, value_0 = -initial
        cost), pnl_series (value_0 then per-step changes; sums to total_pnl), hedge_shares (held after
        each date), cash, costs (total), n_rebalances (excluding the initial trade).
    """
    S = np.asarray(S_path, dtype=float)
    V = np.asarray(option_values, dtype=float)
    D = np.asarray(deltas, dtype=float)
    one = S.ndim == 1
    S, V, D = np.atleast_2d(S), np.atleast_2d(V), np.atleast_2d(D)
    if not (S.shape == V.shape == D.shape):
        raise ValueError("S_path, option_values and deltas must have the same shape")
    if position == "short_option":
        opt = -1.0
    elif position == "long_option":
        opt = 1.0
    else:
        raise ValueError("position must be 'short_option' or 'long_option'")
    c = float(cost_bps) / 1e4
    hf = int(hedge_freq)
    if hf < 1:
        raise ValueError("hedge_freq must be >= 1 (steps between rebalances)")
    m, n1 = S.shape
    n = n1 - 1
    q = contracts
    grow = math.exp(r * dt)
    h = -opt * q * D[:, 0]
    cost0 = np.abs(h) * S[:, 0] * c
    cash = -opt * q * V[:, 0] - h * S[:, 0] - cost0
    costs = cost0.copy()
    value = np.empty((m, n1))
    shares = np.empty((m, n1))
    cash_hist = np.empty((m, n1))
    value[:, 0] = cash + h * S[:, 0] + opt * q * V[:, 0]
    shares[:, 0], cash_hist[:, 0] = h, cash
    n_reb = 0
    for i in range(1, n1):
        cash = cash * grow
        if i < n and i % hf == 0:
            h_new = -opt * q * D[:, i]
            dh = h_new - h
            tc = np.abs(dh) * S[:, i] * c
            cash = cash - dh * S[:, i] - tc
            costs = costs + tc
            h = h_new
            n_reb += 1
        val = cash + h * S[:, i] + opt * q * V[:, i]
        if i == n:
            lc = np.abs(h) * S[:, i] * c
            val = val - lc
            costs = costs + lc
        value[:, i], shares[:, i], cash_hist[:, i] = val, h, cash
    pnl_series = np.concatenate([value[:, :1], np.diff(value, axis=1)], axis=1)
    total = value[:, -1]
    if one:
        return {"total_pnl": float(total[0]), "value": value[0], "pnl_series": pnl_series[0],
                "hedge_shares": shares[0], "cash": cash_hist[0], "costs": float(costs[0]),
                "n_rebalances": n_reb}
    return {"total_pnl": total, "value": value, "pnl_series": pnl_series, "hedge_shares": shares,
            "cash": cash_hist, "costs": costs, "n_rebalances": n_reb}
