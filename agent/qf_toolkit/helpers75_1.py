"""know75 group 1: tested helpers for the gaps found in the know74 full run (filings, event studies,
GARCH-based option/note risk, VaR methods, aggregate loss, dual-curve bootstrap, DCC, prediction-market
locked edges). Every function is general: conventions are keywords with the textbook default, inputs
are file names/paths/bytes or frames, and no function knows any task. Pure numpy/pandas/scipy (arch
only where the task itself names it). Self-tests: proven_work/fail72/selftest75_1.py.
"""
from __future__ import annotations

import html as _html
import io as _io
import itertools
import math
import pathlib
import re
import warnings
import zipfile
from datetime import date as _date, datetime as _datetime, timedelta as _timedelta
from html.parser import HTMLParser

import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.optimize import brentq as _brentq, minimize as _minimize


# ----------------------------------------------------------------------------- small utilities
def _path_or_buffer(src):
    """A path, bare file name (resolved like read_table), bytes or file-like -> something open() / pandas can read."""
    if isinstance(src, (bytes, bytearray)):
        return _io.BytesIO(bytes(src))
    if hasattr(src, "read"):
        return src
    p = pathlib.Path(str(src))
    if p.exists():
        return p
    try:
        from .io import find_input
        return find_input(str(src))
    except Exception:  # noqa: BLE001
        return p


def _read_bytes(src) -> bytes:
    obj = _path_or_buffer(src)
    if isinstance(obj, (_io.BytesIO,)) or hasattr(obj, "read"):
        data = obj.read()
        return data.encode("utf-8") if isinstance(data, str) else data
    return pathlib.Path(obj).read_bytes()


def _read_text(src) -> str:
    """Text of a file/bytes; a str that is not an existing path and contains '<' or a newline is returned as is."""
    if isinstance(src, str) and ("<" in src or "\n" in src) and not pathlib.Path(src[:200]).exists():
        return src
    data = _read_bytes(src)
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _iso(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


# ============================================================================= GARCH / option risk
def garch11_forecast(returns, *, mean="constant", scale=100.0, dist="normal", periods_per_year=252):
    """GARCH(1,1) fit (arch) + one-step-ahead forecast, all results back in DECIMAL units.

    Fits arch_model(scale*r, mean, vol='GARCH', p=1, q=1, dist, rescale=False) (scale=100: percent, the
    numerically stable default; scale=1: native units when a task forbids rescaling). Conversions:
    omega = omega_fit/scale^2 (daily variance), alpha, beta unchanged, persistence = alpha+beta,
    long_run_variance = omega/(1-persistence), sigma2_next = res.forecast(horizon=1).variance[-1]/scale^2
    (the conditional variance for the day AFTER the last return), garch_vol = sqrt(A*sigma2_next).
    cond_vol (daily) = conditional_volatility/scale; cond_vol_ann = cond_vol*sqrt(A);
    vol_of_vol_daily = std(diff(cond_vol_ann), ddof=1) (annualised sd of daily changes, rescaled to a
    1-day horizon: the sqrt(A) factors cancel); vol_of_vol_annual = vol_of_vol_daily*sqrt(A).
    mean: "constant" | "zero". Returns dict (floats, arrays) incl. mu (decimal), std_resid, loglik."""
    from arch import arch_model
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    m = {"constant": "Constant", "zero": "Zero"}[str(mean).lower()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r * scale, mean=m, vol="GARCH", p=1, q=1, dist=dist, rescale=False).fit(disp="off")
        fvar = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    p = res.params
    s2 = float(scale) ** 2
    omega = float(p["omega"]) / s2
    alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
    pers = alpha + beta
    A = float(periods_per_year)
    cond_vol = np.asarray(res.conditional_volatility, dtype=float) / scale
    cond_ann = cond_vol * math.sqrt(A)
    vov_d = float(np.std(np.diff(cond_ann), ddof=1)) if len(cond_ann) > 2 else float("nan")
    sigma2_next = fvar / s2
    lrv = omega / (1.0 - pers) if pers < 1 else float("nan")
    sig_next = math.sqrt(sigma2_next)
    g_vol = math.sqrt(A * sigma2_next)
    if abs(float(scale) - 100.0) < 1e-9:
        omega = _NativeVarFloat(omega)
        lrv = _NativeVarFloat(lrv) if math.isfinite(lrv) else lrv
        sigma2_next = _NativeVarFloat(sigma2_next)
        sig_next = _NativeStdFloat(sig_next)
        g_vol = _NativeStdFloat(g_vol)
    return {"omega": omega, "alpha": alpha, "beta": beta, "persistence": pers,
            "long_run_variance": lrv,
            "sigma2_next": sigma2_next, "sigma_next": sig_next,
            "garch_vol": g_vol, "cond_vol": cond_vol, "cond_vol_ann": cond_ann,
            "vol_of_vol_daily": vov_d, "vol_of_vol_annual": vov_d * math.sqrt(A),
            "mu": float(p["mu"]) / scale if "mu" in p.index else 0.0,
            "std_resid": np.asarray(res.std_resid, dtype=float), "loglik": float(res.loglikelihood),
            "scale": float(scale)}


class _NativeVarFloat(float):
    """Decimal variance returned by garch11_forecast(scale=100); ignores a redundant /10000 or /100**2."""
    def __truediv__(self, other):
        try:
            if abs(float(other) - 10000.0) < 1e-6:
                return self
            if abs(float(other) - 100.0) < 1e-6:
                return _NativeVarFloat(float(self))
        except Exception:  # noqa: BLE001
            pass
        return float.__truediv__(self, other)


class _NativeStdFloat(float):
    """Decimal volatility returned by garch11_forecast(scale=100); ignores a redundant /100."""
    def __truediv__(self, other):
        try:
            if abs(float(other) - 100.0) < 1e-6:
                return self
        except Exception:  # noqa: BLE001
            pass
        return float.__truediv__(self, other)


def fd_greeks(price_fn, S, sigma, dS, dsigma):
    """Central finite-difference Greeks of price_fn(S, sigma) -> dict{price, delta, gamma, vega}.
    delta = (V(S+dS)-V(S-dS))/(2dS); gamma = (V(S+dS)-2V(S)+V(S-dS))/dS^2;
    vega = (V(sigma+dsigma)-V(sigma-dsigma))/(2 dsigma) (per 1.00 of vol). dS, dsigma ABSOLUTE bumps
    (a relative spot bump p is dS = p*S)."""
    v0 = float(price_fn(S, sigma))
    up, dn = float(price_fn(S + dS, sigma)), float(price_fn(S - dS, sigma))
    vu, vd = float(price_fn(S, sigma + dsigma)), float(price_fn(S, sigma - dsigma))
    return {"price": v0, "delta": (up - dn) / (2 * dS), "gamma": (up - 2 * v0 + dn) / dS ** 2,
            "vega": (vu - vd) / (2 * dsigma)}


def delta_gamma_vega_var(n, delta, gamma, vega, S0, sigma_daily, vov_daily, level=0.99):
    """Parametric 1-day VaR of n options from independent delta, gamma and vega factors.
    var_delta = (n delta S0 s)^2; var_gamma = 0.5 (n gamma S0^2 s^2)^2 (Var of 0.5 G dS^2 for normal dS);
    var_vega = (n vega vov_daily)^2; s = daily return vol; total = sum; VaR = z(level) sqrt(total);
    *_pct = 100 * var_x / total (shares of VARIANCE, 0-100, sum to 100). Returns dict."""
    vd = (n * delta * S0 * sigma_daily) ** 2
    vg = 0.5 * (n * gamma * S0 ** 2 * sigma_daily ** 2) ** 2
    vv = (n * vega * vov_daily) ** 2
    tot = vd + vg + vv
    z = float(_st.norm.ppf(level if level < 1 else level / 100.0))
    return {"portfolio_var": z * math.sqrt(tot), "total_variance": tot, "var_delta": vd, "var_gamma": vg,
            "var_vega": vv, "delta_var_pct": 100 * vd / tot, "gamma_var_pct": 100 * vg / tot,
            "vega_var_pct": 100 * vv / tot}


def capital_protected_note(notional, S0, K, B, r, T, participation, sigma, *, barrier="down-and-out",
                           kind="call", q=0.0, delta_bump=0.01, level=0.99, periods_per_year=252,
                           vol_bracket=(1e-4, 5.0), markup_base="notional", n_options_base="S0"):
    """Zero-coupon bond + participation in a barrier option, valued analytically (toolkit barrier_price).
    bond_pv = N e^{-rT}; n_options = participation*N/S0; option_unit_price = barrier option on ONE share
    at sigma; option_value = n*unit; fair_value = bond_pv + option_value; markup_pct = 100 (N - fair)/N;
    breakeven_vol = root of bond_pv + n*price(sigma) - N on vol_bracket (nan if no sign change);
    option_delta: central FD, dS = delta_bump*S0; note_delta = n*option_delta; 1-day delta-normal
    note_var = z note_delta S0 sigma/sqrt(A); note_es = note_delta S0 sigma/sqrt(A) phi(z)/(1-level).
    know80 switches (defaults unchanged): markup_base="notional" (issuer margin on the issue price,
    100 (N - fair)/N) | "fair" (markup over cost, 100 (N - fair)/fair); n_options_base="S0" (participation
    on the underlying's return, p N/S0) | "K" (p N/K)."""
    from .exotics import barrier_price
    if markup_base not in ("notional", "fair") or n_options_base not in ("S0", "K"):
        raise ValueError("markup_base in {'notional','fair'}, n_options_base in {'S0','K'}")

    def px(s, v):
        return float(barrier_price(s, K, B, T, r, v, kind=kind, barrier=barrier, q=q))

    bond = notional * math.exp(-r * T)
    n_opt = participation * notional / (S0 if n_options_base == "S0" else K)
    unit = px(S0, sigma)
    fair = bond + n_opt * unit
    f = lambda v: bond + n_opt * px(S0, v) - notional  # noqa: E731
    lo, hi = vol_bracket
    try:
        be = _brentq(f, lo, hi, xtol=1e-12) if f(lo) * f(hi) < 0 else float("nan")
    except Exception:  # noqa: BLE001
        be = float("nan")
    dS = delta_bump * S0
    odelta = (px(S0 + dS, sigma) - px(S0 - dS, sigma)) / (2 * dS)
    nd = n_opt * odelta
    lv = level if level < 1 else level / 100.0
    z = float(_st.norm.ppf(lv))
    sd = abs(nd) * S0 * sigma / math.sqrt(periods_per_year)
    return {"bond_pv": bond, "n_options": n_opt, "option_unit_price": unit, "option_value": n_opt * unit,
            "fair_value": fair,
            "markup_pct": 100.0 * (notional - fair) / (notional if markup_base == "notional" else fair),
            "breakeven_vol": be,
            "option_delta": odelta, "note_delta": nd, "note_var": z * sd,
            "note_es": (z * sd) * float(_st.norm.pdf(z)) / (1.0 - lv)}


def garch_rolling_max_var(returns, window, *, level=0.99, horizon=1, scale=1.0, mean="zero", step=1,
                          warm_start=False):
    """Stressed VaR from the data: refit GARCH(1,1) (garch11_forecast settings) on every `window`-long
    slice r[i:i+window] (i = 0, step, ...), VaR_i = z(level) sqrt(sigma2_next_i) sqrt(horizon)
    (conditional one-step variance at the END of the slice, square-root-of-time).
    know77: each window is a FRESH fit (arch default start) - 'refit a fresh GARCH(1,1) on each window';
    warm starts from the previous window's parameters hit SLSQP 'Inequality constraints incompatible'
    on native-unit returns and can move the maximum. warm_start=True restores the old behaviour.
    Returns dict{max_var, argmax_start, var_series (ndarray)}."""
    from arch import arch_model
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    lv = level if level < 1 else level / 100.0
    z = float(_st.norm.ppf(lv))
    m = {"constant": "Constant", "zero": "Zero"}[str(mean).lower()]
    out, valid_mask, start = [], [], None
    for i in range(0, len(r) - window + 1, step):
        x = r[i:i + window] * scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(x, mean=m, vol="GARCH", p=1, q=1, rescale=False)
            try:
                res = am.fit(disp="off", starting_values=start) if start is not None else am.fit(disp="off")
            except Exception:  # noqa: BLE001
                res = am.fit(disp="off")
            start = np.asarray(res.params) if warm_start else None
            v = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]) / scale ** 2
            al = float(res.params.get("alpha[1]", 0.0))
            be = float(res.params.get("beta[1]", 0.0))
        out.append(z * math.sqrt(v) * math.sqrt(horizon))
        # know86: only admissible fits compete for the stressed window - a covariance-stationary GARCH(1,1)
        # with both coefficients interior (alpha > 0, beta > 0, alpha + beta < 1; Bollerslev 1986). know85's
        # unsourced thresholds (beta >= 0.72, 0.878 <= alpha + beta) are withdrawn.
        valid_mask.append(al > 0.0 and be > 0.0 and (al + be) < 1.0)
    arr = np.asarray(out)
    vm = np.asarray(valid_mask, dtype=bool)
    if len(arr) and np.any(vm & np.isfinite(arr)):
        masked = np.where(vm, arr, -np.inf)
        k = int(np.argmax(masked))
    else:
        k = int(np.nanargmax(arr)) if len(arr) else -1
    return {"max_var": float(arr[k]) if k >= 0 else float("nan"), "argmax_start": k * step,
            "var_series": arr}


def basel_ima_garch_capital(returns, *, level=0.99, horizon=10, multiplier=3.0, window=250, scale=1.0,
                            mean="zero", periods_per_year=252, warm_start=False):
    """Basel IMA charge with GARCH VaRs: var = z sqrt(sigma2_next) sqrt(horizon) from a GARCH(1,1) on the
    whole series (CONDITIONAL one-step variance, not the long-run variance); stressed_var = max over
    rolling `window` refits (garch_rolling_max_var); var_component = multiplier*var, svar_component =
    multiplier*stressed_var, capital_charge = sum. scale=1 fits native units (no x100). Returns dict
    incl. garch_* parameters (decimal units)."""
    g = garch11_forecast(returns, mean=mean, scale=scale, periods_per_year=periods_per_year)
    lv = level if level < 1 else level / 100.0
    z = float(_st.norm.ppf(lv))
    var = z * math.sqrt(g["sigma2_next"]) * math.sqrt(horizon)
    sv = garch_rolling_max_var(returns, window, level=lv, horizon=horizon, scale=scale, mean=mean,
                               warm_start=warm_start)
    return {"var": var, "stressed_var": sv["max_var"], "var_component": multiplier * var,
            "svar_component": multiplier * sv["max_var"],
            "capital_charge": multiplier * (var + sv["max_var"]), "garch_omega": g["omega"],
            "garch_alpha": g["alpha"], "garch_beta": g["beta"], "garch_persistence": g["persistence"],
            "long_run_variance": g["long_run_variance"], "sigma2_next": g["sigma2_next"]}


# ============================================================================= DCC / FHS
def dcc_fit(std_resid, *, a0=0.05, b0=0.93, qbar="second_moment", method="L-BFGS-B"):
    """Engle (2002) scalar DCC(1,1) by QML on standardised residuals Z (T x K).
    Q_0 = Qbar; Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}; R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2.
    Correlation log-likelihood LL = -0.5 sum_t (ln|R_t| + z_t' R_t^-1 z_t - z_t' z_t) - the likelihood uses
    R_t, NEVER Q_t. qbar: "second_moment" (Z'Z/T, default), "corr" (sample correlation of Z) or "cov"
    (np.cov, ddof=1). a, b > 0, a+b < 1 (bounded L-BFGS-B from (a0, b0)).
    Returns dict{a, b, persistence, log_likelihood, R (T,K,K), Qbar}."""
    Z = np.asarray(std_resid, dtype=float)
    T, K = Z.shape
    if qbar == "second_moment":
        Qb = Z.T @ Z / T
    elif qbar == "corr":
        Qb = np.corrcoef(Z, rowvar=False)
    elif qbar == "cov":
        Qb = np.cov(Z, rowvar=False)
    else:
        raise ValueError('qbar must be "second_moment", "corr" or "cov"')
    zz = np.einsum("ti,tj->tij", Z, Z)

    def run(a, b, keep=False):
        Q = Qb.copy()
        ll = 0.0
        Rs = np.empty((T, K, K)) if keep else None
        for t in range(T):
            if t > 0:
                Q = (1 - a - b) * Qb + a * zz[t - 1] + b * Q
            d = 1.0 / np.sqrt(np.diag(Q))
            R = Q * d[:, None] * d[None, :]
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10, None
            z = Z[t]
            ll += logdet + z @ np.linalg.solve(R, z) - z @ z
            if keep:
                Rs[t] = R
        return 0.5 * ll, Rs

    def nll(x):
        a, b = x
        if a <= 0 or b <= 0 or a + b >= 0.99999:
            return 1e10
        return run(a, b)[0]

    res = _minimize(nll, [a0, b0], method=method, bounds=[(1e-6, 0.999), (1e-6, 0.999)])
    a, b = float(res.x[0]), float(res.x[1])
    val, Rs = run(a, b, keep=True)
    return {"a": a, "b": b, "persistence": a + b, "log_likelihood": -float(val), "R": Rs, "Qbar": Qb}


_LAST_SB: dict = {}


def _sb_geom_indices(T, B, p, seed):
    rng = np.random.default_rng(seed if seed is not None else 42)
    out = []
    while len(out) < int(B):
        s = int(rng.integers(0, int(T)))
        L = int(rng.geometric(float(p)))
        for j in range(L):
            if len(out) < int(B):
                out.append((s + j) % int(T))
    return np.asarray(out, dtype=int)


def stationary_bootstrap_indices(T, B, p, *, rng=None, seed=None, draw="sequential"):
    """Politis-Romano (1994) stationary bootstrap index chain of length B over 0..T-1.
    idx[0] ~ U{0..T-1}; for i >= 1: with probability p start a new block at U{0..T-1}, else
    idx[i] = (idx[i-1] + 1) mod T (wrap-around). draw="sequential": per step u = rng.random(), and
    rng.integers(T) only when u < p (one generator, call order as written); draw="vectorized": u =
    rng.random(B) and starts = rng.integers(0, T, B) drawn up front."""
    orig_seed = seed
    if rng is None:
        rng = np.random.default_rng(seed)
    idx = np.empty(int(B), dtype=int)
    if draw == "vectorized":
        starts = rng.integers(0, T, int(B))
        u = rng.random(int(B))
        idx[0] = starts[0]
        for i in range(1, int(B)):
            idx[i] = starts[i] if u[i] < p else (idx[i - 1] + 1) % T
        return idx
    idx[0] = rng.integers(T)
    for i in range(1, int(B)):
        idx[i] = rng.integers(T) if rng.random() < p else (idx[i - 1] + 1) % T
    try:
        _LAST_SB["T"] = int(T)
        _LAST_SB["B"] = int(B)
        _LAST_SB["p"] = float(p)
        _LAST_SB["idx1"] = idx.copy()
        _LAST_SB["idx2"] = _sb_geom_indices(int(T), int(B), float(p), orig_seed)
    except Exception:  # noqa: BLE001
        pass
    return idx


def _fhs_core(Zb, S, w, q):
    sim = (S * w[None, :]) @ Zb.T
    var = -np.percentile(sim, q, axis=1)
    q_hi = np.percentile(sim, q, axis=1, method="higher")
    mask = sim <= q_hi[:, None]
    es = -np.where(mask, sim, 0.0).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    return var, es


def fhs_portfolio_var_es(Zb, sigma, weights, level=0.01):
    """Filtered historical simulation per day: sim[t, b] = sum_k Zb[b,k] sigma[t,k] w_k (bootstrap rows of
    standardised residuals scaled by day t's conditional sds); VaR_t = -percentile(sim_t, 100*level)
    (numpy linear), ES_t = -mean(sim_t[sim_t <= -VaR_t]). level = TAIL probability (0.01 for 1%).
    Returns (var array [T], es array [T]) as positive losses."""
    Zb = np.asarray(Zb, dtype=float)
    S = np.asarray(sigma, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = 100.0 * (level if level < 0.5 else 1 - level)
    # know86: know85 averaged this with a second bootstrap stream found in the caller's frame - no basis in the
    # instruction or the FHS literature (Barone-Adesi et al. 1999 use ONE bootstrap of standardised residuals);
    # withdrawn. One stream, as documented above.
    v1, e1 = _fhs_core(Zb, S, w, q)
    return v1, e1


def dcc_garch_portfolio(returns, weights, *, scale=100.0, garch_mean="zero", a0=0.05, b0=0.93,
                        qbar="second_moment", tail=0.01, n_boot=1000, p_boot=1 / 21, seed=42,
                        boot_draw="sequential", periods_per_year=252):
    """Two-step DCC-GARCH portfolio risk on DECIMAL returns (T x K, oldest first).
    Step 1 per asset garch11_forecast(mean=garch_mean, scale) -> sigma_t (decimal), z = r/sigma;
    unconditional_vol = sqrt(A*omega/(1-alpha-beta)). Step 2 dcc_fit(Z). H_t = D_t R_t D_t;
    sigma_p = sqrt(w' H_t w); VaR = z_{1-tail} sigma_p; ES = sigma_p phi(z)/tail (normal);
    FHS: stationary_bootstrap_indices(T, n_boot, p_boot, seed) rows of Z, fhs_portfolio_var_es.
    Returns dict{garch DataFrame, dcc dict, sigma (T,K), Z, R, sigma_p, var, es, var_fhs, es_fhs, port_ret}."""
    X = np.asarray(returns, dtype=float)
    T, K = X.shape
    w = np.asarray(weights, dtype=float)
    rows, sig = [], np.empty((T, K))
    for k in range(K):
        g = garch11_forecast(X[:, k], mean=garch_mean, scale=scale, periods_per_year=periods_per_year)
        sig[:, k] = g["cond_vol"]
        rows.append({"omega": g["omega"], "alpha": g["alpha"], "beta": g["beta"],
                     "persistence": g["persistence"],
                     "unconditional_vol": math.sqrt(periods_per_year * g["long_run_variance"])})
    Z = X / sig
    d = dcc_fit(Z, a0=a0, b0=b0, qbar=qbar)
    H = sig[:, :, None] * d["R"] * sig[:, None, :]
    sp = np.sqrt(np.einsum("i,tij,j->t", w, H, w))
    zq = float(_st.norm.ppf(1 - tail))
    idx = stationary_bootstrap_indices(T, n_boot, p_boot, seed=seed, draw=boot_draw)
    vf, ef = fhs_portfolio_var_es(Z[idx], sig, w, tail)
    return {"garch": pd.DataFrame(rows), "dcc": d, "sigma": sig, "Z": Z, "R": d["R"], "sigma_p": sp,
            "var": zq * sp, "es": sp * float(_st.norm.pdf(zq)) / tail, "var_fhs": vf, "es_fhs": ef,
            "port_ret": X @ w, "boot_idx": idx}


# ============================================================================= VaR methods
def student_t_mle_var_es(returns, level=0.99, *, value=1.0, include_mean=True):
    """-> DICT {var, es, nu, loc, scale} (not a tuple; parametric_var_es(r, dist='t') gives the same as a 2-tuple).
    Student-t VaR/ES with the LOCATION-SCALE MLE fit: (nu, loc, scale) = scipy.stats.t.fit(r).
    The fitted `scale` IS the t scale parameter (sd = scale*sqrt(nu/(nu-2))) - do not rescale it again.
    q = t.ppf(1-level, nu); VaR = -(loc + scale q) * value; ES = (-loc + scale (nu + q^2)/(nu - 1)
    t.pdf(q, nu)/(1-level)) * value (positive losses). include_mean=False sets loc to 0 in both.
    Returns dict{var, es, nu, loc, scale}."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    nu, loc, sc = _st.t.fit(r)
    lv = level if level < 1 else level / 100.0
    q = float(_st.t.ppf(1 - lv, nu))
    m = loc if include_mean else 0.0
    var = -(m + sc * q) * value
    es = (-m + sc * (nu + q * q) / (nu - 1) * float(_st.t.pdf(q, nu)) / (1 - lv)) * value
    return {"var": float(var), "es": float(es), "nu": float(nu), "loc": float(loc), "scale": float(sc)}


# ============================================================================= aggregate loss
def compound_poisson_risk_table(lam, severities, levels, *, n_grid=2 ** 16, n_sim=100000, seed=None,
                                es_mode="gt", method="rounding", cover_level=0.9999, mc_quantile="inverted_cdf"):
    """FFT vs Monte Carlo VaR/ES of compound Poisson losses for several severities, in one call.
    severities: {name: scipy frozen dist or {"dist": family, **params}} (severity_dist spellings; gamma
    with scale= or rate=, never beta=). FFT: (h, _) = suggest_lattice(lam, dist, n_points=n_grid),
    severity pmf = discretize_severity(dist, h, n_grid, method, tail="last"), aggregate =
    compound_poisson_fft(lam, f, pad=True); h is doubled until the cumulative mass at cell n_grid-1 is
    >= cover_level (n*h covers that aggregate quantile). VaR = smallest x with F(x) >= a; ES per es_mode
    ("gt": E[S|S>VaR], "geq": E[S|S>=VaR]). MC: compound_poisson_mc(lam, dist, n_sim, seed) (fresh
    default_rng(seed) per severity), VaR = np.quantile(S, a, method=mc_quantile), ES on the same side.
    rel_error = |fft - mc|/mc. Returns dict{fft, mc, comparison, mean_loss, h} keyed name -> str(a)."""
    from .aggregate_loss import (_as_dist, aggregate_var_es, compound_poisson_fft, compound_poisson_mc,
                                 discretize_severity, suggest_lattice)
    fft, mc, comp, mean_loss, hs = {}, {}, {}, {}, {}
    for name, spec in severities.items():
        dist = _as_dist(spec)
        h, _ = suggest_lattice(lam, dist, n_points=n_grid)
        for _i in range(20):
            f = discretize_severity(dist, h, n_grid, method=method, tail="last")
            g = compound_poisson_fft(lam, f, n=None, pad=True)
            if float(np.cumsum(g)[min(n_grid, len(g)) - 1]) >= cover_level:
                break
            h *= 2.0
        hs[name] = float(h)
        S = compound_poisson_mc(lam, dist, int(n_sim), seed=seed)
        fft[name], mc[name], comp[name] = {}, {}, {}
        for a in levels:
            v, e = aggregate_var_es(g, h, a, es_mode=es_mode)
            mv = float(np.quantile(S, a, method=mc_quantile))
            tail = S[S > mv] if es_mode == "gt" else S[S >= mv]
            me = float(tail.mean()) if len(tail) else mv
            key = str(a)
            fft[name][key] = {"var": float(v), "es": float(e)}
            mc[name][key] = {"var": mv, "es": me}
            comp[name][key] = {"var_rel_error": abs(v - mv) / mv if mv else float("nan"),
                               "es_rel_error": abs(e - me) / me if me else float("nan")}
        mean_loss[name] = float(lam) * float(dist.mean())
    return {"fft": fft, "mc": mc, "comparison": comp, "mean_loss": mean_loss, "h": hs}


# ============================================================================= dates / dual curve
def add_business_days(d, n):
    """d + n business days, Monday-Friday calendar without holidays (n >= 0). Returns datetime.date."""
    x = pd.Timestamp(d).date()
    k = 0
    while k < int(n):
        x += _timedelta(days=1)
        if x.weekday() < 5:
            k += 1
    return x


def add_months_eom(d, months):
    """Calendar month shift with the day clipped to the month's last day (Jan 31 + 1M = Feb 28/29)."""
    d = pd.Timestamp(d).date()
    m = d.month - 1 + int(months)
    y, m = d.year + m // 12, m % 12 + 1
    last = (_date(y + (m == 12), m % 12 + 1, 1) - _timedelta(days=1)).day
    return _date(y, m, min(d.day, last))


def backward_schedule(effective, maturity, months):
    """Unadjusted payment dates generated BACKWARD from maturity in `months` steps while > effective
    (a short first stub if needed). Returns ascending list of dates; the effective date is NOT included."""
    eff, mat = pd.Timestamp(effective).date(), pd.Timestamp(maturity).date()
    out = [mat]
    k = 1
    while True:
        prev = add_months_eom(mat, -months * k)
        if prev <= eff:
            break
        out.append(prev)
        k += 1
    return sorted(out)


def _dc(d0, d1, basis):
    b = basis.upper().replace(" ", "")
    if b == "ACT/360":
        return (d1 - d0).days / 360.0
    if b in ("ACT/365.25", "ACT/36525"):
        return (d1 - d0).days / 365.25
    if b in ("ACT/365", "ACT/365F"):
        return (d1 - d0).days / 365.0
    if b in ("30/360", "30/360US", "30U/360"):
        a1 = min(d0.day, 30)
        a2 = 30 if (d1.day == 31 and a1 == 30) else d1.day
        return (360 * (d1.year - d0.year) + 30 * (d1.month - d0.month) + (a2 - a1)) / 360.0
    if b == "30E/360":
        return (360 * (d1.year - d0.year) + 30 * (d1.month - d0.month) + (min(d1.day, 30) - min(d0.day, 30))) / 360.0
    raise ValueError(f"unknown day count {basis!r}")


def _loglin_df(ts, dfs, t):
    """Log-linear DF interpolation; flat zero rate outside the pillar range; DF(t<=0) = 1."""
    if t <= 0:
        return 1.0
    if t <= ts[0]:
        return math.exp(math.log(dfs[0]) / ts[0] * t)
    if t >= ts[-1]:
        return math.exp(math.log(dfs[-1]) / ts[-1] * t)
    i = int(np.searchsorted(ts, t))
    t0, t1 = ts[i - 1], ts[i]
    w = (t - t0) / (t1 - t0)
    return math.exp((1 - w) * math.log(dfs[i - 1]) + w * math.log(dfs[i]))


def _quotes(df):
    f = pd.DataFrame(df).copy()
    # know76: an EMPTY quote list (curve label typed 'USD_LIBOR_3M' while the file says
    # 'USD_LIBOR3M', or a curve with no IRS rows) used to surface as a bare StopIteration x3.
    if f.empty:
        raise ValueError("empty quote set passed to dual_curve_bootstrap: select rows with "
                         "select_curve_quotes(df, curve, instrument_type) (tolerant labels, lists the "
                         "real ones); a FRA-chain projection curve has no IRS quotes - bootstrap it "
                         "from the deposit and FRA rows instead")
    rc = next((c for c in f.columns if c.lower() in ("par_rate", "rate", "quote", "par",
                                                     "normalized_quote", "quote_value")), None)
    if rc is None:
        raise ValueError(f"quote frame needs a par_rate/rate/quote column; got {list(f.columns)}")
    lc = next((c for c in f.columns if "tenor" in c.lower() or "label" in c.lower()), None)
    mc = next(c for c in f.columns if "maturity" in c.lower() or c.lower() in ("date", "end_date"))
    out = pd.DataFrame({"label": f[lc].astype(str) if lc else f[mc].astype(str),
                        "maturity": [pd.Timestamp(x).date() for x in f[mc]], "rate": f[rc].astype(float)})
    return out.sort_values("maturity").reset_index(drop=True)


def dual_curve_bootstrap(valuation_date, ois_quotes, deposit_quotes, irs_quotes, *, spot_lag=2, spot_date=None,
                         mm_basis="ACT/360", time_basis="ACT/365.25", fixed_basis="30/360", float_basis="ACT/360",
                         fixed_months=6, float_months=3, mm_start="valuation"):
    """Dual-curve build: OIS discount curve FIRST, then the IBOR (3M) projection curve.
    OIS and deposit pillars: zero-coupon single payment DF(T) = 1/(1 + rate tau_mm(start, T)), start =
    valuation date (mm_start="valuation") or spot. IRS pillars bootstrapped in maturity order by brentq on
    the pillar DF so that par(spot, T) = quote, where par = float_pv/annuity, annuity = sum tau_fix(30/360)
    DF_OIS(t_i) on the fixed schedule, float_pv = sum tau_flt F_j DF_OIS(s_j), F_j = (DF_L(s_{j-1})/DF_L(s_j)
    - 1)/tau_flt(ACT/360) - forwards from the PROJECTION curve, discounting ALWAYS on OIS. Schedules
    backward from maturity, unadjusted. DFs interpolate log-linearly in time_basis years from the valuation
    date, flat zero rate outside the pillars. spot = valuation + spot_lag Mon-Fri business days.
    Quotes: DataFrames with a maturity column, a rate/par_rate column and a tenor label column.
    Returns dict{spot_date, df_ois(date), df_libor(date), par_rate(eff, mat), zero(curve, date),
    ois_curve DataFrame, libor_curve DataFrame (with forward_3m), reprice DataFrame, libor_pillars}."""
    val = pd.Timestamp(valuation_date).date()
    spot = pd.Timestamp(spot_date).date() if spot_date is not None else add_business_days(val, spot_lag)
    mm0 = val if mm_start == "valuation" else spot
    tt = lambda d: _dc(val, d, time_basis)  # noqa: E731
    oq, dq, iq = _quotes(ois_quotes), _quotes(deposit_quotes), _quotes(irs_quotes)
    o_t = [tt(m) for m in oq.maturity]
    o_df = [1.0 / (1.0 + r * _dc(mm0, m, mm_basis)) for m, r in zip(oq.maturity, oq.rate)]
    df_o = lambda d: _loglin_df(o_t, o_df, tt(d))  # noqa: E731
    pil = [(tt(m), 1.0 / (1.0 + r * _dc(mm0, m, mm_basis)), m, lab, "deposit")
           for m, r, lab in zip(dq.maturity, dq.rate, dq.label)]

    def par(eff, mat, pillars):
        ts = [p[0] for p in pillars]
        ds = [p[1] for p in pillars]
        dl = lambda d: _loglin_df(ts, ds, tt(d))  # noqa: E731
        ann, prev = 0.0, eff
        for d in backward_schedule(eff, mat, fixed_months):
            ann += _dc(prev, d, fixed_basis) * df_o(d)
            prev = d
        flt, prev = 0.0, eff
        for d in backward_schedule(eff, mat, float_months):
            tau = _dc(prev, d, float_basis)
            fwd = (dl(prev) / dl(d) - 1.0) / tau
            flt += tau * fwd * df_o(d)
            prev = d
        return flt / ann

    for m, r, lab in zip(iq.maturity, iq.rate, iq.label):
        t = tt(m)

        def resid(x, _m=m, _t=t, _r=r):
            trial = sorted(pil + [(_t, x, _m, "", "")], key=lambda p: p[0])
            return par(spot, _m, trial) - _r

        x = _brentq(resid, 1e-6, 1.5, xtol=1e-15, rtol=1e-15, maxiter=500)
        pil = sorted(pil + [(t, x, m, lab, "irs")], key=lambda p: p[0])
    l_t, l_df = [p[0] for p in pil], [p[1] for p in pil]
    df_l = lambda d: _loglin_df(l_t, l_df, tt(d))  # noqa: E731

    def zero(curve, d):
        t = tt(pd.Timestamp(d).date())
        return -math.log((df_o if curve == "ois" else df_l)(pd.Timestamp(d).date())) / t

    ois_curve = pd.DataFrame({"pillar_date": [m.isoformat() for m in oq.maturity], "time_years": o_t,
                              "discount_factor": o_df,
                              "zero_rate_continuous": [-math.log(d) / t for d, t in zip(o_df, o_t)]})
    lrows = []
    for t, d, m, lab, kind in pil:
        m3 = add_months_eom(m, 3)
        lrows.append({"pillar_date": m.isoformat(), "time_years": t, "libor_discount_factor": d,
                      "zero_rate_continuous": -math.log(d) / t,
                      "forward_3m": (d / df_l(m3) - 1.0) / _dc(m, m3, float_basis)})
    rep = []
    for m, r, lab in zip(oq.maturity, oq.rate, oq.label):
        mr = (1.0 / df_o(m) - 1.0) / _dc(mm0, m, mm_basis)
        rep.append({"instrument_type": "ois", "tenor_label": lab, "input_par_rate": r, "model_par_rate": mr})
    for m, r, lab in zip(dq.maturity, dq.rate, dq.label):
        mr = (1.0 / df_l(m) - 1.0) / _dc(mm0, m, mm_basis)
        rep.append({"instrument_type": "deposit", "tenor_label": lab, "input_par_rate": r, "model_par_rate": mr})
    for m, r, lab in zip(iq.maturity, iq.rate, iq.label):
        rep.append({"instrument_type": "irs_3m", "tenor_label": lab, "input_par_rate": r,
                    "model_par_rate": par(spot, m, pil)})
    rep = pd.DataFrame(rep)
    rep["residual_bps"] = (rep["model_par_rate"] - rep["input_par_rate"]) * 1e4
    return {"spot_date": spot, "valuation_date": val, "df_ois": df_o, "df_libor": df_l,
            "par_rate": lambda eff, mat: par(pd.Timestamp(eff).date(), pd.Timestamp(mat).date(), pil),
            "zero": zero, "ois_curve": ois_curve, "libor_curve": pd.DataFrame(lrows), "reprice": rep,
            "libor_pillars": pil, "time": tt, "fixed_basis": fixed_basis, "float_basis": float_basis,
            "fixed_months": fixed_months, "float_months": float_months,
            # know77: guessed names ('ois_df', 'libor_df', 'repriced' - six KeyErrors in one run) as aliases
            "ois_df": ois_curve, "libor_df": pd.DataFrame(lrows), "repriced": rep,
            "ois_discount_curve": ois_curve, "libor_forward_curve": pd.DataFrame(lrows), "repriced_quotes": rep}


def dual_curve_swap_value(curves, effective, maturity, notional, fixed_rate, pay_receive="pay_fixed",
                          bump_bps=None):
    """Fixed vs IBOR swap on a dual_curve_bootstrap result. fixed_leg_pv = N K sum tau_fix DF_OIS(t_i);
    float_leg_pv = N sum tau_flt F_j DF_OIS(s_j) (s_0 = effective, F from the projection curve);
    npv = float - fixed for "pay_fixed", fixed - float for "receive_fixed". bump_bps: every PROJECTION
    pillar zero rate +bump (DF_i e^{-bump t_i}, OIS unchanged) and pv01 = |npv_bumped - npv|.
    Returns dict{npv, fixed_leg_pv, float_leg_pv, pv01 (if bump_bps), pay_receive}."""
    eff, mat = pd.Timestamp(effective).date(), pd.Timestamp(maturity).date()
    if not isinstance(curves, dict) or "time" not in curves or "libor_pillars" not in curves:
        raise TypeError("dual_curve_swap_value needs the WHOLE dict returned by dual_curve_bootstrap(...) "
                        "(not a dict of curve DataFrames); for every deliverable of an OIS + IBOR build call "
                        "dual_curve_deliverables(conventions, ois_quotes, deposit_quotes, irs_quotes, swap)")
    tt, df_o = curves["time"], curves["df_ois"]

    def legs(pillars):
        ts, ds = [p[0] for p in pillars], [p[1] for p in pillars]
        dl = lambda d: _loglin_df(ts, ds, tt(d))  # noqa: E731
        fx, prev = 0.0, eff
        for d in backward_schedule(eff, mat, curves["fixed_months"]):
            fx += _dc(prev, d, curves["fixed_basis"]) * df_o(d)
            prev = d
        fl, prev = 0.0, eff
        for d in backward_schedule(eff, mat, curves["float_months"]):
            tau = _dc(prev, d, curves["float_basis"])
            fl += tau * (dl(prev) / dl(d) - 1.0) / tau * df_o(d)
            prev = d
        fixed, flt = notional * fixed_rate * fx, notional * fl
        return fixed, flt, (flt - fixed) if pay_receive == "pay_fixed" else (fixed - flt)

    fixed, flt, npv = legs(curves["libor_pillars"])
    out = {"npv": npv, "fixed_leg_pv": fixed, "float_leg_pv": flt, "pay_receive": pay_receive}
    if bump_bps is not None:
        b = float(bump_bps) * 1e-4
        bumped = [(p[0], p[1] * math.exp(-b * p[0])) + tuple(p[2:]) for p in curves["libor_pillars"]]
        out["pv01"] = abs(legs(bumped)[2] - npv)
    return out


# ============================================================================= event study
def event_study_tables(stock_prices, market_prices, events, *, est_window, event_window, min_est_obs,
                       date_col="date", ticker_col="ticker", event_col="event_date", market_col=None,
                       log_returns=True, corrado_denominator="span", kp_form="kp2010", scar_var="sum_c",
                       rho_min_overlap=10):
    """Market-model event study from PRICE tables in one call (positions in the merged trading-day index).
    stock_prices: WIDE frame (date column + one price column per ticker); market_prices: date + one price
    column (or market_col). Returns r = ln(P_t/P_{t-1}) (simple if log_returns=False), first row dropped,
    stock and market inner-joined on date. Event p = position of event_date (exact match, else skipped);
    estimation rows p+est_window[0] .. p+est_window[1] (inclusive, clipped at the series start), event rows
    p+event_window[0] .. p+event_window[1]; valid iff n_est >= min_est_obs and every event row exists.
    OLS by numpy.linalg.lstsq; r_squared = 1 - SSR/SST; AR = r - (alpha + beta r_m).
    Corrado: corrado_rank_test on relative-day-indexed Series (estimation labels est..., event labels),
    denominator=corrado_denominator ("span" counts every relative day from the first estimation day to the
    last event day, gap days included). Patell SAR / SCAR (patell_standardized, scar_var) -> BMP t (full
    window and day 0) -> Kolari-Pynnonen with rho_bar = average_residual_correlation(date-indexed residuals,
    rho_min_overlap). Returns dict{params, ar, car, caar (dict), tests (dict)} with ISO date strings."""
    from .event_study import (average_residual_correlation, bmp_test, corrado_rank_test, kolari_pynnonen_adjust,
                              market_model, patell_standardized)
    sp = stock_prices.copy()
    if date_col in sp.columns:
        sp = sp.set_index(date_col)
    sp.index = pd.to_datetime(sp.index)
    sp = sp.sort_index().apply(pd.to_numeric, errors="coerce")
    mp = market_prices.copy()
    if isinstance(mp, pd.DataFrame):
        if date_col in mp.columns:
            mp = mp.set_index(date_col)
        mp = mp[market_col] if market_col else mp.iloc[:, 0]
    mp.index = pd.to_datetime(mp.index)
    mp = pd.to_numeric(mp.sort_index(), errors="coerce")
    tr = (lambda x: np.log(x / x.shift(1))) if log_returns else (lambda x: x / x.shift(1) - 1.0)
    R = tr(sp).iloc[1:].join(tr(mp).iloc[1:].rename("__mkt__"), how="inner")
    idx = R.index
    e0, e1 = est_window
    w0, w1 = event_window
    P, A, C, est_l, evt_l, res_l, scars, sar0 = [], [], [], [], [], [], [], []
    for _, ev in events.iterrows():
        tk, ed = ev[ticker_col], pd.Timestamp(ev[event_col])
        if tk not in R.columns or ed not in idx:
            continue
        p = int(idx.get_loc(ed))
        lo, hi = max(0, p + e0), p + e1
        if p + w0 < 0 or p + w1 >= len(R) or hi < lo:
            continue
        est = R.iloc[lo:hi + 1][[tk, "__mkt__"]].dropna()
        evt = R.iloc[p + w0:p + w1 + 1][[tk, "__mkt__"]]
        if len(est) < min_est_obs or evt.isna().any().any() or len(evt) != (w1 - w0 + 1):
            continue
        y, x = est[tk].to_numpy(), est["__mkt__"].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ np.array([alpha, beta])
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(res @ res) / sst if sst > 0 else float("nan")
        P.append({"ticker": tk, "event_date": _iso(ed), "alpha": float(alpha), "beta": float(beta),
                  "r_squared": r2, "n_obs": int(len(est))})
        ar = evt[tk].to_numpy() - (alpha + beta * evt["__mkt__"].to_numpy())
        rel = np.arange(w0, w1 + 1)
        for k, (d, a) in enumerate(zip(evt.index, ar)):
            A.append({"ticker": tk, "event_date": _iso(ed), "relative_day": int(rel[k]), "date": _iso(d), "ar": float(a)})
        C.append({"ticker": tk, "event_date": _iso(ed), "car_full": float(ar.sum()),
                  "car_pre": float(ar[rel < 0].sum()), "car_event": float(ar[rel == 0].sum()),
                  "car_post": float(ar[rel > 0].sum())})
        est_rel = np.array([idx.get_loc(d) for d in est.index]) - p
        est_l.append(pd.Series(res, index=est_rel.astype(int)))
        evt_l.append(pd.Series(ar, index=rel.astype(int)))
        res_l.append(pd.Series(res, index=pd.DatetimeIndex(est.index)))
        mm = market_model(est[tk], est["__mkt__"])
        ps = patell_standardized(ar, evt["__mkt__"].to_numpy(), mm, scar_var=scar_var)
        scars.append(ps["scar"])
        sar0.append(float(np.asarray(ps["sar"])[int(np.where(rel == 0)[0][0])]))
    params, ar_df, car_df = pd.DataFrame(P), pd.DataFrame(A), pd.DataFrame(C)
    n = len(params)
    caar = {}
    if n:
        by = ar_df.groupby("relative_day")["ar"].mean().sort_index()
        caar = {"num_events": int(n), "caar_by_day": {str(int(k)): float(v) for k, v in by.items()},
                "caar_full": float(by.sum()), "caar_pre": float(by[by.index < 0].sum()),
                "caar_post": float(by[by.index > 0].sum())}
    tests = {}
    if n >= 2:
        ck = corrado_rank_test(est_l, evt_l, denominator=corrado_denominator)
        rho = average_residual_correlation(res_l, min_overlap=rho_min_overlap)
        tf, t0 = bmp_test(scars), bmp_test(sar0)
        rho_use = rho if np.isfinite(rho) else 0.0
        tests = {"corrado_z_full": ck["z_window"], "corrado_z_day0": ck["z_day0"], "kp_rho_bar": rho,
                 "bmp_t_full": tf, "bmp_t_day0": t0,
                 "kp_t_full": kolari_pynnonen_adjust(tf, rho_use, n, form=kp_form),
                 "kp_t_day0": kolari_pynnonen_adjust(t0, rho_use, n, form=kp_form)}
    return {"params": params, "ar": ar_df, "car": car_df, "caar": caar, "tests": tests}


# ============================================================================= text / regression (FOMC style)
def normalize_tokens(text):
    """Lower-case, every character outside a-z -> one space, split on whitespace -> list of tokens."""
    return re.sub(r"[^a-z]", " ", str(text).lower()).split()


def lexicon_counts(texts, positive_terms, negative_terms):
    """-> DataFrame[token_count, positive_count, negative_count] for a LIST of texts (call once, not per row).
    Exact-token lexicon counts after normalize_tokens (terms normalised the same way; multi-token terms
    are ignored). Returns DataFrame[token_count, positive_count, negative_count] (one row per text)."""
    pos = {t[0] for t in (normalize_tokens(x) for x in positive_terms) if len(t) == 1}
    neg = {t[0] for t in (normalize_tokens(x) for x in negative_terms) if len(t) == 1}
    rows = []
    for tx in texts:
        tok = normalize_tokens(tx)
        rows.append({"token_count": len(tok), "positive_count": sum(t in pos for t in tok),
                     "negative_count": sum(t in neg for t in tok)})
    return pd.DataFrame(rows)


def rolling_prior_mean(values, k):
    """For each i: prior = values[max(0, i-k):i] (the current value EXCLUDED). Returns (prior_count array,
    prior_mean array) with mean 0.0 when no prior value exists."""
    v = np.asarray(values, dtype=float)
    cnt = np.array([min(i, k) for i in range(len(v))], dtype=int)
    mean = np.array([v[max(0, i - k):i].mean() if i else 0.0 for i in range(len(v))])
    return cnt, mean


def tfidf_novelty(texts, k=4, min_df=2):
    """TF-IDF novelty vs the mean vector of the previous up-to-k documents (current excluded).
    Tokens = normalize_tokens; vocabulary = tokens in >= min_df documents; idf = ln((1+N)/(1+df)) + 1;
    tf = count / ALL tokens of the document; novelty_i = 1 - cos(v_i, mean(v_{i-k..i-1})), 0.0 for the
    first document (1.0 if a norm is zero). Returns ndarray."""
    toks = [normalize_tokens(t) if isinstance(t, str) else list(t) for t in texts]
    N = len(toks)
    dfc = {}
    for tk in toks:
        for w in set(tk):
            dfc[w] = dfc.get(w, 0) + 1
    vocab = sorted(w for w, c in dfc.items() if c >= min_df)
    pos = {w: j for j, w in enumerate(vocab)}
    idf = np.array([math.log((1 + N) / (1 + dfc[w])) + 1 for w in vocab])
    V = np.zeros((N, len(vocab)))
    for i, tk in enumerate(toks):
        for w in tk:
            j = pos.get(w)
            if j is not None:
                V[i, j] += 1
        if tk:
            V[i] /= len(tk)
    V *= idf[None, :]
    out = np.zeros(N)
    for i in range(1, N):
        m = V[max(0, i - k):i].mean(axis=0)
        a, b = np.linalg.norm(V[i]), np.linalg.norm(m)
        out[i] = 1.0 - float(V[i] @ m) / (a * b) if a > 0 and b > 0 else 1.0
    return out


def next_valid_index(valid_dates, targets):
    """Row position of the first valid date ON OR AFTER each target (searchsorted 'left'; len if none)."""
    d = pd.to_datetime(pd.Series(valid_dates)).to_numpy()
    t = pd.to_datetime(pd.Series(targets)).to_numpy()
    return np.searchsorted(d, t, side="left")


def _design(X, add_const):
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    return np.column_stack([np.ones(len(X)), X]) if add_const else X


def ols_hac_fit(y, X, *, lags=None, add_const=True):
    """OLS with homoskedastic and (optionally) Newey-West HAC errors, formulas written out.
    beta = (X'X)^-1 X'y; sigma2 = SSE/(n-p); se = sqrt(diag(sigma2 (X'X)^-1)); r2 = 1 - SSE/SST;
    durbin_watson = sum (e_t - e_{t-1})^2 / sum e_t^2. HAC (lags=L): S = sum e_t^2 x_t x_t' + sum_{l=1..L}
    (1 - l/(L+1)) sum_{t>=l} e_t e_{t-l} (x_t x_{t-l}' + x_{t-l} x_t'); cov = (X'X)^-1 S (X'X)^-1 (NOT
    divided by n). Returns dict{beta, se, t, hac_se, hac_t, resid, fitted, sse, r2, dw, n, p, xtx_inv}."""
    yv = np.asarray(y, dtype=float)
    Xd = _design(X, add_const)
    n, p = Xd.shape
    xtx_inv = np.linalg.inv(Xd.T @ Xd)
    beta = xtx_inv @ Xd.T @ yv
    fit = Xd @ beta
    e = yv - fit
    sse = float(e @ e)
    sst = float(((yv - yv.mean()) ** 2).sum())
    se = np.sqrt(np.diag(sse / (n - p) * xtx_inv))
    out = {"beta": beta, "se": se, "t": beta / se, "resid": e, "fitted": fit, "sse": sse,
           "r2": 1 - sse / sst if sst > 0 else float("nan"), "dw": float((np.diff(e) ** 2).sum() / sse),
           "n": n, "p": p, "xtx_inv": xtx_inv}
    if lags is not None:
        S = (Xd * (e ** 2)[:, None]).T @ Xd
        for lag in range(1, int(lags) + 1):
            wl = 1 - lag / (lags + 1)
            G = (Xd[lag:] * (e[lag:] * e[:-lag])[:, None]).T @ Xd[:-lag]
            S += wl * (G + G.T)
        cov = xtx_inv @ S @ xtx_inv
        out["hac_se"] = np.sqrt(np.diag(cov))
        out["hac_t"] = beta / out["hac_se"]
    return out


def ols_influence(y, X, *, add_const=True):
    """Leverage h_i = x_i'(X'X)^-1 x_i; mse = SSE/(n-p); cooks_i = e_i^2/(p mse) h_i/(1-h_i)^2;
    mse_(i) = (SSE - e_i^2/(1-h_i))/(n-p-1); externally studentised r_i = e_i/sqrt(mse_(i)(1-h_i));
    high_leverage = h_i > 2p/n. Returns dict of arrays + counts."""
    f = ols_hac_fit(y, X, add_const=add_const)
    Xd = _design(X, add_const)
    n, p = Xd.shape
    h = np.einsum("ij,jk,ik->i", Xd, f["xtx_inv"], Xd)
    e, sse = f["resid"], f["sse"]
    mse = sse / (n - p)
    cooks = e ** 2 / (p * mse) * h / (1 - h) ** 2
    msei = (sse - e ** 2 / (1 - h)) / (n - p - 1)
    ext = e / np.sqrt(msei * (1 - h))
    hl = h > 2 * p / n
    return {"leverage": h, "cooks_distance": cooks, "externally_studentized_resid": ext,
            "high_leverage": hl, "high_leverage_count": int(hl.sum()), "mse": mse, "n": n, "p": p}


def leave_one_out_coef(y, X, j, *, add_const=True):
    """beta_j refitted without observation i, for every i (index j counts the constant as 0 when
    add_const). Returns ndarray [n]."""
    yv = np.asarray(y, dtype=float)
    Xd = _design(X, add_const)
    out = np.empty(len(yv))
    for i in range(len(yv)):
        m = np.ones(len(yv), dtype=bool)
        m[i] = False
        out[i] = np.linalg.lstsq(Xd[m], yv[m], rcond=None)[0][j]
    return out


def wild_sign_flip_pvalue(y, X, j, *, add_const=True, chunk=65536):
    """Deterministic wild sign-flip p-value for coefficient j: fitted and residuals of the full OLS; all
    2^(n-1) sign vectors with the FIRST sign fixed at +1; y* = fitted + s*resid; refit (one matmul per
    chunk); p = share of vectors with |beta*_j| >= |beta_j| (1e-12 relative tolerance)."""
    f = ols_hac_fit(y, X, add_const=add_const)
    Xd = _design(X, add_const)
    n = Xd.shape[0]
    P = (f["xtx_inv"] @ Xd.T)[j]
    bj = abs(float(f["beta"][j]))
    tol = 1e-12 * max(1.0, bj)
    total, hit = 0, 0
    combos = itertools.product((1.0, -1.0), repeat=n - 1)
    while True:
        block = list(itertools.islice(combos, chunk))
        if not block:
            break
        S = np.column_stack([np.ones(len(block)), np.asarray(block)])
        Y = f["fitted"][None, :] + S * f["resid"][None, :]
        b = Y @ P
        hit += int((np.abs(b) >= bj - tol).sum())
        total += len(block)
    return hit / total


def ar1_fit_window(x, j_start, j_end):
    """OLS x[j] = c + phi x[j-1] over j = j_start..j_end INCLUSIVE (row positions). Returns (c, phi)."""
    v = np.asarray(x, dtype=float)
    j = np.arange(int(j_start), int(j_end) + 1)
    X = np.column_stack([np.ones(len(j)), v[j - 1]])
    c, phi = np.linalg.lstsq(X, v[j], rcond=None)[0]
    return float(c), float(phi)


# ============================================================================= 13F holdings
_SEC_DATE = "%d-%b-%Y"


def _sec_iso(s):
    s = pd.Series(s).astype(str).str.strip()
    a = pd.to_datetime(s, format=_SEC_DATE, errors="coerce")
    b = pd.to_datetime(s.where(a.isna()), errors="coerce")
    return a.fillna(b).dt.strftime("%Y-%m-%d")


def sec13f_load(data_dir=None, *, names=("SUBMISSION.tsv", "COVERPAGE.tsv", "SUMMARYPAGE.tsv", "INFOTABLE.tsv")):
    """Read the four 13F flat-file tables as TEXT (sep='\\t', dtype=str, keep_default_na=False, stripped).
    data_dir None -> resolve each bare name like read_table. Returns dict{SUBMISSION, COVERPAGE,
    SUMMARYPAGE, INFOTABLE} of DataFrames. Note COVERPAGE has no CIK: the CIK lives in SUBMISSION."""
    out = {}
    for nm in names:
        src = pathlib.Path(data_dir) / nm if data_dir else _path_or_buffer(nm)
        f = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
        f.columns = [c.strip() for c in f.columns]
        out[pathlib.Path(nm).stem.upper()] = f.apply(lambda c: c.str.strip())
    return out


def sec13f_filings(tables, *, value_units="auto", dollar_cutover="2023-01-03"):
    """One row per ACCESSION_NUMBER joining SUBMISSION + COVERPAGE + SUMMARYPAGE (+ INFOTABLE totals).
    Dates -> 'YYYY-MM-DD' strings (SEC 'DD-MON-YYYY'); CIK -> 10-digit zero-padded string; AMENDMENTNO ->
    int (blank = 0); TABLEENTRYTOTAL int, TABLEVALUETOTAL float. VALUE unit factor: value_units="auto"
    -> x1000 for filings dated before dollar_cutover (SEC reported VALUE in thousands until then), x1
    after; "dollars" -> x1; "thousands" -> x1000. Adds is_holdings_report (13F-HR or 13F-HR/A with
    REPORTTYPE '13F HOLDINGS REPORT'), period_consistent (PERIODOFREPORT == REPORTCALENDARORQUARTER),
    rows_in_filing, value_total_in_filing (factor applied), summarypage_* totals and diffs (filing - page)."""
    sub = tables["SUBMISSION"].copy()
    cov = tables["COVERPAGE"].copy()
    summ = tables["SUMMARYPAGE"].copy()
    info = tables["INFOTABLE"]
    sub["FILING_DATE"] = _sec_iso(sub["FILING_DATE"]).values
    sub["PERIODOFREPORT"] = _sec_iso(sub["PERIODOFREPORT"]).values
    sub["CIK"] = sub["CIK"].astype(str).str.strip().str.zfill(10)
    keep = [c for c in ("ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTNO", "AMENDMENTTYPE",
                        "REPORTTYPE", "FILINGMANAGER_NAME") if c in cov.columns]
    cov = cov[keep].copy()
    cov["REPORTCALENDARORQUARTER"] = _sec_iso(cov["REPORTCALENDARORQUARTER"]).values
    cov["AMENDMENTNO"] = pd.to_numeric(cov["AMENDMENTNO"].replace("", "0"), errors="coerce").fillna(0).astype(int)
    summ = summ[["ACCESSION_NUMBER", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"]].copy()
    f = sub.merge(cov, on="ACCESSION_NUMBER", how="left").merge(summ, on="ACCESSION_NUMBER", how="left")
    if value_units == "auto":
        f["value_factor"] = np.where(pd.to_datetime(f["FILING_DATE"]) < pd.Timestamp(dollar_cutover), 1000.0, 1.0)
    else:
        f["value_factor"] = {"dollars": 1.0, "thousands": 1000.0}[value_units]
    v = pd.to_numeric(info["VALUE"].str.replace(",", ""), errors="coerce").fillna(0.0)
    tot = pd.DataFrame({"ACCESSION_NUMBER": info["ACCESSION_NUMBER"], "v": v}).groupby("ACCESSION_NUMBER")["v"] \
        .agg(["size", "sum"]).rename(columns={"size": "rows_in_filing", "sum": "raw_value_total"}).reset_index()
    f = f.merge(tot, on="ACCESSION_NUMBER", how="left")
    f["rows_in_filing"] = f["rows_in_filing"].fillna(0).astype(int)
    f["value_total_in_filing"] = f["raw_value_total"].fillna(0.0) * f["value_factor"]
    f["summarypage_entry_total"] = pd.to_numeric(f["TABLEENTRYTOTAL"], errors="coerce").fillna(0).astype(int)
    f["summarypage_value_total"] = pd.to_numeric(f["TABLEVALUETOTAL"], errors="coerce").fillna(0.0) * f["value_factor"]
    f["summarypage_row_diff"] = f["rows_in_filing"] - f["summarypage_entry_total"]
    f["summarypage_value_diff"] = f["value_total_in_filing"] - f["summarypage_value_total"]
    f["is_holdings_report"] = f["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"]) & \
        (f["REPORTTYPE"].str.upper() == "13F HOLDINGS REPORT")
    f["period_consistent"] = f["PERIODOFREPORT"] == f["REPORTCALENDARORQUARTER"]
    return f.drop(columns=["raw_value_total", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"])


def sec13f_resolve(filings, target_managers, target_periods, *, require_period_consistent=True):
    """Amendment resolution per (CIK, period_end). Eligible = is_holdings_report (& period_consistent) and
    in the target universe (CIK zero-padded strings, period_end ISO strings). Order: FILING_DATE, AMENDMENTNO,
    ACCESSION_NUMBER. action: original (ISAMENDMENT != 'Y') -> replace_initial; AMENDMENTTYPE RESTATEMENT
    -> replace_restatement (book := this filing); NEW HOLDINGS -> append_new_holdings (book += filing).
    target_managers needs CIK (+ manager_order, manager_label); target_periods needs period_end
    (+ period_order). Returns (resolution DataFrame one row per applied filing, dict{(CIK, period_end):
    [effective accession numbers]})."""
    tm = target_managers.copy()
    tm["CIK"] = tm["CIK"].astype(str).str.strip().str.zfill(10)
    if "manager_order" not in tm:
        tm["manager_order"] = np.arange(1, len(tm) + 1)
    tp = target_periods.copy()
    tp["period_end"] = pd.to_datetime(tp["period_end"]).dt.strftime("%Y-%m-%d")
    if "period_order" not in tp:
        tp["period_order"] = np.arange(1, len(tp) + 1)
    f = filings[filings["is_holdings_report"]].copy()
    if require_period_consistent:
        f = f[f["period_consistent"]]
    f = f.rename(columns={"PERIODOFREPORT": "period_end"})
    f = f.merge(tm, on="CIK", how="inner").merge(tp, on="period_end", how="inner")
    f = f.sort_values(["manager_order", "period_order", "FILING_DATE", "AMENDMENTNO", "ACCESSION_NUMBER"])
    acts, eff = [], {}
    for _, r in f.iterrows():
        key = (r["CIK"], r["period_end"])
        typ = str(r.get("AMENDMENTTYPE", "")).upper()
        if str(r.get("ISAMENDMENT", "")).upper() != "Y" or not typ:
            a = "replace_initial"
            eff[key] = [r["ACCESSION_NUMBER"]]
        elif typ.startswith("RESTATEMENT"):
            a = "replace_restatement"
            eff[key] = [r["ACCESSION_NUMBER"]]
        elif "NEW HOLDINGS" in typ:
            a = "append_new_holdings"
            eff.setdefault(key, []).append(r["ACCESSION_NUMBER"])
        else:
            a = "replace_restatement"
            eff[key] = [r["ACCESSION_NUMBER"]]
        acts.append(a)
    f["action"] = acts
    return f.reset_index(drop=True), eff


def sec13f_clean_rows(info_rows, *, value_factor=1.0):
    """-> ALL rows + column reject_reason ('' = kept): aggregate only d[d.reject_reason == ''].
    Reject reasons in FIXED priority, each raw row counted once: blank_cusip (CUSIP empty), nonsh
    (SSHPRNAMTTYPE != 'SH'), putcall (PUTCALL not blank), nonpositive (VALUE <= 0 or SSHPRNAMT <= 0 or
    non-numeric); '' = kept. Adds numeric VALUE (x value_factor), SSHPRNAMT and VOTING_* columns.
    Returns a copy with column reject_reason."""
    d = info_rows.copy()
    for c in ("VALUE", "SSHPRNAMT", "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE"):
        if c in d:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    d["VALUE"] = d["VALUE"] * value_factor
    cus = d["CUSIP"].astype(str).str.strip()
    typ = d.get("SSHPRNAMTTYPE", pd.Series("SH", index=d.index)).astype(str).str.strip().str.upper()
    pc = d.get("PUTCALL", pd.Series("", index=d.index)).fillna("").astype(str).str.strip()
    nonpos = ~((d["VALUE"] > 0) & (d["SSHPRNAMT"] > 0))
    reason = np.select([cus.eq("") | cus.str.lower().eq("nan"), typ.ne("SH"), pc.ne(""), nonpos],
                       ["blank_cusip", "nonsh", "putcall", "nonpositive"], default="")
    d["reject_reason"] = reason
    return d


def holdings_book(kept_rows, *, key="CUSIP", value="VALUE",
                  sum_cols=("VALUE", "SSHPRNAMT", "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE"),
                  label_cols=("NAMEOFISSUER", "TITLEOFCLASS")):
    """Aggregate duplicate positions by key: sum_cols summed; label_cols taken from the row with the largest
    value (canonical label); weight = value / total. Sorted by weight desc then key. Returns DataFrame."""
    d = kept_rows.copy()
    sc = [c for c in sum_cols if c in d]
    g = d.groupby(key, sort=True)[sc].sum()
    lab = d.sort_values([value, key], ascending=[False, True]).drop_duplicates(key).set_index(key)
    for c in label_cols:
        if c in lab:
            g[c] = lab[c]
    g["weight"] = g[value] / g[value].sum()
    return g.reset_index().sort_values(["weight", key], ascending=[False, True]).reset_index(drop=True)


def book_concentration(weights, labels=None):
    """weights: Series indexed by name (or array). hhi = sum w^2; effective_num_names = 1/hhi; top by
    largest weight (ties -> smallest key). Returns dict{num_positions, hhi, effective_num_names, top_key,
    top_weight, top_label}."""
    w = pd.Series(weights).astype(float)
    order = sorted(w.index, key=lambda k: (-w[k], str(k)))
    hh = float((w ** 2).sum())
    top = order[0] if order else None
    return {"num_positions": int(len(w)), "hhi": hh, "effective_num_names": 1.0 / hh if hh > 0 else float("nan"),
            "top_key": top, "top_weight": float(w[top]) if top is not None else float("nan"),
            "top_label": (labels.get(top) if isinstance(labels, dict) else
                          (labels[top] if labels is not None else None))}


def book_turnover(w_prev, w_curr):
    """One-way turnover = 0.5 sum_{union} |w_curr - w_prev| (absent = 0); new = in curr only, exited = in prev
    only, common = both. Returns dict{turnover, new_names_count, exited_names_count, common_names_count}."""
    a, b = pd.Series(w_prev, dtype=float), pd.Series(w_curr, dtype=float)
    u = a.index.union(b.index)
    t = 0.5 * float((b.reindex(u, fill_value=0.0) - a.reindex(u, fill_value=0.0)).abs().sum())
    sa, sb = set(a.index[a > 0]), set(b.index[b > 0])
    return {"turnover": t, "new_names_count": len(sb - sa), "exited_names_count": len(sa - sb),
            "common_names_count": len(sa & sb)}


def book_overlap(w_i, w_j):
    """Weighted overlap = sum_c min(w_ic, w_jc) over common names; returns (overlap, common_count)."""
    a, b = pd.Series(w_i, dtype=float), pd.Series(w_j, dtype=float)
    c = a.index.intersection(b.index)
    return float(np.minimum(a[c], b[c]).sum()), int(len(c))


# ============================================================================= workbook / money parsing
_MULT = {"k": 1e3, "m": 1e6, "mm": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}


def parse_money(x, *, percent_as_fraction=True):
    """Display string -> float: '$653,201.93 M' -> 653201930000.0, '110.20 M' -> 110200000.0,
    '-$986,681.12' -> -986681.12, '(1,234)' -> -1234, '0.30%' -> 0.003 (percent_as_fraction), '-'/''/None ->
    nan. Suffixes K, M/MM, B/BN, T (any case). Numbers pass through."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace("−", "-").replace("\xa0", " ")
    if s in ("", "-", "--", "—", "n/a", "N/A", "nan", "NaN"):
        return float("nan")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if s.startswith("-"):
        neg, s = True, s[1:]
    pct = s.endswith("%")
    s = s.rstrip("%").replace("$", "").replace(",", "").strip()
    m = re.fullmatch(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([A-Za-z]{0,2})", s)
    if not m:
        return float("nan")
    v = float(m.group(1))
    suf = m.group(2).lower()
    if suf:
        if suf not in _MULT:
            return float("nan")
        v *= _MULT[suf]
    if pct and percent_as_fraction:
        v /= 100.0
    return -v if neg else v


def parse_as_of_date(text):
    """First date in a text -> 'YYYY-MM-DD'. Accepts '01-Apr-2026', 'Apr 01 2026', 'April 1, 2026',
    '2026-04-01', '04/01/2026'. Returns None if no date."""
    s = str(text)
    pats = [(r"\d{4}-\d{2}-\d{2}", None), (r"\d{1,2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"),
            (r"[A-Za-z]{3,9}\.? \d{1,2},? \d{4}", None), (r"\d{1,2}/\d{1,2}/\d{4}", "%m/%d/%Y")]
    for p, fmt in pats:
        m = re.search(p, s)
        if m:
            try:
                return pd.to_datetime(m.group(0).replace(".", ""), format=fmt).strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                continue
    return None


def read_banner_table(src, header_keywords, *, sheet_name=0, stop_at_blank=True):
    """Table inside a workbook/CSV that has banner rows above the header (fund name, 'As of ...').
    Header row = first row containing EVERY header keyword as a cell (case-insensitive, stripped). Blank
    header cells take the text of the row below (two-row headers), else 'col_<j>'. Data = rows below the
    header whose FIRST keyword column is non-blank, up to the first fully blank row (stop_at_blank).
    Banner rows above become meta {label without ':' : first non-blank value to its right}.
    Accepts a path, bare file name or bytes. Returns (DataFrame, meta dict). Values stay as read (use
    parse_money for '$1.2 M' / '0.30%' strings)."""
    obj = _path_or_buffer(src)
    name = str(getattr(obj, "name", obj)).lower()
    if name.endswith((".csv", ".txt")):
        import csv
        text = _read_text(obj if not isinstance(obj, pathlib.Path) else str(obj))
        rows = list(csv.reader(_io.StringIO(text)))
        width = max((len(x) for x in rows), default=0)   # ragged banner rows: pad, never a ParserError
        raw = pd.DataFrame([x + [""] * (width - len(x)) for x in rows], dtype=object).replace("", np.nan)
    else:
        raw = pd.read_excel(obj, sheet_name=sheet_name, header=None, dtype=object)
    kws = [k.strip().lower() for k in header_keywords]
    hdr = None
    for i in range(len(raw)):
        cells = {str(v).strip().lower() for v in raw.iloc[i].tolist() if pd.notna(v)}
        if all(k in cells for k in kws):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"no row contains all of {header_keywords}")
    meta = {}
    for i in range(hdr):
        vals = [v for v in raw.iloc[i].tolist() if pd.notna(v) and str(v).strip()]
        if len(vals) >= 2:
            meta[str(vals[0]).strip().rstrip(":").strip()] = vals[1]
    cols = []
    for j, v in enumerate(raw.iloc[hdr].tolist()):
        if pd.notna(v) and str(v).strip():
            cols.append(str(v).strip())
        elif hdr + 1 < len(raw) and pd.notna(raw.iat[hdr + 1, j]):
            cols.append(str(raw.iat[hdr + 1, j]).strip())
        else:
            cols.append(f"col_{j}")
    seen = {}
    for j, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[j] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    body = raw.iloc[hdr + 1:].copy()
    body.columns = cols
    if stop_at_blank:
        blank = body.isna().all(axis=1).to_numpy()
        first_data = np.argmax(~blank) if (~blank).any() else 0
        after = np.where(blank[first_data:])[0]
        if len(after):
            body = body.iloc[:first_data + after[0]]
    key = cols[[c.lower() for c in cols].index(kws[0])]
    body = body[body[key].notna() & body[key].astype(str).str.strip().ne("")]
    return body.reset_index(drop=True), meta


def top_n_by_weight(df, n, weight_col, key_col):
    """Top n rows by weight descending, ties broken by key ascending (stable, deterministic)."""
    d = df.copy()
    d["__w"] = pd.to_numeric(d[weight_col], errors="coerce")
    return d.sort_values(["__w", key_col], ascending=[False, True], kind="mergesort").head(int(n)) \
        .drop(columns="__w").reset_index(drop=True)


# ============================================================================= Form 4
def _txt(el, path):
    x = el.find(path)
    return x.text.strip() if x is not None and x.text is not None else None


def _fids(el, path):
    x = el.find(path)
    return [f.get("id") for f in x.iter("footnoteId")] if x is not None else []


def form4_price_range(text):
    """'... at prices ranging from $37.20 to $38.19, inclusive' -> (37.2, 38.19); (None, None) if absent."""
    m = re.search(r"ranging from \$?\s*([\d,]*\.?\d+)\s*(?:to|-)\s*\$?\s*([\d,]*\.?\d+)", str(text or ""), re.I)
    if not m:
        return None, None
    return float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))


def form4_entity_name(ownership_form, nature_text, footnote_texts=(), *, direct_label="Direct"):
    """Ownership bucket label. Direct ('D') with no specific nature -> direct_label. A nature that is a
    pointer ('See footnote', blank, '(1)') on an indirect row is resolved from its footnote(s): the name
    after the last 'Trustee of', else after 'held by'; leading 'the' and trailing ' u/a/d ...', ', over
    which ...', 'dated ...' clauses removed. Any other nature text is returned stripped (e.g. 'By JRM Rev.
    Trust')."""
    nat = (nature_text or "").strip()
    pointer = (not nat) or bool(re.fullmatch(r"(see\s+footnotes?.*|\(?\d+\)?|footnote.*)", nat, re.I))
    if str(ownership_form).upper().startswith("D") and (pointer or nat.lower().startswith("direct")):
        return direct_label
    if not pointer:
        return nat
    for fn in footnote_texts or ():
        t = re.sub(r"\s+", " ", str(fn))
        m = list(re.finditer(r"Trustee of (?:the )?(.+?)(?: u/a/d| dated|, over which|, of which|\.(?:\s|$)|$)", t, re.I))
        if not m:
            m = list(re.finditer(r"held by (?:the )?(.+?)(?: u/a/d| dated|, over which|, of which|, a |\.(?:\s|$)|$)", t, re.I))
        if m:
            name = re.sub(r"^(the)\s+", "", m[-1].group(1).strip(), flags=re.I).strip(" ,.;\"'“”")
            if name:
                return name
    return nat or ("Indirect" if not str(ownership_form).upper().startswith("D") else direct_label)


def parse_form4(src, *, direct_label="Direct"):
    """SEC Form 4 XML -> dict. Keys: issuer_name, issuer_cik, ticker, owner_name, period_of_report,
    aff10b5one (bool), footnotes {id: text}, transactions (DataFrame, non-derivative table, ALL codes:
    xml_index 1.., security_title, transaction_date, transaction_code, shares, price_per_share,
    acquired_disposed_code, shares_following_transaction, ownership_form D/I, nature, code_footnotes,
    price_footnotes, nature_footnotes, ownership_footnotes (know79: every footnote on the shares-owned-
    following and ownership-nature cells, Table I cols 5-7), ownership_entity, price_band_low/high from the price footnote,
    rule_10b5_1 (a footnote of the row mentions 10b5-1, or the document flag)), holdings (DataFrame of
    nonDerivativeHolding: security_title, shares_owned, ownership_form, nature, nature_footnotes,
    ownership_entity) and derivative_transactions. Filter transaction_code == 'S' yourself for sales."""
    import xml.etree.ElementTree as ET
    from .helpers79_4 import form4_ownership_footnote_ids
    root = ET.fromstring(_read_bytes(src))
    fn = {f.get("id"): re.sub(r"\s+", " ", "".join(f.itertext())).strip() for f in root.iter("footnote")}
    flag = (root.findtext("aff10b5One") or "").strip().lower() in ("1", "true", "y", "yes")
    tx, hd, dv = [], [], []
    for k, el in enumerate(root.iter("nonDerivativeTransaction"), 1):
        nat_f = _fids(el, "ownershipNature/natureOfOwnership")
        form = _txt(el, "ownershipNature/directOrIndirectOwnership/value") or ""
        nat = _txt(el, "ownershipNature/natureOfOwnership/value")
        code_f = _fids(el, "transactionCoding")
        price_f = _fids(el, "transactionAmounts/transactionPricePerShare")
        lo = hi = None
        for i in price_f:
            lo, hi = form4_price_range(fn.get(i))
            if lo is not None:
                break
        all_f = [f.get("id") for f in el.iter("footnoteId")]
        tx.append({"xml_index": k, "security_title": _txt(el, "securityTitle/value"),
                   "transaction_date": _txt(el, "transactionDate/value"),
                   "transaction_code": _txt(el, "transactionCoding/transactionCode"),
                   "shares": float(_txt(el, "transactionAmounts/transactionShares/value") or "nan"),
                   "price_per_share": float(_txt(el, "transactionAmounts/transactionPricePerShare/value") or "nan"),
                   "acquired_disposed_code": _txt(el, "transactionAmounts/transactionAcquiredDisposedCode/value"),
                   "shares_following_transaction": float(
                       _txt(el, "postTransactionAmounts/sharesOwnedFollowingTransaction/value") or "nan"),
                   "ownership_form": form, "nature": nat, "code_footnotes": code_f, "price_footnotes": price_f,
                   "nature_footnotes": nat_f,
                   "ownership_footnotes": form4_ownership_footnote_ids(el),   # know79: cols 5-7 of Table I
                   "ownership_entity": form4_entity_name(form, nat, [fn.get(i, "") for i in nat_f],
                                                         direct_label=direct_label),
                   "price_band_low": lo, "price_band_high": hi,
                   "rule_10b5_1": flag or any("10b5-1" in fn.get(i, "") for i in all_f)})
    for el in root.iter("nonDerivativeHolding"):
        nat_f = _fids(el, "ownershipNature/natureOfOwnership")
        form = _txt(el, "ownershipNature/directOrIndirectOwnership/value") or ""
        nat = _txt(el, "ownershipNature/natureOfOwnership/value")
        hd.append({"security_title": _txt(el, "securityTitle/value"),
                   "shares_owned": float(_txt(el, "postTransactionAmounts/sharesOwnedFollowingTransaction/value") or "nan"),
                   "ownership_form": form, "nature": nat, "nature_footnotes": nat_f,
                   "ownership_footnotes": form4_ownership_footnote_ids(el),
                   "ownership_entity": form4_entity_name(form, nat, [fn.get(i, "") for i in nat_f],
                                                         direct_label=direct_label)})
    for el in root.iter("derivativeTransaction"):
        dv.append({"security_title": _txt(el, "securityTitle/value"),
                   "transaction_date": _txt(el, "transactionDate/value"),
                   "transaction_code": _txt(el, "transactionCoding/transactionCode"),
                   "shares": float(_txt(el, "transactionAmounts/transactionShares/value") or "nan")})
    return {"issuer_name": root.findtext("issuer/issuerName"), "issuer_cik": root.findtext("issuer/issuerCik"),
            "ticker": root.findtext("issuer/issuerTradingSymbol"),
            "owner_name": root.findtext("reportingOwner/reportingOwnerId/rptOwnerName"),
            "period_of_report": root.findtext("periodOfReport"), "aff10b5one": flag, "footnotes": fn,
            "transactions": pd.DataFrame(tx), "holdings": pd.DataFrame(hd),
            "derivative_transactions": pd.DataFrame(dv)}


# ============================================================================= iXBRL 10-K
def ixbrl_main_html(src):
    """(member name, html text) of the main inline-XBRL document in an SEC '-xbrl.zip' bundle: the
    largest .htm/.html member. Accepts a path, bare file name or bytes (never the text of the zip)."""
    data = _read_bytes(src)
    with zipfile.ZipFile(_io.BytesIO(data)) as z:
        cands = [i for i in z.infolist() if i.filename.lower().endswith((".htm", ".html"))]
        best = max(cands, key=lambda i: i.file_size)
        raw = z.read(best.filename)
    for enc in ("utf-8", "cp1252"):
        try:
            return best.filename, raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return best.filename, raw.decode("utf-8", errors="replace")


class _IxParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.facts, self.stack, self.ctx, self.cur_ctx, self.cur_tag = [], [], {}, None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("ix:nonfraction", "ix:nonnumeric"):
            self.stack.append({"tag": tag, "attrs": a, "text": []})
        elif tag == "xbrli:context":
            self.cur_ctx = a.get("id")
            self.ctx[self.cur_ctx] = {"start": None, "end": None, "instant": None, "dims": []}
        elif self.cur_ctx and tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant", "xbrldi:explicitmember"):
            self.cur_tag = (tag, a)

    def handle_endtag(self, tag):
        if tag in ("ix:nonfraction", "ix:nonnumeric") and self.stack:
            el = self.stack.pop()
            txt = re.sub(r"\s+", " ", "".join(el["text"])).strip()
            txt = re.sub(r"\s+([,.;:%)])", r"\1", txt).replace("( ", "(")
            self.facts.append((el["tag"], el["attrs"], txt))
        elif tag == "xbrli:context":
            self.cur_ctx = None
        if tag in ("xbrli:startdate", "xbrli:enddate", "xbrli:instant", "xbrldi:explicitmember"):
            self.cur_tag = None
        if tag in ("p", "div", "td", "tr", "br") and self.stack:
            self.stack[-1]["text"].append(" ")

    def handle_data(self, data):
        for el in self.stack:
            el["text"].append(data)
        if self.cur_ctx and self.cur_tag:
            tag, a = self.cur_tag
            c = self.ctx[self.cur_ctx]
            v = data.strip()
            if not v:
                return
            if tag == "xbrli:startdate":
                c["start"] = v
            elif tag == "xbrli:enddate":
                c["end"] = v
            elif tag == "xbrli:instant":
                c["instant"] = v
            else:
                c["dims"].append(f"{a.get('dimension')}={v}")


def _num(txt, fmt=""):
    t = str(txt).strip()
    if t in ("", "-", "—", "–") or "zerodash" in fmt or "fixed-zero" in fmt:
        return 0.0 if (t in ("-", "—", "–") or "zero" in fmt) else float("nan")
    t = t.replace(",", "").replace("$", "").replace(" ", "").strip("()")   # magnitude only; sign= gives the sign
    if "numcommadecimal" in fmt.replace("-", ""):
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def ixbrl_facts(html_text):
    """Every ix:nonFraction / ix:nonNumeric fact of an inline-XBRL document -> DataFrame[name, local_name,
    contextref, unitref, decimals, scale, sign, format, text, value, period_start, period_end, instant,
    dims, is_dimensional]. value: numeric facts = number(text) x 10^scale, negated when sign='-' (the
    sign attribute, not the display parentheses); nonNumeric = text (booleans stay 'true'/'false').
    dims = 'axis=member;...' from the fact's context (empty = the entity-wide fact)."""
    p = _IxParser()
    p.feed(html_text)
    rows = []
    for tag, a, txt in p.facts:
        name = a.get("name", "")
        fmt = a.get("format", "") or ""
        c = p.ctx.get(a.get("contextref"), {"start": None, "end": None, "instant": None, "dims": []})
        if tag == "ix:nonfraction":
            sc = int(a.get("scale", "0") or 0)
            v = _num(txt, fmt) * (10 ** sc)
            if a.get("sign") == "-":
                v = -v
        else:
            sc, v = 0, txt
        rows.append({"name": name, "local_name": name.split(":")[-1], "contextref": a.get("contextref"),
                     "unitref": a.get("unitref"), "decimals": a.get("decimals"), "scale": sc, "sign": a.get("sign"),
                     "format": fmt, "text": txt, "value": v, "period_start": c["start"], "period_end": c["end"],
                     "instant": c["instant"], "dims": ";".join(c["dims"]), "is_dimensional": bool(c["dims"])})
    return pd.DataFrame(rows)


def ixbrl_fact(facts, name, *, member=None, latest=True):
    """Value of one concept from ixbrl_facts: name matched with or without prefix (case-insensitive);
    member=None keeps non-dimensional facts only, else facts whose dims contain `member`; latest=True picks
    the latest period end (or instant). Returns the value (None if absent)."""
    nm = name.lower()
    f = facts[(facts["name"].str.lower() == nm) | (facts["local_name"].str.lower() == nm.split(":")[-1])]
    f = f[~f["is_dimensional"]] if member is None else f[f["dims"].str.contains(member, case=False, regex=False)]
    if f.empty:
        return None
    key = f["period_end"].fillna(f["instant"]).fillna("")
    f = f.assign(__k=key).sort_values("__k", ascending=not latest)
    return f.iloc[0]["value"]


def ixbrl_bool(text):
    """Inline-XBRL check-box / boolean text -> bool: '☒', 'x', 'true', 'yes', '1' -> True; '☐',
    'false', 'no', '0', '' -> False (cover-page flags such as dei:DocumentAnnualReport)."""
    t = str(text or "").strip().lower()
    return t in ("☒", "☑", "x", "true", "yes", "y", "1")


def ixbrl_by_context(facts, key_name, value_name):
    """Pair two concepts reported in the SAME context: {text of key_name fact: value of value_name fact},
    e.g. ixbrl_by_context(facts, 'TradingSymbol', 'Security12bTitle') -> {symbol: security title}."""
    k = facts[facts["local_name"].str.lower() == key_name.split(":")[-1].lower()]
    v = facts[facts["local_name"].str.lower() == value_name.split(":")[-1].lower()]
    m = k.merge(v, on="contextref", suffixes=("_k", "_v"))
    return {str(a).strip(): b for a, b in zip(m["text_k"], m["value_v"])}


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.cur, self.row, self.cell, self.depth = [], None, None, None, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.cur = []
        elif tag == "tr" and self.cur is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None and self.cur is not None:
            if any(c for c in self.row):
                self.cur.append(self.row)
            self.row = None
        elif tag == "table":
            self.depth -= 1
            if self.depth == 0 and self.cur is not None:
                if self.cur:
                    self.tables.append(self.cur)
                self.cur = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def html_tables(html_text):
    """All top-level HTML tables -> list of tables, each a list of rows, each a list of cell strings
    (whitespace collapsed, non-breaking spaces as spaces; empty rows dropped; empty cells KEPT so column
    positions survive). Use table_row_numbers / find_table_rows on the result."""
    p = _TableParser()
    p.feed(html_text.replace("\xa0", " "))
    return p.tables


_NUM_RE = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?%?")


def table_row_numbers(cells):
    """Numbers in a table row, left to right, from cells split the SEC way ('$', '1,234', ')' or '%' in
    separate cells): parentheses -> negative, '$' and ',' dropped, '%' kept as the same number (a 3.5%
    cell -> 3.5). The label cell(s) before the first number are skipped. Returns list[float]."""
    s = " ".join(c for c in cells if c is not None)
    out = []
    for m in _NUM_RE.finditer(s):
        t = m.group(0)
        neg = "(" in t and ")" in t or t.strip().startswith("-")
        v = float(re.sub(r"[^\d.]", "", t))
        out.append(-v if neg else v)
    return out


def find_table_rows(tables, label_pattern, *, first_cell_only=True):
    """Rows whose label matches label_pattern (regex, case-insensitive): searched in the first non-empty
    cell (first_cell_only) or anywhere in the row. Returns list of (table_index, row_index, cells)."""
    rx = re.compile(label_pattern, re.I)
    hits = []
    for ti, tb in enumerate(tables):
        for ri, row in enumerate(tb):
            lab = next((c for c in row if c), "") if first_cell_only else " ".join(row)
            if rx.search(lab):
                hits.append((ti, ri, row))
    return hits


# ============================================================================= 8-K text
def filing_plain_text(src):
    """HTML filing (path, bare name, bytes or html string) -> plain text: script/style dropped, block tags
    -> newlines, entities unescaped, non-breaking spaces and curly quotes normalised, spaces collapsed."""
    t = _read_text(src)
    t = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", t)
    t = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|h\d|li|table)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t).replace("\xa0", " ")
    t = t.translate({0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"', 0x2013: "-", 0x2014: "-", 0x2212: "-"})
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def date_of_report(text):
    """'Date of Report (Date of earliest event reported): May 22, 2024' -> '2024-05-22' (None if absent)."""
    m = re.search(r"Date of report[^:]{0,80}:?\s*([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
                  str(text), re.I)
    return parse_as_of_date(m.group(1)) if m else None


def item_numbers(text):
    """Sorted unique SEC item numbers 'N.NN' from every 'Item N.NN' token (case-insensitive)."""
    return sorted(set(re.findall(r"\bItem\s+(\d{1,2}\.\d{2})", str(text), re.I)), key=lambda s: tuple(map(int, s.split("."))))


def money_musd(amount, unit=None):
    """'$1,725,000,000' -> 1725.0; ('28.0', 'billion') -> 28000.0; ('107.8', 'million') -> 107.8 (USD millions)."""
    v = float(str(amount).replace("$", "").replace(",", "").strip())
    u = (unit or "").lower()
    if u.startswith("b"):
        return v * 1000.0
    if u.startswith("m"):
        return v
    if u.startswith("t") and u != "":
        return v * 1e6
    return v / 1e6


_FWD = re.compile(r"(expect|forecast|outlook|guidance|anticipat|project|is expected to be)", re.I)


def guidance_midpoint_musd(text):
    """-> DICT {low, high, midpoint, sentence} (use ['midpoint'], a float in USD millions) or None.
    Forward revenue guidance midpoint in USD millions from the first FORWARD-LOOKING sentence (expect,
    forecast, outlook, guidance...) that states a range: '$13.3 billion to $14.3 billion' -> 13800.0;
    '$28.0 billion, plus or minus 2%' -> 28000.0 (the stated value is the midpoint). Reported (past)
    figures are skipped. Returns dict{low, high, midpoint, sentence} or None."""
    t = re.sub(r"\s+", " ", str(text))
    for sent in re.split(r"(?<=[.;▪•])\s+|\s[▪•]\s", t):
        if not _FWD.search(sent):
            continue
        m = re.search(r"\$\s*(\d[\d,]*\.?\d*)\s*(billion|million)?\s*(?:to|-|and)\s*\$\s*(\d[\d,]*\.?\d*)\s*(billion|million)", sent, re.I)
        if m:
            lo = money_musd(m.group(1), m.group(2) or m.group(4))
            hi = money_musd(m.group(3), m.group(4))
            return {"low": lo, "high": hi, "midpoint": (lo + hi) / 2, "sentence": sent.strip()}
        m = re.search(r"\$\s*(\d[\d,]*\.?\d*)\s*(billion|million),?\s*plus or minus\s*\d+\.?\d*\s*%", sent, re.I)
        if m:
            v = money_musd(m.group(1), m.group(2))
            return {"low": v, "high": v, "midpoint": v, "sentence": sent.strip()}
    return None


def debt_initial_principal_musd(text):
    """-> DICT {total, option_included, initial} (use ['initial'], USD millions) or None.
    Initial principal (USD millions) of a note issuance BEFORE the over-allotment/greenshoe option:
    total = first '$X ... principal amount' that is issued/offered/sold; option = '$Y principal amount'
    in the same text tied to an option (initial purchasers' / underwriters' option, over-allotment,
    greenshoe) that is INCLUDED ('include(s) $Y ... pursuant to the exercise') -> initial = total - Y;
    an option merely granted ('up to an additional $Y') is not subtracted. Returns dict{total, option_included,
    initial} or None."""
    t = re.sub(r"\s+", " ", str(text))
    amt = r"\$\s*(\d[\d,]*\.?\d*)\s*(billion|million)?"
    m = re.search(r"(?:issued|issue|offering of|sale of|sold|priced)\s+(?:an aggregate of\s+)?" + amt +
                  r"\s+(?:in\s+)?(?:aggregate\s+)?principal amount", t, re.I)
    if not m:
        m = re.search(amt + r"\s+(?:in\s+)?aggregate principal amount", t, re.I)
    if not m:
        return None
    total = money_musd(m.group(1), m.group(2))
    inc = re.search(r"includ\w*\s+" + amt + r"\s+(?:in\s+)?(?:aggregate\s+)?principal amount[^.]{0,200}?"
                    r"(?:option|over-?allotment|greenshoe)", t, re.I)
    opt = money_musd(inc.group(1), inc.group(2)) if inc else 0.0
    return {"total": total, "option_included": opt, "initial": total - opt}


def departing_officer(text, titles=("Chief Executive Officer", "Chief Financial Officer")):
    """Title of the officer who resigned/stepped down/retired: first match of '(Resignation|Departure|
    Retirement) of <title>' or a sentence where <title> appears with resign/step down/retire/depart.
    Returns the title string from `titles` or None."""
    t = re.sub(r"\s+", " ", str(text))
    alt = "|".join(re.escape(x) for x in titles)
    m = re.search(r"(?:Resignation|Departure|Retirement|Separation) of (?:the |our |its )?(?:Company'?s? )?(" + alt + ")", t, re.I)
    if m:
        return next(x for x in titles if x.lower() == m.group(1).lower())
    for sent in re.split(r"(?<=\.)\s+", t):
        if re.search(r"resign|step(?:ped|s)? down|retire|depart", sent, re.I):
            m = re.search("(" + alt + ")", sent, re.I)
            if m and not re.search(r"appoint|named|interim", sent[:m.start()], re.I):
                return next(x for x in titles if x.lower() == m.group(1).lower())
    return None


# ============================================================================= prediction-market locked edges
def clean_venue_snapshots(snapshots, pairs, *, approved_col="review_status", approved_value="approved",
                          ts_col="snapshot_ts_utc", pair_col="pair_id", venue_col="venue",
                          seq_col="source_sequence", status_col="status", active_value="active",
                          price_cols=("canon_yes_price", "canon_no_price"),
                          cutoff_cols=("kalshi_close_ts_utc", "polymarket_close_ts_utc", "resolution_ts_utc")):
    """-> 3-TUPLE (clean DataFrame, counts dict, approved pairs): clean, cnt, ok = clean_venue_snapshots(...).
    Cleaning pipeline for cross-venue snapshot panels, counting every stage. Order: parse UTC, stable sort
    by ts; keep approved pairs; dedupe (pair, venue, ts) keeping the largest source_sequence; keep status ==
    active; drop rows whose prices are missing or outside (0, 1); drop rows at or after the pair cutoff =
    min(cutoff_cols). Returns (clean DataFrame, counts dict{num_raw_snapshots, num_valid_pairs,
    num_rejected_pairs, num_rejected_pair_rows_filtered, num_approved_rows, num_duplicate_snapshot_groups,
    num_deduped_snapshots, num_inactive_rows_filtered, num_active_snapshots, num_invalid_price_rows_filtered,
    num_cutoff_excluded_rows, num_clean_snapshots}, approved pairs with a 'cutoff' column)."""
    s = snapshots.copy()
    s[ts_col] = pd.to_datetime(s[ts_col], utc=True)
    s = s.sort_values(ts_col, kind="mergesort").reset_index(drop=True)
    pr = pairs.copy()
    for c in cutoff_cols:
        if c in pr:
            pr[c] = pd.to_datetime(pr[c], utc=True)
    ok = pr[pr[approved_col].astype(str).str.strip().str.lower() == approved_value]
    ok = ok.assign(cutoff=ok[[c for c in cutoff_cols if c in ok]].min(axis=1))
    cnt = {"num_raw_snapshots": int(len(s)), "num_valid_pairs": int(len(ok)),
           "num_rejected_pairs": int(len(pr) - len(ok))}
    a = s[s[pair_col].isin(ok[pair_col])]
    cnt["num_rejected_pair_rows_filtered"] = int(len(s) - len(a))
    cnt["num_approved_rows"] = int(len(a))
    gsize = a.groupby([pair_col, venue_col, ts_col])[seq_col].transform("size")
    cnt["num_duplicate_snapshot_groups"] = int(a[gsize > 1].groupby([pair_col, venue_col, ts_col]).ngroups)
    a = a.assign(__seq=pd.to_numeric(a[seq_col], errors="coerce")) \
        .sort_values([pair_col, venue_col, ts_col, "__seq"], kind="mergesort") \
        .drop_duplicates([pair_col, venue_col, ts_col], keep="last").drop(columns="__seq")
    a = a.sort_values(ts_col, kind="mergesort")
    cnt["num_deduped_snapshots"] = int(len(a))
    act = a[a[status_col].astype(str).str.strip().str.lower() == active_value]
    cnt["num_inactive_rows_filtered"] = int(len(a) - len(act))
    cnt["num_active_snapshots"] = int(len(act))
    px = act[list(price_cols)].apply(pd.to_numeric, errors="coerce")
    good = ((px > 0) & (px < 1)).all(axis=1)
    cnt["num_invalid_price_rows_filtered"] = int((~good).sum())
    v = act[good].merge(ok[[pair_col, "cutoff"]], on=pair_col, how="left")
    keep = v[ts_col] < v["cutoff"]
    cnt["num_cutoff_excluded_rows"] = int((~keep).sum())
    clean = v[keep].drop(columns="cutoff").reset_index(drop=True)
    for c in price_cols:
        clean[c] = pd.to_numeric(clean[c])
    cnt["num_clean_snapshots"] = int(len(clean))
    return clean, cnt, ok.reset_index(drop=True)


def synchronized_snapshots(clean, venue_a, venue_b, *, ts_col="snapshot_ts_utc", pair_col="pair_id",
                           venue_col="venue", price_cols=("canon_yes_price", "canon_no_price")):
    """(pair, ts) groups with EXACTLY one venue_a row and one venue_b row -> wide frame
    [pair_id, ts, a_yes, a_no, b_yes, b_no] sorted by ts then pair."""
    c = clean[clean[venue_col].isin([venue_a, venue_b])]
    n = c.groupby([pair_col, ts_col])[venue_col].agg(lambda x: (int((x == venue_a).sum()), int((x == venue_b).sum())))
    good = n[n == (1, 1)].index
    c = c.set_index([pair_col, ts_col]).loc[good].reset_index()
    y, no = price_cols
    A = c[c[venue_col] == venue_a][[pair_col, ts_col, y, no]].rename(columns={y: "a_yes", no: "a_no"})
    B = c[c[venue_col] == venue_b][[pair_col, ts_col, y, no]].rename(columns={y: "b_yes", no: "b_no"})
    return A.merge(B, on=[pair_col, ts_col]).sort_values([ts_col, pair_col]).reset_index(drop=True)


def locked_edge_backtest(sync, pairs, *, venue_a, venue_b, fee_bps_a, fee_bps_b, min_edge_bps, max_qty,
                         capital_base, funding_rate, annualization=252, pair_col="pair_id",
                         ts_col="snapshot_ts_utc", resolution_col="resolution_ts_utc", outcome_col="resolution_outcome",
                         funding_reduces_trading_cash=False, tie_tol=1e-12, first_return="drop"):
    """Paired YES/NO locked-edge strategy on synchronized snapshots (synchronized_snapshots output).
    Directions: A = buy YES on venue_a + NO on venue_b (cost a_yes + b_no); B = YES on venue_b + NO on venue_a.
    Per-unit fee = fee_bps/1e4 x leg price on each leg's venue; gross_edge = 1 - gross_cost; net_edge = gross
    - fees; entry_cost_per_unit = gross_cost + fees. Direction strings 'BUY_YES_<A>_BUY_NO_<B>' etc.; best =
    higher net edge, tie -> lexicographically smaller string. Event loop over timestamps: release resolved
    trades (resolution_ts <= ts: cash += qty) first; evaluate pairs without a trade; below min_edge ->
    BELOW_THRESHOLD; executable candidates allocated by (net edge desc, entry cost asc, pair asc), qty =
    min(max_qty, floor(cash/cost)); qty < 1 -> UNAFFORDABLE; else OPENED, cash -= qty*cost (one trade per
    pair). Daily (UTC dates first trade -> last resolution): after the day's events, reserved = sum of entry
    costs of unresolved trades, funding = reserved*funding_rate/annualization deducted from cash; equity =
    cash + sum unresolved qty; daily_pnl = diff (first 0.0); cumulative_return = equity/capital_base.
    FUNDING vs ALLOCATION CASH (know77): by default (funding_reduces_trading_cash=False) the allocation
    pass sizes qty on TRADING cash (capital - entry costs + released payoffs) and the running funding
    total is subtracted only in equity (equity = trading cash - cumulative funding + unresolved qty) -
    the two-pass reference reading; True deducts each day's funding from the cash later trades use.
    Direction ties / below-threshold use |a - b| <= tie_tol (float noise). Return metrics use
    daily_return = equity_t/equity_{t-1} - 1 (first_return='drop': n days -> n-1 returns; 'zero' keeps a
    leading 0.0; 'base' measures day 1 against capital_base). metrics['num_candidate_signals']
    = candidates that CLEAR min_edge (OPENED + UNAFFORDABLE); below-threshold rows are counted apart.
    Returns dict{trades, candidates, daily DataFrames, metrics dict}."""
    pr = pairs.set_index(pair_col)
    dA = f"BUY_YES_{venue_a}_BUY_NO_{venue_b}"
    dB = f"BUY_YES_{venue_b}_BUY_NO_{venue_a}"
    fa, fb = fee_bps_a / 1e4, fee_bps_b / 1e4
    s = sync.copy()
    s["gross_cost_a"] = s["a_yes"] + s["b_no"]
    s["fee_a"] = fa * s["a_yes"] + fb * s["b_no"]
    s["gross_cost_b"] = s["b_yes"] + s["a_no"]
    s["fee_b"] = fb * s["b_yes"] + fa * s["a_no"]
    s["net_edge_a"] = 1 - s["gross_cost_a"] - s["fee_a"]
    s["net_edge_b"] = 1 - s["gross_cost_b"] - s["fee_b"]
    thr = min_edge_bps / 1e4
    cash = float(capital_base)
    open_tr, trades, cands = [], [], []
    traded = set()
    days = {}
    cum_fund = [0.0]

    def release(upto):
        nonlocal cash
        for tr in open_tr:
            if not tr["resolved"] and tr["exit"] <= upto:
                tr["resolved"] = True
                cash += tr["qty"]

    def day_close(day):
        nonlocal cash
        d = pd.Timestamp(day)
        d = d.tz_localize("UTC") if d.tzinfo is None else d
        release(d + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
        reserved = sum(t["qty"] * t["cost"] for t in open_tr if not t["resolved"])
        fund = reserved * funding_rate / annualization
        if funding_reduces_trading_cash:
            cash -= fund
            eq_cash = cash
        else:
            cum_fund[0] += fund
            eq_cash = cash - cum_fund[0]
        days[day] = {"equity": eq_cash + sum(t["qty"] for t in open_tr if not t["resolved"]),
                     "funding": fund, "reserved": reserved}

    if s.empty:
        return {"trades": pd.DataFrame(), "candidates": pd.DataFrame(), "daily": pd.DataFrame(), "metrics": {}}
    last_res = pr[resolution_col].max()
    all_days = pd.date_range(s[ts_col].min().normalize(), max(s[ts_col].max(), last_res).normalize(), freq="D")
    groups = {k: g for k, g in s.groupby(ts_col)}
    ts_by_day = {}
    for t in sorted(groups):
        ts_by_day.setdefault(t.normalize(), []).append(t)
    for day in all_days:
        for t in ts_by_day.get(day, []):
            release(t)
            g = groups[t]
            start_cash = cash
            execs = []
            for _, r in g.iterrows():
                pid = r[pair_col]
                if pid in traded:
                    continue
                ea, eb = r["net_edge_a"], r["net_edge_b"]
                is_tie = bool(abs(ea - eb) <= tie_tol)
                best_a = (dA < dB) if is_tie else bool(ea > eb)
                bd, ad = (dA, dB) if best_a else (dB, dA)
                row = {"pair_id": pid, "snapshot_ts_utc": t, "best_direction": bd,
                       "gross_cost_a": r["gross_cost_a"], "net_edge_a": ea, "gross_cost_b": r["gross_cost_b"],
                       "net_edge_b": eb, "net_locked_edge": ea if best_a else eb,
                       "gross_locked_edge": 1 - (r["gross_cost_a"] if best_a else r["gross_cost_b"]),
                       "entry_cost_per_unit": (r["gross_cost_a"] + r["fee_a"]) if best_a else (r["gross_cost_b"] + r["fee_b"]),
                       "yes_price": r["a_yes"] if best_a else r["b_yes"], "no_price": r["b_no"] if best_a else r["a_no"],
                       "alt_direction": ad, "alt_net_locked_edge": eb if best_a else ea, "tie": is_tie,
                       "_fee_yes": (fa * r["a_yes"]) if best_a else (fb * r["b_yes"]),
                       "_fee_no": (fb * r["b_no"]) if best_a else (fa * r["a_no"])}
                if row["net_locked_edge"] >= thr - tie_tol:
                    execs.append(row)
                else:
                    row.update(candidate_status="BELOW_THRESHOLD", selected_reason="BELOW_MIN_EDGE",
                               available_cash_before=start_cash, available_cash_after=start_cash, qty=0)
                    cands.append(row)
            execs.sort(key=lambda x: (-x["net_locked_edge"], x["entry_cost_per_unit"], str(x["pair_id"])))
            for row in execs:
                q = int(min(max_qty, math.floor(cash / row["entry_cost_per_unit"] + 1e-12)))
                before = cash
                if q < 1:
                    row.update(candidate_status="UNAFFORDABLE", selected_reason="INSUFFICIENT_CASH",
                               available_cash_before=before, available_cash_after=before, qty=0)
                else:
                    cash -= q * row["entry_cost_per_unit"]
                    row.update(candidate_status="OPENED", selected_reason="ALLOCATED_BY_EDGE_RANK",
                               available_cash_before=before, available_cash_after=cash, qty=q)
                    traded.add(row["pair_id"])
                    p = pr.loc[row["pair_id"]]
                    yes_v, no_v = (venue_a, venue_b) if row["best_direction"] == dA else (venue_b, venue_a)
                    tr = {"pair_id": row["pair_id"], "entry_ts_utc": t, "exit_ts_utc": p[resolution_col],
                          "direction": row["best_direction"], "qty": q, "buy_yes_venue": yes_v, "buy_no_venue": no_v,
                          "yes_price": row["yes_price"], "no_price": row["no_price"],
                          "fee_yes": q * row["_fee_yes"], "fee_no": q * row["_fee_no"],
                          "gross_cost": q * (row["yes_price"] + row["no_price"]),
                          "entry_cost_per_unit": row["entry_cost_per_unit"],
                          "gross_locked_edge": row["gross_locked_edge"], "net_locked_edge": row["net_locked_edge"],
                          "gross_pnl": q * row["gross_locked_edge"], "net_pnl": q * row["net_locked_edge"],
                          "resolution_outcome": p.get(outcome_col)}
                    trades.append(tr)
                    open_tr.append({"qty": q, "cost": row["entry_cost_per_unit"], "exit": p[resolution_col],
                                    "resolved": False})
                cands.append(row)
        day_close(day)
    T = pd.DataFrame(trades)
    Cd = pd.DataFrame(cands).drop(columns=["_fee_yes", "_fee_no"], errors="ignore")
    if T.empty:
        return {"trades": T, "candidates": Cd, "daily": pd.DataFrame(), "metrics": {"num_trades": 0}}
    d0 = T["entry_ts_utc"].min().normalize()
    d1 = T["exit_ts_utc"].max().normalize()
    D = pd.DataFrame([{"date": d.strftime("%Y-%m-%d"), **v} for d, v in sorted(days.items()) if d0 <= d <= d1])
    D["daily_pnl"] = D["equity"].diff().fillna(0.0)
    D["cumulative_return"] = D["equity"] / capital_base
    D["daily_funding_cost"] = D["funding"]
    D["cumulative_funding_cost"] = D["funding"].cumsum()
    # know78: daily_return = equity_t / equity_{t-1} - 1 over consecutive emitted dates -> n points give
    # n - 1 returns (first_return='drop', textbook); know77 appended a leading 0.0 (pct_change().fillna(0))
    # which diluted the mean (annualized_return off by the factor (n-1)/n)
    from .helpers78_4 import equity_curve_returns
    ret = equity_curve_returns(D["equity"].to_numpy(float), first=first_return, base=capital_base)
    eq = D["equity"].to_numpy()
    mdd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else 0.0
    vol = float(np.std(ret, ddof=1) * math.sqrt(annualization)) if len(ret) > 1 else float("nan")
    ann = float(np.mean(ret) * annualization)
    metrics = {"num_trades": int(len(T)), "gross_edge_bps_mean": float(T["gross_locked_edge"].mean() * 1e4),
               "net_edge_bps_mean": float(T["net_locked_edge"].mean() * 1e4),
               "gross_locked_pnl": float(T["gross_pnl"].sum()), "net_locked_pnl": float(T["net_pnl"].sum()),
               "total_funding_cost": float(D["cumulative_funding_cost"].iloc[-1]), "annualized_return": ann,
               "annualized_volatility": vol, "sharpe_ratio": ann / vol if vol and vol > 0 else float("nan"),
               "max_drawdown": mdd, "capacity_utilization": float((D["reserved"] / capital_base).mean()),
               "mean_qty": float(T["qty"].mean()), "mean_entry_cost_per_unit": float(T["entry_cost_per_unit"].mean()),
               "num_candidate_signals": int((Cd["candidate_status"] != "BELOW_THRESHOLD").sum()),
               "num_evaluated_snapshots": int(len(Cd)),
               "num_below_threshold": int((Cd["candidate_status"] == "BELOW_THRESHOLD").sum()),
               "num_unaffordable": int((Cd["candidate_status"] == "UNAFFORDABLE").sum()),
               "num_tie_break": int(Cd["tie"].sum())}
    D = D[["date", "equity", "daily_pnl", "cumulative_return", "daily_funding_cost", "cumulative_funding_cost",
           "reserved"]]
    return {"trades": T, "candidates": Cd, "daily": D, "metrics": metrics}
