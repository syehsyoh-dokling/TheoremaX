"""know87 round 2 (group A): Merton (1976) jump-diffusion - Poisson-mixture MLE on log returns, series pricing,
BS comparison at the total vol and the implied-vol smile.

pf86b wrote the mixture likelihood by hand with the observations on axis 0 and the jump counts on axis 1, then
took logsumexp over axis 0 - a sum over the RETURNS instead of over the jump counts. The optimiser then ran on a
meaningless objective (sigma at its lower bound, log-likelihood -386 instead of a few thousand) and every price
built on it was wrong. merton_jd_loglik fixes the orientation once (jump counts n on axis 0, observations on
axis 1, logsumexp over n: one density per return), and merton_jump_diffusion_report runs the whole chain.

Conventions are keywords with textbook defaults or switches read from the task's words, no task names, no data:
- drift of the per-period density: 'compensated' (mu - sigma^2/2 - lambda k) dt + n mu_J (Ito on the model
  dS/S = (mu - lambda k) dt + sigma dW + (J - 1) dN) | 'plain' mu dt + n mu_J (a free location, Press 1967);
- Poisson rate of the pricing series: merton_series_rate_rule(text) -> 'lambda_prime' (lambda (1 + k), Merton
  1976 / Haug 2007, consistent with r_n = r - lambda k + n ln(1 + k)/T) or 'lambda' (when the task's own
  formula weights the terms with exp(-lambda tau) (lambda tau)^n / n!);
- sample skewness / excess kurtosis: moment_bias 'adjusted' (G1/G2, pandas) | 'population' (g1/g2, scipy default).
Conventions table: theory_library/conventions.jsonl 'merton_series_poisson_rate', 'sample_skew_kurtosis'.
Self-tests: proven_work/fail72/selftest87_A.py.

API lines (one per function):
     merton_series_rate_rule(text=None) -> 'lambda' | 'lambda_prime' (Poisson rate of the Merton series weights, from the task's formula)
     merton_jd_loglik(x, sigma, lam, mu_J, sigma_J, dt, n_max, mu=None, drift="compensated") -> float sum_i log sum_{n<=n_max} Pois(n; lam dt) N(x_i; m_n, v_n) (logsumexp over n for EACH return)
     merton_jd_calibrate(x, dt, n_max, bounds, n_starts, seed=0, mu=None, drift="compensated") -> dict{sigma, lambda, mu_J, sigma_J, log_likelihood, mu, n_starts, n_success}; L-BFGS-B from uniform starts inside bounds ([(lo, hi)] x 4 in that order, or a dict by parameter name), best LL kept
     merton_series_call(S0, K, T, r, sigma, lam, mu_J, sigma_J, q=0.0, poisson_rate=None, n_series=None) -> price (K scalar or array)
     merton_jump_diffusion_report(close, dt, n_max, bounds, n_starts, r, maturities, moneyness, smile_threshold, seed=0, mu=None, drift="compensated", poisson_rate=None, n_series=None, moment_bias="adjusted", iv_method="newton") -> ResultDict{'calibration.json': dict, 'merton_prices.csv': DataFrame[T, K, moneyness, merton_price, bs_price, merton_implied_vol], 'summary.json': dict}; for name, obj in rep.items(): emit(name, obj)
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


_PRIME = r"lambda\s*'|lambda_prime|lambda\s*\*?\s*\(\s*1\s*\+\s*(?:k|kbar|kappa)\w*\s*\)"
_PLAIN = (r"exp\(\s*-\s*lambda(?:_star|\*)?\s*\*?\s*(?:tau|t)\s*\)|\(\s*lambda(?:_star|\*)?\s*\*?\s*(?:tau|t)\s*\)\s*\^\s*n"
          r"|poisson\.pmf\(\s*n\s*,\s*lambda(?:_star)?\s*\*\s*(?:tau|t)\s*\)")


def merton_series_rate_rule(text=None):
    """Poisson rate of the Merton series weights. 'lambda_prime' = lambda (1 + k) when the task names it, and when
    the task is silent (Merton 1976 eq. 16; Haug 2007 s.6.10: C = sum e^{-lambda' T} (lambda' T)^n / n! BS(r_n,
    sigma_n), lambda' = lambda (1 + k), r_n = r - lambda k + n ln(1 + k)/T); 'lambda' when the task's own formula
    weights the terms with exp(-lambda tau) (lambda tau)^n / n!."""
    t = _text(text).lower()
    if re.search(_PRIME, t):
        return "lambda_prime"
    if re.search(_PLAIN, t):
        return "lambda"
    return "lambda_prime"


def _kbar(mu_J, sigma_J):
    return math.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0


def merton_jd_loglik(x, sigma, lam, mu_J, sigma_J, *, dt, n_max, mu=None, drift="compensated"):
    """Log-likelihood of per-period log returns x under Merton jump-diffusion, truncated at n_max jumps per period:
    log p(x_i) = logsumexp_n [n ln(lam dt) - lam dt - ln n! + ln N(x_i; m_n, v_n)], m_n = (mu - sigma^2/2 - lam k) dt
    + n mu_J ('compensated'; 'plain': mu dt + n mu_J), v_n = sigma^2 dt + n sigma_J^2, k = e^{mu_J + sigma_J^2/2} - 1;
    mu defaults to mean(x)/dt (the physical drift held fixed). Jump counts on axis 0, returns on axis 1, the
    logsumexp runs over axis 0 (the jump counts) - one density per return - and the result is summed over returns.
    -inf for non-positive sigma / lam / sigma_J."""
    from scipy.special import gammaln, logsumexp
    x = np.asarray(x, float).ravel()
    if not (sigma > 0 and lam > 0 and sigma_J > 0):
        return -np.inf
    if mu is None:
        mu = float(np.mean(x)) / dt
    n = np.arange(int(n_max) + 1, dtype=float)[:, None]                  # [N+1, 1]  jump counts
    logw = n * math.log(lam * dt) - lam * dt - gammaln(n + 1.0)           # [N+1, 1]
    if drift == "plain":
        m = mu * dt + n * mu_J
    else:
        m = (mu - 0.5 * sigma ** 2 - lam * _kbar(mu_J, sigma_J)) * dt + n * mu_J
    v = sigma ** 2 * dt + n * sigma_J ** 2
    logphi = -0.5 * np.log(2.0 * np.pi * v) - 0.5 * (x[None, :] - m) ** 2 / v   # [N+1, n_obs]
    per_obs = logsumexp(logw + logphi, axis=0)                              # [n_obs]
    return float(np.sum(per_obs))


_MERTON_ORDER = ("sigma", "lambda", "mu_J", "sigma_J")
_MERTON_ALIAS = {"sigma": "sigma", "vol": "sigma", "sigma_diffusion": "sigma", "lambda": "lambda", "lam": "lambda",
                 "intensity": "lambda", "jump_intensity": "lambda", "mu_j": "mu_J", "mu_jump": "mu_J",
                 "jump_mean": "mu_J", "sigma_j": "sigma_J", "sigma_jump": "sigma_J", "jump_vol": "sigma_J",
                 "jump_std": "sigma_J"}


def _merton_bounds(bounds):
    """[(lo, hi)] x 4 in the order (sigma, lambda, mu_J, sigma_J). know92: a DICT keyed by the parameter names
    ({'sigma': [lo, hi], 'lambda': ...}, the usual aliases accepted) is put in that order; iterating the dict
    unpacked its key strings ('too many values to unpack') on every repair round. A flat [lo, hi, lo, hi, ...]
    list of 8 numbers is paired up."""
    if isinstance(bounds, dict):
        named = {}
        for k, v in bounds.items():
            key = _MERTON_ALIAS.get(str(k).strip().lower().replace(" ", "_"), _MERTON_ALIAS.get(str(k).strip()))
            if key is None and str(k) in _MERTON_ORDER:
                key = str(k)
            if key is None:
                raise ValueError(f"merton bounds: unknown parameter {k!r}; use {_MERTON_ORDER}")
            named[key] = v
        missing = [p for p in _MERTON_ORDER if p not in named]
        if missing:
            raise ValueError(f"merton bounds: missing {missing}; give (lo, hi) for each of {_MERTON_ORDER}")
        bounds = [named[p] for p in _MERTON_ORDER]
    else:
        bounds = list(bounds)
        if len(bounds) == 8 and all(np.isscalar(v) for v in bounds):
            bounds = [bounds[i:i + 2] for i in range(0, 8, 2)]
    out = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(out) != 4:
        raise ValueError(f"merton bounds: need 4 (lo, hi) pairs in the order {_MERTON_ORDER}, got {len(out)}")
    return out


def merton_jd_calibrate(x, *, dt, n_max, bounds, n_starts, seed=0, mu=None, drift="compensated", method="L-BFGS-B"):
    """MLE of (sigma, lambda, mu_J, sigma_J) by scipy.optimize.minimize(method) on -merton_jd_loglik within `bounds`
    [(lo, hi) x 4, in that order], from n_starts starting points drawn uniformly inside the bounds
    (RandomState(seed), one uniform per parameter in order); the converged result with the highest log-likelihood
    is kept (the best unconverged one only when none converged)."""
    from scipy.optimize import minimize
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if mu is None:
        mu = float(np.mean(x)) / dt
    b = _merton_bounds(bounds)

    def nll(p):
        ll = merton_jd_loglik(x, p[0], p[1], p[2], p[3], dt=dt, n_max=n_max, mu=mu, drift=drift)
        return -ll if np.isfinite(ll) else 1e10

    rs = np.random.RandomState(seed)
    best, best_any, n_ok = None, None, 0
    for _ in range(int(n_starts)):
        x0 = [rs.uniform(lo, hi) for lo, hi in b]
        res = minimize(nll, x0, method=method, bounds=b)
        if best_any is None or res.fun < best_any.fun:
            best_any = res
        if res.success:
            n_ok += 1
            if best is None or res.fun < best.fun:
                best = res
    use = best if best is not None else best_any
    s, lam, mj, sj = (float(v) for v in use.x)
    return {"sigma": s, "lambda": lam, "mu_J": mj, "sigma_J": sj, "log_likelihood": float(-use.fun),
            "mu": float(mu), "n_starts": int(n_starts), "n_success": int(n_ok)}


def merton_series_call(S0, K, T, r, sigma, lam, mu_J, sigma_J, *, q=0.0, poisson_rate=None, n_series=None,
                       tail_tol=1e-14):
    """Merton (1976) European call: sum_n w_n BS_call(S0, K, T, r_n, sigma_n, q), sigma_n^2 = sigma^2 + n sigma_J^2/T,
    r_n = r - lam k + n ln(1 + k)/T, k = e^{mu_J + sigma_J^2/2} - 1; w_n = Poisson(n; rate T) with rate = lam (1 + k)
    ('lambda_prime') or lam ('lambda'); None reads the task's formula (merton_series_rate_rule). n_series None: run
    until the Poisson tail is below tail_tol and at least ceil(3 rate T) terms (converged sum)."""
    from scipy.stats import poisson
    from .options import bs_price
    k = _kbar(mu_J, sigma_J)
    rule = poisson_rate or merton_series_rate_rule()
    rate = lam * (1.0 + k) if rule == "lambda_prime" else lam
    lt = rate * T
    if n_series is None:
        N = max(int(math.ceil(3 * lt)), 1)
        while poisson.sf(N, lt) > tail_tol and N < 10000:
            N += 1
    else:
        N = int(n_series)
    K = np.asarray(K, float)
    total = np.zeros_like(K, dtype=float)
    for n in range(N + 1):
        w = float(poisson.pmf(n, lt))
        if w == 0.0:
            continue
        sn = math.sqrt(sigma ** 2 + n * sigma_J ** 2 / T)
        rn = r - lam * k + n * math.log1p(k) / T
        total = total + w * np.asarray(bs_price(S0, K, T, rn, sn, kind="call", q=q), float)
    return float(total) if total.ndim == 0 else total


def _iv_newton(price, S, K, T, r, sigma0, q=0.0, tol=1e-10, max_iter=100):
    from scipy.stats import norm
    from .options import bs_price
    s = float(sigma0)
    for _ in range(max_iter):
        d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / (s * math.sqrt(T))
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        diff = float(bs_price(S, K, T, r, s, kind="call", q=q)) - price
        if vega < 1e-12:
            break
        step = diff / vega
        s_new = s - step
        if not np.isfinite(s_new) or s_new <= 0:
            break
        s = s_new
        if abs(step) < tol:
            return s
    from .helpers75_3 import implied_vol_vec
    return float(implied_vol_vec(price, S, K, T, r=r, q=q, kind="call"))


def merton_jump_diffusion_report(close, *, dt, n_max, bounds, n_starts, r, maturities, moneyness, smile_threshold,
                                 seed=0, mu=None, drift="compensated", poisson_rate=None, n_series=None,
                                 moment_bias="adjusted", iv_method="newton", date_col="date", close_col="close"):
    """Whole chain: closes (DataFrame with date/close, sorted by date, or an array in time order) -> log returns
    x_i = ln(c_i / c_{i-1}) -> moments (mean, std ddof=1, skewness, excess kurtosis; moment_bias 'adjusted' = G1/G2
    as pandas, 'population' = g1/g2 as scipy's default) -> merton_jd_calibrate -> S0 = last close -> for every T in
    maturities and m in moneyness, K = S0 m: merton_series_call, bs_price at sigma_total = sqrt(sigma^2 + lambda
    (mu_J^2 + sigma_J^2)), implied vol of the Merton price by Newton from sigma_total (Brent fallback) -> summary:
    smile_present = any maturity's max(iv) - min(iv) > smile_threshold, max_iv_spread, ATM price per T (key
    atm_merton_price_T<round(100 T):03d>), mean_price_ratio_merton_over_bs. Returns a bundle {file name: object}."""
    from ._rdict import ResultDict
    from .options import bs_price
    if isinstance(close, pd.DataFrame):
        df = close.copy()
        cols = {str(c).lower(): c for c in df.columns}
        if date_col.lower() in cols:
            df[cols[date_col.lower()]] = pd.to_datetime(df[cols[date_col.lower()]])
            df = df.sort_values(cols[date_col.lower()]).reset_index(drop=True)
        c = df[cols.get(close_col.lower(), close_col)].to_numpy(float)
    else:
        c = np.asarray(close, float).ravel()
    c = c[np.isfinite(c) & (c > 0)]
    x = np.diff(np.log(c))
    s = pd.Series(x)
    if moment_bias == "population":
        from scipy.stats import kurtosis, skew
        sk, ku = float(skew(x)), float(kurtosis(x, fisher=True))
    else:
        sk, ku = float(s.skew()), float(s.kurt())
    cal = merton_jd_calibrate(x, dt=dt, n_max=n_max, bounds=bounds, n_starts=n_starts, seed=seed, mu=mu, drift=drift)
    S0 = float(c[-1])
    sig, lam, mj, sj = cal["sigma"], cal["lambda"], cal["mu_J"], cal["sigma_J"]
    sigma_total = math.sqrt(sig ** 2 + lam * (mj ** 2 + sj ** 2))
    rule = poisson_rate or merton_series_rate_rule()
    rows = []
    for T in maturities:
        for m in moneyness:
            K = S0 * float(m)
            pm = merton_series_call(S0, K, float(T), r, sig, lam, mj, sj, poisson_rate=rule, n_series=n_series)
            pb = float(bs_price(S0, K, float(T), r, sigma_total, kind="call"))
            if iv_method == "newton":
                iv = _iv_newton(pm, S0, K, float(T), r, sigma_total)
            else:
                from .helpers75_3 import implied_vol_vec
                iv = float(implied_vol_vec(pm, S0, K, float(T), r=r, kind="call"))
            rows.append({"T": float(T), "K": K, "moneyness": float(m), "merton_price": float(pm), "bs_price": pb,
                         "merton_implied_vol": float(iv)})
    table = pd.DataFrame(rows).sort_values(["T", "K"]).reset_index(drop=True)
    spreads = table.groupby("T")["merton_implied_vol"].agg(lambda v: float(v.max() - v.min()))
    calib = {"n_returns": int(len(x)), "return_mean": float(np.mean(x)), "return_std": float(np.std(x, ddof=1)),
             "return_skewness": sk, "return_excess_kurtosis": ku, "sigma": sig, "lambda": lam, "mu_J": mj,
             "sigma_J": sj, "log_likelihood": cal["log_likelihood"], "S0": S0}
    summ = {"sigma_total": float(sigma_total), "n_maturities": int(table["T"].nunique()),
            "n_strikes": int(table["moneyness"].nunique()), "n_prices": int(len(table)),
            "smile_present": bool((spreads > smile_threshold).any())}
    for T in maturities:
        atm = table[np.isclose(table["T"], float(T)) & np.isclose(table["moneyness"], 1.0)]
        if len(atm):
            summ[f"atm_merton_price_T{int(round(float(T) * 100)):03d}"] = float(atm["merton_price"].iloc[0])
    summ["max_iv_spread"] = float(spreads.max())
    summ["mean_price_ratio_merton_over_bs"] = float((table["merton_price"] / table["bs_price"]).mean())
    extras = {"log_returns": x, "calibration": cal, "poisson_rate": rule, "iv_spread_by_T": spreads.to_dict()}
    return ResultDict({"calibration.json": ResultDict(calib), "merton_prices.csv": table,
                       "summary.json": ResultDict(summ)}, extras=extras)
