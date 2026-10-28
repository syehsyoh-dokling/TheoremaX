"""Extended VaR / ES toolkit: every convention is an explicit keyword with a documented default.

SIGN CONVENTION: every VaR and ES returned here is a POSITIVE loss number, as a fraction of
value (returns in, fraction out). Pass `value=` (portfolio value / notional) to get currency.
Loss L = -return. VaR_c = Q_c(L) = -Q_{1-c}(r). ES_c = E[L | L >= VaR_c] >= VaR_c.

`level` accepts 0.99, 99 (percent) or 0.01 (tail probability, anything < 0.5); all mean 99%.

Returns are in chronological order (oldest first, NEWEST LAST) wherever time matters
(EWMA, age-weighted, filtered HS). NaNs are dropped.

Already in qf_toolkit.risk (not duplicated here): var_es_historical, var_es_normal,
var_es_student_t, cornish_fisher_var, kupiec_pof, fit_garch11, dcc_correlation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import binom, chi2, gaussian_kde, norm, t as student_t

from .risk import kupiec_pof

__all__ = [
    "parametric_var_es", "ewma_var_es", "ewma_sigma_series", "ewma_vol", "historical_var_es",
    "age_weighted_var_es", "filtered_hs_var_es", "kernel_var_es", "scale_horizon",
    "portfolio_returns", "var_from_portfolio", "exceedances", "christoffersen_independence",
    "christoffersen_cc", "basel_traffic_light", "component_var", "kelly_fraction",
    "fractional_kelly", "var_constrained_fraction",
]

_EPS = 1e-12


# ----------------------------------------------------------------------------- helpers
def _level(level) -> float:
    """Normalise a confidence level: 99 -> 0.99, 0.01 -> 0.99, 0.99 -> 0.99."""
    c = float(level)
    if c > 1.0:
        c = c / 100.0
    if c < 0.5:
        c = 1.0 - c
    if not 0.5 <= c < 1.0:
        raise ValueError(f"level must be a confidence in [0.5, 1): got {level}")
    return c


def _tail(c: float) -> float:
    """Tail probability 1-c rounded to kill float noise (1-0.99 = 0.010000000000000009)."""
    return round(1.0 - c, 12)


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("no finite returns")
    return r


def _xlogy(a, b) -> float:
    """a*log(b) with 0*log(anything) = 0."""
    return 0.0 if a == 0 else a * math.log(b)


def _std_t_es_factor(q: float, nu: float, tail: float) -> float:
    """ES of a standard (unit-scale) Student-t loss beyond its quantile q."""
    return student_t.pdf(q, nu) * (nu + q * q) / ((nu - 1.0) * tail)


# ----------------------------------------------------------------------------- 1. parametric
def parametric_var_es(returns=None, *, mu=None, sigma=None, level=0.99, dist="normal", nu=None,
                      include_mean=True, horizon=1, scaling="sqrt", ddof=1, value=1.0,
                      t_scale="std"):
    """Parametric (variance-covariance) VaR and ES. Returns (VaR, ES), POSITIVE losses.

    Inputs: either `returns` (mu = sample mean, sigma = sample std with `ddof`, default 1 =
    pandas/sample; use ddof=0 for "MLE") or explicit `mu`/`sigma` (explicit values override).
    mu defaults to 0 when not derivable.

    include_mean=True : VaR = -(h*mu) + sqrt(h) * z_c * sigma  (i.e. -(mu + z_{1-c} sigma) at h=1)
    include_mean=False: VaR = sqrt(h) * z_c * sigma            (zero-mean / RiskMetrics)
    ES normal: sigma * phi(z_c)/(1-c) (same mean handling).

    dist="t": q = t_nu.ppf(c), ES factor = f_nu(q) (nu + q^2) / ((nu-1)(1-c)).
      t_scale="std"  : sigma is the STANDARD DEVIATION -> t scale = sigma*sqrt((nu-2)/nu) (nu>2)
      t_scale="scale": sigma is already the t scale parameter (e.g. from scipy t.fit).
      nu=None with `returns` -> full MLE scipy t.fit(returns) -> (nu, loc, scale): loc replaces
      mu and scale is used directly (t_scale ignored).
    horizon h, scaling="sqrt": sigma*sqrt(h), mu*h (iid). scaling="linear": sigma*h, mu*h.
    value multiplies both outputs (currency).
    """
    c = _level(level)
    tail = _tail(c)
    fitted_t = False
    if returns is not None:
        r = _clean(returns)
        if dist == "t" and nu is None:
            nu, loc, scl = student_t.fit(r)
            fitted_t = True
            if mu is None:
                mu = loc
            if sigma is None:
                sigma = scl
        if mu is None:
            mu = float(r.mean())
        if sigma is None:
            sigma = float(r.std(ddof=ddof))
    if sigma is None:
        raise ValueError("need returns or sigma")
    mu = 0.0 if (mu is None or not include_mean) else float(mu)
    sigma = float(sigma)

    if scaling == "sqrt":
        s_h, m_h = sigma * math.sqrt(horizon), mu * horizon
    elif scaling == "linear":
        s_h, m_h = sigma * horizon, mu * horizon
    else:
        raise ValueError("scaling must be 'sqrt' or 'linear'")

    if dist == "normal":
        z = norm.ppf(c)
        var = s_h * z - m_h
        es = s_h * norm.pdf(z) / tail - m_h
    elif dist == "t":
        if nu is None:
            raise ValueError("dist='t' needs nu (or returns to fit it)")
        nu = float(nu)
        if fitted_t or t_scale == "scale":
            scl = s_h
        elif t_scale == "std":
            if nu <= 2:
                raise ValueError("t_scale='std' needs nu > 2 (finite variance)")
            scl = s_h * math.sqrt((nu - 2.0) / nu)
        else:
            raise ValueError("t_scale must be 'std' or 'scale'")
        q = student_t.ppf(c, nu)
        var = scl * q - m_h
        es = scl * _std_t_es_factor(q, nu, tail) - m_h
    else:
        raise ValueError("dist must be 'normal' or 't'")
    return float(var * value), float(es * value)


# ----------------------------------------------------------------------------- 2. EWMA
def ewma_sigma_series(returns, lam=0.94, *, init="first", init_window=None):
    """RiskMetrics variance recursion. Returns (sigma2_series, sigma2_next).

    sigma2[t] = lam * sigma2[t-1] + (1-lam) * r[t-1]^2  for t = 1..n-1 (element t is the
    forecast for day t using data up to t-1); sigma2_next = forecast for the day AFTER the last
    return = lam*sigma2[n-1] + (1-lam)*r[n-1]^2.
    Seed sigma2[0] (`init`):
      "first"  : r[0]^2 (same as qf_toolkit.stats.ewma_variance)
      "sample" : sample variance (ddof=1) of the first `init_window` returns (None = all returns)
      "meansq" : mean of r^2 over the first `init_window` returns (zero-mean sample variance)
      float    : given variance.
    """
    r = _clean(returns)
    n = r.size
    if isinstance(init, str):
        w = r if init_window is None else r[:int(init_window)]
        if init == "first":
            s0 = r[0] ** 2
        elif init == "sample":
            s0 = float(np.var(w, ddof=1)) if w.size > 1 else float(w[0] ** 2)
        elif init == "meansq":
            s0 = float(np.mean(w ** 2))
        else:
            raise ValueError("init must be 'first', 'sample', 'meansq' or a float")
    else:
        s0 = float(init)
    s2 = np.empty(n)
    s2[0] = s0
    for i in range(1, n):
        s2[i] = lam * s2[i - 1] + (1.0 - lam) * r[i - 1] ** 2
    s2_next = lam * s2[-1] + (1.0 - lam) * r[-1] ** 2
    return s2, float(s2_next)


def ewma_vol(returns, lam=0.94, *, init="first", init_window=None, as_of="last",
             periods_per_year=None):
    """Current EWMA VOLATILITY (a float, not a variance, not a tuple).

    as_of="last": sigma for the last observed day (sqrt of sigma2[n-1] after updating with r[n-1],
    i.e. the variance including the last return = sigma2_next); as_of="forecast" is the same number
    read as the next-day forecast; as_of="before_last": sqrt(sigma2[n-1]) (data up to n-2).
    periods_per_year=252 annualizes (x sqrt(252)). 1 Oct: ewma_sigma_series returns VARIANCES in a
    tuple and a daily variance 1.5e-4 was shipped as an 'annualized vol'."""
    s2, s2_next = ewma_sigma_series(returns, lam, init=init, init_window=init_window)
    var = s2[-1] if as_of == "before_last" else s2_next
    vol = float(np.sqrt(var))
    return vol * float(np.sqrt(periods_per_year)) if periods_per_year else vol


def ewma_var_es(returns, lam=0.94, level=0.99, *, include_mean=False, init="first",
                init_window=None, dist="normal", nu=None, t_scale="std", horizon=1,
                value=1.0, return_series=False):
    """EWMA (RiskMetrics) one-day-ahead VaR/ES for the day after the last return.

    sigma = sqrt(sigma2_next) from ewma_sigma_series (see its seed rules; default r[0]^2).
    include_mean=False (RiskMetrics default: mu = 0); True uses the sample mean.
    Then VaR/ES as parametric_var_es(dist, nu, t_scale, horizon sqrt-scaled).
    Returns (VaR, ES) POSITIVE; with return_series=True returns
    (VaR, ES, sigma_next, sigma_series) where sigma_series[t] is the vol forecast for day t.
    """
    r = _clean(returns)
    s2, s2n = ewma_sigma_series(r, lam, init=init, init_window=init_window)
    mu = float(r.mean()) if include_mean else 0.0
    var, es = parametric_var_es(mu=mu, sigma=math.sqrt(s2n), level=level, dist=dist, nu=nu,
                                include_mean=include_mean, horizon=horizon, value=value,
                                t_scale=t_scale)
    if return_series:
        return var, es, math.sqrt(s2n), np.sqrt(s2)
    return var, es


# ----------------------------------------------------------------------------- 3. historical
def historical_var_es(returns, level=0.99, *, method="linear", on="returns", es_tail="geq",
                      value=1.0):
    """Historical-simulation VaR/ES. Returns (VaR, ES) POSITIVE.

    method (numpy quantile methods, applied per `on`):
      on="returns" (default, the usual code): q = np.quantile(r, 1-c, method); VaR = -q.
      on="losses": VaR = np.quantile(-r, c, method).
      Identical for "linear"/"midpoint"; for "lower"/"higher"/"nearest" the two differ
      ("lower" on returns == "higher" on losses = the more conservative order statistic).
      Any other numpy method ("inverted_cdf", "hazen", ...) is passed through.
    method="empirical": order-statistic convention (Jorion/Basel "k-th worst loss"):
      k = ceil(n(1-c)) (float-noise guarded), VaR = k-th largest loss =
      sorted_losses_desc[k-1]; e.g. n=250, c=0.99 -> k=3 (3rd worst), n=100 -> k=1.
      (numpy "inverted_cdf" on RETURNS at 1-c gives the same order statistic.)
    es_tail: "geq" (default) ES = mean of losses >= VaR (r <= q, non-strict, ties included);
             "gt" strict losses > VaR (falls back to VaR if empty);
             "k_worst" mean of the k = ceil(n(1-c)) worst losses (Basel-style count).
    """
    c = _level(level)
    tail = _tail(c)
    r = _clean(returns)
    n = r.size
    losses = -r
    k = max(1, math.ceil(n * tail - 1e-9))
    if method == "empirical":
        var = float(np.sort(losses)[::-1][k - 1])
    elif on == "returns":
        var = -float(np.quantile(r, tail, method=method))
    elif on == "losses":
        var = float(np.quantile(losses, c, method=method))
    else:
        raise ValueError("on must be 'returns' or 'losses'")
    if es_tail == "geq":
        sel = losses[losses >= var - _EPS * max(1.0, abs(var))]
    elif es_tail == "gt":
        sel = losses[losses > var + _EPS * max(1.0, abs(var))]
    elif es_tail == "k_worst":
        sel = np.sort(losses)[::-1][:k]
    else:
        raise ValueError("es_tail must be 'geq', 'gt' or 'k_worst'")
    es = float(sel.mean()) if sel.size else var
    return var * value, es * value


# ----------------------------------------------------------------------------- 4. BRW
def age_weighted_var_es(returns, lam=0.98, level=0.99, *, interpolate=False, es_mode="tail",
                        value=1.0):
    """Age-weighted HS (Boudoukh-Richardson-Whitelaw 1998). Returns (VaR, ES) POSITIVE.

    returns in time order, NEWEST LAST. Age i = 1 for the newest observation, n for the oldest:
      w(i) = lam^(i-1) (1-lam) / (1-lam^n)   (sums to 1; lam=1 -> equal weights 1/n).
    Sort losses descending (worst first, stable), carry weights, cumulate.
    interpolate=False (default): VaR = the first loss whose cumulative weight >= 1-c
      (lam -> 1 reproduces method="empirical", the ceil(n(1-c))-th worst loss).
    interpolate=True: linear interpolation in cumulative weight between the previous sorted
      loss (cum_{k-1} < 1-c) and the first loss reaching 1-c (cum_k >= 1-c).
    es_mode="tail" (default): ES = sum(w L)/sum(w) over the worst losses up to AND including
      the VaR observation. es_mode="exact": Acerbi-Tasche, the VaR observation only gets the
      weight needed to make the tail mass exactly 1-c, ES = that sum / (1-c).
    """
    c = _level(level)
    tail = _tail(c)
    r = _clean(returns)
    n = r.size
    age = np.arange(n, 0, -1, dtype=float)          # oldest first -> age n ... newest age 1
    if abs(1.0 - lam) < 1e-15:
        w = np.full(n, 1.0 / n)
    else:
        w = lam ** (age - 1.0) * (1.0 - lam) / (1.0 - lam ** n)
    losses = -r
    order = np.argsort(-losses, kind="stable")
    L, W = losses[order], w[order]
    cum = np.cumsum(W)
    k = int(np.searchsorted(cum, tail - 1e-12, side="left"))
    k = min(k, n - 1)
    var = float(L[k])
    if interpolate and k > 0:
        c0, c1 = cum[k - 1], cum[k]
        var = float(L[k - 1] + (tail - c0) / (c1 - c0) * (L[k] - L[k - 1]))
    if es_mode == "tail":
        es = float(np.sum(W[:k + 1] * L[:k + 1]) / np.sum(W[:k + 1]))
    elif es_mode == "exact":
        prev = cum[k - 1] if k > 0 else 0.0
        es = float((np.sum(W[:k] * L[:k]) + (tail - prev) * L[k]) / tail)
    else:
        raise ValueError("es_mode must be 'tail' or 'exact'")
    return var * value, max(es, var) * value


# ----------------------------------------------------------------------------- 5. FHS
def filtered_hs_var_es(returns, lam=0.94, level=0.99, *, init="first", init_window=None,
                       method="linear", es_tail="geq", demean=False, value=1.0,
                       return_scenarios=False):
    """Filtered HS (Hull-White 1998 / Barone-Adesi 1999) with EWMA volatility.

    sigma_t from ewma_sigma_series (sigma_t uses data up to t-1; seed `init`, default r0^2);
    z_t = (r_t - m)/sigma_t (m = sample mean if demean else 0); scenarios
    r*_t = m + z_t * sigma_{n} where sigma_{n} is the forecast for the day after the last
    return; then historical_var_es(r*, level, method, es_tail). Returns (VaR, ES) POSITIVE
    (plus the scenario array if return_scenarios=True). Observations with sigma_t = 0 dropped.
    """
    r = _clean(returns)
    s2, s2n = ewma_sigma_series(r, lam, init=init, init_window=init_window)
    m = float(r.mean()) if demean else 0.0
    sig = np.sqrt(s2)
    ok = sig > 0
    z = (r[ok] - m) / sig[ok]
    scen = m + z * math.sqrt(s2n)
    var, es = historical_var_es(scen, level, method=method, es_tail=es_tail, value=value)
    if return_scenarios:
        return var, es, scen
    return var, es


# ----------------------------------------------------------------------------- 6. KDE
def kernel_var_es(returns, level=0.99, *, bandwidth="silverman", value=1.0):
    """Gaussian kernel density VaR/ES. Returns (VaR, ES) POSITIVE.

    Density of RETURNS f(x) = (1/n) sum phi((x - r_i)/h)/h.
    bandwidth: "silverman" = scipy gaussian_kde rule h = std(ddof=1)*(3n/4)^(-1/5)
      (~1.06 sd n^-1/5); "scott" = scipy scott (1-D: std*n^-1/5); "silverman_robust" =
      0.9 min(sd, IQR/1.34) n^-1/5; a float = the ABSOLUTE bandwidth h (NOT scipy's factor).
    VaR: q solves F(q) = 1-c by brentq on the exact mixture CDF; VaR = -q.
    ES = -(1/(1-c)) int_{-inf}^{q} x f(x) dx, integrated in closed form per kernel
    (mean(r_i Phi(a_i) - h phi(a_i)), a_i = (q - r_i)/h); equals numerical quadrature.
    """
    c = _level(level)
    tail = _tail(c)
    r = _clean(returns)
    n = r.size
    sd = float(np.std(r, ddof=1))
    if isinstance(bandwidth, str):
        bw = bandwidth.lower()
        if bw in ("silverman", "scott"):
            h = float(np.sqrt(gaussian_kde(r, bw_method=bw).covariance[0, 0]))
        elif bw == "silverman_robust":
            iqr = float(np.subtract(*np.percentile(r, [75, 25])))
            spread = min(sd, iqr / 1.34) if iqr > 0 else sd
            h = 0.9 * spread * n ** (-0.2)
        else:
            raise ValueError("bandwidth: 'silverman', 'scott', 'silverman_robust' or float h")
    else:
        h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")

    def cdf(x):
        return float(np.mean(norm.cdf((x - r) / h)))

    lo, hi = r.min() - 10 * h, r.max() + 10 * h
    while cdf(lo) > tail:
        lo -= 10 * h
    q = optimize.brentq(lambda x: cdf(x) - tail, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    a = (q - r) / h
    partial = float(np.mean(r * norm.cdf(a) - h * norm.pdf(a)))   # int_{-inf}^q x f(x) dx
    var = -q
    es = -partial / tail
    return float(var * value), float(max(es, var) * value)


# ----------------------------------------------------------------------------- 7. horizon
def scale_horizon(var_1d, horizon, method="sqrt", *, mu=0.0):
    """Scale a 1-day VaR (or ES) to `horizon` days.

    method="sqrt" (Basel iid, zero drift): VaR_h = VaR_1 * sqrt(h).
      With a daily mean mu (VaR_1 = z s - mu): VaR_h = (VaR_1 + mu) sqrt(h) - h mu.
    method="linear": VaR_h = VaR_1 * h (perfectly autocorrelated / conservative).
    For simulation methods on h-day overlapping returns, build those returns instead.
    """
    h = float(horizon)
    if method == "sqrt":
        return float((var_1d + mu) * math.sqrt(h) - h * mu)
    if method == "linear":
        return float(var_1d * h)
    raise ValueError("method must be 'sqrt' or 'linear'")


def portfolio_returns(prices_df, weights, *, kind="simple"):
    """Daily-rebalanced fixed-weight portfolio returns from a wide price DataFrame.

    Index sorted ascending, rows with any NaN price dropped (listwise), non-positive prices
    dropped. weights: dict {column: w} or array in column order (not renormalised).
    kind="simple": r_p = sum w_i (P_t/P_{t-1} - 1)
    kind="log_avg": r_p = sum w_i ln(P_t/P_{t-1}) ("average of the individual log-returns")
    kind="log_portfolio": r_p = ln(sum w_i P_t/P_{t-1}) = ln(1 + simple r_p) (log of portfolio
    gross return). kind="log" is refused as ambiguous.
    Returns a pd.Series indexed by date.
    """
    df = pd.DataFrame(prices_df).sort_index()
    if isinstance(weights, dict):
        cols = list(weights.keys())
        w = np.array([weights[k] for k in cols], dtype=float)
        df = df[cols]
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != df.shape[1]:
            raise ValueError("weights length must equal number of price columns")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[(df > 0).all(axis=1)].dropna()
    simple = df.pct_change(fill_method=None).iloc[1:]
    # 1 Oct: kind="log" was read as "average of the log-returns" and silently gave ln(1 + simple
    # r_p) - a 4% bias in the mean. The two log definitions must be named explicitly.
    if kind == "simple":
        rp = simple.to_numpy() @ w
    elif kind == "log_portfolio":
        rp = np.log1p(simple.to_numpy() @ w)
    elif kind in ("log_avg", "avg_log"):
        rp = np.log(df / df.shift(1)).iloc[1:].to_numpy() @ w
    elif kind == "log":
        raise ValueError(
            "kind='log' is ambiguous. Use kind='log_avg' when the task says the portfolio return is "
            "the (weighted/simple) average of the individual log-returns: r_p = sum w_i ln(P_t/P_t-1). "
            "Use kind='log_portfolio' only for the log of the portfolio gross return: "
            "r_p = ln(sum w_i P_t/P_t-1). Quote the task sentence that decides it.")
    else:
        raise ValueError("kind must be 'simple', 'log_avg' or 'log_portfolio'")
    return pd.Series(rp, index=simple.index, name="portfolio")


def var_from_portfolio(prices_df, weights, level=0.99, *, kind="simple", method="historical",
                       value=1.0, **kwargs):
    """VaR/ES of a fixed-weight portfolio built by portfolio_returns(kind). Returns (VaR, ES).

    method: "historical" (historical_var_es; kwargs e.g. method_q -> its `method`),
    "parametric" (parametric_var_es kwargs: include_mean, dist, nu, horizon, ddof),
    "ewma", "age_weighted", "filtered", "kernel" -> the matching function here.
    """
    rp = portfolio_returns(prices_df, weights, kind=kind).to_numpy()
    if method == "historical":
        qm = kwargs.pop("method_q", kwargs.pop("quantile_method", "linear"))
        return historical_var_es(rp, level, method=qm, value=value, **kwargs)
    if method == "parametric":
        return parametric_var_es(rp, level=level, value=value, **kwargs)
    if method == "ewma":
        return ewma_var_es(rp, level=level, value=value, **kwargs)
    if method == "age_weighted":
        return age_weighted_var_es(rp, level=level, value=value, **kwargs)
    if method == "filtered":
        return filtered_hs_var_es(rp, level=level, value=value, **kwargs)
    if method == "kernel":
        return kernel_var_es(rp, level, value=value, **kwargs)
    raise ValueError("unknown method")


# ----------------------------------------------------------------------------- 8. backtests
def exceedances(returns, var_series, *, strict=True):
    """Boolean hits I_t = 1{loss_t > VaR_t} (strict, default) or >= if strict=False.

    loss_t = -returns_t; var_series are POSITIVE VaR forecasts for the SAME day t, built only
    from data before t. pandas Series are aligned on their index (inner) first.
    Returns a numpy bool array.
    """
    if isinstance(returns, pd.Series) and isinstance(var_series, pd.Series):
        a, b = returns.align(var_series, join="inner")
        r, v = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    else:
        r = np.asarray(returns, dtype=float).ravel()
        v = np.asarray(var_series, dtype=float).ravel()
        if v.size == 1:
            v = np.full_like(r, float(v[0]))
        if r.size != v.size:
            raise ValueError("returns and var_series length differ")
    # 1 Oct standard-var: returns (~0.01) were compared with VaR in DOLLARS (~20000) and 0 hits
    # shipped over 440 days. Both sides must be in the same unit.
    fin_r, fin_v = r[np.isfinite(r)], v[np.isfinite(v)]
    if fin_r.size and fin_v.size and np.max(np.abs(fin_r)) < 5 and np.median(np.abs(fin_v)) > 50 * max(
            np.max(np.abs(fin_r)), 1e-12):
        raise ValueError("exceedances: var_series (median %.4g) is in currency units but returns are "
                         "fractions (max |r| %.4g). Divide the VaR by the notional (or multiply the "
                         "returns by it) so both are in the same unit." % (np.median(np.abs(fin_v)),
                                                                          np.max(np.abs(fin_r))))
    loss = -r
    return (loss > v) if strict else (loss >= v)


def _transitions(hits):
    h = np.asarray(hits).astype(int).ravel()
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    return n00, n01, n10, n11


def christoffersen_independence(hits):
    """Christoffersen (1998) independence LR over the T-1 consecutive pairs. Returns (LR, p).

    pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/(T-1); 0*ln0 = 0.
    LR_ind = -2[ln L(pi) - ln L(pi01, pi11)] ~ chi2(1); p = chi2.sf(LR, 1).
    """
    n00, n01, n10, n11 = _transitions(hits)
    tot = n00 + n01 + n10 + n11
    if tot == 0:
        return 0.0, 1.0
    pi = (n01 + n11) / tot
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
           + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen_cc(hits, level=0.99):
    """Conditional coverage LR_cc = LR_uc (Kupiec, all T days, p = 1-c) + LR_ind (T-1 pairs).

    Returns (LR_cc, p) with p = chi2.sf(LR_cc, 2).
    """
    h = np.asarray(hits).astype(int).ravel()
    lr_uc, _ = kupiec_pof(int(h.sum()), int(h.size), _level(level))
    lr_ind, _ = christoffersen_independence(h)
    lr = lr_uc + lr_ind
    return float(lr), float(chi2.sf(lr, 2))


_BASEL_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_traffic_light(exceptions, n=250, level=0.99):
    """Basel (1996) traffic light. Returns (zone, plus_factor, multiplier = 3 + plus_factor).

    n=250, level=0.99 (standard): green 0-4 (k=0), yellow 5-9 (k=.40,.50,.65,.75,.85),
    red >= 10 (k=1.0).
    Other n/level: zones from the binomial cumulative probability P(X <= x) with p = 1-c:
    green < 95%, red >= 99.99%, else yellow; the yellow plus factor uses the Basel table's
    cumulative-probability cut points (95.88/98.63/99.60/99.89/99.97%).
    """
    x = int(exceptions)
    c = _level(level)
    if int(n) == 250 and abs(c - 0.99) < 1e-12:
        if x <= 4:
            return "green", 0.0, 3.0
        if x >= 10:
            return "red", 1.0, 4.0
        k = _BASEL_PLUS[x]
        return "yellow", k, 3.0 + k
    cp = float(binom.cdf(x, int(n), _tail(c)))
    if cp < 0.95:
        return "green", 0.0, 3.0
    if cp >= 0.9999:
        return "red", 1.0, 4.0
    k = 0.40
    for cut, kk in ((0.9863, 0.50), (0.9960, 0.65), (0.9989, 0.75), (0.9997, 0.85)):
        if cp >= cut:
            k = kk
    return "yellow", k, 3.0 + k


# ----------------------------------------------------------------------------- 9. component
def component_var(weights, cov, level=0.99, *, value=1.0, mu=None, measure="var"):
    """Delta-normal marginal / component VaR (Euler allocation).

    x = weights * value (currency positions if value is the portfolio value);
    sigma_p = sqrt(x' S x); total = k sigma_p - x'mu with k = z_c (measure="var") or
    phi(z_c)/(1-c) (measure="es"); mu=None -> zero mean.
    marginal_i = d total / d x_i = k (S x)_i / sigma_p - mu_i
    component_i = x_i * marginal_i (sum == total exactly); pct_i = component_i / total.
    Returns dict(total=float, marginal=array, component=array, pct=array, sigma_p=float).
    """
    c = _level(level)
    x = np.asarray(weights, dtype=float).ravel() * float(value)
    S = np.atleast_2d(np.asarray(cov, dtype=float))
    m = np.zeros_like(x) if mu is None else np.asarray(mu, dtype=float).ravel()
    sx = S @ x
    sp = float(math.sqrt(x @ sx))
    if measure == "var":
        k = norm.ppf(c)
    elif measure == "es":
        k = norm.pdf(norm.ppf(c)) / _tail(c)
    else:
        raise ValueError("measure must be 'var' or 'es'")
    total = k * sp - float(x @ m)
    marginal = k * sx / sp - m
    comp = x * marginal
    return {"total": float(total), "marginal": marginal, "component": comp,
            "pct": comp / total, "sigma_p": sp}


# ----------------------------------------------------------------------------- 10. Kelly
def kelly_fraction(mu=None, sigma=None, rf=0.0, *, p=None, b=None):
    """Kelly fraction.

    Discrete bet (p, b given): f = p - (1-p)/b, b = NET odds (win b per 1 staked).
    Continuous: f = (mu - rf)/sigma^2 with sigma the std (same period as mu, rf).
    Multi-asset: mu vector and sigma a COVARIANCE matrix -> f = S^-1 (mu - rf).
    Can be negative (short) or > 1 (leverage).
    """
    if p is not None or b is not None:
        if p is None or b is None:
            raise ValueError("discrete Kelly needs both p and b")
        return float(p - (1.0 - p) / b)
    if mu is None or sigma is None:
        raise ValueError("need mu and sigma (or p and b)")
    mu_a = np.asarray(mu, dtype=float)
    S = np.asarray(sigma, dtype=float)
    if S.ndim == 2:
        return np.linalg.solve(S, mu_a.ravel() - rf)
    return float((float(mu_a) - rf) / float(S) ** 2)


def fractional_kelly(f, fraction=0.5):
    """fraction * Kelly (half Kelly = 0.5). Works on scalars and arrays."""
    return f * fraction if np.ndim(f) == 0 else np.asarray(f, dtype=float) * fraction


def var_constrained_fraction(f_kelly, returns_or_sigma, var_limit, level=0.99, *,
                             kind="auto", method="normal", mu=0.0, horizon=1,
                             hs_method="linear", value=1.0, return_scale=False):
    """Cap a Kelly position so that its VaR <= var_limit.

    VaR is positively homogeneous of degree 1 in the position, so the capped position is
    s * f with s = min(1, var_limit / VaR(f)) (s = 1 if VaR(f) <= 0). VaR(s f) == var_limit
    whenever s < 1; direction is kept.
    VaR(f) (fraction of capital; `value` converts to currency to compare with var_limit):
      kind="sigma"  : scalar vol  -> |f| (z sqrt(h) sigma) - f*mu*h (method must be normal)
      kind="cov"    : f vector, cov S -> z sqrt(h) sqrt(f'Sf) - h f'mu
      kind="returns": 1-D series (scalar f) or T x N matrix (f vector): position P&L = f*r
                      (resp. r @ f); method="normal" -> parametric with sample mean if mu is
                      None else given mu; method="historical" -> historical_var_es(hs_method).
      kind="auto": scalar -> sigma; square 2-D matching len(f) -> cov; else returns.
    Returns the capped fraction (plus s if return_scale=True).
    """
    c = _level(level)
    z = norm.ppf(c)
    f = np.asarray(f_kelly, dtype=float)
    X = np.asarray(returns_or_sigma, dtype=float)
    if kind == "auto":
        if X.ndim == 0:
            kind = "sigma"
        elif X.ndim == 2 and X.shape[0] == X.shape[1] == f.size and f.size > 1 \
                and np.allclose(X, X.T):
            kind = "cov"
        else:
            kind = "returns"
    if kind == "sigma":
        mu_v = 0.0 if mu is None else float(mu)
        fv = float(f)
        var = abs(fv) * z * float(X) * math.sqrt(horizon) - fv * mu_v * horizon
    elif kind == "cov":
        fv = f.ravel()
        mu_v = np.zeros_like(fv) if mu is None else np.broadcast_to(np.asarray(mu, float),
                                                                      fv.shape)
        var = z * math.sqrt(horizon) * math.sqrt(float(fv @ X @ fv)) - horizon * float(fv @ mu_v)
    elif kind == "returns":
        pnl = X.ravel() * float(f) if X.ndim == 1 else X @ f.ravel()
        pnl = pnl[np.isfinite(pnl)]
        if method == "normal":
            m = float(pnl.mean()) if mu is None else float(np.sum(np.atleast_1d(mu) * f))
            var = parametric_var_es(mu=m, sigma=float(pnl.std(ddof=1)), level=c,
                                    horizon=horizon)[0]
        elif method == "historical":
            var = historical_var_es(pnl, c, method=hs_method)[0] * math.sqrt(horizon)
        else:
            raise ValueError("method must be 'normal' or 'historical'")
    else:
        raise ValueError("kind must be 'auto', 'sigma', 'cov' or 'returns'")
    var *= float(value)
    s = 1.0 if var <= 0 else min(1.0, float(var_limit) / var)
    out = float(f) * s if f.ndim == 0 else f * s
    return (out, s) if return_scale else out
