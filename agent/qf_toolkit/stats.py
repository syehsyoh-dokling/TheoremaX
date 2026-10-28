"""Estimators generated code rewrites (and breaks) most often, with conventions as arguments.

Nothing here guesses a periodicity or a ddof: pass the ones the task states.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ols(y, X, *, add_const=True, hac_lags=None) -> dict:
    """OLS with classical or Newey-West (Bartlett) standard errors.

    Returns dict: beta, se, t, resid, r2, nobs. `hac_lags=None` gives classical SEs (ddof=k);
    an int gives Newey-West with that many lags (no small-sample scaling).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[mask], X[mask]
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if hac_lags is None:
        s2 = resid @ resid / (n - k)
        cov = s2 * xtx_inv
    else:
        u = X * resid[:, None]
        S = u.T @ u
        for lag in range(1, int(hac_lags) + 1):
            w = 1 - lag / (hac_lags + 1)
            G = u[lag:].T @ u[:-lag]
            S += w * (G + G.T)
        cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tss = ((y - y.mean()) ** 2).sum()
    return {"beta": beta, "se": se, "t": beta / se, "resid": resid,
            "r2": 1 - (resid @ resid) / tss if tss > 0 else float("nan"), "nobs": n}


def newey_west_tstat(series, lags: int):
    """t-statistic of the mean of a series with Newey-West (Bartlett) standard error.
    Returns (mean, se, t)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    res = ols(x, np.empty((len(x), 0)), add_const=True, hac_lags=lags)
    return float(res["beta"][0]), float(res["se"][0]), float(res["t"][0])


def annualized_vol(returns, periods_per_year, ddof=1):
    """Std of a RETURN SERIES x sqrt(periods_per_year). A single number is refused (1 Oct: a daily
    vol passed here came back as nan/0); to annualize one daily vol use vol * sqrt(periods)."""
    if np.ndim(returns) == 0 or np.asarray(returns).size < 2:
        raise ValueError("annualized_vol takes a return SERIES (std is computed). To annualize one "
                         "daily volatility number use: daily_vol * math.sqrt(periods_per_year)")
    return float(np.nanstd(np.asarray(returns, dtype=float), ddof=ddof) * math.sqrt(periods_per_year))


def annualized_mean(returns, periods_per_year):
    """Arithmetic annualisation: mean periodic return * periods_per_year."""
    return float(np.nanmean(np.asarray(returns, dtype=float)) * periods_per_year)


def cagr(returns, periods_per_year):
    """Geometric annualised return from simple periodic returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    growth = float(np.prod(1 + r))
    return growth ** (periods_per_year / len(r)) - 1


def sharpe(returns, periods_per_year, rf_per_period=0.0, ddof=1):
    """Annualised Sharpe ratio of simple periodic returns (arithmetic mean / std * sqrt(ppy))."""
    ex = np.asarray(returns, dtype=float) - rf_per_period
    ex = ex[np.isfinite(ex)]
    sd = ex.std(ddof=ddof)
    return float(ex.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")


def drawdown_series(returns=None, *, equity=None) -> pd.Series:
    """Drawdown (<= 0) from simple returns or an equity curve: equity / running_max - 1."""
    if equity is None:
        equity = (1 + pd.Series(returns, dtype=float).fillna(0.0)).cumprod()
    equity = pd.Series(equity, dtype=float)
    return equity / equity.cummax() - 1


def max_drawdown(returns=None, *, equity=None) -> float:
    """Maximum drawdown as a NEGATIVE number (e.g. -0.23). Negate if the task wants a magnitude."""
    return float(drawdown_series(returns, equity=equity).min())


def ar1_fit(x, *, add_const=True) -> dict:
    """x_t = a + phi * x_{t-1} + e_t by OLS. Returns a, phi, sigma_e, nobs, add_const.

    add_const=False fits through the origin (a = 0; for de-meaned / deseasonalised residuals);
    sigma_e uses ddof = number of fitted coefficients (2 with constant, 1 without).
    1 Oct cme-hdd: add_const was passed twice and crashed; it is accepted now."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    res = ols(x[1:], x[:-1], add_const=add_const)
    if add_const:
        a, phi = res["beta"]
    else:
        a, phi = 0.0, res["beta"][0]
    k = 2 if add_const else 1
    sigma_e = math.sqrt(res["resid"] @ res["resid"] / (res["nobs"] - k))
    return {"a": float(a), "phi": float(phi), "sigma_e": sigma_e, "nobs": res["nobs"],
            "add_const": bool(add_const)}


def ou_from_ar1(a, phi, sigma_e, dt=1.0) -> dict:
    """Exact discretisation of dX = kappa(theta - X)dt + sigma dW from AR(1) coefficients.

    kappa = -ln(phi)/dt (per unit of dt's time unit), theta = a/(1-phi),
    sigma = sigma_e * sqrt(2 kappa / (1 - phi^2)), half_life = ln 2 / kappa (same unit as dt).
    With dt=1 the half-life is in PERIODS (observations); multiply by the period length for days.
    """
    if not 0 < phi < 1:
        return {"kappa": float("nan"), "theta": float("nan"), "sigma": float("nan"),
                "half_life": float("nan"), "mean_reverting": False}
    kappa = -math.log(phi) / dt
    return {"kappa": kappa, "theta": a / (1 - phi),
            "sigma": sigma_e * math.sqrt(2 * kappa / (1 - phi ** 2)),
            "half_life": math.log(2) / kappa, "mean_reverting": True}


def ewma_variance(returns, lam=0.94, *, init=None) -> np.ndarray:
    """RiskMetrics recursion s2_t = lam*s2_{t-1} + (1-lam)*r_{t-1}^2; s2_0 = init or r_0^2.
    Element t is the variance forecast made with information up to t-1."""
    r = np.asarray(returns, dtype=float)
    out = np.empty_like(r)
    out[0] = r[0] ** 2 if init is None else init
    for i in range(1, len(r)):
        out[i] = lam * out[i - 1] + (1 - lam) * r[i - 1] ** 2
    return out


def spearman_ic(signal, forward_return) -> float:
    """Rank IC of one cross-section (NaNs dropped pairwise)."""
    s = pd.Series(signal, dtype=float)
    f = pd.Series(forward_return, dtype=float)
    ok = s.notna() & f.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(s[ok], f[ok]).statistic)
