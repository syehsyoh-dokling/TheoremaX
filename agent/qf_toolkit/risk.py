"""Tail risk with one sign convention: VaR and ES are POSITIVE loss magnitudes.

Inputs are returns or P&L where gains are positive. A task that reports VaR as a negative
return quantile needs the result negated explicitly - read the task, then decide.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm, t as student_t


def var_es_historical(returns, level=0.99, *, method="linear"):
    """Historical VaR/ES at confidence `level` (0.99 = 99%).

    VaR = `level`-quantile of losses (loss = -return) using numpy.quantile(method=...);
    ES = mean of losses >= VaR. Returns (var, es) as positive losses.
    Quantile method matters on small samples: 'linear' (numpy default), 'lower', 'higher',
    'inverted_cdf' (empirical, no interpolation). Use the one the task states.
    """
    losses = -np.asarray(returns, dtype=float)
    losses = losses[np.isfinite(losses)]
    var = float(np.quantile(losses, level, method=method))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return var, es


def var_es_normal(mu, sigma, level=0.99):
    """Parametric normal VaR/ES of a return distribution N(mu, sigma^2). Positive losses."""
    z = norm.ppf(level)
    var = -mu + sigma * z
    es = -mu + sigma * norm.pdf(z) / (1 - level)
    return float(var), float(es)


def var_es_student_t(mu, scale, nu, level=0.99):
    """Student-t VaR/ES for returns mu + scale * T_nu (scale is the t scale, not the std).

    If you have the standard deviation s, scale = s * sqrt((nu - 2) / nu).
    """
    q = student_t.ppf(level, nu)
    var = -mu + scale * q
    es = -mu + scale * (student_t.pdf(q, nu) * (nu + q ** 2) / ((nu - 1) * (1 - level)))
    return float(var), float(es)


def cornish_fisher_var(mu, sigma, skew, excess_kurt, level=0.99):
    """Modified (Cornish-Fisher) VaR, positive loss. skew/excess_kurt of RETURNS."""
    z = norm.ppf(1 - level)
    zcf = (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * excess_kurt / 24
           - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)
    return float(-(mu + sigma * zcf))


def kupiec_pof(violations: int, n: int, level=0.99):
    """Kupiec proportion-of-failures test. Returns (lr_statistic, p_value).
    level: the VaR CONFIDENCE (0.99 or 99); know77: a TAIL probability (< 0.5, e.g. 0.01 =
    'expected_rate') is read as such - kupiec_pof(x, n, 0.01) gave LR 17883 instead of 8."""
    lv = float(level)
    if lv > 1.0:
        lv /= 100.0
    p = lv if lv < 0.5 else 1 - lv
    x = int(violations)
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("need n > 0 and 0 <= violations <= n")
    phat = x / n

    def xlogy(a, b):                      # a*log(b) with 0*log(0) = 0
        return 0.0 if a == 0 else a * math.log(b)

    ll_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(chi2.sf(lr, 1))


def fit_garch11(returns, initial_guess=None):
    """Quasi-maximum likelihood GARCH(1,1) with consensus initial guess and stationarity bounds.

    Model: sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
    Returns (omega, alpha, beta, conditional_variances).
    """
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    var0 = float(np.var(r))
    if initial_guess is None:
        x0 = [0.05 * var0, 0.05, 0.90]
    else:
        x0 = list(initial_guess)

    def neg_log_lik(params):
        omega, alpha, beta = params
        if alpha + beta >= 1.0 or omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        T = len(r)
        h = np.empty(T)
        h[0] = var0
        for t in range(1, T):
            h[t] = omega + alpha * (r[t - 1] ** 2) + beta * h[t - 1]
            if h[t] <= 0:
                return 1e10
        return 0.5 * np.sum(np.log(h) + (r ** 2) / h)

    bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.999)]
    cons = ({"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]})
    res = minimize(neg_log_lik, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)

    omega, alpha, beta = res.x
    h = np.empty(len(r))
    h[0] = var0
    for t in range(1, len(r)):
        h[t] = omega + alpha * (r[t - 1] ** 2) + beta * h[t - 1]
    return float(omega), float(alpha), float(beta), h


def dcc_correlation(std_residuals, a0=0.05, b0=0.93, method="L-BFGS-B"):
    """Standard Engle (2002) Dynamic Conditional Correlation (DCC).

    Returns (a, b, log_likelihood, dynamic_correlation_array).
    """
    from scipy.optimize import minimize

    eps = np.asarray(std_residuals, dtype=float)
    T, K = eps.shape
    Q_bar = np.cov(eps, rowvar=False)

    def neg_log_lik(params):
        a, b = params
        if a <= 1e-6 or b <= 1e-6 or a + b >= 0.9999:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(T):
            e_t = eps[t:t + 1].T
            if t > 0:
                e_prev = eps[t - 1:t - 1 + 1].T
                Q = (1.0 - a - b) * Q_bar + a * (e_prev @ e_prev.T) + b * Q
            inv_diag = 1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12))
            R = np.diag(inv_diag) @ Q @ np.diag(inv_diag)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10
            invR = np.linalg.pinv(R)
            ll += 0.5 * (logdet + float(e_t.T @ invR @ e_t) - float(e_t.T @ e_t))
        return ll

    bounds = [(1e-4, 0.20), (0.70, 0.999)]
    res = minimize(neg_log_lik, [a0, b0], method=method, bounds=bounds,
                   options={"ftol": 1e-8, "gtol": 1e-6})
    if not res.success or res.x[0] == a0:
        # Fallback to Nelder-Mead if L-BFGS-B stuck
        res = minimize(neg_log_lik, [a0, b0], method="Nelder-Mead", bounds=bounds)

    a, b = float(res.x[0]), float(res.x[1])
    log_lik = -float(res.fun)

    # Compute correlation series
    R_series = np.empty((T, K, K))
    Q = Q_bar.copy()
    for t in range(T):
        if t > 0:
            e_prev = eps[t - 1:t - 1 + 1].T
            Q = (1.0 - a - b) * Q_bar + a * (e_prev @ e_prev.T) + b * Q
        inv_diag = 1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12))
        R_series[t] = np.diag(inv_diag) @ Q @ np.diag(inv_diag)
    return a, b, log_lik, R_series


def simulate_jump_diffusion(S0, r, q, sigma, lambda_j, mu_j, sigma_j, T,
                            N_steps=252, N_paths=50000, seed=42):
    """Merton jump diffusion simulation with antithetic variance reduction."""
    rng = np.random.default_rng(seed)
    dt = T / N_steps
    half_n = N_paths // 2

    k = math.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = (r - q - lambda_j * k - 0.5 * sigma ** 2) * dt

    log_S = np.full(N_paths, math.log(S0))
    for _ in range(N_steps):
        # Antithetic normal increments
        z_half = rng.standard_normal(half_n)
        z = np.concatenate([z_half, -z_half])
        # Poisson jumps
        n_jumps = rng.poisson(lambda_j * dt, size=N_paths)
        jump_sizes = np.zeros(N_paths)
        for i in range(N_paths):
            if n_jumps[i] > 0:
                jump_sizes[i] = rng.normal(mu_j * n_jumps[i], sigma_j * math.sqrt(n_jumps[i]))
        log_S += drift + sigma * math.sqrt(dt) * z + jump_sizes
    return np.exp(log_S)
