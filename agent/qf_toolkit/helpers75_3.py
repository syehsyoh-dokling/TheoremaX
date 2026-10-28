"""know75 group-3 helpers: regime models and portfolio construction (Gaussian HMM, Black-Litterman,
drifting-weight Brinson-Fachler, Kelly Monte-Carlo paths, CTA vol targeting, absorption-ratio
risk parity), option surfaces (vectorised implied vol, SVI, Dupire on a grid, coin-margined
forwards, local-vol Monte-Carlo barrier, two-factor Heston), closed-form implied-vol
approximations, Crank-Nicolson PSOR with cash dividends, exact ACT/365 time to expiry, OU fits
(plug-in Euler, OU-with-jumps moments) and rolling OLS with Newey-West HAC.

Every function is general (no task names), pure numpy/pandas/scipy (arch only inside
garch11_conditional_vol), and states its formula and conventions in the docstring. Values a task
fixes (windows, tolerances, seeds, thresholds) are arguments - pass the task's own values.
Self-tests: proven_work/fail72/selftest75_3.py.

API lines (one per function):
     gaussian_hmm_fit(x, n_states=2, mu0=None, sigma0=None, A0=None, pi0=None, max_iter=100, tol=1e-6, stop="mu"|"loglik") -> dict{mu,sigma,A,pi,gamma[T,K],loglik,n_iter,converged,bull}; log-space Baum-Welch, init = first-half/second-half split (ddof=1)
     gaussian_hmm_posteriors(x, mu, sigma, A, pi) -> dict{gamma smoothed, filtered, loglik} (log-space forward-backward)
     black_litterman(Sigma, w_mkt, delta, tau, P=None, Q=None, Omega=None) -> dict{pi, mu_bl, M, w_raw=(delta Sigma)^-1 mu_bl, w_gross (sum|w|=1), w_sum1 (sum w=1)}
     bl_absolute_views(views, n_assets, periods_per_year=1, confidence_is="std") -> (P, Q, Omega); Q = annual/ppy, Omega = diag(std^2/ppy)
     month_end_returns(prices, start=None, end=None) -> DataFrame of simple returns, PeriodIndex months in [start,end], anchored on the prior month's last trading day
     brinson_fachler_drift(rb, rp, wb, w_target, cash_target=0.0, rc=None, rebalance=None) -> dict{alloc,sel,inter [M,N], cash_alloc [M], Rp, Rb, weights_bom [M,N+1], totals}; weights drift between rebalances
     mvn_return_cube(mean, cov, n_paths, n_steps, seed=None, method="mvn"|"cholesky") -> [n_paths,n_steps,n_assets]; "mvn" = default_rng(seed).multivariate_normal(mean, cov, size=(n_paths,n_steps))
     portfolio_path_returns(cube, weights, rf_per_period=0.0, cash_leg=False) -> [n_paths,n_steps] = cube @ w (+ rf (1 - sum w) if cash_leg)
     wealth_path_metrics(port_returns, initial_capital=1.0, periods_per_year=252, ddof=1) -> dict{terminal,sharpe,max_drawdown POSITIVE (per path), median_terminal, mean_terminal, median_sharpe, median_max_drawdown}
     kelly_portfolio_var(f, cov, mu=None, level=0.99, include_mean=True) -> z sqrt(f'Sf) - f'mu (POSITIVE loss fraction)
     sma_seeded_ema(x, span) -> EMA[span-1] = mean(x[:span]), EMA[t] = a x[t] + (1-a) EMA[t-1], a = 2/(span+1), NaN before; column-wise for 2-D
     ewma_variance_alpha(r, span=None, alpha=None, seed="first") -> s2[0] = r[0]^2, s2[t] = a r[t]^2 + (1-a) s2[t-1] (weight on the CURRENT square), column-wise
     garch11_conditional_vol(returns, mean="Constant", scale=100.0) -> dict{sigma DECIMAL daily [T], omega DECIMAL^2, alpha, beta, persistence, long_run_variance DECIMAL^2, mu}
     percentile_regimes(x, lo_pct, hi_pct) -> dict{codes 0 low/1 mid/2 high, labels, thresh_lo, thresh_hi, frac_low, frac_mid, frac_high}; low x<=P_lo, high x>=P_hi (np.percentile)
     vol_target_weights(signal, vol, target, scale=None, max_leverage=None) -> clip(signal*target*scale/vol, -L, L), NaN -> 0
     lagged_portfolio_returns(weights, returns, lag=1) -> rp[t] = sum_i w[t-lag,i] r[t,i], first `lag` entries 0 (full length T)
     log_return_performance(x, periods_per_year=252, rf_annual=0.0, ddof=0) -> dict{annualized_log_return, annualized_return=exp(.)-1, annualized_volatility, sharpe, max_drawdown (<=0), max_drawdown_abs, calmar}
     rolling_absorption_ratio(X, window, k, window_end="inclusive") -> dict{ar, largest_eigenvalue, end_idx, mp_threshold}; eigvalsh of the rolling CORRELATION matrix, AR = top-k / N
     classify_upper_thresholds(x, thresholds, labels) -> labels[i] for the first thresholds[i] (descending) with x > thr, else labels[-1] (strict >)
     month_end_flags(dates) -> bool array True on the last row of each calendar month present (final row True)
     inverse_vol_weights(window_returns, ddof=1, budget=1.0) -> (1/sd)/sum(1/sd) * budget
     hold_weights_returns(X, rebalance_idx, weights) -> (returns, day_idx): day t > first rebalance earns w(last rebalance STRICTLY before t) . X[t]
     tail_var_cvar(returns, level) -> (VaR, CVaR) POSITIVE; VaR = -quantile(r, 1-c) (linear), CVaR = -mean(r[r <= -VaR])
     loss_var_cvar_sorted(returns, level, clamp=True) -> (VaR, CVaR); losses=-r sorted, VaR = losses[ceil(c n)-1] (clamped >= 0), CVaR = mean(losses > VaR)
     implied_vol_vec(price, S, K, T, r=0.0, q=0.0, kind="call", lo=1e-4, hi=5.0, xtol=1e-8) -> ndarray (Series/arrays ok, kind may be an array of 'call'/'put'/'C'/'P'); NaN outside no-arbitrage bounds
     svi_raw(k, a, b, rho, m, sigma) -> total variance w(k) | svi_derivatives(k, a,b,rho,m,sigma) -> (w, dw/dk, d2w/dk2)
     svi_fit(k, w, T=None, a_min=1e-6, x0=None) -> dict{a,b,rho,m,sigma,rmse_w,rmse_vol (if T),success}; least squares on total variance with bounds, a > 0
     gatheral_local_var(k, w, w_k, w_kk, w_T) -> local variance = w_T / [1 - k w_k/w + 0.25(-0.25 - 1/w + k^2/w^2) w_k^2 + 0.5 w_kk], k = ln(K/F)
     dupire_local_vol_grid(C, strikes, ttms, r=0.0, q=0.0, var_floor=None, vol_cap=None, edge_order=2, den_min=None) -> dict{dC_dT,dC_dK,d2C_dK2,local_var,local_vol} [nT,nK]; numpy.gradient for every derivative, d2C/dK2 = gradient twice
     forward_from_coin_parity(K, call_coin, put_coin) -> F = K / (1 - (c - p)) for premiums quoted in the underlying coin, DF = 1 (USD premium = coin premium x F)
     parse_option_instrument(name, sep="-") -> dict{underlying, expiry_label, strike, kind 'call'/'put'} from 'BTC-24APR26-64000-C'
     local_vol_interpolator(strikes, ttms, lv_grid) -> f(t, S) bilinear in (T, K), flat beyond the grid
     localvol_mc_barrier(S0, local_vol, T, n_monitor, n_sub, n_paths, K, barrier, r=0.0, q=0.0, seed=None, antithetic=True, kind="down_and_out_call") -> dict{barrier_price, barrier_se, vanilla_price, vanilla_se, hit_probability, survival_probability}; barrier checked only at the n_monitor period ends
     heston2_char_func(u, T, S0, r, q, factors) -> E[exp(iu ln S_T)] for a sum of independent Heston variance factors [(kappa, theta, sigma, rho, v0), ...] (little-trap form)
     heston2_price(K, T, S0, r, q, factors, kind="call", method="gauss_legendre"|"quad", n_nodes=2000, upper=100.0, quad_lower=1e-8, quad_upper=50.0, quad_limit=200) -> price array over K; put priced DIRECTLY from P1, P2
     brenner_subrahmanyam_iv(C, S, K, T, r=0, q=0) | li_atm_iv(C, S, K, T, r=0, q=0) | li_nonatm_iv(C, S, K, T, r=0, q=0, rho_cut=1.4, disc_tol=1e-10, negative_disc="nan") | cmh_iv(C, S, K, T, r=0, q=0, negative_disc="floor") -> sigma arrays, discounted S~ = S e^-qT, K~ = K e^-rT
     newton_iv_vec(price, S, K, T, r=0.0, q=0.0, kind="call", sigma0=0.3, tol=1e-10, max_iter=100) -> (sigma, converged PER-POINT mask (truth value = all), n_iter); stops when ALL |step| < tol
     ttm_years_exact(start, end, days_per_year=365.0) -> (end - start) seconds / (days_per_year*86400); ISO strings, Timestamps (tz or naive), Series, or epoch-ms ints
     cn_fd_option(K, T, r, sigma, q=0.0, kind="put", american=True, S_max=None, N_S=300, N_T=600, dividends=(), omega=1.2, tol=1e-8, max_sweeps=10000, S0=None, boundary_tol=1e-6) -> dict{price, delta, S, t, V [N_T+1,N_S+1], boundary [N_T+1]}; Crank-Nicolson + PSOR, cash dividends by V(S,t_d-) = V(S-D,t_d+)
     ou_jump_moments(x0, kappa, theta, sigma, tau, lam=0.0, mu_J=0.0, sigma_J=0.0, jumps_in_mean=True) -> dict{mean, var (jump variance INCLUDED), E_S, Var_S, stationary_mean, stationary_var}
     ou_fit_euler_plugin(x, dt, ddof_resid=1) -> dict{alpha, beta, kappa=-beta/dt, mu=-alpha/beta, sigma=resid_std/sqrt(dt), half_life (time units), half_life_periods, stationary_std, r_squared, resid_std, nobs}; OLS dx_t = alpha + beta x_{t-1}
     ou_exact_paths(x0, kappa, mu, sigma, dt, n_steps, n_paths, seed=None, rng=None) -> [n_paths, n_steps+1]; exact transition, one standard_normal(n_paths) draw per step
     monthly_last(s, label="start") -> month-end last valid value, index relabelled to the first day of the month
     align_monthly(series, base=None, ffill=(), dropna=(), start=None, end=None) -> DataFrame; LEFT-join on base's months (outer if base None), ffill AFTER the join, then dropna, then clip to [start, end]
     rolling_ols_hac(y, X, window, lags=3, add_const=True, index=None) -> DataFrame per window end: beta_*, r_squared, t_hac_* (Bartlett, no small-sample factor), fitted components at the last row, residual, pct_* = Cov(b_k X_k, y)/Var(y)*100, pct_residual = (1-R2)*100
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize as _opt
from scipy.special import logsumexp as _lse
from scipy.stats import norm as _norm


# ============================================================================ HMM
def gaussian_hmm_posteriors(x, mu, sigma, A, pi) -> dict:
    """Log-space forward-backward for a Gaussian HMM (Rabiner 1989).

    log b_k(x_t) = norm.logpdf(x_t, mu_k, sigma_k)
    alpha_t(k) = log b_k(x_t) + lse_j[alpha_{t-1}(j) + log A_jk], alpha_0 = log pi + log b(x_0)
    beta_t(j)  = lse_k[log A_jk + log b_k(x_{t+1}) + beta_{t+1}(k)], beta_{T-1} = 0
    gamma = exp(alpha + beta - logL) (smoothed), filtered = softmax(alpha_t).
    Returns dict{gamma [T,K], filtered [T,K], loglik, log_alpha, log_beta, logB}."""
    x = np.asarray(x, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    with np.errstate(divide="ignore"):
        logA = np.log(np.asarray(A, dtype=float))
        logpi = np.log(np.asarray(pi, dtype=float).ravel())
    T, K = len(x), len(mu)
    logB = _norm.logpdf(x[:, None], mu[None, :], sigma[None, :])
    la = np.empty((T, K))
    lb = np.zeros((T, K))
    la[0] = logpi + logB[0]
    lae = np.logaddexp.reduce
    for t in range(1, T):
        la[t] = logB[t] + lae(la[t - 1][:, None] + logA, axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = lae(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    loglik = float(_lse(la[-1]))
    gamma = np.exp(la + lb - loglik)
    filtered = np.exp(la - _lse(la, axis=1)[:, None])
    return {"gamma": gamma, "filtered": filtered, "loglik": loglik, "log_alpha": la,
            "log_beta": lb, "logB": logB, "logA": logA}


def gaussian_hmm_fit(x, n_states=2, mu0=None, sigma0=None, A0=None, pi0=None, max_iter=100,
                     tol=1e-6, stop="mu") -> dict:
    """Baum-Welch EM for a univariate Gaussian HMM, every pass in log space.

    Init (when mu0/sigma0 are None): split x into n_states consecutive chunks (2 states: first
    N//2 observations and the rest); mu_k = chunk mean, sigma_k = chunk std (ddof=1).
    A0 default: 0.9 on the diagonal, rest equal; pi0 default uniform.
    Each pass: E-step with the CURRENT (mu, sigma, A, pi); M-step
      pi = gamma_0; A_ij = sum_{t<T-1} xi_t(i,j) / sum_{t<T-1} gamma_t(i);
      mu_k = sum gamma x / sum gamma; sigma_k^2 = sum gamma (x - mu_k_new)^2 / sum gamma (MLE).
    stop="mu": converged when max|mu_new - mu_old| < tol; "loglik": |logL - logL_prev| < tol.
    n_iter counts the passes actually run (the pass that meets the rule included); after the loop
    a final E-step with the final parameters gives gamma and loglik. bull = argmax(mu).
    """
    x = np.asarray(x, dtype=float).ravel()
    T, K = len(x), int(n_states)
    if mu0 is None or sigma0 is None:
        bounds = [0] + [(T * (j + 1)) // K for j in range(K)]
        if K == 2:
            bounds = [0, T // 2, T]
        chunks = [x[bounds[j]:bounds[j + 1]] for j in range(K)]
        mu = np.array([c.mean() for c in chunks]) if mu0 is None else np.asarray(mu0, float).copy()
        sigma = (np.array([c.std(ddof=1) for c in chunks]) if sigma0 is None
                 else np.asarray(sigma0, float).copy())
    else:
        mu, sigma = np.asarray(mu0, float).copy(), np.asarray(sigma0, float).copy()
    if A0 is None:
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9)
    else:
        A = np.asarray(A0, dtype=float).copy()
    pi = np.full(K, 1.0 / K) if pi0 is None else np.asarray(pi0, dtype=float).copy()
    prev_ll, converged, n_iter = -np.inf, False, 0
    for it in range(1, int(max_iter) + 1):
        post = gaussian_hmm_posteriors(x, mu, sigma, A, pi)
        g, la, lb, logB, logA = post["gamma"], post["log_alpha"], post["log_beta"], post["logB"], post["logA"]
        ll = post["loglik"]
        log_xi = (la[:-1, :, None] + logA[None, :, :] + logB[1:, None, :] + lb[1:, None, :] - ll)
        xi = np.exp(log_xi)
        gsum = g.sum(axis=0)
        mu_new = (g * x[:, None]).sum(axis=0) / gsum
        sigma_new = np.sqrt((g * (x[:, None] - mu_new[None, :]) ** 2).sum(axis=0) / gsum)
        A_new = xi.sum(axis=0) / g[:-1].sum(axis=0)[:, None]
        A_new = A_new / A_new.sum(axis=1, keepdims=True)
        pi_new = g[0] / g[0].sum()
        if stop == "mu":
            conv = bool(np.max(np.abs(mu_new - mu)) < tol)
        else:
            conv = bool(abs(ll - prev_ll) < tol)
        mu, sigma, A, pi, prev_ll, n_iter = mu_new, sigma_new, A_new, pi_new, ll, it
        if conv:
            converged = True
            break
    post = gaussian_hmm_posteriors(x, mu, sigma, A, pi)
    return {"mu": mu, "sigma": sigma, "A": A, "pi": pi, "gamma": post["gamma"],
            "filtered": post["filtered"], "loglik": post["loglik"], "n_iter": int(n_iter),
            "converged": converged, "bull": int(np.argmax(mu))}


# ============================================================================ Black-Litterman
def bl_absolute_views(views, n_assets, periods_per_year=1, confidence_is="std", asset_key="asset",
                      return_key="return", confidence_key="confidence"):
    """Absolute views -> (P, Q, Omega) in the data frequency.

    Each view: {asset: 0-based index, return: ANNUAL expected return, confidence: ANNUAL std
    of the view error}. P row = one-hot at the asset; Q = return / ppy;
    Omega = diag(confidence^2 / ppy) when confidence_is="std" (variance scales with time, so
    daily std = annual std / sqrt(ppy)), diag(confidence / ppy) when it is a variance."""
    V = len(views)
    P = np.zeros((V, int(n_assets)))
    Q = np.zeros(V)
    om = np.zeros(V)
    for i, v in enumerate(views):
        P[i, int(v[asset_key])] = 1.0
        Q[i] = float(v[return_key]) / periods_per_year
        c = float(v[confidence_key])
        om[i] = (c ** 2 if confidence_is == "std" else c) / periods_per_year
    return P, Q, np.diag(om)


def black_litterman(Sigma, w_mkt, delta, tau, P=None, Q=None, Omega=None) -> dict:
    """Black-Litterman (1992), He-Litterman (1999) form.

    pi = delta Sigma w_mkt (reverse optimisation, same frequency as Sigma)
    M = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 ; mu_bl = M [(tau Sigma)^-1 pi + P' Omega^-1 Q]
    w_raw = (delta Sigma)^-1 mu_bl ; w_gross = w_raw / sum|w_raw| ; w_sum1 = w_raw / sum(w_raw).
    No views -> mu_bl = pi and w_raw = w_mkt. Posterior covariance = Sigma + M."""
    S = np.asarray(Sigma, dtype=float)
    w = np.asarray(w_mkt, dtype=float).ravel()
    pi = delta * S @ w
    tS_inv = np.linalg.inv(tau * S)
    if P is None or len(np.atleast_1d(Q)) == 0:
        M = np.linalg.inv(tS_inv)
        mu = pi.copy()
    else:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        Qv = np.asarray(Q, dtype=float).ravel()
        Om_inv = np.linalg.inv(np.atleast_2d(np.asarray(Omega, dtype=float)))
        M = np.linalg.inv(tS_inv + P.T @ Om_inv @ P)
        mu = M @ (tS_inv @ pi + P.T @ Om_inv @ Qv)
    w_raw = np.linalg.solve(delta * S, mu)
    return {"pi": pi, "mu_bl": mu, "M": M, "posterior_cov": S + M, "w_raw": w_raw,
            "w_gross": w_raw / np.abs(w_raw).sum(), "w_sum1": w_raw / w_raw.sum()}


# ============================================================================ Brinson with drift
def month_end_returns(prices, start=None, end=None) -> pd.DataFrame:
    """Monthly simple returns from DAILY prices (DatetimeIndex or a 'date'/'Date' column).

    Month-end price = last available row of each calendar month; R_m = P_m / P_{m-1} - 1 where
    P_{m-1} is the PRIOR month's last trading day (so the month before `start` anchors the first
    return). Only months with start <= month <= end are returned (index: PeriodIndex 'M').
    Raises if the anchor month is missing."""
    df = prices.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        dcol = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "time")), None)
        if dcol is None:
            raise ValueError("prices needs a DatetimeIndex or a date column")
        df[dcol] = pd.to_datetime(df[dcol])
        df = df.set_index(dcol)
    df = df.sort_index()
    me = df.groupby(df.index.to_period("M")).last()
    ret = me.pct_change()
    lo = pd.Period(pd.Timestamp(start), "M") if start is not None else ret.index[1]
    hi = pd.Period(pd.Timestamp(end), "M") if end is not None else ret.index[-1]
    if (lo - 1) not in me.index:
        raise ValueError(f"no prices in the anchor month {lo - 1} before {lo}")
    out = ret.loc[(ret.index >= lo) & (ret.index <= hi)]
    return out


def brinson_fachler_drift(rb, rp, wb, w_target, cash_target=0.0, rc=None, rebalance=None) -> dict:
    """Brinson-Fachler per period with drifting beginning-of-period portfolio weights.

    rb, rp: [M, N] benchmark-ETF and portfolio-ETF returns per sector; wb: [M, N] benchmark
    weights (sum 1 per row); w_target: [N] portfolio sector targets; cash_target: cash weight;
    rc: [M] cash return per period; rebalance: [M] bool, True = reset to targets at the START of
    that period (period 0 is always reset).
    Per period: Rb = wb.rb; Rp = w.rp + wc rc;
      allocation_i = (w_i - wb_i)(rb_i - Rb); selection_i = wb_i (rp_i - rb_i);
      interaction_i = (w_i - wb_i)(rp_i - rb_i); cash allocation = wc (rc - Rb).
    Drift to next period: w_i <- w_i (1 + rp_i) / G, wc <- wc (1 + rc)/G, G = total growth.
    Sum of all effects per period = Rp - Rb. Totals are sums over periods; compound returns
    prod(1+R) - 1."""
    rb = np.atleast_2d(np.asarray(rb, dtype=float))
    rp = np.atleast_2d(np.asarray(rp, dtype=float))
    wb = np.atleast_2d(np.asarray(wb, dtype=float))
    M, N = rb.shape
    if wb.shape[0] == 1 and M > 1:
        wb = np.repeat(wb, M, axis=0)
    wt = np.asarray(w_target, dtype=float).ravel()
    rc = np.zeros(M) if rc is None else np.asarray(rc, dtype=float).ravel()
    if rebalance is None:
        reb = np.zeros(M, dtype=bool)
    else:
        rv = np.asarray(rebalance).ravel()
        if rv.dtype != bool and len(rv) != M and np.all(np.mod(rv, 1) == 0) and np.all((rv >= 0) & (rv < M)):
            # know77: rebalance=[0, 3, 6, 9] (period INDICES) used to be read as 4 flags -> IndexError
            reb = np.zeros(M, dtype=bool)
            reb[rv.astype(int)] = True
        else:
            reb = rv.astype(bool)
            if len(reb) != M:
                raise ValueError(f"rebalance needs {M} flags or a list of period indices; got {len(reb)} values")
    reb[0] = True
    alloc, sel, inter = (np.zeros((M, N)) for _ in range(3))
    cash_alloc, Rp, Rb = np.zeros(M), np.zeros(M), np.zeros(M)
    w_bom = np.zeros((M, N + 1))
    w, wc = wt.copy(), float(cash_target)
    for m in range(M):
        if reb[m]:
            w, wc = wt.copy(), float(cash_target)
        w_bom[m, :N], w_bom[m, N] = w, wc
        Rb[m] = float(wb[m] @ rb[m])
        Rp[m] = float(w @ rp[m] + wc * rc[m])
        alloc[m] = (w - wb[m]) * (rb[m] - Rb[m])
        sel[m] = wb[m] * (rp[m] - rb[m])
        inter[m] = (w - wb[m]) * (rp[m] - rb[m])
        cash_alloc[m] = wc * (rc[m] - Rb[m])
        g = w * (1.0 + rp[m])
        gc = wc * (1.0 + rc[m])
        G = g.sum() + gc
        w, wc = g / G, gc / G
    monthly_alloc = alloc.sum(axis=1) + cash_alloc
    return {"alloc": alloc, "sel": sel, "inter": inter, "cash_alloc": cash_alloc, "Rp": Rp,
            "Rb": Rb, "weights_bom": w_bom, "monthly_allocation": monthly_alloc,
            "monthly_selection": sel.sum(axis=1), "monthly_interaction": inter.sum(axis=1),
            "total_allocation": float(monthly_alloc.sum()), "total_selection": float(sel.sum()),
            "total_interaction": float(inter.sum()),
            "portfolio_return": float(np.prod(1 + Rp) - 1), "benchmark_return": float(np.prod(1 + Rb) - 1),
            # know77: the model read res['allocation'] (KeyError ended a run) - long-name aliases
            "allocation": alloc, "selection": sel, "interaction": inter, "cash_allocation": cash_alloc}


# ============================================================================ Kelly MC paths
def mvn_return_cube(mean, cov, n_paths, n_steps, seed=None, method="mvn", rng=None):
    """Simulated returns [n_paths, n_steps, n_assets], one draw shared by every scheme.

    method="mvn": np.random.default_rng(seed).multivariate_normal(mean, cov, size=(n_paths, n_steps))
    (numpy's own SVD factorisation - what "draw from a multivariate normal" literally does);
    method="cholesky": mean + Z @ L.T with Z = standard_normal((n_paths, n_steps, n)), L = chol(cov).
    The two give DIFFERENT random numbers from the same seed. seed is cast to int."""
    mean = np.asarray(mean, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    g = rng if rng is not None else np.random.default_rng(None if seed is None else int(seed))
    if method == "mvn":
        return g.multivariate_normal(mean, cov, size=(int(n_paths), int(n_steps)))
    L = np.linalg.cholesky(cov)
    Z = g.standard_normal((int(n_paths), int(n_steps), len(mean)))
    return mean + Z @ L.T


def portfolio_path_returns(cube, weights, rf_per_period=0.0, cash_leg=False):
    """Portfolio return per path and period: cube @ w; with cash_leg the unlevered remainder
    (1 - sum w) earns rf_per_period (negative remainder = borrowing at rf)."""
    w = np.asarray(weights, dtype=float).ravel()
    out = np.asarray(cube, dtype=float) @ w
    if cash_leg:
        out = out + rf_per_period * (1.0 - w.sum())
    return out


def wealth_path_metrics(port_returns, initial_capital=1.0, periods_per_year=252, ddof=1) -> dict:
    """Per-path wealth W_t = W0 prod(1 + r), terminal wealth, Sharpe = mean/std(ddof) sqrt(ppy)
    (no rf), max drawdown = max_t (1 - W_t / max(W0, max_{s<=t} W_s)) POSITIVE; plus medians."""
    R = np.atleast_2d(np.asarray(port_returns, dtype=float))
    W = initial_capital * np.cumprod(1.0 + R, axis=1)
    peak = np.maximum(np.maximum.accumulate(W, axis=1), initial_capital)
    mdd = np.max(1.0 - W / peak, axis=1)
    sd = R.std(axis=1, ddof=ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = R.mean(axis=1) / sd * math.sqrt(periods_per_year)
    term = W[:, -1]
    # know76: 'terminal_wealth' alias (four KeyError crashes guessed that name)
    return {"terminal": term, "terminal_wealth": term, "sharpe": sh, "max_drawdown": mdd, "wealth": W,
            "median_terminal": float(np.median(term)), "mean_terminal": float(np.mean(term)),
            "median_sharpe": float(np.median(sh)), "median_max_drawdown": float(np.median(mdd))}


def kelly_portfolio_var(f, cov, mu=None, level=0.99, include_mean=True) -> float:
    """Parametric one-period VaR of the position vector f (fraction of capital, POSITIVE loss):
    z_c sqrt(f' Sigma f) - f' mu (mean term only when include_mean and mu given)."""
    f = np.asarray(f, dtype=float).ravel()
    c = level / 100.0 if level > 1 else level
    v = _norm.ppf(c) * math.sqrt(float(f @ np.asarray(cov, float) @ f))
    if include_mean and mu is not None:
        v -= float(f @ np.asarray(mu, dtype=float).ravel())
    return float(v)


# ============================================================================ CTA vol targeting
def sma_seeded_ema(x, span):
    """SMA-seeded EMA: a = 2/(span+1); EMA[span-1] = mean(x[0:span]);
    EMA[t] = a x[t] + (1-a) EMA[t-1] for t >= span; NaN before the seed. Works column-wise on
    2-D input; returns the same shape as x (compute on PRICES, then align to returns with [1:])."""
    X = np.asarray(x, dtype=float)
    one = X.ndim == 1
    X2 = X[:, None] if one else X
    span = int(span)
    a = 2.0 / (span + 1.0)
    out = np.full(X2.shape, np.nan)
    if X2.shape[0] >= span:
        out[span - 1] = X2[:span].mean(axis=0)
        for t in range(span, X2.shape[0]):
            out[t] = a * X2[t] + (1.0 - a) * out[t - 1]
    return out[:, 0] if one else out


def ewma_variance_alpha(r, span=None, alpha=None, seed="first"):
    """EWMA variance with weight alpha on the CURRENT squared return:
    s2[0] = r[0]^2 (seed="first") or the given float; s2[t] = a r[t]^2 + (1-a) s2[t-1];
    a = 2/(span+1) when span is given. Column-wise for 2-D. Annualised vol = sqrt(A * s2)."""
    R = np.asarray(r, dtype=float)
    one = R.ndim == 1
    R2 = R[:, None] if one else R
    a = float(alpha) if alpha is not None else 2.0 / (float(span) + 1.0)
    s2 = np.empty(R2.shape)
    s2[0] = R2[0] ** 2 if seed == "first" else float(seed)
    for t in range(1, R2.shape[0]):
        s2[t] = a * R2[t] ** 2 + (1.0 - a) * s2[t - 1]
    return s2[:, 0] if one else s2


def garch11_conditional_vol(returns, mean="Constant", scale=100.0, dist="normal") -> dict:
    """GARCH(1,1) by arch on scale*returns (percent when scale=100), results back in DECIMAL:
    sigma_t = conditional_volatility/scale (daily); omega = omega_pct/scale^2; persistence =
    alpha+beta; long_run_variance = omega/(1 - alpha - beta) (decimal^2 per period)."""
    from arch import arch_model
    r = np.asarray(returns, dtype=float).ravel()
    res = arch_model(r * scale, mean=mean, vol="GARCH", p=1, q=1, dist=dist,
                     rescale=False).fit(disp="off")
    p = res.params
    omega = float(p["omega"]) / scale ** 2
    a, b = float(p["alpha[1]"]), float(p["beta[1]"])
    mu = float(p["mu"]) / scale if "mu" in p.index else 0.0
    sg = np.asarray(res.conditional_volatility, dtype=float) / scale
    # know76: aliases for the names the model guessed (conditional_volatility, volatility, alpha[1])
    return {"sigma": sg, "conditional_volatility": sg, "volatility": sg, "omega": omega,
            "alpha": a, "beta": b, "alpha[1]": a, "beta[1]": b, "persistence": a + b,
            "long_run_variance": omega / (1.0 - a - b) if a + b < 1 else float("nan"), "mu": mu}


def percentile_regimes(x, lo_pct, hi_pct, labels=("low", "mid", "high")) -> dict:
    """Regimes from time-series percentiles: thresh = np.percentile(x, pct) (linear);
    low if x <= thresh_lo, high if x >= thresh_hi, mid otherwise (high wins a tie only when
    thresholds coincide)."""
    v = np.asarray(x, dtype=float).ravel()
    lo, hi = float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
    codes = np.ones(len(v), dtype=int)
    codes[v <= lo] = 0
    codes[v >= hi] = 2
    lab = np.array(labels, dtype=object)[codes]
    n = len(v)
    return {"codes": codes, "labels": lab, "thresh_lo": lo, "thresh_hi": hi,
            "frac_low": float((codes == 0).sum() / n), "frac_mid": float((codes == 1).sum() / n),
            "frac_high": float((codes == 2).sum() / n)}


def vol_target_weights(signal, vol, target, scale=None, max_leverage=None):
    """w[t,i] = signal[t,i] * target * scale[t] / vol[t,i]; clipped to [-L, L]; NaN/inf -> 0.
    scale: None (static) or a length-T vector applied to every asset at the same t."""
    s = np.asarray(signal, dtype=float)
    v = np.asarray(vol, dtype=float)
    sc = 1.0 if scale is None else np.asarray(scale, dtype=float)
    if np.ndim(sc) == 1 and s.ndim == 2:
        sc = sc[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        w = s * target * sc / v
    w = np.where(np.isfinite(w), w, 0.0)
    if max_leverage is not None:
        w = np.clip(w, -max_leverage, max_leverage)
    return w


def lagged_portfolio_returns(weights, returns, lag=1):
    """rp[t] = sum_i w[t-lag, i] * r[t, i] (weights decided at the close of t-lag earn day t);
    rp[:lag] = 0; output length = len(returns)."""
    W = np.asarray(weights, dtype=float)
    R = np.asarray(returns, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    if R.ndim == 1:
        R = R[:, None]
    out = np.zeros(R.shape[0])
    out[lag:] = np.nansum(W[:-lag] * R[lag:], axis=1)
    return out


def log_return_performance(x, periods_per_year=252, rf_annual=0.0, ddof=0) -> dict:
    """Metrics of a LOG-return series: ann_log = mean*A; annualized_return = exp(ann_log)-1;
    annualized_volatility = std(ddof)*sqrt(A); sharpe = (ann_log - rf)/vol;
    wealth = exp(cumsum x); max_drawdown = min(wealth/cummax - 1) (<= 0) and its magnitude;
    calmar = annualized_return / |max_drawdown|."""
    v = np.asarray(x, dtype=float).ravel()
    ann_log = float(v.mean() * periods_per_year)
    vol = float(v.std(ddof=ddof) * math.sqrt(periods_per_year))
    wealth = np.exp(np.cumsum(v))
    dd = wealth / np.maximum.accumulate(wealth) - 1.0
    mdd = float(dd.min())
    ann = math.exp(ann_log) - 1.0
    sh = (ann_log - rf_annual) / vol if vol > 0 else float("nan")
    cal = ann / abs(mdd) if mdd != 0 else float("nan")
    # know76: *_ratio / annualized_vol aliases (KeyError 'sharpe_ratio' ended a run)
    return {"annualized_log_return": ann_log, "annualized_return": ann, "annualized_volatility": vol,
            "annualized_vol": vol, "sharpe": sh, "sharpe_ratio": sh, "max_drawdown": mdd,
            "max_drawdown_abs": abs(mdd), "calmar": cal, "calmar_ratio": cal}


# ============================================================================ eigen regimes / risk parity
def rolling_absorption_ratio(X, window, k, window_end="inclusive") -> dict:
    """Absorption ratio on rolling CORRELATION matrices (Kritzman-Page): AR = sum of the k
    largest eigenvalues (np.linalg.eigvalsh) / N. window_end="inclusive": window t uses rows
    t-W+1..t for t = W-1..T-1 (T-W+1 windows); "exclusive": rows t-W..t-1 for t = W..T-1.
    mp_threshold = (1 + sqrt(N/W))^2 (Marchenko-Pastur upper edge, diagnostic only)."""
    A = np.asarray(X, dtype=float)
    T, N = A.shape
    W = int(window)
    ends = range(W - 1, T) if window_end == "inclusive" else range(W, T)
    ar, big, idx = [], [], []
    for t in ends:
        rows = A[t - W + 1:t + 1] if window_end == "inclusive" else A[t - W:t]
        lam = np.sort(np.linalg.eigvalsh(np.corrcoef(rows, rowvar=False)))[::-1]
        ar.append(lam[:k].sum() / lam.sum())
        big.append(lam[0])
        idx.append(t)
    return {"ar": np.array(ar), "largest_eigenvalue": np.array(big), "end_idx": np.array(idx),
            "mp_threshold": (1.0 + math.sqrt(N / W)) ** 2}


def classify_upper_thresholds(x, thresholds, labels):
    """labels[i] for the first i (thresholds in DESCENDING order) with x > thresholds[i];
    labels[-1] when x is at or below every threshold (strict >: equality falls lower)."""
    v = np.asarray(x, dtype=float).ravel()
    out = np.full(len(v), labels[-1], dtype=object)
    done = np.zeros(len(v), dtype=bool)
    for thr, lab in zip(thresholds, labels):
        hit = (~done) & (v > thr)
        out[hit] = lab
        done |= hit
    return out


def month_end_flags(dates):
    """True on the last row of each calendar month present in `dates` (sorted); the final row
    of the sample is always True (its month ends there as far as the data go)."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    p = d.dt.to_period("M")
    return (p != p.shift(-1)).to_numpy()


def inverse_vol_weights(window_returns, ddof=1, budget=1.0):
    """w_i = (1/sd_i) / sum_j (1/sd_j) * budget with sd = column std(ddof) of the window."""
    sd = np.asarray(window_returns, dtype=float).std(axis=0, ddof=ddof)
    iv = 1.0 / sd
    return iv / iv.sum() * budget


def hold_weights_returns(X, rebalance_idx, weights):
    """Buy-and-hold-the-weights backtest without drift: for each day t after the FIRST rebalance
    index, r_p[t] = w(most recent rebalance index STRICTLY before t) . X[t]. Returns
    (r_p array, day indices)."""
    A = np.asarray(X, dtype=float)
    rb = np.asarray(rebalance_idx, dtype=int)
    Wt = np.atleast_2d(np.asarray(weights, dtype=float))
    days = np.arange(rb[0] + 1, A.shape[0])
    pos = np.searchsorted(rb, days, side="left") - 1
    rp = np.einsum("ij,ij->i", Wt[pos], A[days])
    return rp, days


def tail_var_cvar(returns, level):
    """Historical VaR/CVaR of a return sample as POSITIVE losses: VaR = -np.quantile(r, 1-c)
    (linear interpolation), CVaR = -mean(r[r <= -VaR])."""
    r = np.asarray(returns, dtype=float).ravel()
    c = level / 100.0 if level > 1 else level
    var = -float(np.quantile(r, 1.0 - c))
    tail = r[r <= -var]
    return var, (-float(tail.mean()) if len(tail) else var)


def loss_var_cvar_sorted(returns, level, clamp=True):
    """Order-statistic VaR: losses = -r sorted ascending; VaR = losses[ceil(c n) - 1]
    (clamped at 0 when clamp); CVaR = mean of losses STRICTLY greater than VaR (VaR if none)."""
    L = np.sort(-np.asarray(returns, dtype=float).ravel())
    c = level / 100.0 if level > 1 else level
    i = min(max(int(math.ceil(c * len(L))) - 1, 0), len(L) - 1)
    var = float(L[i])
    if clamp:
        var = max(var, 0.0)
    worse = L[L > var]
    return var, (float(worse.mean()) if len(worse) else var)


# ============================================================================ implied vol / surfaces
def _bs(S, K, T, r, q, sig, call):
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sq)
    d2 = d1 - sig * sq
    c = S * np.exp(-q * T) * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)
    return np.where(call, c, c - S * np.exp(-q * T) + K * np.exp(-r * T))


def _is_call(kind):
    arr = np.asarray(kind, dtype=object)
    f = np.vectorize(lambda s: str(s).strip().lower() not in ("put", "p"), otypes=[bool])
    out = f(arr)
    return bool(out) if out.ndim == 0 else out


def implied_vol_vec(price, S, K, T, r=0.0, q=0.0, kind="call", lo=1e-4, hi=5.0, xtol=1e-8):
    """Black-Scholes(-Merton) implied vol element-wise by Brent on [lo, hi] (xtol).

    Accepts scalars, arrays or pandas Series (broadcast). kind: 'call'/'put'/'C'/'P' or an array
    of them. NaN when T <= 0, price is not finite, or price is outside
    (BS(lo), BS(hi)) - i.e. below intrinsic / above the no-arbitrage bound."""
    P, S_, K_, T_, r_, q_ = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, S, K, T, r, q)))
    kd = np.broadcast_to(np.asarray(kind, dtype=object), P.shape)
    out = np.full(P.shape, np.nan)
    for idx in np.ndindex(P.shape):
        p, s, k, t, rr, qq = (float(a[idx]) for a in (P, S_, K_, T_, r_, q_))
        call = str(kd[idx]).strip().lower() not in ("put", "p")
        if not (np.isfinite(p) and t > 0 and s > 0 and k > 0):
            continue
        f = lambda v: float(_bs(s, k, t, rr, qq, v, call)) - p
        flo, fhi = f(lo), f(hi)
        if flo > 0 or fhi < 0:
            continue
        out[idx] = _opt.brentq(f, lo, hi, xtol=xtol, maxiter=500) if flo < 0 < fhi else (lo if flo == 0 else hi)
    return out if out.ndim else float(out)


def svi_raw(k, a, b, rho, m, sigma):
    """Raw SVI total implied variance (Gatheral 2004): w(k) = a + b[rho (k-m) + sqrt((k-m)^2 + sigma^2)],
    k = ln(K/F)."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_derivatives(k, a, b, rho, m, sigma):
    """(w, dw/dk, d2w/dk2) of raw SVI: w_k = b[rho + (k-m)/R], w_kk = b sigma^2 / R^3, R = sqrt((k-m)^2+sigma^2)."""
    k = np.asarray(k, dtype=float)
    R = np.sqrt((k - m) ** 2 + sigma ** 2)
    return svi_raw(k, a, b, rho, m, sigma), b * (rho + (k - m) / R), b * sigma ** 2 / R ** 3


def svi_fit(k, w, T=None, a_min=1e-6, x0=None, weights=None) -> dict:
    """Least-squares raw-SVI fit to observed total variance w(k) with bounds
    a in [a_min, max w], b in [1e-6, 10], rho in [-0.999, 0.999], m in [min k - 1, max k + 1],
    sigma in [1e-4, 5]; several starts, L-BFGS-B; constraint a + b sigma sqrt(1-rho^2) >= 0
    and b(1+|rho|) <= 4 by penalty. rmse_w on total variance; rmse_vol = RMSE of sqrt(w/T) when T
    is given (vol units)."""
    k = np.asarray(k, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    wt = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float).ravel()
    bnds = [(a_min, max(float(w.max()), a_min * 10)), (1e-6, 10.0), (-0.999, 0.999),
            (float(k.min()) - 1.0, float(k.max()) + 1.0), (1e-4, 5.0)]

    def obj(p):
        a, b, rho, m, s = p
        err = svi_raw(k, a, b, rho, m, s) - w
        pen = max(0.0, -(a + b * s * math.sqrt(1 - rho * rho))) + max(0.0, b * (1 + abs(rho)) - 4.0)
        return float(np.sum(wt * err * err)) + 1e3 * pen * pen

    starts = [x0] if x0 is not None else []
    wmin = max(float(w.min()), a_min * 2)
    for rho0 in (-0.5, 0.0, 0.3):
        for s0 in (0.05, 0.2):
            starts.append([wmin * 0.5, 0.1, rho0, float(k[np.argmin(w)]), s0])
    best = None
    for st in starts:
        st = [min(max(v, lo_), hi_) for v, (lo_, hi_) in zip(st, bnds)]
        res = _opt.minimize(obj, st, method="L-BFGS-B", bounds=bnds)
        if best is None or res.fun < best.fun:
            best = res
    a, b, rho, m, s = best.x
    fit = svi_raw(k, a, b, rho, m, s)
    out = {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(s),
           "rmse_w": float(np.sqrt(np.mean((fit - w) ** 2))), "success": bool(best.success)}
    if T is not None:
        out["rmse_vol"] = float(np.sqrt(np.mean((np.sqrt(np.maximum(fit, 0) / T) - np.sqrt(np.maximum(w, 0) / T)) ** 2)))
    return out


def gatheral_local_var(k, w, w_k, w_kk, w_T):
    """Dupire local variance in total-variance form (Gatheral 2006, eq. 1.10), k = ln(K/F_T):
    sigma_loc^2 = w_T / [1 - (k/w) w_k + 0.25 (-0.25 - 1/w + k^2/w^2) w_k^2 + 0.5 w_kk]."""
    k, w, w_k, w_kk, w_T = (np.asarray(v, dtype=float) for v in (k, w, w_k, w_kk, w_T))
    den = 1.0 - k / w * w_k + 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * w_k ** 2 + 0.5 * w_kk
    return w_T / den


def dupire_local_vol_grid(C, strikes, ttms, r=0.0, q=0.0, var_floor=None, vol_cap=None,
                          edge_order=2, den_min=None, kk_scheme="gradient") -> dict:
    """Dupire on a call-price grid C[nT, nK] (rows = maturities ascending, cols = strikes):
    sigma^2 = [dC/dT + (r - q) K dC/dK + q C] / [0.5 K^2 d2C/dK2]   (r = q = 0: 2 C_T / (K^2 C_KK)).
    Every derivative by numpy.gradient on the given (possibly non-uniform) axis with edge_order
    (edge_order=1: central inside, one-sided at the ends; 2 needs >= 3 points, else 1);
    d2C/dK2 = gradient applied twice. den_min: NaN where the denominator < den_min.
    var_floor / vol_cap: local_var <- max(local_var, var_floor) (non-finite -> floor), then vol
    capped. Returns dict of [nT, nK] arrays.
    know79 kk_scheme: 'gradient' (default here: gradient twice with edge_order, as tasks that NAME
    numpy.gradient require) or 'three_point' = the central second difference (C_{i+1} - 2C_i + C_{i-1})/h^2
    (non-uniform form), NaN at the first/last strike - the textbook reading of "central differences at
    interior points" (helpers79_2.fd_second_derivative)."""
    C = np.asarray(C, dtype=float)
    K = np.asarray(strikes, dtype=float).ravel()
    t = np.asarray(ttms, dtype=float).ravel()
    eo_t = edge_order if len(t) > edge_order else 1
    eo_k = edge_order if len(K) > edge_order else 1
    dT = np.gradient(C, t, axis=0, edge_order=eo_t) if len(t) > 1 else np.full_like(C, np.nan)
    dK = np.gradient(C, K, axis=1, edge_order=eo_k)
    if str(kk_scheme).lower() in ("three_point", "3pt", "central"):
        from .helpers79_2 import fd_second_derivative
        dKK = fd_second_derivative(C, K, axis=1, scheme="three_point")
    else:
        dKK = np.gradient(dK, K, axis=1, edge_order=eo_k)
    num = dT + (r - q) * K[None, :] * dK + q * C
    den = 0.5 * K[None, :] ** 2 * dKK
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = num / den
    if den_min is not None:
        lv = np.where(den < den_min, np.nan, lv)
    if var_floor is not None:
        lv = np.where(np.isfinite(lv), np.maximum(lv, var_floor), var_floor if den_min is None else lv)
    vol = np.sqrt(lv)
    if vol_cap is not None:
        vol = np.minimum(vol, vol_cap)
    return {"dC_dT": dT, "dC_dK": dK, "d2C_dK2": dKK, "local_var": lv, "local_vol": vol}


def forward_from_coin_parity(K, call_coin, put_coin):
    """Forward implied by a call-put pair whose premiums are quoted in the UNDERLYING coin
    (Deribit-style inverse options), discount factor 1: C_usd = c F, P_usd = p F and
    C_usd - P_usd = F - K  =>  F = K / (1 - (c - p))."""
    K, c, p = (np.asarray(v, dtype=float) for v in (K, call_coin, put_coin))
    F = K / (1.0 - (c - p))
    return F if F.ndim else float(F)


def parse_option_instrument(name, sep="-") -> dict:
    """'BTC-24APR26-64000-C' -> {underlying, expiry_label/expiry '24APR26', strike, kind/option_type/type 'call'|'put'}.

    Last token C/P or CALL/PUT; strike may contain 'd' as decimal point. know76: 'option_type',
    'type' and 'expiry' are aliases (a run crashed 4x on KeyError 'option_type')."""
    parts = str(name).strip().split(sep)
    if len(parts) < 4:
        raise ValueError(f"instrument {name!r} is not UNDERLYING{sep}EXPIRY{sep}STRIKE{sep}TYPE")
    kind = parts[-1].strip().upper()
    k = "call" if kind in ("C", "CALL") else "put"
    return {"underlying": parts[0], "expiry_label": parts[-3], "expiry": parts[-3],
            "strike": float(parts[-2].replace("d", ".")),
            "kind": k, "option_type": k, "type": k}


def local_vol_interpolator(strikes, ttms, lv_grid):
    """f(t, S) -> local vol by bilinear interpolation on the (T, K) grid, flat (clamped)
    outside it. lv_grid shape [len(ttms), len(strikes)]."""
    from scipy.interpolate import RegularGridInterpolator
    K = np.asarray(strikes, dtype=float).ravel()
    t = np.asarray(ttms, dtype=float).ravel()
    G = np.asarray(lv_grid, dtype=float)
    if len(t) == 1:
        t = np.array([t[0], t[0] + 1.0])
        G = np.vstack([G, G])
    f = RegularGridInterpolator((t, K), G, method="linear", bounds_error=False, fill_value=None)

    def lv(tt, S):
        S = np.asarray(S, dtype=float)
        pts = np.column_stack([np.full(S.size, min(max(float(tt), t[0]), t[-1])),
                               np.clip(S.ravel(), K[0], K[-1])])
        return f(pts).reshape(S.shape)
    return lv


def localvol_mc_barrier(S0, local_vol, T, n_monitor, n_sub, n_paths, K, barrier=None, r=0.0, q=0.0,
                        seed=None, antithetic=True, kind="down_and_out_call", H=None, strikes=None,
                        ttms=None) -> dict:
    """Low-level local-vol barrier MC (keys barrier_price, vanilla_price, ...); six-file coin-snapshot deliverables: localvol_barrier_report.

    A coin-snapshot task's barrier_results.json + summary.json in contract keys come from
    localvol_barrier_report(raw, task_rules, barrier_contract, snapshot_meta), which calls this engine.

    know79: local_vol may be a callable sigma(t, S), a constant, a long local-vol TABLE (DataFrame with
    ttm_years/T, strike, local_vol columns) or a 2-D grid with strikes=/ttms= - all go through
    helpers79_2.local_vol_callable (bilinear, flat outside). barrier= or H= (alias). S0, T, K, barrier may be
    1-element arrays/Series (a know78 run passed a Series T and a DataFrame local_vol and crashed 3x).
    T split into n_monitor equal monitoring periods (e.g. days) of n_sub sub-steps each,
    dt = T/(n_monitor n_sub). Log-Euler step S <- S exp((r-q-0.5 s^2) dt + s sqrt(dt) Z),
    s = local_vol(t, S) (callable, or a constant). The barrier is checked ONLY at the end of
    each monitoring period (daily close). kind: down/up and out/in call/put. Antithetic pairs
    (Z, -Z) when antithetic; SE from the pair averages. Discount exp(-rT).
    Returns barrier and vanilla prices/SE on the same paths, hit and survival probabilities."""
    from .helpers79_2 import local_vol_callable

    def _sc(v, name):
        a = np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=float).ravel()
        if a.size != 1:
            raise ValueError(f"localvol_mc_barrier: {name} must be ONE number, got {a.size} values")
        return float(a[0])
    if barrier is None:
        barrier = H
    if barrier is None:
        raise TypeError("localvol_mc_barrier: pass the barrier level (barrier= or H=)")
    S0, T, K, barrier = _sc(S0, "S0"), _sc(T, "T"), _sc(K, "K"), _sc(barrier, "barrier")
    n_monitor, n_sub, n_paths = (max(1, int(round(_sc(v, nm)))) for v, nm in
                                 ((n_monitor, "n_monitor"), (n_sub, "n_sub"), (n_paths, "n_paths")))
    local_vol = local_vol_callable(local_vol, strikes=strikes, ttms=ttms)
    g = np.random.default_rng(None if seed is None else int(seed))
    n_half = int(n_paths) // 2 if antithetic else int(n_paths)
    n = 2 * n_half if antithetic else n_half
    dt = T / (n_monitor * n_sub)
    S = np.full(n, float(S0))
    hit = np.zeros(n, dtype=bool)
    down = kind.startswith("down")
    lvf = local_vol if callable(local_vol) else (lambda t, s: np.full(np.shape(s), float(local_vol)))
    t = 0.0
    for _ in range(n_monitor):
        for _ in range(n_sub):
            Z = g.standard_normal(n_half)
            if antithetic:
                Z = np.concatenate([Z, -Z])
            s = np.asarray(lvf(t, S), dtype=float)
            S = S * np.exp((r - q - 0.5 * s * s) * dt + s * math.sqrt(dt) * Z)
            t += dt
        hit |= (S <= barrier) if down else (S >= barrier)
    pay = np.maximum(S - K, 0.0) if kind.endswith("call") else np.maximum(K - S, 0.0)
    disc = math.exp(-r * T)
    knocked = pay * (~hit) if "_out_" in kind else pay * hit
    def _est(x):
        x = disc * x
        if antithetic:
            x = 0.5 * (x[:n_half] + x[n_half:])
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    bp, bse = _est(knocked)
    vp, vse = _est(pay)
    hp = float(hit.mean())
    return {"barrier_price": bp, "barrier_se": bse, "vanilla_price": vp, "vanilla_se": vse,
            "hit_probability": hp, "survival_probability": 1.0 - hp, "n_paths": n}


# ============================================================================ two-factor Heston
def _norm_factors(factors):
    """know76: factors may be dicts {kappa, theta, sigma (xi/eta), rho, v0 (nu0/v_0/var0)} - a run
    unpacked dict KEYS as numbers ("can't multiply sequence by non-int of type 'str'") x6."""
    out = []
    for f in factors:
        if isinstance(f, dict):
            lo = {str(k).lower(): v for k, v in f.items()}

            def g(*names, _lo=lo, _f=f):
                for n in names:
                    if n in _lo:
                        return float(_lo[n])
                raise KeyError(f"heston factor {_f!r} lacks one of {names}")
            out.append((g("kappa"), g("theta"), g("sigma", "xi", "eta", "vol_of_vol"), g("rho"),
                        g("v0", "nu0", "v_0", "nu_0", "var0", "variance0")))
        else:
            out.append(tuple(float(x) for x in f))
    return out


def heston2_char_func(u, T, S0, r, q, factors):
    """Characteristic function of ln S_T, E[exp(i u ln S_T)], for dS/S = (r-q)dt + sum_j sqrt(v_j) dZ_j
    with independent CIR factors dv_j = kappa_j(theta_j - v_j)dt + sigma_j sqrt(v_j) dW_j,
    corr(dZ_j, dW_j) = rho_j (Christoffersen-Heston-Jacobs 2009). factors: [(kappa, theta, sigma,
    rho, v0), ...]. Little-trap form per factor: b = kappa - rho sigma i u,
    d = sqrt(b^2 + sigma^2 (i u + u^2)), g = (b-d)/(b+d),
    C_j = kappa theta/sigma^2 [(b-d)T - 2 ln((1 - g e^{-dT})/(1-g))], D_j = (b-d)/sigma^2 (1-e^{-dT})/(1-g e^{-dT}),
    phi = exp(i u (ln S0 + (r-q)T) + sum_j (C_j + D_j v0_j)). u may be complex/array."""
    u = np.asarray(u, dtype=complex)
    out = 1j * u * (math.log(S0) + (r - q) * T)
    for kappa, theta, sigma, rho, v0 in _norm_factors(factors):
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b * b + sigma * sigma * (1j * u + u * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        Cj = kappa * theta / sigma ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
        Dj = (b - d) / sigma ** 2 * (1.0 - e) / (1.0 - g * e)
        out = out + Cj + Dj * v0
    return np.exp(out)


_LEGGAUSS = {}


def _leggauss_cached(n):
    """np.polynomial.legendre.leggauss(n), computed once per n (know77: leggauss(2000) is a 2000x2000
    eigenproblem, ~0.5 s - it was recomputed for every maturity and every put/call, 24 times a surface)."""
    if n not in _LEGGAUSS:
        _LEGGAUSS[n] = np.polynomial.legendre.leggauss(n)
    return _LEGGAUSS[n]


def heston2_price(K, T, S0, r, q, factors, kind="call", method="gauss_legendre", n_nodes=2000,
                  upper=100.0, quad_lower=1e-8, quad_upper=50.0, quad_limit=200):
    """European price from the probabilities
    P1 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u - i) / (i u phi(-i))] du,
    P2 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u) / (i u)] du;
    call = S0 e^{-qT} P1 - K e^{-rT} P2; put (DIRECT) = K e^{-rT}(1 - P2) - S0 e^{-qT}(1 - P1).
    method="gauss_legendre": np.polynomial.legendre.leggauss(n_nodes) mapped to [0, upper]
    (eta = 0.5(x+1) upper, w = 0.5 upper w_x); "quad": scipy.integrate.quad on
    [quad_lower, quad_upper] with limit=quad_limit. Returns an array over K (float for scalar K)."""
    from scipy import integrate
    Ks = np.atleast_1d(np.asarray(K, dtype=float))
    phi_mi = heston2_char_func(-1j, T, S0, r, q, factors)

    def f1(u, lk):
        return np.real(np.exp(-1j * u * lk) * heston2_char_func(u - 1j, T, S0, r, q, factors) / (1j * u * phi_mi))

    def f2(u, lk):
        return np.real(np.exp(-1j * u * lk) * heston2_char_func(u, T, S0, r, q, factors) / (1j * u))

    P1, P2 = np.empty(len(Ks)), np.empty(len(Ks))
    method = {"gl": "gauss_legendre", "gauss-legendre": "gauss_legendre", "scipy_quad": "quad",
              "scipy.integrate.quad": "quad"}.get(str(method).lower(), str(method).lower())
    if method == "gauss_legendre":
        x, wx = _leggauss_cached(int(n_nodes))
        eta, w = 0.5 * (x + 1.0) * upper, 0.5 * upper * wx
        # know77: phi does not depend on K - evaluate it ONCE per maturity and vectorise over strikes
        # (same formula; a 12x12 surface went from ~14 s to well under 1 s, the cell had timed out).
        ph1 = heston2_char_func(eta - 1j, T, S0, r, q, factors) / (1j * eta * phi_mi)
        ph2 = heston2_char_func(eta, T, S0, r, q, factors) / (1j * eta)
        E = np.exp(-1j * eta[None, :] * np.log(Ks)[:, None])
        P1 = 0.5 + np.real(E * ph1[None, :]) @ w / math.pi
        P2 = 0.5 + np.real(E * ph2[None, :]) @ w / math.pi
    else:
        for i, kk in enumerate(Ks):
            lk = math.log(kk)
            P1[i] = 0.5 + integrate.quad(lambda u: float(f1(u, lk)), quad_lower, quad_upper, limit=quad_limit)[0] / math.pi
            P2[i] = 0.5 + integrate.quad(lambda u: float(f2(u, lk)), quad_lower, quad_upper, limit=quad_limit)[0] / math.pi
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    if str(kind).lower().startswith("c"):
        px = S0 * dq * P1 - Ks * dr * P2
    else:
        px = Ks * dr * (1.0 - P2) - S0 * dq * (1.0 - P1)
    return px if np.ndim(K) else float(px[0])


# ============================================================================ IV approximations
def _disc(C, S, K, T, r, q):
    C, S, K, T = (np.asarray(v, dtype=float) for v in (C, S, K, T))
    return C, S * np.exp(-q * T), K * np.exp(-r * T), T


def brenner_subrahmanyam_iv(C, S, K, T, r=0.0, q=0.0):
    """sigma = sqrt(2 pi / T) C / S~, S~ = S e^{-qT} (ATM approximation; apply on ATM rows)."""
    C, St, Kt, T = _disc(C, S, K, T, r, q)
    return np.sqrt(2.0 * np.pi / T) * C / St


def _li_cubic(alpha):
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = 3.0 * alpha / math.sqrt(32.0)
        z = np.cos(np.arccos(np.where(np.abs(arg) <= 1.0, arg, np.nan)) / 3.0)
        return 2.0 * math.sqrt(2.0) * z - np.sqrt(8.0 * z * z - 6.0 * alpha / (math.sqrt(2.0) * z))


def li_atm_iv(C, S, K, T, r=0.0, q=0.0):
    """Li (2005) ATM: alpha = sqrt(2 pi) C/S~; z = cos(arccos(3 alpha/sqrt 32)/3);
    sigma sqrt T = 2 sqrt2 z - sqrt(8 z^2 - 6 alpha/(sqrt2 z)); NaN outside the arccos domain."""
    C, St, Kt, T = _disc(C, S, K, T, r, q)
    return _li_cubic(math.sqrt(2.0 * math.pi) * C / St) / np.sqrt(T)


def li_nonatm_iv(C, S, K, T, r=0.0, q=0.0, rho_cut=0.5, disc_tol=1e-10, negative_disc="nan"):
    """Li (2005) / Chin-Olafsson-Nel non-ATM: eta = S~/K~, c = C/K~, alpha = sqrt(2 pi)(2c + 1 - eta)/(1 + eta),
    rho = |eta - 1| / c^2. rho <= rho_cut: the ATM cubic with this alpha; rho > rho_cut:
    sigma sqrt T = [alpha + sqrt(alpha^2 - 4 (eta - 1)^2/(1 + eta))]/2. A discriminant within
    disc_tol of 0 is set to 0; a clearly negative one -> NaN (negative_disc="nan") or 0 ("floor")."""
    C, St, Kt, T = _disc(C, S, K, T, r, q)
    eta, c = St / Kt, C / Kt
    alpha = math.sqrt(2.0 * math.pi) * (2.0 * c + 1.0 - eta) / (1.0 + eta)
    rho = np.abs(eta - 1.0) / c ** 2
    disc = alpha ** 2 - 4.0 * (eta - 1.0) ** 2 / (1.0 + eta)
    disc = np.where(np.abs(disc) < disc_tol, 0.0, disc)
    disc = np.where(disc < 0, np.nan if negative_disc == "nan" else 0.0, disc)
    with np.errstate(invalid="ignore"):
        quad = (alpha + np.sqrt(disc)) / 2.0
    return np.where(rho <= rho_cut, _li_cubic(alpha), quad) / np.sqrt(T)


def cmh_iv(C, S, K, T, r=0.0, q=0.0, negative_disc="floor"):
    """Corrado-Miller-Hallerbach rational approximation (Chin-Olafsson-Nel Vol II 7.2.1):
    sigma sqrt T = sqrt(2 pi)/(S~ + K~) [2C - (S~ - K~) + sqrt((2C - (S~ - K~))^2 - 2(S~ - K~)^2/pi)];
    a negative discriminant floored at 0 ("floor") or NaN ("nan")."""
    C, St, Kt, T = _disc(C, S, K, T, r, q)
    t1 = 2.0 * C - (St - Kt)
    disc = t1 * t1 - 2.0 * (St - Kt) ** 2 / math.pi
    disc = np.where(disc < 0, 0.0 if negative_disc == "floor" else np.nan, disc)
    return math.sqrt(2.0 * math.pi) / (St + Kt) * (t1 + np.sqrt(disc)) / np.sqrt(T)


def newton_iv_vec(price, S, K, T, r=0.0, q=0.0, kind="call", sigma0=0.3, tol=1e-10, max_iter=100):
    """RETURNS A 3-TUPLE (sigma array, converged, iterations) - unpack it: iv, ok, n = newton_iv_vec(...).

    Vectorised Newton-Raphson on BS: sigma <- sigma - (BS(sigma) - V)/vega,
    vega = S e^{-qT} n(d1) sqrt T; iterate ALL points together until all(|step| < tol) or
    max_iter. Returns (sigma array, converged, iterations).
    know77: `converged` is a PER-POINT boolean array whose truth value is all() - so both
    `if ok:` and `df['converged'] = ok` / `iv[ok]` work (a scalar flag broadcast over a frame
    deleted every row once one deep-OTM or zero-maturity point diverged)."""
    V, S_, K_, T_ = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (price, S, K, T)))
    call = _is_call(kind)
    sig = np.full(V.shape, float(sigma0))
    step = np.full(V.shape, np.inf)
    with np.errstate(all="ignore"):
        for it in range(1, int(max_iter) + 1):
            sq = np.sqrt(T_)
            d1 = (np.log(S_ / K_) + (r - q + 0.5 * sig * sig) * T_) / (sig * sq)
            vega = S_ * np.exp(-q * T_) * _norm.pdf(d1) * sq
            step = (_bs(S_, K_, T_, r, q, sig, call) - V) / vega
            sig = sig - step
            if np.all(np.abs(step) < tol):
                return sig, _ConvergedMask.of(np.isfinite(sig)), it
    ok = np.isfinite(sig) & (np.abs(step) < tol) & (sig > 0)
    return sig, _ConvergedMask.of(ok), int(max_iter)


class _ConvergedMask(np.ndarray):
    """Boolean per-point mask whose truth value is all() (works in `if ok:` and as a row mask)."""

    @classmethod
    def of(cls, a):
        return np.asarray(a, dtype=bool).view(cls)

    def __bool__(self):
        return bool(np.all(np.asarray(self)))

    def __repr__(self):
        return f"converged({np.asarray(self).tolist() if self.size <= 10 else f'{int(np.asarray(self).sum())}/{self.size}'})"


# ============================================================================ time to expiry
def _to_utc(x):
    if isinstance(x, pd.Series):
        s = x
    elif np.ndim(x) == 0:
        s = pd.Series([x])
    else:
        s = pd.Series(np.asarray(x))
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if np.nanmax(np.abs(s.to_numpy(dtype=float))) > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, format="mixed")


def ttm_years_exact(start, end, days_per_year=365.0):
    """Exact year fraction (end - start).total_seconds() / (days_per_year * 86400) - ACT/365 on
    timestamps, NOT on calendar dates (a 15:30 -> 16:00 expiry counts its half hour).
    start/end: ISO strings ('...Z' ok), Timestamps (tz-aware or naive = UTC), datetime64,
    pandas Series, or epoch numbers (> 1e11 read as milliseconds, else seconds). Broadcasts;
    scalar in -> float."""
    a, b = _to_utc(start), _to_utc(end)
    if len(a) == 1 and len(b) > 1:
        a = pd.Series([a.iloc[0]] * len(b))
    if len(b) == 1 and len(a) > 1:
        b = pd.Series([b.iloc[0]] * len(a))
    yrs = (b.reset_index(drop=True) - a.reset_index(drop=True)).dt.total_seconds().to_numpy() / (days_per_year * 86400.0)
    return float(yrs[0]) if (np.ndim(start) == 0 and np.ndim(end) == 0
                             and not isinstance(start, pd.Series) and not isinstance(end, pd.Series)) else yrs


# ============================================================================ Crank-Nicolson PSOR
def cn_fd_option(K, T, r, sigma, q=0.0, kind="put", american=True, S_max=None, N_S=300, N_T=600,
                 dividends=(), omega=1.2, tol=1e-8, max_sweeps=10000, S0=None, boundary_tol=1e-6) -> dict:
    """Crank-Nicolson on S_i = i dS (i = 0..N_S, S_max default 3K), t_n = n dt (n = 0..N_T),
    solved backward from the payoff at t = T.

    alpha_i = 0.25 dt (sigma^2 i^2 - (r-q) i), beta_i = -0.5 dt (sigma^2 i^2 + r),
    gamma_i = 0.25 dt (sigma^2 i^2 + (r-q) i);
    (-alpha, 1-beta, -gamma) V^n = (alpha, 1+beta, gamma) V^{n+1} on interior nodes,
    Dirichlet rows: put V(0) = K e^{-r tau} (American K), V(S_max) = 0; call V(0) = 0,
    V(S_max) = S_max e^{-q tau} - K e^{-r tau} (American: max with S_max - K), tau = T - t_n.
    European: banded direct solve. American: PSOR (red-black Gauss-Seidel ordering, numpy
    vectorised) with projection max(v, payoff), relaxation omega, stop when max|dv| < tol.
    Cash dividends [(amount, time), ...] at step n_d = round(time/dt): after V^{n_d} is
    computed, V(S, t_d-) = V(S - D, t_d+) by linear interpolation (S - D floored at 0), then the
    American projection again. boundary[n]: put = largest S_i with payoff > 0 and
    |V - payoff| < boundary_tol (0.0 if none); call = smallest such S_i.
    Returns dict{price at S0 (interpolated), delta central (V(S0+dS)-V(S0-dS))/(2 dS) at t = 0,
    S, t, V [N_T+1, N_S+1] (row n = time t_n), boundary, payoff}."""
    from scipy.linalg import solve_banded
    S_max = 3.0 * K if S_max is None else float(S_max)
    NS, NT = int(N_S), int(N_T)
    dS, dt = S_max / NS, T / NT
    S = np.arange(NS + 1) * dS
    t = np.arange(NT + 1) * dt
    call = str(kind).lower().startswith("c")
    pay = np.maximum(S - K, 0.0) if call else np.maximum(K - S, 0.0)
    i = np.arange(NS + 1, dtype=float)
    al = 0.25 * dt * (sigma ** 2 * i ** 2 - (r - q) * i)
    be = -0.5 * dt * (sigma ** 2 * i ** 2 + r)
    ga = 0.25 * dt * (sigma ** 2 * i ** 2 + (r - q) * i)
    I = slice(1, NS)
    lo, di, up = -al[I], 1.0 - be[I], -ga[I]
    ab = np.zeros((3, NS - 1))
    ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], di, lo[1:]
    divs = {}
    # know76: dividends may be {'time': t, 'amount': D} dicts; a tuple whose 'time' lies outside
    # [0, T] while its 'amount' lies inside is read as (time, amount) - a (time, amount) list meant
    # "amount at time" and was silently dropped (dividend prices equal to no-dividend prices).
    _pairs = []
    for d in dividends:
        if isinstance(d, dict):
            _pairs.append((float(d["amount"]), float(d["time"])))
            continue
        amt, td = float(d[0]), float(d[1])
        if not (0.0 <= td <= T) and 0.0 <= amt <= T:
            amt, td = td, amt
        _pairs.append((amt, td))
    for amt, td in _pairs:
        nd = int(round(float(td) / dt))
        if 0 <= nd <= NT:
            divs[nd] = divs.get(nd, 0.0) + float(amt)
    V = np.empty((NT + 1, NS + 1))
    V[NT] = pay
    bnd = np.zeros(NT + 1)
    red = np.arange(0, NS - 1, 2)
    blk = np.arange(1, NS - 1, 2)
    payI = pay[I]

    def _bound(vrow):
        m = (pay > 0) & (np.abs(vrow - pay) < boundary_tol)
        if not m.any():
            return 0.0
        return float(S[m].min()) if call else float(S[m].max())

    if american:
        bnd[NT] = _bound(V[NT])
    for n in range(NT - 1, -1, -1):
        tau = T - t[n]
        old = V[n + 1]
        if call:
            v0 = 0.0
            vN = S_max * math.exp(-q * tau) - K * math.exp(-r * tau)
            if american:
                vN = max(vN, S_max - K)
        else:
            v0 = K if american else K * math.exp(-r * tau)
            vN = 0.0
        rhs = al[I] * old[:-2] + (1.0 + be[I]) * old[1:-1] + ga[I] * old[2:]
        rhs[0] += al[1] * v0
        rhs[-1] += ga[NS - 1] * vN
        if not american:
            x = solve_banded((1, 1), ab, rhs)
        else:
            x = np.maximum(old[1:-1].copy(), payI)
            for _ in range(int(max_sweeps)):
                prev = x.copy()
                for idx in (red, blk):
                    left = np.where(idx > 0, x[np.maximum(idx - 1, 0)], 0.0)
                    right = np.where(idx < NS - 2, x[np.minimum(idx + 1, NS - 2)], 0.0)
                    y = (rhs[idx] - lo[idx] * left - up[idx] * right) / di[idx]
                    x[idx] = np.maximum(payI[idx], x[idx] + omega * (y - x[idx]))
                if np.max(np.abs(x - prev)) < tol:
                    break
        row = np.concatenate([[v0], x, [vN]])
        if n in divs and n > 0:
            row = np.interp(np.maximum(S - divs[n], 0.0), S, row)
            if american:
                row = np.maximum(row, pay)
        V[n] = row
        if american:
            bnd[n] = _bound(row)
    s0 = K if S0 is None else float(S0)
    price = float(np.interp(s0, S, V[0]))
    delta = float((np.interp(s0 + dS, S, V[0]) - np.interp(s0 - dS, S, V[0])) / (2 * dS))
    return {"price": price, "delta": delta, "S": S, "t": t, "V": V, "boundary": bnd, "payoff": pay}


# ============================================================================ OU fits / moments
def ou_jump_moments(x0, kappa, theta, sigma, tau, lam=0.0, mu_J=0.0, sigma_J=0.0, jumps_in_mean=True) -> dict:
    """-> {mean, var, E_S, Var_S, stationary_mean, stationary_var}; STATIONARY moments = keys stationary_* (or tau=np.inf), never tau=0 (var 0).

    Conditional moments of dX = kappa(theta - X)dt + sigma dW + J dN, J ~ N(mu_J, sigma_J^2),
    N Poisson(lam):
      E[X_tau] = theta + (x0 - theta) e^{-kappa tau} + (lam mu_J / kappa)(1 - e^{-kappa tau})  (jump drift when jumps_in_mean)
      Var[X_tau] = (sigma^2 + lam (mu_J^2 + sigma_J^2)) (1 - e^{-2 kappa tau}) / (2 kappa)   (jump variance INCLUDED)
    Lognormal approximation for S = e^X: E_S = exp(m + v/2), Var_S = (e^v - 1) e^{2m + v}.
    Stationary: theta + lam mu_J/kappa and (sigma^2 + lam(mu_J^2 + sigma_J^2))/(2 kappa).
    Set lam = 0 for the pure OU moments."""
    e = np.exp(-kappa * np.asarray(tau, dtype=float))
    jm = lam * mu_J / kappa
    m = theta + (x0 - theta) * e + (jm * (1 - e) if jumps_in_mean else 0.0)
    q2 = sigma ** 2 + lam * (mu_J ** 2 + sigma_J ** 2)
    v = q2 * (1 - e * e) / (2 * kappa)
    return {"mean": m, "var": v, "E_S": np.exp(m + 0.5 * v), "Var_S": (np.exp(v) - 1) * np.exp(2 * m + v),
            "stationary_mean": theta + (jm if jumps_in_mean else 0.0), "stationary_var": q2 / (2 * kappa)}


def ou_fit_euler_plugin(x, dt, ddof_resid=1) -> dict:
    """OU by OLS on the Euler-discretised SDE: dx_t = x_t - x_{t-1} = alpha + beta x_{t-1} + e_t.
    PLUG-IN Euler: kappa = -beta/dt, mu = -alpha/beta, sigma = std(e, ddof_resid)/sqrt(dt)
    (NOT the exact -ln(1+beta)/dt). half_life = ln2/kappa (time units of dt),
    half_life_periods = half_life/dt, stationary_std = sigma/sqrt(2 kappa); NaN when kappa <= 0.
    nobs = number of differences (len(x) - 1)."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    y, xl = np.diff(v), v[:-1]
    X = np.column_stack([np.ones_like(xl), xl])
    (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ np.array([alpha, beta])
    rs = float(np.std(e, ddof=ddof_resid))
    kappa = -beta / dt
    sig = rs / math.sqrt(dt)
    ok = kappa > 0
    hl = math.log(2) / kappa if ok else float("nan")
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return {"alpha": float(alpha), "beta": float(beta), "kappa": float(kappa),
            "mu": float(-alpha / beta) if beta != 0 else float("nan"), "sigma": sig,
            "half_life": hl, "half_life_periods": hl / dt if ok else float("nan"),
            "stationary_std": sig / math.sqrt(2 * kappa) if ok else float("nan"),
            "r_squared": r2, "resid_std": rs, "nobs": int(len(y))}


def ou_exact_paths(x0, kappa, mu, sigma, dt, n_steps, n_paths, seed=None, rng=None):
    """Exact OU transition x_{t+dt} = mu + (x_t - mu) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) Z,
    one rng.standard_normal(n_paths) draw per step in time order (rng = default_rng(seed)).
    Returns [n_paths, n_steps + 1] with column 0 = x0."""
    g = rng if rng is not None else np.random.default_rng(None if seed is None else int(seed))
    e = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1.0 - e * e) / (2.0 * kappa))
    out = np.empty((int(n_paths), int(n_steps) + 1))
    out[:, 0] = x0
    for t in range(int(n_steps)):
        out[:, t + 1] = mu + (out[:, t] - mu) * e + sd * g.standard_normal(int(n_paths))
    return out


# ============================================================================ monthly alignment / rolling HAC
def dated_index(s, name="series"):
    """The Series' index as dates. know90: a series whose index is a plain row NUMBER (0, 1, 2 ...) has lost its
    dates - typically `for df in [a, b]: df = df.sort_values(...); df.set_index('date', inplace=True)`, which
    re-binds the loop variable and leaves a, b unchanged - and pd.to_datetime(0, 1, 2 ...) silently dated every
    row 1970-01-01, so a date filter emptied everything. Such a series raises a ValueError that says so."""
    idx = s.index
    if isinstance(idx, pd.DatetimeIndex):
        return idx
    if isinstance(idx, pd.RangeIndex) or pd.api.types.is_integer_dtype(idx):
        mag = float(np.nanmax(np.abs(np.asarray(idx, float)))) if len(idx) else 0.0
        if mag < 1e8:       # row numbers, not epoch stamps
            raise ValueError(
                f"{name!r} is indexed by row numbers (0..{len(idx) - 1}), not dates - index it by its date column "
                "first: s = frame.set_index('<date column>')['<value column>']. A loop `for df in [...]: df = "
                "df.sort_values(...)` re-binds `df` and leaves the original frames unchanged; assign the result back "
                "(frames = {name: f.set_index(...) for name, f in ...}).")
    return pd.to_datetime(idx)


def monthly_last(s, label="start"):
    """Last valid observation of each calendar month (resample('ME').last() skips NaN), index
    relabelled to the first day of the month (label="start") or kept at month end ("end")."""
    obj = s.copy()
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj.index = dated_index(obj, getattr(obj, "name", None) or "series")
    m = obj.sort_index().resample("ME").last()
    if label == "start":
        m.index = m.index.to_period("M").to_timestamp()
    return m


def align_monthly(series, base=None, ffill=(), dropna=(), start=None, end=None) -> pd.DataFrame:
    """Merge monthly series {name: Series indexed by month start}. base given: LEFT-join every
    series onto base's month index; else outer join. Then forward-fill the `ffill` columns ON
    THE JOINED FRAME (a discontinued series carries its last value into later months), then drop
    rows missing any `dropna` column, then keep start <= month <= end.
    know76: a series with more than one observation in some month (daily data) or dated off the
    first of the month is collapsed with monthly_last first - joining raw daily values on month
    starts kept only the 1st-of-month business days (about half the months survived)."""
    series = dict(series)
    for _n, _s in list(series.items()):
        _s = _s.copy()
        if not isinstance(_s.index, pd.DatetimeIndex):
            _s.index = dated_index(_s, _n)
        _per = _s.index.to_period("M")
        if _per.duplicated().any() or not (_s.index == _per.to_timestamp()).all():
            _s = monthly_last(_s.dropna(), label="start")
        series[_n] = _s
    names = list(series)
    if base is not None:
        df = series[base].to_frame(base)
        for n in names:
            if n != base:
                df = df.join(series[n].rename(n), how="left")
    else:
        df = pd.concat({n: series[n] for n in names}, axis=1, join="outer").sort_index()
    for c in ffill:
        df[c] = df[c].ffill()
    if dropna:
        df = df.dropna(subset=list(dropna))
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def rolling_ols_hac(y, X, window, lags=3, add_const=True, index=None, names=None) -> pd.DataFrame:
    """Rolling OLS on windows of `window` rows ending at t (t = window-1..T-1), one row per window.

    beta = lstsq([1, X], y); e = y - Xb; R2 = 1 - SSE/SST.
    Newey-West: S = G0 + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j'), G0 = sum e_t^2 x_t x_t',
    G_j = sum_{t>j} e_t e_{t-j} x_t x_{t-j}'; V = (X'X)^-1 S (X'X)^-1 (no small-sample factor);
    t_hac_k = b_k / sqrt(V_kk).
    comp_k = b_k X_k[last]; residual = y[last] - a - sum comp_k;
    pct_k = Cov(b_k X_k, y)/Var(y) * 100 (ddof=1 both), pct_residual = (1 - R2) * 100 (sum = 100).
    names: factor names (default x1..xk); index: labels for the rows (window end label)."""
    yv = np.asarray(y, dtype=float).ravel()
    Xv = np.asarray(X, dtype=float)
    Xv = Xv[:, None] if Xv.ndim == 1 else Xv
    k = Xv.shape[1]
    names = list(names) if names is not None else [f"x{j + 1}" for j in range(k)]
    lab = np.arange(len(yv)) if index is None else np.asarray(index)
    rows = []
    for t in range(window - 1, len(yv)):
        yy = yv[t - window + 1:t + 1]
        Z = Xv[t - window + 1:t + 1]
        D = np.column_stack([np.ones(window), Z]) if add_const else Z
        b, *_ = np.linalg.lstsq(D, yy, rcond=None)
        e = yy - D @ b
        sst = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - float(e @ e) / sst
        XtXi = np.linalg.pinv(D.T @ D)
        Xe = D * e[:, None]
        Sm = Xe.T @ Xe
        for j in range(1, int(lags) + 1):
            G = Xe[j:].T @ Xe[:-j]
            Sm += (1.0 - j / (lags + 1.0)) * (G + G.T)
        V = XtXi @ Sm @ XtXi
        off = 1 if add_const else 0
        row = {"label": lab[t], "r_squared": r2, "intercept": float(b[0]) if add_const else 0.0}
        vy = float(np.var(yy, ddof=1))
        fitted_last = 0.0
        for j, nm in enumerate(names):
            bj = float(b[j + off])
            row[f"beta_{nm}"] = bj
            with np.errstate(divide="ignore", invalid="ignore"):
                row[f"t_hac_{nm}"] = bj / math.sqrt(V[j + off, j + off]) if V[j + off, j + off] > 0 else float("nan")
            row[f"comp_{nm}"] = bj * float(Z[-1, j])
            fitted_last += row[f"comp_{nm}"]
            row[f"pct_{nm}"] = float(np.cov(bj * Z[:, j], yy, ddof=1)[0, 1]) / vy * 100.0
        row["y"] = float(yy[-1])
        row["residual"] = float(yy[-1]) - row["intercept"] - fitted_last
        row["pct_residual"] = (1.0 - r2) * 100.0
        rows.append(row)
    out = pd.DataFrame(rows)
    # know76: the index carries the window-end labels too (results.index was 0..n-1 and the dates
    # were written as integers); the 'label' column stays.
    if index is not None and len(out):
        out.index = pd.Index(out["label"].to_numpy(), name="date")
    return out
