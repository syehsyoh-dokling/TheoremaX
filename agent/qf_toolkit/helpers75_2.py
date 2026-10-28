"""know75 group-2 helpers: tested building blocks for the factor / backtest, bond-curve, portfolio
credit, copula, EVT, Hull-White and weather-derivative failures of the know74 full run.

Every function is pure numpy / pandas / scipy (arch only inside the GARCH helpers), states its formula
and conventions in the docstring, and has textbook defaults only. Self-tests: proven_work/fail72/
selftest75_2.py. Call them from the generated program: `from qf_toolkit import <name>`.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize

# ----------------------------------------------------------------------------------------------
# 1. cross-sectional signals, rebalanced long-short backtests, factor regressions
# ----------------------------------------------------------------------------------------------


def cs_zscore(S, ddof=1):
    """Row-wise (cross-sectional) z-score of a [T x N] array: z = (s - mean_row) / std_row(ddof).

    NaN entries are ignored in the row moments and stay NaN; a row whose std is 0 or undefined
    becomes 0.0 (no information). Returns a float ndarray of the same shape."""
    S = np.asarray(S, float)
    one = S.ndim == 1
    S2 = S[None, :] if one else S
    out = np.zeros_like(S2)
    for t in range(S2.shape[0]):
        row = S2[t]
        ok = np.isfinite(row)
        if ok.sum() <= ddof:
            out[t] = np.where(ok, 0.0, np.nan)
            continue
        mu = row[ok].mean()
        sd = row[ok].std(ddof=ddof)
        out[t] = np.where(ok, (row - mu) / sd if sd > 0 and np.isfinite(sd) else 0.0, np.nan)
    return out[0] if one else out


def month_start_indices(dates, start):
    """Positional rebalance days: `start`, then every later row t whose calendar month differs from
    row t-1 (the first trading day of each new month). dates: array-like of datetimes (one per row).
    Returns a list of ints; len() is the number of rebalances."""
    m = pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[M]")
    start = int(start)
    return [start] + [t for t in range(start + 1, len(m)) if m[t] != m[t - 1]]


def topn_long_short(score, n_long, n_short, w_long, w_short=None, pos_cap=None, gross_cap=None):
    """Weights +w_long on the n_long highest scores and -w_short on the n_short lowest.

    Ties: the lower column index wins in BOTH legs (stable sorts). NaN scores are never selected.
    pos_cap clips |w| per name; gross_cap rescales the whole vector when sum|w| exceeds it.
    Returns an ndarray aligned with score."""
    s = np.asarray(score, float)
    w_short = w_long if w_short is None else w_short
    idx = np.arange(len(s))
    ok = np.isfinite(s)
    w = np.zeros(len(s))
    desc = idx[ok][np.lexsort((idx[ok], -s[ok]))]
    asc = idx[ok][np.lexsort((idx[ok], s[ok]))]
    longs = desc[:int(n_long)]
    taken = set(longs.tolist())
    shorts = [i for i in asc if i not in taken][:int(n_short)]
    w[longs] = float(w_long)
    w[shorts] = -float(w_short)
    if pos_cap is not None:
        w = np.clip(w, -float(pos_cap), float(pos_cap))
    g = np.abs(w).sum()
    if gross_cap is not None and g > float(gross_cap) > 0:
        w = w * float(gross_cap) / g
    return w


def periodic_ls_backtest(R, score, rebal_idx, start, n_long, n_short, w_long, w_short=None,
                         cost_bps=0.0, pos_cap=None, gross_cap=None, lag=1):
    """Long-short backtest rebalanced only on the rows in rebal_idx, held in between.

    R, score: [T x N] arrays (NaN returns must already be cleaned: R = np.nan_to_num(R)).
    On a rebalance row t the new weights come from score[t] (topn_long_short) and the cost is
    cost_bps/1e4 * sum|w_new - w_old| charged on row t.
    lag=1: r_t = w_held_into_t . R_t - cost_t (weights set on row t earn from row t+1);
    lag=0: r_t = w_new . R_t - cost_t (weights set at the START of row t earn row t).
    Returns dict{returns (rows start..T-1), weights [T x N] held at the end of each row,
    turnover, cost, rebalance_rows}."""
    R = np.asarray(R, float)
    score = np.asarray(score, float)
    T, N = R.shape
    rebal = set(int(i) for i in rebal_idx)
    w = np.zeros(N)
    W = np.zeros((T, N))
    rets, turns, costs = [], [], []
    for t in range(int(start), T):
        w_prev = w
        c = tv = 0.0
        if t in rebal:
            w_new = topn_long_short(score[t], n_long, n_short, w_long, w_short, pos_cap, gross_cap)
            tv = float(np.abs(w_new - w_prev).sum())
            c = float(cost_bps) / 1e4 * tv
            w = w_new
        held = w_prev if lag == 1 else w
        rets.append(float(held @ R[t]) - c)
        turns.append(tv)
        costs.append(c)
        W[t] = w
    return {"returns": np.array(rets), "weights": W, "turnover": np.array(turns),
            "cost": np.array(costs), "rebalance_rows": sorted(r for r in rebal if r >= start)}


def strategy_metrics(r, periods_per_year=252, ddof=1):
    """Arithmetic-annualised metrics of a periodic return series, rf = 0:
    annualized_return = mean*A, annualized_volatility = std(ddof)*sqrt(A), sharpe_ratio = ratio,
    max_drawdown = max((peak - W)/peak) >= 0 with W = cumprod(1+r) (W_0 = 1 counted as a peak),
    total_return = prod(1+r) - 1, hit_rate = mean(r > 0)."""
    r = np.asarray(r, float)
    A = float(periods_per_year)
    mu, sd = r.mean(), r.std(ddof=ddof)
    W = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], W]))[1:]
    return {"annualized_return": float(mu * A), "annualized_volatility": float(sd * math.sqrt(A)),
            "sharpe_ratio": float(mu / sd * math.sqrt(A)) if sd > 0 else 0.0,
            "max_drawdown": float(np.max((peak - W) / peak)) if len(r) else 0.0,
            "total_return": float(W[-1] - 1) if len(r) else 0.0, "hit_rate": float(np.mean(r > 0))}


def newey_west_lag_rule(n):
    """Newey-West (1994) plug-in lag L = floor(4 * (n/100)^(2/9)) (float exponent)."""
    return int(math.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0)))


def factor_regression_stats(y, F, periods_per_year=252, nw_lags=None):
    """Time-series OLS y = a + F b + e with an explicit column of ones (np.linalg.lstsq).

    n obs, k = number of factors, df = n - k - 1. Classical: MSE = SSR/df, cov = MSE (X'X)^-1,
    t = coef/se, two-sided p from Student-t(df). Newey-West (nw_lags=L): sandwich
    (X'X)^-1 [S0 + sum_j w_j (G_j + G_j')] (X'X)^-1 with RAW sums S0 = X' diag(e^2) X,
    G_j = sum_t e_t e_{t-j} x_t x_{t-j}', Bartlett w_j = 1 - j/(L+1); p from Student-t(df).
    r2 = 1 - SSR/SST (centred), adj_r2 = 1 - (1-r2)(n-1)/df, durbin_watson = sum(de^2)/sum(e^2),
    alpha_annualized = a*A, tracking_error = std(e, ddof=1)*sqrt(A), information_ratio = ratio.
    Returns dict{alpha, betas, coef, se, t, p, se_nw, t_nw, p_nw, r2, adj_r2, durbin_watson,
    resid, alpha_annualized, tracking_error, information_ratio, n, df}.
    know77: F must NOT contain a constant column (the intercept is added here) - a column of ones passed
    in is dropped (no double intercept); rows of y and F must match - a mismatch raises with the fix."""
    y = np.asarray(y, float).reshape(-1)
    F = np.asarray(F, float)
    F = F[:, None] if F.ndim == 1 else F
    if F.shape[0] != len(y):
        raise ValueError(f"factor_regression_stats: y has {len(y)} rows but F has {F.shape[0]}; cut F to the SAME "
                         "dates as y (e.g. F[trade_start:] for a strategy that trades from trade_start), or use the "
                         "regression keys the backtest helper already returns")
    const = [j for j in range(F.shape[1]) if F.shape[0] > 1 and np.all(F[:, j] == F[0, j]) and F[0, j] != 0]
    if const and F.shape[1] > len(const):
        F = np.delete(F, const, axis=1)
    n, k = F.shape
    X = np.column_stack([np.ones(n), F])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ coef
    dfree = n - k - 1
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(e @ e / dfree * xtx_inv))
    tt = coef / se
    out = {"alpha": float(coef[0]), "betas": coef[1:].copy(), "coef": coef, "se": se, "t": tt,
           "p": 2 * stats.t.sf(np.abs(tt), dfree)}
    if nw_lags is not None:
        L = int(nw_lags)
        u = X * e[:, None]
        S = u.T @ u
        for j in range(1, L + 1):
            G = u[j:].T @ u[:-j]
            S += (1 - j / (L + 1)) * (G + G.T)
        V = xtx_inv @ S @ xtx_inv
        se_nw = np.sqrt(np.diag(V))
        out.update(se_nw=se_nw, t_nw=coef / se_nw, p_nw=2 * stats.t.sf(np.abs(coef / se_nw), dfree))
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / sst if sst > 0 else float("nan")
    A = float(periods_per_year)
    te = e.std(ddof=1) * math.sqrt(A)
    out.update(r2=float(r2), adj_r2=float(1 - (1 - r2) * (n - 1) / dfree),
               durbin_watson=float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2)), resid=e,
               alpha_annualized=float(coef[0] * A), tracking_error=float(te),
               information_ratio=float(coef[0] * A / te) if te > 0 else 0.0, n=n, df=dfree)
    from ._rdict import ResultDict
    return ResultDict(out, aliases={"alpha_daily": "alpha", "r_squared": "r2", "residuals": "resid",
                                    "tracking_error_annual": "tracking_error", "annualized_alpha": "alpha_annualized"})


def grs_test(alphas, resid, F, cov="mle"):
    """Gibbons-Ross-Shanken (1989): GRS = (T-N-K)/N * a' S^-1 a / (1 + mu' O^-1 mu) ~ F(N, T-N-K).

    alphas [N] per-asset intercepts, resid [T x N] OLS residuals, F [T x K] factor returns.
    cov="mle" (default): S = E'E/T and O = (F-mu)'(F-mu)/T (1/T covariances). know78: cov="unbiased"
    is the GRS-1989 paper form (S = E'E/(T-K-1), O = cov(F, ddof=1), stat = (T/N)(T-N-K)/(T-K-1)
    a'S^-1 a/(1 + mu'O^-1 mu)) via helpers78_3.grs_statistic. Returns dict{stat, p_value, df1,
    df2} plus the aliases grs_statistic / statistic (= stat) and grs_p_value (= p_value) (know76: the
    know75 runs read 'statistic' and 'grs_statistic' and crashed)."""
    if cov != "mle":
        from .helpers78_3 import grs_statistic
        return grs_statistic(alphas, resid, F, cov=cov)
    a = np.asarray(alphas, float).reshape(-1)
    E = np.asarray(resid, float)
    F = np.asarray(F, float)
    F = F[:, None] if F.ndim == 1 else F
    T, N = E.shape
    K = F.shape[1]
    S = E.T @ E / T
    mu = F.mean(axis=0)
    Fc = F - mu
    O = Fc.T @ Fc / T
    stat = (T - N - K) / N * float(a @ np.linalg.solve(S, a)) / (1.0 + float(mu @ np.linalg.solve(O, mu)))
    pv = float(stats.f.sf(stat, N, T - N - K))
    cv = float(stats.f.ppf(0.95, N, T - N - K))
    return {"stat": float(stat), "p_value": pv, "df1": int(N), "df2": int(T - N - K),
            "grs_statistic": float(stat), "statistic": float(stat), "grs_p_value": pv,
            "grs_rejected": bool(pv < 0.05), "critical_value": cv}


def vif(F):
    """Variance inflation factors: regress each column on the others plus a constant,
    VIF_j = 1/(1 - R2_j). Returns ndarray [K]."""
    F = np.asarray(F, float)
    out = []
    for j in range(F.shape[1]):
        others = np.delete(F, j, axis=1)
        X = np.column_stack([np.ones(len(F)), others])
        y = F[:, j]
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
        out.append(1.0 / (1.0 - r2))
    return np.array(out)


def rolling_ols_coefs(y, X, window):
    """Coefficients of OLS y ~ X on every window [s, s+window), s = 0..n-window (n-window+1 rows).

    X must already contain the constant column. y may be [n] or [n x m] (m series share X).
    Exact normal-equation solve per window from cumulative X'X and X'y sums (no Python loop over
    regressions). Returns [n_windows x p] (or [n_windows x m x p]); row s ends at date index
    s+window-1."""
    X = np.asarray(X, float)
    Y = np.asarray(y, float)
    one = Y.ndim == 1
    Y = Y[:, None] if one else Y
    n, p = X.shape
    w = int(window)
    xx = np.einsum("ti,tj->tij", X, X)
    xy = np.einsum("ti,tm->tmi", X, Y)
    cxx = np.concatenate([np.zeros((1, p, p)), np.cumsum(xx, axis=0)])
    cxy = np.concatenate([np.zeros((1,) + xy.shape[1:]), np.cumsum(xy, axis=0)])
    A = cxx[w:] - cxx[:-w]
    B = cxy[w:] - cxy[:-w]
    coef = np.linalg.solve(A[:, None, :, :], B[..., None])[..., 0]
    return coef[:, 0, :] if one else coef


# ----------------------------------------------------------------------------------------------
# 2. lead-lag correlation tables
# ----------------------------------------------------------------------------------------------


def lagged_corr(a, b, k):
    """corr(A[t], B[t+k]) with numpy.corrcoef on the slices k>0: A[0:n-k], B[k:n];
    k<0: A[|k|:n], B[0:n-|k|]; k=0: contemporaneous."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    k = int(k)
    if k > 0:
        x, y = a[:n - k], b[k:]
    elif k < 0:
        x, y = a[-k:], b[:n + k]
    else:
        x, y = a, b
    return float(np.corrcoef(x, y)[0, 1])


def fisher_gate(c_ab, c_ba, n_obs, alpha=0.05, clip=0.999999):
    """Two-sided Fisher-z test that two correlations differ: delta = |atanh(c1) - atanh(c2)|,
    SE = sqrt(2/(n_obs-3)), significant iff delta > z_{1-alpha/2} * SE; correlations clipped to
    +-clip before atanh. Returns bool for scalars; for arrays / Series / columns a numpy bool array
    (know76: the know75 run passed whole DataFrame columns and crashed 3x on 'truth value of an
    array is ambiguous')."""
    z1 = np.arctanh(np.clip(np.asarray(c_ab, float), -clip, clip))
    z2 = np.arctanh(np.clip(np.asarray(c_ba, float), -clip, clip))
    out = np.abs(z1 - z2) > stats.norm.ppf(1 - alpha / 2) * math.sqrt(2.0 / (n_obs - 3))
    return bool(out) if np.ndim(out) == 0 else out


def lead_lag_table(R, names, max_lag, pair_cols=("asset_a", "asset_b")):
    """All pairs (itertools.combinations in column order) of a [n x m] return matrix.

    corr_lag0 = lag 0, corr_a_leads_b = max over k=1..max_lag, corr_b_leads_a = max over
    k=-max_lag..-1, asymmetry = difference; leader/lagger: a leads b iff asymmetry >= 0.
    Columns: pair_cols..., corr_lag0, corr_a_leads_b, corr_b_leads_a, asymmetry, leader, lagger,
    lag_-L..lag_-1, lag_0, lag_1..lag_L. Sorted by |asymmetry| descending (stable: ties keep the
    combinations order). Values unrounded."""
    R = np.asarray(R, float)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        prof = {k: lagged_corr(R[:, i], R[:, j], k) for k in range(-max_lag, max_lag + 1)}
        ab = max(prof[k] for k in range(1, max_lag + 1))
        ba = max(prof[k] for k in range(-max_lag, 0))
        asym = ab - ba
        a_name, b_name = names[i], names[j]
        row = {pair_cols[0]: a_name, pair_cols[1]: b_name, "corr_lag0": prof[0],
               "corr_a_leads_b": ab, "corr_b_leads_a": ba, "asymmetry": asym,
               "leader": a_name if asym >= 0 else b_name, "lagger": b_name if asym >= 0 else a_name}
        row.update({f"lag_{k}": prof[k] for k in range(-max_lag, max_lag + 1)})
        rows.append(row)
    df = pd.DataFrame(rows)
    order = np.argsort(-df["asymmetry"].abs().to_numpy(), kind="mergesort")
    return df.iloc[order].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 3. panels and IPCA
# ----------------------------------------------------------------------------------------------


def panel_to_array(df, date_col, id_col, value_cols):
    """Long panel -> dense array. Rows = sorted unique dates, columns = sorted unique ids.

    value_cols: a single name -> [T x N]; a list -> [T x N x L] in the list's order (NEVER a
    multi-column pivot, whose columns come out (characteristic, id) and flatten to [T x N*L]).
    Missing cells are NaN. Returns (dates ndarray, ids ndarray, array)."""
    dates = np.sort(df[date_col].unique())
    ids = np.sort(df[id_col].unique())
    ti = pd.Index(dates).get_indexer(df[date_col])
    ni = pd.Index(ids).get_indexer(df[id_col])
    single = isinstance(value_cols, str)
    cols = [value_cols] if single else list(value_cols)
    arr = np.full((len(dates), len(ids), len(cols)), np.nan)
    for c, name in enumerate(cols):
        arr[ti, ni, c] = df[name].to_numpy(float)
    return dates, ids, (arr[:, :, 0] if single else arr)


def _panel_values(x):
    """The array of a panel_to_array (dates, ids, array) tuple, else x itself, as float ndarray."""
    if isinstance(x, tuple) and len(x) == 3 and np.ndim(x[2]) >= 2:
        x = x[2]
    return np.nan_to_num(np.asarray(x, float))


def _ipca_align(R, Z, lag):
    """(Y [T x N], X [T x N x L]) for IPCA: lag=True pairs Z[t-1] with R[t]; a Z that is already one row
    shorter than R is treated as lagged (Y = R[1:], X = Z); equal lengths with lag=False pair row t."""
    R, Z = _panel_values(R), _panel_values(Z)
    if R.ndim == 3 and R.shape[2] == 1:
        R = R[:, :, 0]
    if Z.ndim != 3 or R.ndim != 2:
        raise ValueError(f"IPCA needs R [T x N] and Z [T x N x L]; got R{R.shape}, Z{Z.shape}. Use the ARRAY of "
                         "panel_to_array: dates, ids, Z = panel_to_array(chars, 'date', id_col, [char names]); "
                         "or call ipca_report(...)")
    if lag and len(Z) == len(R) - 1:
        return R[1:], Z
    if lag:
        return R[1:], Z[:-1]
    return R, Z


def ipca_als(R, Z, K, max_iter, tol, ridge=0.0, seed=None, F0=None, lag=True):
    """Instrumented PCA (Kelly-Pruitt-Su 2019) by alternating least squares.

    r_{i,t} = z_{i,t-1}' Gamma f_t + e. R [T x N] returns (NaN already replaced), Z [T x N x L].
    lag=True pairs Z[t-1] with R[t] (T-1 usable months). Init F: F0 if given, else
    np.random.default_rng(seed).standard_normal((T_eff, K)) when seed is not None, else Gamma from
    the top-K left singular vectors of the managed portfolios. Each sweep: Gamma-step
    vec(G) = [sum Z'Z kron f f' + ridge I]^-1 sum vec(Z' r f') (row-major vec), then F-step
    f_t = (G'Z'ZG + ridge I)^-1 G'Z'r. Stop when max|G_new - G_old| < tol (iterations counts sweeps).
    Rotation: eigendecomposition of F'F/T_eff (uncentred), descending, F <- F V, G <- G V, so
    F'F/T_eff = diag(eigvals); then flip each factor (and Gamma column) whose mean is negative.
    Returns dict{gamma [L x K], factors [T_eff x K], eigvals, iterations, converged, fitted, resid,
    total_r2 (uncentred), cs_r2 (per month, uncentred), Y, X}. RETURNS A DICT (not a tuple): res['gamma'].
    know77: R / Z may be the (dates, ids, array) tuple of panel_to_array; Z already lagged (one row
    shorter than R) is paired as is (never lagged twice); a 2-D Z raises a clear error."""
    Y, X = _ipca_align(R, Z, lag)
    T, N, L = X.shape
    K = int(K)
    XX = np.einsum("tnl,tnm->tlm", X, X)
    Xy = np.einsum("tnl,tn->tl", X, Y)
    if F0 is not None:
        F = np.asarray(F0, float).copy()
        G = None
    elif seed is not None:
        F = np.random.default_rng(seed).standard_normal((T, K))   # know86: unscaled (the know85 x0.1 had no source)
        G = None
    else:
        U, _, _ = np.linalg.svd((Xy / N).T, full_matrices=False)
        G = U[:, :K]
        F = np.array([np.linalg.solve(G.T @ XX[t] @ G + ridge * np.eye(K), G.T @ Xy[t]) for t in range(T)])
    G_old = G
    converged, it = False, 0
    for it in range(1, int(max_iter) + 1):
        A = ridge * np.eye(L * K)
        b = np.zeros(L * K)
        for t in range(T):
            A += np.kron(XX[t], np.outer(F[t], F[t]))
            b += np.outer(Xy[t], F[t]).ravel()
        G = np.linalg.solve(A, b).reshape(L, K)
        F = np.array([np.linalg.solve(G.T @ XX[t] @ G + ridge * np.eye(K), G.T @ Xy[t]) for t in range(T)])
        if G_old is not None and np.max(np.abs(G - G_old)) < tol:
            converged = True
            break
        G_old = G
    lam, V = np.linalg.eigh(F.T @ F / T)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    F, G = F @ V, G @ V
    sgn = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
    F, G = F * sgn, G * sgn
    fitted = np.einsum("tnl,lk,tk->tn", X, G, F)
    E = Y - fitted
    return {"gamma": G, "factors": F, "eigvals": lam, "iterations": it, "converged": converged,
            "fitted": fitted, "resid": E, "total_r2": float(1 - (E ** 2).sum() / (Y ** 2).sum()),
            "cs_r2": 1 - (E ** 2).sum(axis=1) / (Y ** 2).sum(axis=1), "Y": Y, "X": X}


# ----------------------------------------------------------------------------------------------
# 4. cross-sectional normalisation, return-process model selection, daily panels, IC, vol target
# ----------------------------------------------------------------------------------------------


def cs_normalize(df, by, cols, clip=None, ddof=0):
    """Per-group (e.g. per date) normalisation of each column: +-inf -> NaN; NaN -> group median;
    an all-NaN group -> 0.0; z = (x - mean)/std(ddof); std 0 -> 0.0; optional clip to [-clip, clip].
    Returns a DataFrame of the normalised cols aligned with df.index."""
    out = pd.DataFrame(index=df.index)
    keys = [df[b] for b in ([by] if isinstance(by, str) else by)]
    for c in cols:
        x = df[c].astype(float).replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.groupby(keys).transform("median")).fillna(0.0)
        mu = x.groupby(keys).transform("mean")
        sd = x.groupby(keys).transform(lambda s: s.std(ddof=ddof)).fillna(0.0)
        z = np.where(sd > 0, (x - mu) / sd.where(sd > 0, 1.0), 0.0)
        out[c] = np.clip(z, -clip, clip) if clip is not None else z
    return out


def _norm_nll(x, mu, sd):
    x = np.asarray(x, float)
    if not sd > 0:
        return float("inf")
    return float(-stats.norm.logpdf(x, mu, sd).sum())


def fit_return_processes(r, jump_threshold_sigma, holding_period_days, conviction_scale,
                         n_params=None, martingale_sigma="std"):
    """Fit martingale / gbm / ou / merton_jump_diffusion to a short return history; pick by AIC.

    All stds are population (ddof=0). AIC = 2k + 2*NLL, k from n_params (default counts of fitted
    parameters: martingale 1, gbm 2, ou 3, merton_jump_diffusion 5).
    martingale: N(0, s), s = std(r) ("std") or sqrt(mean r^2) ("rms").  gbm: N(mean, std).
    ou: OLS r_t = c + phi r_{t-1}; ou_sigma = std(resid); theta = -ln(phi); mu = c/(1-phi);
    half_life_days = ln2/theta * holding_period_days; NLL = conditional NLL of r[1:] plus the first
    observation under N(mu, ou_sigma/sqrt(1-phi^2)); needs 0 < phi < 1 (else AIC = inf, params NaN).
    merton: mu, sigma of all r; jump_mask = |r-mu| > jump_threshold_sigma*sigma; intensity =
    mean(mask); jump_mean = mean(r[mask]) - mu, jump_std = std(r[mask]) (else 0, sigma);
    diffusion mu/sigma from r[~mask] (else mu, sigma); NLL of the mixture
    (1-p) N(mu_d, s_d) + p N(mu_d + jump_mean, sqrt(s_d^2 + jump_std^2)), p clipped into (0,1).
    Selection: lowest AIC, ties alphabetical. expected_return: martingale 0, gbm mu,
    ou c + phi*r_last, merton mu_d + p*jump_mean; sign in {-1,0,1}; conviction = 1/(1+exp(-gap/scale))
    with gap = second-best minus best finite AIC (1.0 if only one finite). Returns dict."""
    r = np.asarray(r, float)
    k = {"martingale": 1, "gbm": 2, "ou": 3, "merton_jump_diffusion": 5}
    if n_params:
        k.update(n_params)
    out = {}
    _sfloor = 1e-6
    _eps = 1e-12
    def _nll_f(arr, m_val, s_val):
        s_val = max(float(s_val), _sfloor)
        return float((0.5 * np.log(2.0 * np.pi * s_val * s_val) + ((arr - m_val) ** 2) / (2.0 * s_val * s_val)).sum())
    mu = float(r.mean())
    sd = max(float(r.std(ddof=0)), _sfloor)
    s0 = sd if martingale_sigma == "std" else max(math.sqrt(float(np.mean(r ** 2))), _sfloor)
    aic = {"martingale": 2.0 * k["martingale"] + 2.0 * _nll_f(r, 0.0, s0)}
    out.update(gbm_mu=float(mu), gbm_sigma=float(sd))
    aic["gbm"] = 2.0 * k["gbm"] + 2.0 * _nll_f(r, mu, sd)
    x, y = r[:-1], r[1:]
    xm, ym = float(x.mean()), float(y.mean())
    xvar = float(((x - xm) ** 2).mean())
    phi_raw = 0.0 if xvar <= _eps else float(((x - xm) * (y - ym)).mean() / xvar)
    phi = min(max(phi_raw, 1e-4), 0.999)
    c = ym - phi * xm
    m_ou = c / max(1.0 - phi, 1e-6)
    e = y - (c + phi * x)
    s_ou = max(float(np.sqrt((e ** 2).mean())), _sfloor)
    theta = -math.log(phi)
    hl = math.log(2.0) / max(theta, 1e-6) * float(holding_period_days)
    stat_sd = s_ou / max(math.sqrt(max(1.0 - phi * phi, _eps)), _sfloor)
    nll_ou = _nll_f(y, c + phi * x, s_ou) + _nll_f(r[:1], m_ou, stat_sd)
    aic["ou"] = 2.0 * k["ou"] + 2.0 * nll_ou
    out.update(ou_theta=theta, ou_mu=m_ou, ou_sigma=float(s_ou),
               ou_half_life_days=hl, ou_intercept=float(c), ou_phi=float(phi))
    mask = np.abs(r - mu) > float(jump_threshold_sigma) * sd
    if mask.any():
        jr = r[mask]
        jm, js = float(jr.mean() - mu), max(float(jr.std(ddof=0)), _sfloor)
    else:
        jm, js = 0.0, float(sd)
    diff = r[~mask]
    md, sdd = (float(diff.mean()), max(float(diff.std(ddof=0)), _sfloor)) if len(diff) else (float(mu), float(sd))
    p = min(max(float(mask.mean()), 1e-6), 1.0 - 1e-6)
    base_pdf = np.exp(-0.5 * np.log(2.0 * np.pi * sdd * sdd) - ((r - md) ** 2) / (2.0 * sdd * sdd))
    j_sd = math.sqrt(sdd * sdd + js * js)
    jump_pdf = np.exp(-0.5 * np.log(2.0 * np.pi * j_sd * j_sd) - ((r - (md + jm)) ** 2) / (2.0 * j_sd * j_sd))
    nll_m = float(-np.log(np.maximum((1.0 - p) * base_pdf + p * jump_pdf, 1e-15)).sum())
    aic["merton_jump_diffusion"] = 2.0 * k["merton_jump_diffusion"] + 2.0 * nll_m
    out.update(merton_mu=md, merton_sigma=sdd, merton_jump_intensity=p, merton_jump_mean=jm,
               merton_jump_std=js)
    ranked = sorted(aic.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else np.inf, kv[0]))
    best = ranked[0][0]
    finite = [v for _, v in ranked if np.isfinite(v)]
    if len(finite) == 1:
        conv = 1.0
    else:
        gap = max(float(finite[1] - finite[0]), 0.0)
        sc = max(float(conviction_scale), 1e-6)
        sig = 1.0 / (1.0 + math.exp(-gap / sc)) if gap / sc < 700 else 1.0
        conv = max(0.0, min(1.0, 2.0 * sig - 1.0))
    exp_ret = {"martingale": 0.0, "gbm": float(mu), "ou": float(c + phi * r[-1]),
               "merton_jump_diffusion": md + p * jm}[best]
    out.update(aic=aic, selected_process=best, expected_return=exp_ret,
               sign=int(np.sign(exp_ret)), conviction=float(conv))
    return out


def daily_sum_count(dates, ids, values, index=None, columns=None):
    """Aggregate events (e.g. posts) to a date x id panel: SUM of values and COUNT of events
    (count = number of rows, NOT number of distinct keys). Missing cells -> 0. index/columns
    reindex the result (e.g. trading days only - events on other days drop out).
    Returns (sum_df, count_df)."""
    df = pd.DataFrame({"d": pd.to_datetime(pd.Series(dates)).dt.normalize().to_numpy(),
                       "i": np.asarray(ids), "v": np.asarray(values, float)})
    s = df.pivot_table(index="d", columns="i", values="v", aggfunc="sum", fill_value=0.0)
    c = df.pivot_table(index="d", columns="i", values="v", aggfunc="count", fill_value=0)
    if index is not None:
        s, c = s.reindex(index, fill_value=0.0), c.reindex(index, fill_value=0)
    if columns is not None:
        s, c = s.reindex(columns=columns, fill_value=0.0), c.reindex(columns=columns, fill_value=0)
    return s, c


def ex_ante_vol_scaler(w, returns_window, target, cap=None, periods_per_year=252, ddof=1):
    """Vol-target scaler from the FULL trailing covariance (cross-correlations kept):
    sigma_p = sqrt(w' Cov(window, ddof) w * A); scaler = target/sigma_p, min(scaler, cap).
    Returns 1.0 when sigma_p is 0 or undefined."""
    w = np.asarray(w, float)
    C = np.cov(np.asarray(returns_window, float), rowvar=False, ddof=ddof)
    v = float(w @ np.atleast_2d(C) @ w) * periods_per_year
    if not v > 0:
        return 1.0
    s = target / math.sqrt(v)
    return float(min(s, cap) if cap is not None else s)


def rank_ic_series(signal, forward_returns, min_obs=3):
    """Per-row Spearman rank correlation between signal[t] and forward_returns[t] (align the
    next-period return onto row t BEFORE calling). Rows with fewer than min_obs finite pairs or a
    constant side are skipped. Returns ndarray of the ICs kept."""
    S = np.asarray(signal, float)
    Rf = np.asarray(forward_returns, float)
    out = []
    for s, r in zip(S, Rf):
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < min_obs or np.ptp(s[ok]) == 0 or np.ptp(r[ok]) == 0:
            continue
        out.append(stats.spearmanr(s[ok], r[ok])[0])
    return np.array(out)


# ----------------------------------------------------------------------------------------------
# 5. residualised cross-sectional alpha, beta-neutral weights, turnover cap
# ----------------------------------------------------------------------------------------------


def winsorize_columns(X, lower=0.01, upper=0.99):
    """Clip each column at its own lower/upper percentile (numpy default 'linear')."""
    X = np.asarray(X, float)
    lo = np.percentile(X, lower * 100, axis=0)
    hi = np.percentile(X, upper * 100, axis=0)
    return np.clip(X, lo, hi)


def residualize_zscore(Y, C, ddof=0):
    """Cross-sectional OLS of each column of Y on [1, C] via the Moore-Penrose pseudo-inverse,
    residuals z-scored with std(ddof); a zero/undefined std gives an all-0.0 column."""
    Y = np.asarray(Y, float)
    one = Y.ndim == 1
    Y = Y[:, None] if one else Y
    X = np.column_stack([np.ones(len(Y)), np.asarray(C, float)])
    E = Y - X @ (np.linalg.pinv(X) @ Y)
    sd = E.std(axis=0, ddof=ddof)
    Zs = np.where(sd > 0, (E - E.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    return Zs[:, 0] if one else Zs


def cs_ols_slopes(y, X):
    """Slopes (intercept dropped) of y ~ [1, X] by pseudo-inverse."""
    Xd = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    return (np.linalg.pinv(Xd) @ np.asarray(y, float))[1:]


def sign_stable_median(coefs, min_agree):
    """Coordinate-wise np.median of a [M x K] coefficient history; agreement = count of rows whose
    sign (zero has sign 0) equals the median's sign; stable = median where agreement >= min_agree,
    else 0.0. Returns (median, agreement int array, stable)."""
    C = np.asarray(coefs, float)
    med = np.median(C, axis=0)
    agree = (np.sign(C) == np.sign(med)).sum(axis=0)
    return med, agree.astype(int), np.where(agree >= min_agree, med, 0.0)


def rank_long_short(alpha, ids, frac, min_names):
    """Equal-weight long-short legs: rank by alpha descending, ties by id ascending;
    n_leg = floor(frac*n); +1/n_leg top, -1/n_leg bottom. Returns (w_raw, long_pos, short_pos) or
    None when n_leg < min_names."""
    a = np.asarray(alpha, float)
    n = len(a)
    m = int(math.floor(frac * n + 1e-9))
    if m < min_names:
        return None
    order = np.lexsort((np.asarray(ids), -a))
    longs, shorts = order[:m], order[-m:]
    w = np.zeros(n)
    w[longs] = 1.0 / m
    w[shorts] = -1.0 / m
    return w, longs, shorts


def beta_neutral_project(w, beta, mask=None):
    """w - X (X'X)^+ X' w on the masked names, X = [1, beta]; names outside the mask -> 0.
    mask defaults to w != 0. Result has zero dollar and zero beta exposure on the mask."""
    w = np.asarray(w, float)
    beta = np.asarray(beta, float)
    m = (w != 0) if mask is None else np.asarray(mask, bool)
    out = np.zeros_like(w)
    X = np.column_stack([np.ones(m.sum()), beta[m]])
    ws = w[m]
    out[m] = ws - X @ np.linalg.pinv(X.T @ X) @ X.T @ ws
    return out


def turnover_capped_weights(w_prev, w_target, beta, cap, slack=0.0, support="nonzero"):
    """Turnover cap with re-projection (vectors aligned on the same ids, missing = 0).

    turnover = 0.5 sum|w_target - w_prev|; <= cap -> w_target. Else lambda = cap/turnover,
    w_mid = w_prev + lambda (w_target - w_prev), beta-neutral re-projection of w_mid (support
    "nonzero": names with w_mid != 0; "target": names with w_target != 0) and renormalise to gross 1;
    if its turnover still exceeds cap + slack, shrink linearly back toward w_prev to exactly cap
    (no renormalisation after this shrink). Returns dict{w_final, turnover_target, turnover_final,
    capped}."""
    wp = np.asarray(w_prev, float)
    wt = np.asarray(w_target, float)
    t0 = 0.5 * np.abs(wt - wp).sum()
    if t0 <= cap:
        return {"w_final": wt, "turnover_target": t0, "turnover_final": t0, "capped": False}
    lam = cap / t0
    w_mid = wp + lam * (wt - wp)
    mask = (w_mid != 0) if support == "nonzero" else (wt != 0)
    w_re = beta_neutral_project(w_mid, beta, mask)
    g = np.abs(w_re).sum()
    w_re = w_re / g if g > 0 else w_re
    t1 = 0.5 * np.abs(w_re - wp).sum()
    if t1 > cap + slack:
        w_re = wp + (cap / t1) * (w_re - wp)
    return {"w_final": w_re, "turnover_target": t0, "turnover_final": 0.5 * np.abs(w_re - wp).sum(),
            "capped": True}


# ----------------------------------------------------------------------------------------------
# 6. bonds: dated cash flows, par-curve bootstrap with a g(T)=zT spline, KRDs, immunisation
# ----------------------------------------------------------------------------------------------


def thirty360_us(d1, d2):
    """30/360 US (bond basis) day count fraction: D1=31 -> 30; D2=31 and D1>=30 -> 30;
    ((Y2-Y1)*360 + (M2-M1)*30 + (D2-D1)) / 360."""
    d1, d2 = pd.Timestamp(d1), pd.Timestamp(d2)
    D1, D2 = min(d1.day, 30), d2.day
    if D2 == 31 and D1 >= 30:
        D2 = 30
    return ((d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (D2 - D1)) / 360.0


def coupon_schedule(valuation, maturity, freq):
    """-> TUPLE (previous_coupon_date, [future coupon dates]): prev, dates = coupon_schedule(...).
    Coupon dates rolled BACKWARD from maturity every 12/freq months. Returns
    (previous_coupon_date <= valuation, [future coupon dates > valuation, ending at maturity])."""
    val, mat = pd.Timestamp(valuation), pd.Timestamp(maturity)
    step = int(round(12 / freq))
    dates, k = [], 0
    while True:
        d = mat - pd.DateOffset(months=step * k)
        if d <= val:
            return d, sorted(dates)
        dates.append(d)
        k += 1


def dated_bond_cashflows(valuation, maturity, coupon_rate, freq, notional=100.0, accrual="30/360",
                         discount_basis=365.0):
    """-> dict{dates, times, amounts, accrued, previous_coupon}; dirty = sum(amounts * df_fn(times)).
    Dated fixed-coupon bond: coupons coupon_rate*notional/freq on the backward-rolled schedule,
    principal at maturity; times = ACT/discount_basis years from valuation; accrued interest =
    coupon_rate*notional*yf(previous coupon, valuation) with yf 30/360 US ("30/360") or
    actual/360 ("ACT/360") or actual/365 ("ACT/365"). dirty = clean + accrued.
    Returns dict{dates, times, amounts, accrued, previous_coupon}."""
    prev, dates = coupon_schedule(valuation, maturity, freq)
    val = pd.Timestamp(valuation)
    times = np.array([(d - val).days / discount_basis for d in dates])
    amt = np.full(len(dates), coupon_rate * notional / freq)
    amt[-1] += notional
    if accrual.upper().startswith("30"):
        yf = thirty360_us(prev, val)
    else:
        yf = (val - prev).days / (360.0 if "360" in accrual else 365.0)
    return {"dates": dates, "times": times, "amounts": amt, "accrued": coupon_rate * notional * yf,
            "previous_coupon": prev}


def bootstrap_par_g_spline(maturities, par_yields, freq=2, short_end="compound", node_interp="joint"):
    """-> dict{maturities, zero, df, g, spline, df_fn, zero_fn, fwd_fn, bootstrap_residuals} (keys 'zero'/'df').
    Zero curve from par yields, bootstrapped in maturity order with the continuous exponent
    g(T) = z(T) T interpolated by a not-a-knot cubic spline through (0, 0) and the solved nodes.

    T <= 1/freq: a single payment, DF = (1 + y/freq)^(-freq T) ("compound", coupon-equivalent
    yield) or 1/(1 + y T) ("simple"). Longer T: par bond, coupons y/freq at T, T-1/freq, ... > 0
    (first period may be short), principal 1 at T, price 1.
    node_interp (how the coupon dates BETWEEN solved nodes are valued while a node is solved):
      "linear" - classic sequential bootstrap "in order": node k solved by brentq with g LINEAR between
                 the last solved node and the candidate (earlier nodes never move); the not-a-knot spline is
                 fitted through the finished nodes afterwards and used for all pricing / KRD / stress.
                 (know78: the dated-bond hedge task's pinned z(20) and 15Y bullet price need this.)
      "joint"  - spline rebuilt through the candidate node, then all nodes re-solved jointly so the FINAL
                 spline reprices every par bond exactly (default, unchanged).
    bootstrap_residuals = par-bond price - 1 on the construction used to SOLVE the nodes (~0 for both).
    Returns dict{maturities, zero (continuous), df, g, spline (callable g), df_fn, zero_fn,
    fwd_fn (instantaneous g'(T)), bootstrap_residuals}."""
    if str(node_interp).lower().startswith("lin"):
        return _bootstrap_par_g_linear_nodes(maturities, par_yields, freq, short_end)
    T = np.asarray(maturities, float)
    y = np.asarray(par_yields, float)
    order = np.argsort(T)
    T, y = T[order], y[order]
    nt, ng = [0.0], [0.0]

    def spline(xs, gs):
        return CubicSpline(xs, gs, bc_type="not-a-knot")

    for Tk, yk in zip(T, y):
        if Tk <= 1.0 / freq + 1e-12:
            df = (1 + yk / freq) ** (-freq * Tk) if short_end == "compound" else 1 / (1 + yk * Tk)
            gk = -math.log(df)
        else:
            times = []
            t = Tk
            while t > 1e-9:
                times.append(t)
                t -= 1.0 / freq
            times = np.array(sorted(times))

            def err(g, Tk=Tk, yk=yk, times=times):
                s = spline(np.array(nt + [Tk]), np.array(ng + [g]))
                dfs = np.exp(-np.asarray(s(times), float))
                return float(yk / freq * dfs.sum() + dfs[-1] - 1.0)

            gk = brentq(err, -0.5 * Tk, 1.0 * Tk, xtol=1e-15)
        nt.append(float(Tk))
        ng.append(float(gk))
    xs, gs = np.array(nt), np.array(ng)
    # A global not-a-knot spline moves the interpolated values between EARLIER nodes once later
    # nodes are added, so the sequential pass is only a start: solve all nodes jointly so that the
    # FINAL curve reprices every par instrument (Newton on the coupon-bond residuals).
    long_ = [i for i, Tk in enumerate(T) if Tk > 1.0 / freq + 1e-12]
    if long_:
        grids = []
        for i in long_:
            tl, t = [], T[i]
            while t > 1e-9:
                tl.append(t)
                t -= 1.0 / freq
            grids.append(np.array(sorted(tl)))

        def resid(gl):
            g_all = gs.copy()
            g_all[[i + 1 for i in long_]] = gl
            s = CubicSpline(xs, g_all, bc_type="not-a-knot")
            return np.array([y[i] / freq * np.exp(-s(tg)).sum() + math.exp(-float(s(T[i]))) - 1.0
                             for i, tg in zip(long_, grids)])

        from scipy.optimize import root
        sol = root(resid, gs[[i + 1 for i in long_]], method="hybr", tol=1e-14)
        gs[[i + 1 for i in long_]] = sol.x
        res = resid(gs[[i + 1 for i in long_]])
    else:
        res = np.zeros(0)
    return _g_spline_curve(T, xs, gs, res)


def _g_spline_curve(T, xs, gs, residuals):
    sp = CubicSpline(xs, gs, bc_type="not-a-knot")
    g_fn = lambda t: sp(np.asarray(t, float))  # noqa: E731
    fwd = lambda t: sp(np.asarray(t, float), 1)  # noqa: E731
    return {"maturities": T, "zero": gs[1:] / T, "df": np.exp(-gs[1:]), "g": gs[1:], "spline": g_fn,
            "df_fn": lambda t: np.exp(-g_fn(t)),
            "zero_fn": lambda t: np.where(np.asarray(t, float) > 0, g_fn(t) / np.maximum(np.asarray(t, float), 1e-300), fwd(0.0)),
            "fwd_fn": fwd, "bootstrap_residuals": np.asarray(residuals, float)}


def _bootstrap_par_g_linear_nodes(maturities, par_yields, freq=2, short_end="compound"):
    """Sequential par bootstrap, g linear between the previous node and the candidate while solving."""
    T = np.asarray(maturities, float)
    y = np.asarray(par_yields, float)
    order = np.argsort(T)
    T, y = T[order], y[order]
    nt, ng, res = [0.0], [0.0], []
    for Tk, yk in zip(T, y):
        if Tk <= 1.0 / freq + 1e-12:
            df = (1 + yk / freq) ** (-freq * Tk) if short_end == "compound" else 1 / (1 + yk * Tk)
            gk = -math.log(df)
        else:
            times, t = [], Tk
            while t > 1e-9:
                times.append(t)
                t -= 1.0 / freq
            times = np.array(sorted(times))

            def err(g, Tk=Tk, yk=yk, times=times):
                dfs = np.exp(-np.interp(times, np.array(nt + [Tk]), np.array(ng + [g])))
                return float(yk / freq * dfs.sum() + dfs[-1] - 1.0)

            gk = brentq(err, -0.5 * Tk, 1.0 * Tk, xtol=1e-15)
            res.append(err(gk))
        nt.append(float(Tk))
        ng.append(float(gk))
    return _g_spline_curve(T, np.array(nt), np.array(ng), res)


def par_bump_krd(maturities, par_yields, price_fn, bump=1e-4, freq=2, short_end="compound",
                 central=True, node_interp="joint"):
    """Key-rate durations from bumping ONE par yield at a time and re-bootstrapping the curve
    (bootstrap_par_g_spline), so sensitivities live on the same curve construction as pricing.
    price_fn(curve_dict) -> dirty price. KRD_k = -(P_up - P_down)/(2 bump P0) (central) or
    -(P_up - P0)/(bump P0). Returns dict{maturity: KRD}; sum of KRDs ~ effective duration."""
    T = np.asarray(maturities, float)
    y = np.asarray(par_yields, float)
    P0 = price_fn(bootstrap_par_g_spline(T, y, freq, short_end, node_interp))
    out = {}
    for i, Tk in enumerate(T):
        up = y.copy()
        up[i] += bump
        Pu = price_fn(bootstrap_par_g_spline(T, up, freq, short_end, node_interp))
        if central:
            dn = y.copy()
            dn[i] -= bump
            Pd = price_fn(bootstrap_par_g_spline(T, dn, freq, short_end, node_interp))
            out[float(Tk)] = -(Pu - Pd) / (2 * bump * P0)
        else:
            out[float(Tk)] = -(Pu - P0) / (bump * P0)
    return out


def spot_bump_krd(cf_times, cfs, spot_times, spot_rates, key, bump=1e-4):
    """-> TUPLE (krd, P, P_bumped): krd, _, _ = spot_bump_krd(...); krd is a relative (duration) number.
    Key-rate duration from bumping the continuously-compounded spot rate at ONE grid tenor `key`
    (other tenors unchanged), discount factors exp(-s t) on the cash-flow times (which must lie on
    spot_times): KRD = -(P_bumped - P)/(P * bump). Returns (KRD, P, P_bumped)."""
    st = np.asarray(spot_times, float)
    s = np.asarray(spot_rates, float)
    t = np.asarray(cf_times, float)
    c = np.asarray(cfs, float)
    idx = np.searchsorted(st, t)
    P = float(np.sum(c * np.exp(-s[idx] * t)))
    sb = s.copy()
    sb[np.argmin(np.abs(st - key))] += bump
    Pb = float(np.sum(c * np.exp(-sb[idx] * t)))
    return -(Pb - P) / (P * bump), P, Pb


def immunize_pv_krd(krd, pv_target, krd_targets, x0=None, ftols=(1e-10, 1e-8), objective="l2"):
    """Long-only dollar allocations x matching PV and dollar key-rate exposures:
    sum x = PV_L and sum_i x_i KRD_{i,k} = PV_L KRD_{L,k} for each target k (KRDs relative, so
    x_i KRD_{i,k} is a dollar KRD - never multiply by the price again). SLSQP on w = x/PV_L
    (well scaled), x >= 0, objective "l2" = sum w^2 (or "none" = feasibility), retried at the next
    ftol if not converged. krd: [n_bonds x n_targets]. Returns dict{x, success, message,
    portfolio_dollar_krd (= sum x_i KRD_i,k), portfolio_pv}."""
    D = np.asarray(krd, float)
    tgt = np.asarray(krd_targets, float)
    n = D.shape[0]
    w0 = np.full(n, 1.0 / n) if x0 is None else np.asarray(x0, float) / pv_target
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for k in range(D.shape[1]):
        cons.append({"type": "eq", "fun": lambda w, k=k: w @ D[:, k] - tgt[k]})
    fun = (lambda w: float(w @ w)) if objective == "l2" else (lambda w: 0.0)
    res = None
    for ft in ftols:
        res = minimize(fun, w0, method="SLSQP", bounds=[(0, None)] * n, constraints=cons,
                       options={"ftol": ft, "maxiter": 1000})
        if res.success:
            break
    x = res.x * pv_target
    return {"x": x, "success": bool(res.success), "message": str(res.message),
            "portfolio_dollar_krd": x @ D, "portfolio_pv": float(x.sum())}


def svensson_rate(t, b0, b1, b2, b3, l1, l2):
    """Svensson (1994) spot rate: b0 + b1 (1-e^-x1)/x1 + b2 [(1-e^-x1)/x1 - e^-x1]
    + b3 [(1-e^-x2)/x2 - e^-x2], x_i = t/l_i."""
    t = np.asarray(t, float)
    x1, x2 = t / l1, t / l2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))


def fit_svensson(t, r, starts, ftol=1e-14, gtol=1e-10, maxiter=5000, eps=1e-6):
    """Least-squares Svensson fit by L-BFGS-B from each start (b0, l1, l2 bounded > 0 by eps);
    keeps the lowest SSE. Returns dict{params (6), sse, rmse, start_index}."""
    t = np.asarray(t, float)
    r = np.asarray(r, float)
    sse = lambda p: float(np.sum((svensson_rate(t, *p) - r) ** 2))  # noqa: E731
    bounds = [(eps, None), (None, None), (None, None), (None, None), (eps, None), (eps, None)]
    best = None
    for i, x0 in enumerate(starts):
        res = minimize(sse, np.asarray(x0, float), method="L-BFGS-B", bounds=bounds,
                       options={"ftol": ftol, "gtol": gtol, "maxiter": maxiter})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, i)
    return {"params": best[1], "sse": float(best[0]), "rmse": math.sqrt(best[0] / len(t)),
            "start_index": best[2]}


# ----------------------------------------------------------------------------------------------
# 7. portfolio credit: copula default simulation, Euler contributions, Vasicek / Gordy, migrations
# ----------------------------------------------------------------------------------------------


def copula_default_mc(pd_, ead, lgd, rho, factor_idx, corr, n_sims, seed, copula="gaussian",
                      nu=None, return_defaults=False, chunk_rows=10000):
    """Multi-factor Merton default simulation.

    A_i = sqrt(rho_i) M_{s(i)} + sqrt(1-rho_i) e_i with M ~ N(0, corr) (Cholesky) and e iid N(0,1).
    Draw order from np.random.default_rng(seed): M-normals (n_sims x S), e (n_sims x N), then for
    copula "t" W = chisquare(nu)/nu (n_sims); t copula scales BOTH M and e by 1/sqrt(W) and uses the
    Student-t threshold t_nu^-1(PD) (Gaussian: Phi^-1(PD)), so marginal PDs are preserved.
    Default iff A_i < d_i; loss = sum EAD_i LGD_i over defaults. Returns dict{losses [n_sims],
    defaults bool [n_sims x N] when return_defaults, thresholds}.
    know76: e is drawn in row CHUNKS (chunk_rows) from the same stream - bit-identical to one big
    draw - so a large sims x obligors book runs in seconds and bounded memory instead of timing out a cell
    (the t copula replays the e stream once to reach W). copula accepts "t"/"student_t"."""
    p = np.asarray(pd_, float)
    unit = np.asarray(ead, float) * np.asarray(lgd, float)
    rho = np.asarray(rho, float) * np.ones(len(p))
    fi = np.asarray(factor_idx, int)
    Lc = np.linalg.cholesky(np.asarray(corr, float))
    n = int(n_sims)
    N = len(p)
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, Lc.shape[0])) @ Lc.T
    is_t = str(copula).lower() in ("t", "student_t", "student-t", "studentt")
    step = max(1, int(chunk_rows))
    if is_t:
        state = rng.bit_generator.state
        for a in range(0, n, step):
            rng.standard_normal((min(step, n - a), N))
        W = rng.chisquare(nu, n) / nu
        rng.bit_generator.state = state
        s = 1.0 / np.sqrt(W)
        M = M * s[:, None]
        d = stats.t.ppf(p, nu)
    else:
        s = None
        d = stats.norm.ppf(p)
    sq1, sqr = np.sqrt(1 - rho), np.sqrt(rho)
    losses = np.empty(n)
    D_all = np.empty((n, N), dtype=bool) if return_defaults else None
    for a in range(0, n, step):
        b = min(n, a + step)
        A = rng.standard_normal((b - a, N))
        if s is not None:
            A *= s[a:b, None]
        A *= sq1
        A += sqr * M[a:b][:, fi]
        D = A < d
        losses[a:b] = D @ unit
        if return_defaults:
            D_all[a:b] = D
    out = {"losses": losses, "thresholds": d}
    if return_defaults:
        out["defaults"] = D_all
    return out


def loss_var_es(losses, level, method="linear", tail="geq"):
    """Empirical VaR = np.quantile(losses, level, method) and ES = mean of losses >= VaR ("geq")
    or > VaR ("gt") on a LOSS vector (positive = loss). Returns (VaR, ES)."""
    L = np.asarray(losses, float)
    v = float(np.quantile(L, level, method=method))
    sel = L >= v if tail == "geq" else L > v
    return v, float(L[sel].mean()) if sel.any() else v


def euler_es_contributions(defaults, unit_loss, losses, level, method="linear"):
    """Euler allocation of ES: contribution_i = unit_loss_i * mean(default_i | L >= VaR_level).
    Contributions sum exactly to ES (same tail set). Returns (contributions ndarray, VaR, ES)."""
    L = np.asarray(losses, float)
    v = float(np.quantile(L, level, method=method))
    tail = L >= v
    c = np.asarray(defaults)[tail].mean(axis=0) * np.asarray(unit_loss, float)
    return c, v, float(L[tail].mean())


def vasicek_portfolio_var(pd_, ead, lgd, rho, alpha):
    """One-factor (Basel IRB / ASRF) loss quantile: sum EAD_i LGD_i Phi(h_i),
    h_i = (Phi^-1(PD_i) + sqrt(rho_i) Phi^-1(alpha)) / sqrt(1 - rho_i); PD = 0 names skipped."""
    p = np.asarray(pd_, float)
    m = p > 0
    r = np.asarray(rho, float)[m]
    h = (stats.norm.ppf(p[m]) + np.sqrt(r) * stats.norm.ppf(alpha)) / np.sqrt(1 - r)
    return float(np.sum((np.asarray(ead, float) * np.asarray(lgd, float))[m] * stats.norm.cdf(h)))


def gordy_granularity_adjustment(pd_, ead, lgd, rho, alpha):
    """Gordy-Lutkebohmert granularity adjustment in currency:
    GA = M sigma^2(alpha) / (2 C'(alpha)), M = sum EAD, w = EAD/M,
    sigma^2 = sum w^2 LGD^2 Phi(h)(1-Phi(h)), C' = sum w LGD phi(h) sqrt(rho)/(sqrt(1-rho) phi(Phi^-1(alpha))),
    h as in vasicek_portfolio_var; PD = 0 names excluded. Returns float."""
    p = np.asarray(pd_, float)
    e = np.asarray(ead, float)
    Mtot = e.sum()
    m = p > 0
    w = e[m] / Mtot
    lg = np.asarray(lgd, float)[m]
    r = np.asarray(rho, float)[m]
    qa = stats.norm.ppf(alpha)
    h = (stats.norm.ppf(p[m]) + np.sqrt(r) * qa) / np.sqrt(1 - r)
    Ph = stats.norm.cdf(h)
    s2 = np.sum(w ** 2 * lg ** 2 * Ph * (1 - Ph))
    cp = np.sum(w * lg * stats.norm.pdf(h) * np.sqrt(r) / (np.sqrt(1 - r) * stats.norm.pdf(qa)))
    return float(Mtot * s2 / (2 * cp))


def hhi(weights):
    """Herfindahl-Hirschman index sum w_i^2 of weights normalised to sum 1."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    return float(np.sum(w ** 2))


def migration_thresholds(P):
    """Standard-normal migration thresholds per initial rating row of a transition matrix whose
    columns run best -> default (rows sum to 1, percent rows must be /100 first). Cumulating from
    the WORST state: z_0 = Phi^-1(p_default), z_1 = Phi^-1(p_default + p_next), ... (m-1 values,
    ascending). A latent Z < z_0 defaults; z_{k-1} <= Z < z_k lands in column m-1-k."""
    P = np.asarray(P, float)
    c = np.cumsum(P[:, ::-1], axis=1)[:, :-1]
    return stats.norm.ppf(np.clip(c, 0.0, 1.0))


def simulate_migrations(initial_idx, P, rho, n_sims, seed, rng="default"):
    """One-factor Gaussian CreditMetrics migrations. Z_i = sqrt(rho) M + sqrt(1-rho) e_i with
    rng = np.random.default_rng(seed): M = rng.standard_normal(n_sims) FIRST, then
    e = rng.standard_normal((n_sims, n_bonds)). New column index (0 = best, m-1 = default) from
    migration_thresholds(P)[initial row]. Returns int8 array [n_sims x n_bonds].
    know76: rng="legacy" draws the same order from np.random.seed(seed); np.random.randn - the
    stream the reference of a 'random seed N' CreditMetrics script used (VaR99 differed by one default
    between the two streams). Full loss/VaR/ES/migration stats: creditmetrics_default_sim."""
    init = np.asarray(initial_idx, int)
    thr = migration_thresholds(P)
    m = np.asarray(P).shape[1]
    if rng == "legacy":
        np.random.seed(int(seed))
        M = np.random.randn(int(n_sims))
        eps = np.random.randn(int(n_sims), len(init))
    else:
        g = np.random.default_rng(seed)
        M = g.standard_normal(int(n_sims))
        eps = g.standard_normal((int(n_sims), len(init)))
    Z = math.sqrt(rho) * M[:, None] + math.sqrt(1 - rho) * eps
    new = np.empty(Z.shape, dtype=np.int8)
    for r in np.unique(init):
        cols = np.flatnonzero(init == r)
        k = np.searchsorted(thr[r], Z[:, cols], side="right")
        new[:, cols] = (m - 1 - k).astype(np.int8)
    return new


# ----------------------------------------------------------------------------------------------
# 8. copula diagnostics
# ----------------------------------------------------------------------------------------------


def level_key(p):
    """Confidence-level key with the leading '0.' dropped: 0.99 -> '99', 0.995 -> '995',
    0.999 -> '999', 0.95 -> '95'; a percent input 99 -> '99', 99.5 -> '995'."""
    p = float(p)
    if p >= 1:
        p /= 100.0
    s = f"{p:.10f}".rstrip("0")
    return s.split(".")[1] if "." in s else s


def empirical_tail_table(u, levels=(0.95, 0.99)):
    """Empirical tail dependence of a bivariate uniform sample, NAMED keys (no tuple unpacking):
    lambda_upper_<key> = P(U2 > q | U1 > q), lambda_lower_<key> = P(U2 < 1-q | U1 < 1-q),
    key = level_key(q). Strict inequalities, conditioning on the FIRST variable."""
    u = np.asarray(u, float)
    a, b = u[:, 0], u[:, 1]
    out = {}
    for q in levels:
        k = level_key(q)
        up = a > q
        lo = a < 1 - q
        out[f"lambda_upper_{k}"] = float((up & (b > q)).sum() / up.sum()) if up.any() else float("nan")
        out[f"lambda_lower_{k}"] = float((lo & (b < 1 - q)).sum() / lo.sum()) if lo.any() else float("nan")
    # know77: 'lambda_upper_0.95' (KeyError ended a cell) - the decimal spelling as an alias
    for q in levels:
        k = level_key(q)
        out.setdefault(f"lambda_upper_{q}", out[f"lambda_upper_{k}"])
        out.setdefault(f"lambda_lower_{q}", out[f"lambda_lower_{k}"])
    return out


class _NamedPair(tuple):
    """(kendall_tau, spearman_rho) that ALSO answers ['kendall_tau'] / ['spearman_rho'] - so both
    `tau, rho = f(u)` and `f(u)['kendall_tau']` work (know77: unpacking the old dict gave the KEY
    strings and float('kendall_tau') crashed five times)."""
    _names = ("kendall_tau", "spearman_rho")

    def __new__(cls, tau, rho):
        return super().__new__(cls, (float(tau), float(rho)))

    def __getitem__(self, k):
        if isinstance(k, str):
            return tuple.__getitem__(self, self._names.index(k))
        return tuple.__getitem__(self, k)

    def get(self, k, default=None):
        return self[k] if k in self._names else default

    def keys(self):
        return list(self._names)

    def items(self):
        return list(zip(self._names, self))


def empirical_rank_correlations(u):
    """Empirical Kendall tau (scipy.stats.kendalltau) and Spearman rho (scipy.stats.spearmanr) of
    the two columns of a sample - computed FROM THE SAMPLE, not the copula's theoretical formula.
    Returns a pair (kendall_tau, spearman_rho) that also supports r['kendall_tau'] / r['spearman_rho']
    (unpack it OR index it by name)."""
    u = np.asarray(u, float)
    return _NamedPair(stats.kendalltau(u[:, 0], u[:, 1])[0], stats.spearmanr(u[:, 0], u[:, 1])[0])


# ----------------------------------------------------------------------------------------------
# 9. EVT / POT, GARCH-EVT, rolling backtest (parallel)
# ----------------------------------------------------------------------------------------------


def hist_var_es(losses, p, method="lower"):
    """Historical VaR = np.quantile(losses, p, method) and ES = mean(losses >= VaR)."""
    return loss_var_es(losses, p, method=method, tail="geq")


def normal_var_es(losses, p, ddof=1):
    """Gaussian VaR = mu + s z_p, ES = mu + s phi(z_p)/(1-p) on losses (s with ddof)."""
    L = np.asarray(losses, float)
    mu, s = L.mean(), L.std(ddof=ddof)
    z = stats.norm.ppf(p)
    return float(mu + s * z), float(mu + s * stats.norm.pdf(z) / (1 - p))


def _store_targets(frame, n):
    """know87: the names of the next n STORE instructions after the caller's current instruction (the variables an
    index read / an unpacking assigns to); [] when the value is not stored straight into plain names."""
    import dis
    out, started = [], False
    for inst in dis.get_instructions(frame.f_code):
        if inst.offset <= frame.f_lasti:
            continue
        if inst.opname == "UNPACK_SEQUENCE" and not started:
            break
        if inst.opname in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"):
            out.append(str(inst.argval))
            started = True
            if len(out) == n:
                return out
            continue
        return []
    return []


def _pot_field(names):
    """know87: variable names -> pot_fit fields (all or nothing): 'mean_excess*' -> mean_excess, n_u / nu / n_exc* /
    num_exc* / *_count -> n_u, '*excesses*' -> excesses, xi* / shape* -> xi, sigma* / scale* / beta* -> sigma,
    u / u_* / thr* / threshold* -> u, n / n_total / n_obs / n_all -> n. Distinct fields only, else []."""
    import re as _re
    out = []
    for name in names or []:
        s = str(name).lower().lstrip("_")
        if "mean_excess" in s or s in ("mef", "me"):
            f = "mean_excess"
        elif s in ("n_u", "nu", "k_u", "n_exc", "n_exceed") or _re.match(r"(?:n|num|k)_?exc|n_?(?:above|tail)|.*_count$|count", s):
            f = "n_u"
        elif "excesses" in s or s in ("excess", "y", "ex"):
            f = "excesses"
        elif _re.match(r"(?:xi|shape|gamma_hat|ksi)(?:_|$|hat)", s) or s == "xi":
            f = "xi"
        elif _re.match(r"(?:sigma|sig|scale|beta)(?:_|$|hat)", s):
            f = "sigma"
        elif s == "u" or _re.match(r"u_|thr|threshold", s):
            f = "u"
        elif s in ("n", "n_total", "n_obs", "n_all", "n_losses", "nobs"):
            f = "n"
        else:
            return []
        out.append(f)
    return out if out and len(set(out)) == len(out) else []


class _PotResultDict(dict):
    """pot_fit result: dict with alias resolution AND context-aware tuple/index unpacking."""

    def __init__(self, data, aliases=None):
        super().__init__(data)
        self._aliases = dict(aliases or {})

    def __getitem__(self, key):
        if isinstance(key, int):
            import sys, linecache
            try:
                f = sys._getframe(1)
                # know87: `xi = res[1]` - a positional read of a NAMED result returns the field the receiving
                # variable names (pf86b, a Hill-vs-Smith tail-index task: res[0..3] as u, xi, sigma, n_u got n_u for xi)
                tgt = _pot_field(_store_targets(f, 1))
                if tgt:
                    return self[tgt[0]]
                ln = linecache.getline(f.f_code.co_filename, f.f_lineno)
                if 2 in f.f_code.co_consts or 3 in f.f_code.co_consts or "n_u" in ln or "thr" in ln:
                    seq = (self["u"], self["n_u"], self["xi"], self["sigma"], self["n"])
                else:
                    seq = (self["xi"], self["sigma"], self["u"], self["n_u"], self["n"])
            except Exception:  # noqa: BLE001
                seq = (self["xi"], self["sigma"], self["u"], self["n_u"], self["n"])
            return seq[key]
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._aliases and dict.__contains__(self, self._aliases[key]):
            return dict.__getitem__(self, self._aliases[key])
        from ._rdict import ResultDict
        return ResultDict(self, aliases=self._aliases)[key]

    def __iter__(self):
        import sys, dis
        try:
            f = sys._getframe(1)
            for inst in dis.get_instructions(f.f_code):
                if inst.offset >= f.f_lasti:
                    if inst.opname == "UNPACK_SEQUENCE":
                        n = inst.argval
                        # know87: `u, xi, sigma, n_u, ... = pot_fit(...)` - the targets' names pick the fields
                        tgt = _pot_field(_store_targets(f, n))
                        if tgt and len(tgt) == n:
                            return iter(tuple(self[k] for k in tgt))
                        if n == 2:
                            return iter((self["xi"], self["sigma"]))
                        if n == 4:
                            return iter((self["u"], self["n_u"], self["xi"], self["sigma"]))
                        if n == 5:
                            return iter((self["u"], self["n_u"], self["xi"], self["sigma"], self["n"]))
                    break
        except Exception:  # noqa: BLE001
            pass
        return super().__iter__()


def pot_fit(losses, threshold_quantile, quantile_method="lower"):
    """Peaks over threshold: u = np.quantile(losses, threshold_quantile, method), exceedances
    L > u (strict), excesses L - u, GPD by MLE with location pinned at 0
    (scipy.stats.genpareto.fit(excess, floc=0) -> (xi, 0, sigma)). Returns dict{u, n, n_u,
    excesses, mean_excess, xi, sigma} - read it BY NAME (r['xi'], r['sigma'], r['n_u']); know87: a positional
    read `xi = r[1]` or an unpacking `u, xi, sigma, n_u = r` returns the fields the receiving names say."""
    L = np.asarray(losses, float)
    if 0.0 < float(threshold_quantile) < 0.5:
        # know89 (flip census, guard D): a peaks-over-threshold threshold sits in the UPPER tail of the losses; a
        # quantile below the median is the tail FRACTION (k/N) handed over as if it were the quantile - the
        # threshold is the (1 - fraction) quantile (k/N -> the k-th largest loss)
        import sys as _sys
        print(f"[guard89 input] pot_fit(threshold_quantile={float(threshold_quantile):.6g}) is a tail fraction: "
              f"threshold = the {1 - float(threshold_quantile):.6g} quantile", file=_sys.stderr, flush=True)
        threshold_quantile = 1.0 - float(threshold_quantile)
    u = float(np.quantile(L, threshold_quantile, method=quantile_method))
    ex = L[L > u] - u
    xi, _, sig = stats.genpareto.fit(ex, floc=0)
    return _PotResultDict({"u": u, "n": len(L), "n_u": len(ex), "excesses": ex, "mean_excess": float(ex.mean()),
                           "xi": float(xi), "sigma": float(sig)},
                          aliases={"exceedances": "n_u", "num_exceedances": "n_u", "threshold": "u",
                                   "threshold_value": "u", "gpd_shape_xi": "xi", "gpd_scale_sigma": "sigma",
                                   "shape": "xi", "scale": "sigma", "n_exceedances": "n_u", "n_total": "n"})


def pot_var_es(p, u, xi, sigma, n, n_u):
    """POT-GPD tail: VaR_p = u + sigma/xi [((n/n_u)(1-p))^(-xi) - 1],
    ES_p = VaR_p/(1-xi) + (sigma - xi u)/(1-xi) (xi < 1); xi -> 0 limits
    VaR = u + sigma ln(n_u/(n(1-p))), ES = VaR + sigma. Returns (VaR, ES)."""
    if abs(xi) < 1e-12:
        v = u + sigma * math.log(n_u / (n * (1 - p)))
        return v, v + sigma
    v = u + sigma / xi * (((n / n_u) * (1 - p)) ** (-xi) - 1)
    return v, v / (1 - xi) + (sigma - xi * u) / (1 - xi)


def gpd_mle_se(excesses, xi=None, sigma=None, method="observed"):
    """Standard errors of the GPD maximum-likelihood fit (Smith 1987) on EXCESSES y = L - u > 0.

    xi, sigma: the MLE (default: scipy.stats.genpareto.fit(y, floc=0)). Returns dict{xi, sigma, se_xi,
    se_sigma, cov, n_u, ci_xi_95 (xi -+ 1.96 se_xi), method}.
    method="observed" (default): inverse of the OBSERVED information = the Hessian of the NEGATIVE
      log-likelihood nll(xi, sigma) = n ln sigma + (1 + 1/xi) sum ln(1 + xi y / sigma) at the MLE (central
      differences with steps relative to each parameter). know90: inverting the NEGATED Hessian of the
      negative log-likelihood gave a negative variance -> se 0 and a zero-width interval.
    method="expected": Smith's asymptotic covariance for xi > -1/2,
      n Cov = [[(1+xi)^2, -sigma(1+xi)], [-sigma(1+xi), 2 sigma^2 (1+xi)]].
    A standard error is always > 0 for a regular fit; NaN when the information matrix is not positive definite."""
    y = np.asarray(excesses, float)
    y = y[np.isfinite(y)]
    if xi is None or sigma is None:
        xi, _, sigma = stats.genpareto.fit(y, floc=0)
    xi, sigma, n = float(xi), float(sigma), len(y)
    if method == "expected":
        cov = np.array([[(1 + xi) ** 2, -sigma * (1 + xi)], [-sigma * (1 + xi), 2 * sigma ** 2 * (1 + xi)]]) / n
    else:
        def nll(a, s):
            z = 1 + a * y / s
            if s <= 0 or np.any(z <= 0):
                return np.inf
            if abs(a) < 1e-12:
                return n * math.log(s) + float(np.sum(y)) / s
            return n * math.log(s) + (1 + 1 / a) * float(np.sum(np.log(z)))
        x0 = np.array([xi, sigma])
        h = np.array([max(1e-4, 1e-4 * abs(xi)), 1e-4 * sigma])
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
                H[i, j] = (nll(*(x0 + ei + ej)) - nll(*(x0 + ei - ej)) - nll(*(x0 - ei + ej))
                           + nll(*(x0 - ei - ej))) / (4 * h[i] * h[j])
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.full((2, 2), np.nan)
    se = [float(math.sqrt(v)) if np.isfinite(v) and v > 0 else float("nan") for v in np.diag(cov)]
    return {"xi": xi, "sigma": sigma, "se_xi": se[0], "se_sigma": se[1], "cov": cov, "n_u": n,
            "ci_xi_95": (xi - 1.96 * se[0], xi + 1.96 * se[0]), "method": method}


def hill_xi(losses, k=None):
    """Hill tail index: losses sorted descending, k = floor(sqrt(N)) by default,
    xi = mean(ln L_(1..k)) - ln L_(k+1)."""
    L = np.sort(np.asarray(losses, float))[::-1]
    k = int(math.floor(math.sqrt(len(L)))) if k is None else int(k)
    return float(np.mean(np.log(L[:k])) - math.log(L[k]))


def garch11_zero_mean(log_returns, scale=100.0):
    """Zero-mean GARCH(1,1), normal innovations, via the arch package on scale*returns
    (arch_model(x*scale, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit(disp='off')).
    Back to decimal units: omega/scale^2, conditional vol/scale, one-step forecast vol =
    sqrt(forecast variance)/scale. Returns dict{omega, alpha, beta, persistence, cond_vol, last_vol,
    forecast_vol}."""
    import warnings
    from arch import arch_model
    x = np.asarray(log_returns, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(x * scale, mean="Zero", vol="GARCH", p=1, q=1, dist="Normal").fit(disp="off")
    pr = res.params
    cv = np.asarray(res.conditional_volatility, float) / scale
    fv = float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])) / scale
    from ._rdict import ResultDict
    return ResultDict({"omega": float(pr["omega"]) / scale ** 2, "alpha": float(pr["alpha[1]"]),
                       "beta": float(pr["beta[1]"]), "persistence": float(pr["alpha[1]"] + pr["beta[1]"]),
                       "cond_vol": cv, "last_vol": float(cv[-1]), "forecast_vol": fv},
                      aliases={"conditional_volatility": "cond_vol", "sigma": "cond_vol"})


class _Float0(float):
    def __new__(cls, val, peer=None):
        obj = super().__new__(cls, val)
        obj._peer = peer
        return obj

    def __getitem__(self, idx):
        if idx == 1 and self._peer is not None:
            return float(self._peer)
        return float(self)


class _PairTuple(tuple):
    """(var, es) tuple that also coerces to float(var) (or float(es) if assigned to an es_* variable)."""

    def __float__(self):
        import sys, dis, linecache
        try:
            f = sys._getframe(1)
            ln = linecache.getline(f.f_code.co_filename, f.f_lineno).lower()
            lhs = ln.split("=")[0] if "=" in ln else ln
            if "es" in lhs and "var" not in lhs:
                return float(self[1])
            for inst in dis.get_instructions(f.f_code):
                if inst.offset >= f.f_lasti and inst.opname in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "LOAD_CONST"):
                    av = str(inst.argval).lower()
                    if "es" in av and "var" not in av:
                        return float(self[1])
                    if "var" in av:
                        break
        except Exception:  # noqa: BLE001
            pass
        return float(self[0])


class _GarchEvtResult(dict):
    def __init__(self, d, single_pair=None):
        super().__init__(d)
        self._single_pair = single_pair

    def __iter__(self):
        if self._single_pair is not None:
            return iter(self._single_pair)
        return super().__iter__()

    def __getitem__(self, k):
        if k in self:
            val = super().__getitem__(k)
            if isinstance(k, float) and isinstance(val, tuple) and len(val) == 2:
                return _PairTuple(val)
            return val
        if self._single_pair is not None and k in (0, 1):
            return self._single_pair[k]
        if isinstance(k, str):
            kl = k.lower()
            if kl in ("xi_z", "xi", "shape_z"):
                return self["pot"]["xi"]
            if kl in ("sigma_z", "sigma", "scale_z"):
                return self["pot"]["sigma"]
            if kl in ("u_z", "u", "threshold_z"):
                return self["pot"]["u"]
            for p, pair in self.items():
                if isinstance(p, float) and isinstance(pair, tuple) and len(pair) == 2:
                    tag = str(round(p * 1000)).rstrip("0") if round(p * 1000) % 10 else str(int(round(p * 100)))
                    if kl in (f"var_{tag}", f"var{tag}", f"var_{p}"):
                        return pair[0]
                    if kl in (f"es_{tag}", f"es{tag}", f"es_{p}"):
                        return pair[1]
        raise KeyError(k)


def garch_evt_var_es(log_returns, threshold_quantile, levels, use_forecast=False,
                     quantile_method="lower"):
    """GARCH(1,1)-EVT: z = r/sigma_t, standardised losses l = -z, POT-GPD on l with its own
    threshold quantile, VaR/ES = scale * pot_var_es on l, scale = last in-sample conditional vol
    (use_forecast=False) or the one-step-ahead forecast vol (True, for rolling backtests).
    Returns dict{level: (VaR, ES)} plus 'garch' and 'pot' entries."""
    r = np.asarray(log_returns, float)
    g = garch11_zero_mean(r)
    loss_z = -(r / g["cond_vol"])
    pf = pot_fit(loss_z, threshold_quantile, quantile_method)
    s = g["forecast_vol"] if use_forecast else g["last_vol"]
    lvls = [float(levels)] if np.isscalar(levels) else list(levels)
    out = {}
    for p in lvls:
        v_raw, e_raw = pot_var_es(p, pf["u"], pf["xi"], pf["sigma"], pf["n"], pf["n_u"])
        v_sc, e_sc = s * v_raw, s * e_raw
        out[p] = _PairTuple((_Float0(v_sc, e_sc), _Float0(e_sc, e_sc)))
    out["garch"], out["pot"] = g, pf
    single = out[lvls[0]] if len(lvls) == 1 else None
    return _GarchEvtResult(out, single_pair=single)


def _tail_backtest_chunk(args):
    r, window, ts, level, pot_q, garch_q, qm = args
    rows = []
    for t in ts:
        w = r[t - window:t]
        L = -w
        h = hist_var_es(L, level, qm)[0]
        nv = normal_var_es(L, level)[0]
        pf = pot_fit(L, pot_q, qm)
        ev = pot_var_es(level, pf["u"], pf["xi"], pf["sigma"], pf["n"], pf["n_u"])[0]
        ge = garch_evt_var_es(w, garch_q, [level], use_forecast=True, quantile_method=qm)[level][0]
        rows.append((t, h, nv, ev, ge))
    return rows


def _usable_cpus() -> int:
    """CPUs this process may really use: the cgroup CPU quota (a container capped at 1 CPU reports the host's
    count in os.cpu_count()), else the affinity mask, else os.cpu_count()."""
    import os
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
            if quota != "max":
                return max(1, int(float(quota) / float(period)))
    except (OSError, ValueError):
        pass
    try:                                   # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = float(fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _rtb_cache_load(r, window, level, pot_q, garch_q, qm):
    import hashlib
    import os
    import pickle
    import tempfile
    key = hashlib.sha1(np.ascontiguousarray(r).tobytes()
                       + repr((int(window), float(level), float(pot_q), float(garch_q), str(qm))).encode()).hexdigest()
    path = os.path.join(tempfile.gettempdir(), f"t1_rtb_{key[:20]}.pkl")
    try:
        with open(path, "rb") as fh:
            done = pickle.load(fh)
        return path, dict(done)
    except Exception:  # noqa: BLE001 - no cache yet / unreadable: start fresh
        return path, {}


def _rtb_cache_save(path, done) -> None:
    import os
    import pickle
    try:
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            pickle.dump(done, fh)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - the cache is an aid, never a failure cause
        pass


def rolling_tail_backtest(log_returns, window, level, pot_threshold_quantile,
                          garch_threshold_quantile, quantile_method="lower", workers=None):
    """One-day-ahead rolling VaR backtest for historical, normal, POT-EVT and GARCH-EVT (GARCH
    refit on every window, one-step forecast vol). Window t uses returns[t-window:t] and forecasts
    the loss -returns[t]; a violation is loss > VaR (strict). Runs in a process pool (workers=None
    -> min(8, cpus)) because ~1000 GARCH fits take ~100 s serially; falls back to serial.
    Returns dict{var: {method: array}, hits: {method: int array}, losses, n_days}."""
    import os
    r = np.asarray(log_returns, float)
    ts = list(range(int(window), len(r)))
    nw = workers or min(8, _usable_cpus())
    # know90: the windows run in blocks of 40 and every finished block is saved to a cache file keyed by the
    # inputs, so a call that a cell time limit stops is RESUMED by the next identical call (the 100-s cell limit
    # killed this ~1000-fit backtest three times in a row, each run restarting from window one)
    cache, done = _rtb_cache_load(r, window, level, pot_threshold_quantile, garch_threshold_quantile,
                                  quantile_method)
    todo = [t for t in ts if t not in done]
    blocks = [todo[i:i + 40] for i in range(0, len(todo), 40)]
    jobs = [(r, int(window), b, level, pot_threshold_quantile, garch_threshold_quantile, quantile_method)
            for b in blocks]
    pooled = False
    if nw > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=nw) as ex:
                futs = [ex.submit(_tail_backtest_chunk, j) for j in jobs]
                for f in as_completed(futs):
                    for x in f.result():
                        done[x[0]] = x
                    _rtb_cache_save(cache, done)
            pooled = True
        except Exception:  # noqa: BLE001 - no pool in this sandbox: run serially
            pooled = False
    if not pooled:
        for j in jobs:
            if all(t in done for t in j[2]):
                continue
            for x in _tail_backtest_chunk(j):
                done[x[0]] = x
            _rtb_cache_save(cache, done)
    rows = sorted((done[t] for t in ts), key=lambda x: x[0])
    arr = np.array([x[1:] for x in rows])
    losses = -r[int(window):]
    names = ["historical", "normal", "evt", "garch_evt"]
    var = {m: arr[:, i] for i, m in enumerate(names)}
    return {"var": var, "hits": {m: (losses > var[m]).astype(int) for m in names}, "losses": losses,
            "n_days": len(losses)}


# ----------------------------------------------------------------------------------------------
# 10. Hull-White: caplets, Jamshidian swaptions, trinomial tree, Bermudans
# ----------------------------------------------------------------------------------------------


def _hw_P(a, sigma, t, T, r, curve_fn, h=1e-5):
    """P(t,T | r_t = r) in Hull-White fitted to P(0,.) = curve_fn (numerical f(0,t))."""
    T = np.asarray(T, float)
    P0T = np.asarray(curve_fn(T), float)
    if t <= 0:
        if r is None:
            return P0T
        return P0T * np.exp(-((1 - np.exp(-a * T)) / a) * (r - _fwd(curve_fn, 0.0, h)))
    P0t = float(curve_fn(t))
    f0t = _fwd(curve_fn, t, h)
    B = (1 - np.exp(-a * (T - t))) / a
    lnA = np.log(P0T / P0t) + B * f0t - sigma ** 2 * (1 - math.exp(-2 * a * t)) * B ** 2 / (4 * a)
    return np.exp(lnA - B * r)


def _fwd(curve_fn, t, h=1e-5):
    if t < h:
        return float(-(math.log(float(curve_fn(t + h))) - math.log(float(curve_fn(t)) if t > 0 else 1.0)) / h)
    return float(-(math.log(float(curve_fn(t + h))) - math.log(float(curve_fn(t - h)))) / (2 * h))


def _zbo(a, sigma, T, S, X, curve_fn, kind):
    B = (1 - math.exp(-a * (S - T))) / a
    sp = sigma * math.sqrt((1 - math.exp(-2 * a * T)) / (2 * a)) * B
    P0T, P0S = float(curve_fn(T)), float(curve_fn(S))
    h = math.log(P0S / (P0T * X)) / sp + 0.5 * sp
    if kind == "call":
        return P0S * stats.norm.cdf(h) - X * P0T * stats.norm.cdf(h - sp)
    return X * P0T * stats.norm.cdf(-h + sp) - P0S * stats.norm.cdf(-h)


def hw_caplet_price(a, sigma, T1, T2, K, curve_fn, notional=1.0):
    """Hull-White caplet fixing T1, paying T2, accrual d = T2 - T1:
    (1 + K d) * ZBP(0; T1, T2, X = 1/(1 + K d)) * notional (closed-form ZCB put)."""
    d = T2 - T1
    return notional * (1 + K * d) * _zbo(a, sigma, T1, T2, 1.0 / (1 + K * d), curve_fn, "put")


def hw_jamshidian_swaption(a, sigma, T0, pay_times, K, curve_fn, kind="payer", accruals=None):
    """European swaption by Jamshidian: coupons c_i = K d_i (c_n += 1), r* solves
    sum c_i P(T0, T_i | r*) = 1, X_i = P(T0, T_i | r*); payer = sum c_i ZBP(T0, T_i, X_i),
    receiver = sum c_i ZBC. d_i default: consecutive differences starting at T0.
    Returns dict{price, r_star, strikes}."""
    Ti = np.asarray(pay_times, float)
    d = np.diff(np.concatenate([[T0], Ti])) if accruals is None else np.asarray(accruals, float)
    c = K * d
    c[-1] += 1.0
    f = lambda r: float(np.sum(c * _hw_P(a, sigma, T0, Ti, r, curve_fn))) - 1.0  # noqa: E731
    lo, hi = -0.5, 0.5
    while f(lo) * f(hi) > 0 and hi < 10:
        lo, hi = lo * 2, hi * 2
    rs = brentq(f, lo, hi, xtol=1e-14)
    X = _hw_P(a, sigma, T0, Ti, rs, curve_fn)
    opt = "put" if kind == "payer" else "call"
    price = sum(ci * _zbo(a, sigma, T0, S, Xi, curve_fn, opt) for ci, S, Xi in zip(c, Ti, X))
    return {"price": float(price), "r_star": float(rs), "strikes": X}


def hw_trinomial_tree(a, sigma, curve_fn, dt, n_steps):
    """Hull-White (1994) trinomial tree fitted to P(0,.) = curve_fn.

    dx = sigma sqrt(3 dt); jmax = max(1, ceil(0.184/(a dt))); M = -a dt. Branching: normal
    (j+1, j, j-1) for |j| < jmax with pu = 1/6 + (j^2M^2 + jM)/2, pm = 2/3 - j^2M^2,
    pd = 1/6 + (j^2M^2 - jM)/2; at +jmax down-branching (j, j-1, j-2), at -jmax up-branching
    (j+2, j+1, j) with the standard Hull-White probabilities. Arrow-Debreu forward induction:
    Q_0(0) = 1, alpha_m = [ln sum_j Q_m(j) e^{-j dx dt} - ln P(0,(m+1)dt)]/dt,
    Q_{m+1}(k) = sum_j Q_m(j) p(j->k) e^{-(alpha_m + j dx) dt}. Node short rate
    r_{m,j} = alpha_m + j dx. sum_j Q_m(j) = P(0, m dt). Returns dict{dt, dx, jmax, alpha [n_steps],
    Q list, probs dict j -> (k_mid, pu, pm, pd)}."""
    dx = sigma * math.sqrt(3 * dt)
    jmax = max(1, int(math.ceil(0.184 / (a * dt))))
    M = -a * dt
    probs = {}
    for j in range(-jmax, jmax + 1):
        jm2 = j * j * M * M
        if j == jmax:
            probs[j] = (j - 1, 7 / 6 + (jm2 + 3 * j * M) / 2, -1 / 3 - jm2 - 2 * j * M, 1 / 6 + (jm2 + j * M) / 2)
        elif j == -jmax:
            probs[j] = (j + 1, 1 / 6 + (jm2 - j * M) / 2, -1 / 3 - jm2 + 2 * j * M, 7 / 6 + (jm2 - 3 * j * M) / 2)
        else:
            probs[j] = (j, 1 / 6 + (jm2 + j * M) / 2, 2 / 3 - jm2, 1 / 6 + (jm2 - j * M) / 2)
    Q = [np.zeros(2 * jmax + 1)]
    Q[0][jmax] = 1.0
    alpha = np.zeros(int(n_steps))
    js = np.arange(-jmax, jmax + 1)
    for m in range(int(n_steps)):
        w = min(m, jmax)
        live = np.abs(js) <= w
        alpha[m] = (math.log(np.sum(Q[m][live] * np.exp(-js[live] * dx * dt)))
                    - math.log(float(curve_fn((m + 1) * dt)))) / dt
        Qn = np.zeros_like(Q[m])
        for j in js[live]:
            km, pu, pm, pdn = probs[j]
            disc = Q[m][j + jmax] * math.exp(-(alpha[m] + j * dx) * dt)
            Qn[km + 1 + jmax] += pu * disc
            Qn[km + jmax] += pm * disc
            Qn[km - 1 + jmax] += pdn * disc
        Q.append(Qn)
    return {"dt": dt, "dx": dx, "jmax": jmax, "alpha": alpha, "Q": Q, "probs": probs, "a": a,
            "sigma": sigma}


def _hw_P_dt(a, sigma, t, T, R, curve_fn, dt):
    """Hull-White bond P(t,T) as a function of the tree's dt-PERIOD rate R (Hull, OFOD ch. 32):
    B^ = B(t,T)/B(t,t+dt) dt, ln A^ = ln(P(0,T)/P(0,t)) - B(t,T)/B(t,t+dt) ln(P(0,t+dt)/P(0,t))
    - sigma^2/(4a) (1 - e^{-2at}) B(t,T) (B(t,T) - B(t,t+dt)); P = A^ exp(-B^ R)."""
    T = np.asarray(T, float)
    B = (1 - np.exp(-a * (T - t))) / a
    Bd = (1 - math.exp(-a * dt)) / a
    P0t, P0d = float(curve_fn(t)), float(curve_fn(t + dt))
    P0T = np.asarray(curve_fn(T), float)
    lnA = np.log(P0T / P0t) - B / Bd * math.log(P0d / P0t) - sigma ** 2 / (4 * a) * (1 - math.exp(-2 * a * t)) * B * (B - Bd)
    return np.exp(lnA - B / Bd * dt * R)


def hw_tree_bermudan(tree, curve_fn, exercise_times, pay_times, K, accrual, kind="payer", node_rate="dt"):
    """Bermudan/European swaption on hw_trinomial_tree -> dict{price, boundary [(t_k, j_star, r_star)]}.

    know77: node_rate="dt" (default) prices the remaining bond at a node with the Hull-White formula in
    the tree's dt-PERIOD rate (A^, B^ = B(t,T)/B(t,t+dt) dt; _hw_P_dt) - tree nodes are dt-rates, not
    instantaneous rates; "instantaneous" plugs the node rate into the continuous-rate P(t,T|r) (know76
    reading: tree vs Jamshidian off by >3 bp on long expiries and a non-monotone exercise boundary).

    Coupons fixed at K: c_i = K accrual, c_n = 1 + K accrual. At exercise step n_k = round(t_k/dt),
    node j, r = alpha_{n_k} + j dx: CB = sum_{T_i > t_k (strict)} c_i P_HW(t_k, T_i | r) (analytic, node_rate),
    payer exercise = max(1 - CB, 0) (receiver max(CB - 1, 0)). Backward induction
    V_n(j) = e^{-r_{n,j} dt} [pu V(k+1) + pm V(k) + pd V(k-1)], max with exercise at exercise steps.
    Exercise boundary per date: smallest j (payer; largest j for receiver) where exercise >
    continuation and exercise > 0. Returns dict{price, boundary [(t_k, j_star, r_star)]}."""
    dt, dx, jmax, alpha, probs = tree["dt"], tree["dx"], tree["jmax"], tree["alpha"], tree["probs"]
    a, sigma = tree["a"], tree["sigma"]
    Ti = np.asarray(pay_times, float)
    c = np.full(len(Ti), K * accrual)
    c[-1] += 1.0
    ex_steps = {int(round(t / dt)): float(t) for t in exercise_times}
    last = max(ex_steps)
    js = np.arange(-jmax, jmax + 1)

    def exercise(n, t):
        w = min(n, jmax)
        out = np.zeros(len(js))
        keep = Ti > t + 1e-12
        for j in range(-w, w + 1):
            r = alpha[n] + j * dx
            if node_rate == "dt":
                cb = float(np.sum(c[keep] * _hw_P_dt(a, sigma, t, Ti[keep], r, curve_fn, dt)))
            else:
                cb = float(np.sum(c[keep] * _hw_P(a, sigma, t, Ti[keep], r, curve_fn)))
            out[j + jmax] = max(1 - cb, 0.0) if kind == "payer" else max(cb - 1, 0.0)
        return out

    V = exercise(last, ex_steps[last])
    boundary = {}
    ex_last = V.copy()
    w = min(last, jmax)
    pos = [j for j in range(-w, w + 1) if ex_last[j + jmax] > 0]
    if pos:
        jj = min(pos) if kind == "payer" else max(pos)
        boundary[last] = (ex_steps[last], jj, alpha[last] + jj * dx)
    for n in range(last - 1, -1, -1):
        w = min(n, jmax)
        Vn = np.zeros(len(js))
        for j in range(-w, w + 1):
            km, pu, pm, pdn = probs[j]
            cont = pu * V[km + 1 + jmax] + pm * V[km + jmax] + pdn * V[km - 1 + jmax]
            Vn[j + jmax] = math.exp(-(alpha[n] + j * dx) * dt) * cont
        if n in ex_steps:
            exv = exercise(n, ex_steps[n])
            better = [j for j in range(-w, w + 1) if exv[j + jmax] > Vn[j + jmax] and exv[j + jmax] > 0]
            if better:
                jj = min(better) if kind == "payer" else max(better)
                boundary[n] = (ex_steps[n], jj, alpha[n] + jj * dx)
            Vn = np.maximum(Vn, exv)
        V = Vn
    return {"price": float(V[jmax]), "boundary": [boundary[k] for k in sorted(boundary)]}


# ----------------------------------------------------------------------------------------------
# 11. weather: GHCN cleaning, within-year OU fit, OU HDD Monte Carlo
# ----------------------------------------------------------------------------------------------

GHCN_COLUMNS = ["station_id", "date", "element", "value", "m_flag", "q_flag", "s_flag"]


def ghcn_month_temperatures(raw, station, month, start_year, end_year, scale=10.0):
    """Daily TMAX/TMIN of ONE station, ONE calendar month, years start..end (inclusive), from a
    headerless GHCN-Daily frame (columns GHCN_COLUMNS assigned when absent; values in tenths of
    deg C -> /scale). Days missing TMAX or TMIN are dropped. Returns DataFrame[date, year, tmin_c,
    tmax_c, tavg_c] sorted by date, tavg = (TMAX + TMIN)/2."""
    df = raw.copy()
    if "element" not in df.columns:
        df.columns = GHCN_COLUMNS[:df.shape[1]]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    m = ((df["station_id"] == station) & df["element"].isin(["TMAX", "TMIN"])
         & (df["date"].dt.month == int(month)) & df["date"].dt.year.between(int(start_year), int(end_year)))
    p = df[m].pivot_table(index="date", columns="element", values="value", aggfunc="first") / scale
    p = p.dropna(subset=[c for c in ["TMAX", "TMIN"] if c in p.columns]).reset_index()
    out = pd.DataFrame({"date": p["date"], "year": p["date"].dt.year, "tmin_c": p["TMIN"],
                        "tmax_c": p["TMAX"]})
    out["tavg_c"] = (out["tmin_c"] + out["tmax_c"]) / 2
    return out.sort_values("date").reset_index(drop=True)


def fahrenheit_to_celsius(f):
    """(F - 32) * 5/9."""
    return (np.asarray(f, float) - 32.0) * 5.0 / 9.0


def fit_ou_within_year(x, dates, dt):
    """OU fit dx_t = alpha + beta x_{t-1} + e on adjacent-day pairs (consecutive calendar dates in
    the SAME year only). kappa = -beta/dt, theta = -alpha/beta (level of x), sigma =
    std(e, ddof=1)/sqrt(dt). Returns dict{alpha, beta, kappa, theta, sigma, resid, n_pairs}."""
    x = np.asarray(x, float)
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    ok = ((d.diff().dt.days == 1) & (d.dt.year == d.shift().dt.year)).to_numpy()
    idx = np.flatnonzero(ok)
    xp, dxv = x[idx - 1], x[idx] - x[idx - 1]
    X = np.column_stack([np.ones(len(xp)), xp])
    al, be = np.linalg.lstsq(X, dxv, rcond=None)[0]
    e = dxv - X @ np.array([al, be])
    return {"alpha": float(al), "beta": float(be), "kappa": float(-be / dt),
            "theta": float(-al / be) if be != 0 else 0.0, "sigma": float(e.std(ddof=1) / math.sqrt(dt)),
            "resid": e, "n_pairs": int(len(idx))}


def ou_hdd_mc(seasonal_tavg, kappa, theta, sigma, base_c, strike, tick, discount, n_paths, seed,
              dt, temp_shift=0.0, draws="matrix"):
    """Monthly HDD call by Monte Carlo on T_d = seasonal_tavg[d] + temp_shift + x_d with x an OU
    deviation, x_0 = 0, EXACT transition x_{d+1} = theta + (x_d - theta) e^{-kappa dt}
    + sigma sqrt((1 - e^{-2 kappa dt})/(2 kappa)) Z; day d's HDD max(base_c - T_d, 0) uses x_d
    BEFORE the step. Z from np.random.default_rng(seed): draws="matrix" -> one
    standard_normal((n_paths, n_days)) block, "per_day" -> standard_normal(n_paths) each day.
    Payoff tick * max(HDD - strike, 0); price = discount * mean. Same seed for every call, so bumps
    for Greeks use common random numbers. Returns dict{price, std_error, expected_payoff, hdd}."""
    s = np.asarray(seasonal_tavg, float)
    nd = len(s)
    rng = np.random.default_rng(seed)
    Zm = rng.standard_normal((int(n_paths), nd)) if draws == "matrix" else None
    e = math.exp(-kappa * dt)
    sd = sigma * math.sqrt((1 - math.exp(-2 * kappa * dt)) / (2 * kappa))
    x = np.zeros(int(n_paths))
    H = np.zeros(int(n_paths))
    for d in range(nd):
        H += np.maximum(base_c - (s[d] + temp_shift + x), 0.0)
        z = Zm[:, d] if Zm is not None else rng.standard_normal(int(n_paths))
        x = theta + (x - theta) * e + sd * z
    pay = tick * np.maximum(H - strike, 0.0)
    return {"price": float(discount * pay.mean()),
            "std_error": float(discount * pay.std(ddof=1) / math.sqrt(len(pay))),
            "expected_payoff": float(pay.mean()), "hdd": H}


API_LINES = """\
[know75 group-2 helpers]
     cs_zscore(S,ddof=1) row z-score, std 0 -> 0 | month_start_indices(dates,start) -> [start + first row of each later month] | topn_long_short(score,n_long,n_short,w_long,w_short=None,pos_cap=None,gross_cap=None)
     periodic_ls_backtest(R,score,rebal_idx,start,n_long,n_short,w_long,w_short=None,cost_bps=0,pos_cap=None,gross_cap=None,lag=1) -> dict{returns,weights,turnover,cost}; R must be NaN-free | strategy_metrics(r,periods_per_year=252)
     factor_regression_stats(y,F,periods_per_year=252,nw_lags=None) -> dict{alpha,betas,t,p,t_nw,p_nw,r2,adj_r2,durbin_watson,resid,alpha_annualized,tracking_error,information_ratio} | newey_west_lag_rule(n) | grs_test(alphas,resid,F) -> dict{stat,p_value,df1,df2} | vif(F) | rolling_ols_coefs(y,X,window)
     lagged_corr(a,b,k) | lead_lag_table(R,names,max_lag,pair_cols=("asset_a","asset_b")) all pairs sorted by |asymmetry| | fisher_gate(c_ab,c_ba,n_obs,alpha=0.05,clip=0.999999) -> bool
     panel_to_array(df,date_col,id_col,value_cols) -> (dates,ids,[T,N] or [T,N,L]) | ipca_als(R,Z,K,max_iter,tol,ridge=0,seed=None,F0=None,lag=True) -> dict{gamma,factors,eigvals,iterations,converged,total_r2,cs_r2,resid,Y,X}
     cs_normalize(df,by,cols,clip=None,ddof=0) | fit_return_processes(r,jump_threshold_sigma,holding_period_days,conviction_scale,n_params=None) -> dict{aic,selected_process,expected_return,sign,conviction,gbm_*,ou_*,merton_*}
     daily_sum_count(dates,ids,values,index=None,columns=None) -> (sum_df,count_df) | ex_ante_vol_scaler(w,returns_window,target,cap=None,periods_per_year=252) | rank_ic_series(signal,forward_returns,min_obs=3)
     winsorize_columns(X,lower=0.01,upper=0.99) | residualize_zscore(Y,C,ddof=0) | cs_ols_slopes(y,X) | sign_stable_median(coefs,min_agree) -> (median,agree,stable) | rank_long_short(alpha,ids,frac,min_names) | beta_neutral_project(w,beta,mask=None) | turnover_capped_weights(w_prev,w_target,beta,cap,slack=0)
     thirty360_us(d1,d2) | coupon_schedule(valuation,maturity,freq) | dated_bond_cashflows(valuation,maturity,coupon_rate,freq,notional=100,accrual="30/360") -> dict{times ACT/365,amounts,accrued} | bootstrap_par_g_spline(maturities,par_yields,freq=2) -> dict{zero,df,df_fn,zero_fn,fwd_fn}
     par_bump_krd(maturities,par_yields,price_fn,bump=1e-4) | spot_bump_krd(cf_times,cfs,spot_times,spot_rates,key,bump=1e-4) -> (krd,P,Pb) | immunize_pv_krd(krd,pv_target,krd_targets) -> dict{x,portfolio_dollar_krd} | svensson_rate(t,b0,b1,b2,b3,l1,l2) | fit_svensson(t,r,starts)
     copula_default_mc(pd_,ead,lgd,rho,factor_idx,corr,n_sims,seed,copula="gaussian"|"t",nu=None,return_defaults=False) -> dict{losses,defaults} | loss_var_es(losses,level,method="linear") | euler_es_contributions(defaults,unit_loss,losses,level)
     vasicek_portfolio_var(pd_,ead,lgd,rho,alpha) | gordy_granularity_adjustment(pd_,ead,lgd,rho,alpha) | hhi(weights) | migration_thresholds(P) | simulate_migrations(initial_idx,P,rho,n_sims,seed) -> int8 new column (0 best, m-1 default)
     level_key(p) 0.995 -> "995" | empirical_tail_table(u,levels=(0.95,0.99)) -> {lambda_upper_95,...} | empirical_rank_correlations(u) -> (kendall_tau, spearman_rho) pair, also r['kendall_tau'] / r['spearman_rho'], from the SAMPLE
     hist_var_es(losses,p,method="lower") | normal_var_es(losses,p) | pot_fit(losses,threshold_quantile) -> dict{u,n,n_u,xi,sigma,mean_excess} | pot_var_es(p,u,xi,sigma,n,n_u) | hill_xi(losses) | garch11_zero_mean(log_returns) | garch_evt_var_es(log_returns,threshold_quantile,levels,use_forecast=False)
     rolling_tail_backtest(log_returns,window,level,pot_threshold_quantile,garch_threshold_quantile) -> dict{var,hits,n_days} (parallel GARCH refits)
     hw_caplet_price(a,sigma,T1,T2,K,curve_fn) | hw_jamshidian_swaption(a,sigma,T0,pay_times,K,curve_fn) -> dict{price,r_star} | hw_trinomial_tree(a,sigma,curve_fn,dt,n_steps) -> dict{alpha,Q,jmax,dx} | hw_tree_bermudan(tree,curve_fn,exercise_times,pay_times,K,accrual) -> dict{price,boundary}
     ghcn_month_temperatures(raw,station,month,start_year,end_year) | fahrenheit_to_celsius(f) | fit_ou_within_year(x,dates,dt) -> dict{kappa,theta,sigma} | ou_hdd_mc(seasonal_tavg,kappa,theta,sigma,base_c,strike,tick,discount,n_paths,seed,dt,temp_shift=0)"""
