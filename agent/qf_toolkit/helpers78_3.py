"""know78 group-3 helpers: conventions that decided regime / factor / execution units.

Each function is the textbook form of a step that the earlier helpers did in a non-standard way
(unstable tie order of same-millisecond events, punctuation-attached tokens, an ALS sweep without
the Kelly-Pruitt-Su identification, the MLE instead of the GRS-1989 covariance scaling). The earlier
report helpers call these in place, so a one-call program picks them up. No task names, no data
values. Self-tests: proven_work/fail72/selftest78_3.py.

API lines (one per function):
     event_order(df, time_col, id_col=None) -> DataFrame sorted by (time, id) with a stable sort; 'first event at or after t' is then the lowest sequence id among same-time events
     alpha_tokens(text) -> list of lower-case alphabetic tokens re.findall('[a-z]+', text.lower()) ('Long,' -> 'long', 'rally!' -> 'rally')
     kps_ipca_als(R, Z, K, max_iter, tol, ridge=0.0, seed=None, lag=True) -> dict{gamma, factors, eigvals, iterations, converged, fitted, resid, total_r2, cs_r2, Y, X}; KPS sweep Gamma-step -> identification (chol + eigen) -> F-step, stop on max(|dGamma|, |dF|) < tol
     grs_statistic(alphas, resid, factors, cov="unbiased") -> dict{stat, p_value, df1, df2, grs_statistic, grs_p_value}; GRS (1989) with Sigma = E'E/(T-K-1), Omega = cov(F, ddof=1) ("mle": both 1/T)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

_ALPHA = re.compile(r"[a-z]+")


def event_order(df, time_col, id_col=None):
    """Stable event order for an event log (quotes, trades, book updates).

    Several events can share one millisecond; the exchange sequence number (update id, trade id)
    gives their true order. Sort by (time_col, id_col) with kind='mergesort' when id_col is present,
    else by time_col alone (stable, so the file order breaks ties). A caller's earlier unstable
    sort (default quicksort) can no longer change which event is 'the first at or after t'.
    Returns a new DataFrame with a fresh RangeIndex."""
    keys = [time_col] + ([id_col] if id_col is not None and id_col in df.columns else [])
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def alpha_tokens(text):
    """Bag-of-words tokens for a lexicon score: lower-case, maximal runs of the letters a-z.

    Punctuation, digits, '$' cashtags and emoji separate tokens ('Long,' -> 'long',
    'moon!!' -> 'moon', '$TSLA' -> 'tsla'). Non-strings give []. This is the tokenizer of the
    usual (#positive - #negative) lexicon count; str.split() keeps punctuation attached and
    misses 'long,'."""
    if not isinstance(text, str):
        return []
    return _ALPHA.findall(text.lower())


def _align(R, Z, lag):
    from .helpers75_2 import _ipca_align
    return _ipca_align(R, Z, lag)


def kps_ipca_als(R, Z, K, max_iter, tol, ridge=0.0, seed=None, lag=True, sweep_sign=True):
    """Instrumented PCA (Kelly, Pruitt, Su 2019) by ALS with the KPS identification EVERY sweep.

    Model r_{i,t} = z_{i,t-1}' Gamma f_t + e. R [T x N] (NaN already replaced), Z [T x N x L];
    lag=True pairs Z[t-1] with R[t] (T_eff = T-1 months). Init F = default_rng(seed).standard_normal
    ((T_eff, K)). One sweep:
      1. Gamma-step: [sum_t Z_t'Z_t kron f_t f_t' + ridge I] vec(Gamma) = sum_t vec(Z_t' r_t f_t')
         (row-major vec, Gamma [L x K]);
      2. identification: Gamma'Gamma = U'U (Cholesky, U upper), Gamma <- Gamma U^-1, F <- F U';
         eigenvectors V of F'F/T_eff in DESCENDING eigenvalue order: Gamma <- Gamma V, F <- F V;
      3. F-step: f_t = (Gamma'Z_t'Z_t Gamma + ridge I)^-1 Gamma'Z_t' r_t;
      4. sign: flip factor k and Gamma column k when mean(F_k) < 0 (eigenvector signs are free and
         otherwise alternate between sweeps, so the iteration could never meet tol).
         know79 sweep_sign=False skips step 4 (a task that flips signs only after the final rotation): the
         identical final F / Gamma; where the eigen step flips signs between sweeps max|dGamma| stays O(1),
         so converged=False and iterations=max_iter (the literal loop's own flag and count).
    Converged when max(max|Gamma - Gamma_prev|, max|F - F_prev|) < tol (iterations = sweeps done).
    Final: eigen-rotation of F'F/T_eff (descending) applied to F and Gamma, then each factor (and its
    Gamma column) is flipped when its time-series mean is negative; F'F/T_eff = diag(eigvals).
    Returns dict{gamma, factors, eigvals, iterations, converged, fitted, resid, total_r2 (uncentred),
    cs_r2 (per month, uncentred), Y, X}."""
    Y, X = _align(R, Z, lag)
    T, N, L = X.shape
    K = int(K)
    XX = np.einsum("tnl,tnm->tlm", X, X)
    Xy = np.einsum("tnl,tn->tl", X, Y)
    F = np.random.default_rng(seed).standard_normal((T, K))
    G_prev, F_prev = None, None
    converged, it = False, 0
    eyeK = np.eye(K)
    for it in range(1, int(max_iter) + 1):
        A = ridge * np.eye(L * K)
        b = np.zeros(L * K)
        for t in range(T):
            A += np.kron(XX[t], np.outer(F[t], F[t]))
            b += np.outer(Xy[t], F[t]).ravel()
        G = np.linalg.solve(A, b).reshape(L, K)
        U = np.linalg.cholesky(G.T @ G).T
        G = G @ np.linalg.inv(U)
        F = F @ U.T
        lam, V = np.linalg.eigh(F.T @ F / T)
        V = V[:, np.argsort(lam)[::-1]]
        G = G @ V
        F = np.array([np.linalg.solve(G.T @ XX[t] @ G + ridge * eyeK, G.T @ Xy[t]) for t in range(T)])
        # eigh leaves each eigenvector's sign free and it alternates between sweeps; without fixing it
        # max|dGamma| stays O(1) forever. Same sign rule as the final step: mean(F_k) >= 0.
        if sweep_sign:
            sg = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
            G, F = G * sg, F * sg
        if G_prev is not None:
            d = max(float(np.max(np.abs(G - G_prev))), float(np.max(np.abs(F - F_prev))))
            if d < tol:
                converged = True
                G_prev, F_prev = G, F
                break
        G_prev, F_prev = G, F
    lam, V = np.linalg.eigh(F.T @ F / T)
    o = np.argsort(lam)[::-1]
    V = V[:, o]
    F, G = F @ V, G @ V
    s = np.where(F.mean(axis=0) < 0, -1.0, 1.0)
    F, G = F * s, G * s
    lam = np.diag(F.T @ F / T).copy()
    fitted = np.einsum("tnl,lk,tk->tn", X, G, F)
    E = Y - fitted
    return {"gamma": G, "factors": F, "eigvals": lam, "iterations": it, "converged": converged, "fitted": fitted,
            "resid": E, "total_r2": float(1 - (E ** 2).sum() / (Y ** 2).sum()),
            "cs_r2": 1 - (E ** 2).sum(axis=1) / (Y ** 2).sum(axis=1), "Y": Y, "X": X}


def grs_statistic(alphas, resid, factors, cov="unbiased"):
    """Gibbons-Ross-Shanken (1989) joint test that N time-series alphas are zero.

    T months, N assets, K factors, alphas [N], resid [T x N] OLS residuals, factors [T x K]:
      cov="unbiased" (GRS 1989): Sigma = E'E/(T-K-1), Omega = cov(F, ddof=1),
        GRS = (T/N) (T-N-K)/(T-K-1) a'Sigma^-1 a / (1 + mu'Omega^-1 mu)
      cov="mle" (Campbell-Lo-MacKinlay): Sigma = E'E/T, Omega = (F-mu)'(F-mu)/T,
        GRS = (T-N-K)/N a'Sigma^-1 a / (1 + mu'Omega^-1 mu)
    The two numerators are identical; only the Omega scaling in the denominator differs.
    p-value = F(N, T-N-K) survival function. Returns dict{stat, p_value, df1, df2, grs_statistic,
    grs_p_value, statistic}."""
    a = np.asarray(alphas, float).reshape(-1)
    E = np.asarray(resid, float)
    F = np.asarray(factors, float)
    F = F[:, None] if F.ndim == 1 else F
    T, N = E.shape
    K = F.shape[1]
    mu = F.mean(axis=0)
    Fc = F - mu
    if cov == "unbiased":
        S = E.T @ E / (T - K - 1)
        O = Fc.T @ Fc / (T - 1)
        stat = (T / N) * (T - N - K) / (T - K - 1) * float(a @ np.linalg.solve(S, a)) / (
            1.0 + float(mu @ np.linalg.solve(O, mu)))
    elif cov == "mle":
        S = E.T @ E / T
        O = Fc.T @ Fc / T
        stat = (T - N - K) / N * float(a @ np.linalg.solve(S, a)) / (1.0 + float(mu @ np.linalg.solve(O, mu)))
    else:
        raise ValueError("cov: 'unbiased' (GRS 1989) | 'mle'")
    pv = float(stats.f.sf(stat, N, T - N - K))
    cv = float(stats.f.ppf(0.95, N, T - N - K))
    return {"stat": float(stat), "p_value": pv, "df1": int(N), "df2": int(T - N - K), "grs_statistic": float(stat),
            "grs_p_value": pv, "statistic": float(stat), "grs_rejected": bool(pv < 0.05), "critical_value": cv}
