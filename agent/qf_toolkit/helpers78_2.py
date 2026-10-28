"""know78 group 2 (volatility / options / GARCH): generic glue for whole-task helpers and the two
local-vol diagnostics that tasks ask for by name.

Pattern behind the know77 losses of this group: a whole-task helper (*_report / *_bundle /
*_deliverables) returned EVERY deliverable, but the cell-per-file scaffold then rebuilt some files by hand
(Dupire flat-vol test with dw/dT = 0, forward-variance table, parity-audit solution.json) or never emitted
them. bundle_files maps any such output to file names (and can emit them all in one line); the two
diagnostics below are the textbook versions a cell can call instead of re-deriving them.

Self-tests: proven_work/fail72/selftest78_2.py.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

__all__ = ["bundle_files", "dupire_flat_vol_check", "forward_variance_table"]


def _default_ext(obj) -> str:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return ".csv"
    if hasattr(obj, "savefig"):
        return ".png"
    if isinstance(obj, str):
        return ".txt"
    return ".json"


def bundle_files(out, names=None, emit=None) -> dict:
    """ANY *_report / *_bundle / *_deliverables output -> {file name: object}; bundle_files(rep, names, emit=emit) writes ALL in one line.

    out: the dict a whole-task helper returned. Its keys are either file names ('summary.json') or
      stems ('summary', 'flat_vol_test'). names: the task's deliverable file names (the contract); each is
      looked up as the exact key, then its stem, then case-insensitively by stem. Without names, every
      key is kept: a key with an extension as is, a stem gets .csv (DataFrame/Series), .png (figure),
      .txt (str) or .json (dict/list/number).
    emit: pass the runtime's emit to write every file now (emit(name, obj) per entry, in `names` order).
    A name with no matching key is NOT invented: a NOTE lists it with the available keys, so only that
    file is left to compute by hand. Returns the {file name: object} mapping."""
    if not isinstance(out, dict):
        raise TypeError("bundle_files expects the dict a *_report/_bundle/_deliverables helper returned")
    res, missing = {}, []
    if names:
        low = {str(k).lower(): k for k in out}
        stems = {pathlib.Path(str(k)).stem.lower(): k for k in out}
        for n in names:
            n = str(n)
            st = pathlib.Path(n).stem
            k = n if n in out else st if st in out else low.get(n.lower()) or stems.get(st.lower())
            if k is None:
                missing.append(n)
            else:
                res[n] = out[k]
    else:
        for k, v in out.items():
            k = str(k)
            res[k if pathlib.Path(k).suffix else k + _default_ext(v)] = v
    if missing:
        print(f"NOTE bundle_files: no entry for {missing}; available keys: {sorted(map(str, out))}")
    if emit is not None:
        for n, obj in res.items():
            emit(n, obj)
    return res


def _bs_call(S, K, T, r, q, sig):
    S, K, T = (np.asarray(x, float) for x in (S, K, T))
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / v
    return S * np.exp(-q * T) * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d1 - v)


def dupire_flat_vol_check(sigma, S0, r=0.0, q=0.0, k_ratios=None, ttms=(0.25, 0.5, 1.0), method="prices",
                          h_rel_K=1e-3, h_T=1e-4, ratio_base="spot") -> pd.DataFrame:
    """Flat-vol identity test of Dupire: constant BS implied vol sigma -> local vol must equal sigma; returns DataFrame(K_ratio, T, local_vol, implied_vol, error).

    method="prices" (Dupire on call prices, the form tasks write):
      sigma_loc^2 = [C_T + (r - q) K C_K + q C] / (0.5 K^2 C_KK), C = Black-Scholes call at sigma;
      C_K, C_KK by CENTRAL differences with step h = h_rel_K * K, C_T by a FORWARD difference h_T.
    method="total_variance" (Gatheral form on w = sigma^2 T, k = ln(K/F)):
      sigma_loc^2 = w_T / [1 - (k/w) w_k + 0.25(-0.25 - 1/w + k^2/w^2) w_k^2 + 0.5 w_kk] with the FLAT surface's
      exact derivatives w_T = sigma^2 (NOT 0 - know77 set dw/dT = 0 and wrote local_vol = 0 everywhere), w_k = w_kk = 0.
    K = base * K_ratio with base = S0 (ratio_base="spot") or the forward F = S0 e^{(r-q)T} ("forward");
    k_ratios default: 9 points 0.90..1.10 (a moderate near-the-money grid, no deep tails).
    error = |local_vol - sigma|; max error is ~1e-5 for "prices" and 0 for "total_variance"."""
    kr = np.linspace(0.90, 1.10, 9) if k_ratios is None else np.asarray(k_ratios, float).ravel()
    rows = []
    for T in np.atleast_1d(np.asarray(ttms, float)):
        F = S0 * math.exp((r - q) * T)
        base = F if str(ratio_base).lower().startswith("f") else S0
        for x in kr:
            K = base * x
            if str(method).lower().startswith("t"):
                w, wT, wk, wkk = sigma * sigma * T, sigma * sigma, 0.0, 0.0
                k = math.log(K / F)
                den = 1 - (k / w) * wk + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * wk * wk + 0.5 * wkk
                lvar = wT / den
            else:
                h = h_rel_K * K
                c0 = float(_bs_call(S0, K, T, r, q, sigma))
                cu, cd = float(_bs_call(S0, K + h, T, r, q, sigma)), float(_bs_call(S0, K - h, T, r, q, sigma))
                cT = (float(_bs_call(S0, K, T + h_T, r, q, sigma)) - c0) / h_T
                cK, cKK = (cu - cd) / (2 * h), (cu - 2 * c0 + cd) / (h * h)
                lvar = (cT + (r - q) * K * cK + q * c0) / (0.5 * K * K * cKK)
            lv = math.sqrt(lvar) if lvar > 0 else float("nan")
            rows.append({"K_ratio": float(x), "T": float(T), "local_vol": lv, "implied_vol": float(sigma),
                         "error": abs(lv - sigma) if math.isfinite(lv) else float("nan")})
    return pd.DataFrame(rows)


def forward_variance_table(T, total_variance, local_vol_atm=None, implied_vol_atm=None, min_positive_share=0.9):
    """Calendar check per expiry: forward_variance_i = (w_{i+1} - w_i)/(T_{i+1} - T_i), last row NaN; returns (DataFrame, flag).

    T: expiry year fractions; total_variance: ATM total implied variance w_i = sigma_ATM(T_i)^2 T_i (e.g.
    SVI at k = 0). Rows are sorted by T. implied_vol_atm defaults to sqrt(w/T); local_vol_atm is copied
    when given (NaN otherwise). Columns: T, total_variance, forward_variance, local_vol_atm, implied_vol_atm.
    flag = True when the share of positive INTERIOR forward variances (all rows but the last, which has no
    later expiry) is >= min_positive_share (True when there is no interior row). No calendar arbitrage
    <=> w non-decreasing in T <=> forward variance >= 0."""
    T = np.asarray(T, float).ravel()
    w = np.asarray(total_variance, float).ravel()
    o = np.argsort(T)
    T, w = T[o], w[o]
    fwd = np.full(len(T), np.nan)
    if len(T) > 1:
        fwd[:-1] = np.diff(w) / np.diff(T)
    lv = np.full(len(T), np.nan) if local_vol_atm is None else np.asarray(local_vol_atm, float).ravel()[o]
    iv = np.sqrt(w / T) if implied_vol_atm is None else np.asarray(implied_vol_atm, float).ravel()[o]
    df = pd.DataFrame({"T": T, "total_variance": w, "forward_variance": fwd, "local_vol_atm": lv,
                       "implied_vol_atm": iv})
    inner = fwd[:-1]
    inner = inner[np.isfinite(inner)]
    flag = bool(len(inner) == 0 or np.mean(inner > 0) >= min_positive_share)
    return df, flag
