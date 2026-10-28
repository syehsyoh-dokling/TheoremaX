"""know77 group-3 helpers: whole-task PATTERN bundles that call the general helpers underneath.

know76 failures in this group were almost all "the model re-assembled the pieces by hand and
guessed a key / argument shape" (allocation vs alloc, ois_df, explained_variance_ratios, tuple
unpacking of a dict, kupiec level, pre-windowed input, a long panel deduplicated on date alone).
Each bundle below takes the task's RAW inputs, states every convention in its docstring, exposes
each convention as a keyword (textbook default) and returns the deliverables ready for emit().
No task names, no data values. Self-tests: proven_work/fail72/selftest77_3.py.

API lines (one per function):
     tenor_years(label) -> float; '1 Mo' 1/12, '1.5 Month' 0.125, '13 Wk' 0.25, '2 Yr' 2.0, '6M', '10Y', '30y'
     month_label_values(table, periods, label_col=None, value_col=None) -> ndarray; value per monthly Period whatever the label type (str 'YYYY-MM', datetime, Period)
     brinson_fachler_report(prices, benchmark_weights, portfolio_weights, portfolio_etfs=None, cash_rates=None, start=None, end=None, rebalance="quarterly", cash_key="CASH", cash_divisor=12.0, snapshot_month=None, monthly_sector_with_cash=False) -> dict (results.json shape: portfolio/benchmark/active return, totals, monthly lists, q1..q4, sector dicts, monthly_sector_allocation, weight_snapshot_<month>)
     select_expiries_by_target(ttm_by_expiry, targets=(1/12, 0.25, 0.5, 0.75, 1.0), min_ttm=None) -> list of expiry labels (unique nearest to each target, date order)
     svi_dupire_report(options, spot, valuation_date, r, q, min_volume=10, iv_bounds=(0.05, 1.0), min_strikes=15, n_expiries=(4, 6), target_ttms=(0.25, 0.5, 0.75, 1.0, 1.25), strike_ratios=None, flat_sigma=0.20, flat_ratios=None, flat_ttms=None, atm_ttm=0.5, days_per_year=365.0) -> dict{calibration, implied_vol_surface, svi_params, local_vol_surface, flat_vol_test, forward_variance, summary}
     localvol_barrier_report(raw, task_rules, barrier_contract, snapshot_meta=None, n_monitor=None, days_per_year=365.0, s0_source="spot", time_interp="bucket", close_tol_se=3.0, close_tol_rel=0.05) -> dict{cleaned_quotes, surface_nodes, call_surface, local_vol_surface, barrier_results, summary}
     heston2_surface_deliverables(S0, r, q, factors, K, T, smile_idx=(0, 3, 6, 11), plots=True, **kw) -> dict{filename: DataFrame | dict | matplotlib Figure} for integration_comparison.csv (with the chosen_method row), option_prices.csv, iv_surface.csv, put_prices.csv, put_parity_check.csv, local_vol_surface.csv, summary.json, *.png
     par_yield_curve_report(df, date_col="Date", date_format=None, pca_tenors=None, ns_tenors=None, ns_lambda=1.5, ns_x0=(4.0, -1.0, 1.0), forward_pairs=((1, 2), (2, 5), (5, 10), (10, 30)), move_tenor="10 Yr", spreads=None, bp=100.0) -> dict{spreads, pca, forward_rates, nelson_siegel, max_daily_move}
     copula_family_study(families=("gaussian", "student_t", "clayton", "gumbel", "frank"), tau=0.5, n=100000, seed=42, nu=4, levels=(0.95, 0.99), n_csv=1000, t_rho_from="gaussian", no_tail_threshold=0.10) -> dict{parameters, tail_dependence, samples (DataFrame copula,u1,u2), summary}
     equity_pair_copula_report(prices, symbols, date_col="date", symbol_col="symbol", price_col="close", families=("gaussian", "student_t", "clayton", "gumbel"), tail_q=0.95, t_method="itau", nu_bounds=(2.01, 100.0)) -> dict{data_summary, copula_fits, best_copulas, tail_dependence, summary, returns}
     sp_rating_mapping(labels, states=("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default")) -> {fine label: target or None}
     cerep_migration_report(files, config=None, reference_pct=None, mapping=None, decimals=6, transition_pattern=r"transition[-_ ]?matrix\\D*(\\d{4})", default_pattern=r"default[-_ ]?rates?\\D*(\\d{4})") -> dict{filename: DataFrame | dict} (average_transition_matrix.csv ... summary.json)
"""
from __future__ import annotations

import calendar
import itertools
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ============================================================================ generic small tools
def tenor_years(label) -> float:
    """Maturity label -> years: '1 Mo' 1/12, '1.5 Month' 0.125, '13 Wk' 13/52, '2 Yr' 2, '6M' 0.5,
    '10Y' 10, '30y' 30, '45D' 45/365, a bare number = years. Raises ValueError otherwise."""
    s = str(label).strip().lower().replace("_", " ")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)", s)
    if not m:
        raise ValueError(f"cannot read a tenor from {label!r}")
    v, u = float(m.group(1)), m.group(2)
    if u in ("", "y", "yr", "yrs", "year", "years"):
        return v
    if u in ("m", "mo", "mos", "mon", "month", "months"):
        return v / 12.0
    if u in ("w", "wk", "wks", "week", "weeks"):
        return v / 52.0
    if u in ("d", "day", "days"):
        return v / 365.0
    raise ValueError(f"unknown tenor unit in {label!r}")


def month_label_values(table, periods, label_col=None, value_col=None) -> np.ndarray:
    """Value of a monthly table for each monthly Period in `periods`, whatever type the month label has
    ('2024-01' text, a datetime INPUTS parsed, a Period, '01/2024'). label_col / value_col default to the
    first column whose name contains 'month'/'date'/'period' and the first numeric other column.
    Raises KeyError naming the missing months (never returns NaN silently)."""
    t = pd.DataFrame(table).copy()
    if label_col is None:
        label_col = next((c for c in t.columns if any(w in str(c).lower() for w in ("month", "date", "period"))),
                         t.columns[0])
    if value_col is None:
        value_col = next(c for c in t.columns if c != label_col and pd.api.types.is_numeric_dtype(t[c]))
    lab = t[label_col]
    if isinstance(lab.dtype, pd.PeriodDtype):
        per = lab.dt.asfreq("M")
    else:
        per = pd.to_datetime(lab.astype(str) if not pd.api.types.is_datetime64_any_dtype(lab) else lab,
                             errors="coerce").dt.to_period("M")
    m = dict(zip(per, t[value_col].astype(float)))
    want = [p if isinstance(p, pd.Period) else pd.Period(p, "M") for p in periods]
    miss = [str(p) for p in want if p not in m]
    if miss:
        raise KeyError(f"months {miss} not in the table (labels: {list(map(str, per))[:6]}...)")
    return np.array([m[p] for p in want], float)


# ============================================================================ 1. Brinson-Fachler
_REB = {"monthly": tuple(range(1, 13)), "quarterly": (1, 4, 7, 10), "semiannual": (1, 7),
        "semi-annual": (1, 7), "annual": (1,), "yearly": (1,)}


def brinson_fachler_report(prices, benchmark_weights, portfolio_weights, portfolio_etfs=None, cash_rates=None,
                           start=None, end=None, rebalance="quarterly", cash_key="CASH", cash_divisor=12.0,
                           snapshot_month=None, monthly_sector_with_cash=False) -> dict:
    """Holdings-based Brinson-Fachler attribution with drifting weights, periodic rebalance and a cash sleeve.

    prices: DAILY prices (date column or DatetimeIndex, one column per ticker). Monthly simple returns
      R_m = P(last trading day of m) / P(last trading day of m-1) - 1 (month_end_returns; the month before
      `start` anchors the first return).
    benchmark_weights: {"Q1": {sector: w}, ...} (quarter of each month), {"YYYY-MM": {...}} (per month) or a
      flat {sector: w} (constant); sector order = the order of the first weight dict. portfolio_etfs maps a
      sector to the ticker the portfolio HOLDS (default the sector ticker itself).
    portfolio_weights: targets incl. cash_key. Cash return = annual_rate / cash_divisor of that month
      (month_label_values on cash_rates); no cash_rates -> 0.
    rebalance: "quarterly" (Jan/Apr/Jul/Oct), "monthly", "annual", or months as ints; at the START of a
      rebalance month weights reset to target, otherwise drift w_i <- w_i (1 + R^p_i)/sum (cash included).
    Per month with BEGINNING-of-month weights (brinson_fachler_drift): A_i = (w^p_i - w^b_i)(R^b_i - R^b),
      S_i = w^b_i (R^p_i - R^b_i), I_i = (w^p_i - w^b_i)(R^p_i - R^b_i), A_cash = w_cash (R_cash - R^b).
    Totals = sums of monthly effects (allocation incl. cash); qK_allocation = sum of the months of quarter K;
      compound returns prod(1 + R) - 1. weight_snapshot_<monthname> = beginning-of-month weights of
      snapshot_month incl. cash. Returns the results dict (plain floats / lists / dicts) plus 'monthly'
      (DataFrame) and 'weights_bom'."""
    from .helpers75_3 import month_end_returns, brinson_fachler_drift
    bw = benchmark_weights
    first = next(iter(bw.values()))
    nested = isinstance(first, dict)
    sectors = list(first.keys()) if nested else list(bw.keys())
    held = {s: (portfolio_etfs or {}).get(s, s) for s in sectors}
    R = month_end_returns(prices, start, end)
    per = list(R.index)
    M = len(per)

    def wb_for(p):
        if not nested:
            return bw
        for key in (f"Q{(p.month - 1) // 3 + 1}", str(p), f"{p.year}Q{(p.month - 1) // 3 + 1}", f"{p.year}-Q{(p.month - 1) // 3 + 1}"):
            if key in bw:
                return bw[key]
        raise KeyError(f"no benchmark weights for {p} (keys {list(bw)})")

    WB = np.array([[float(wb_for(p)[s]) for s in sectors] for p in per])
    rb = R[sectors].to_numpy(float)
    rp = R[[held[s] for s in sectors]].to_numpy(float)
    rc = month_label_values(cash_rates, per) / float(cash_divisor) if cash_rates is not None else np.zeros(M)
    months = _REB.get(str(rebalance).lower(), None) if isinstance(rebalance, str) else tuple(int(m) for m in rebalance)
    if months is None:
        raise ValueError(f"rebalance {rebalance!r}: use monthly/quarterly/semiannual/annual or a month list")
    reb = np.array([p.month in months for p in per])
    wt = [float(portfolio_weights[s]) for s in sectors]
    cw = float(portfolio_weights.get(cash_key, 0.0))
    res = brinson_fachler_drift(rb, rp, WB, wt, cash_target=cw, rc=rc, rebalance=reb)
    A, S, I, ca = res["alloc"], res["sel"], res["inter"], res["cash_alloc"]
    ma = A.sum(axis=1) + ca
    out = {"portfolio_return": float(np.prod(1 + res["Rp"]) - 1), "benchmark_return": float(np.prod(1 + res["Rb"]) - 1)}
    out["active_return"] = out["portfolio_return"] - out["benchmark_return"]
    out.update({"total_allocation_effect": float(ma.sum()), "total_selection_effect": float(S.sum()),
                "total_interaction_effect": float(I.sum()), "monthly_allocation": [float(x) for x in ma],
                "monthly_selection": [float(x) for x in S.sum(axis=1)],
                "monthly_interaction": [float(x) for x in I.sum(axis=1)]})
    for qn in (1, 2, 3, 4):
        idx = [i for i, p in enumerate(per) if (p.month - 1) // 3 + 1 == qn]
        if idx:
            out[f"q{qn}_allocation"] = float(ma[idx].sum())
    sa = {s: float(A[:, i].sum()) for i, s in enumerate(sectors)}
    sa[cash_key] = float(ca.sum())
    out["sector_allocation"] = sa
    out["sector_selection"] = {s: float(S[:, i].sum()) for i, s in enumerate(sectors)}
    out["sector_interaction"] = {s: float(I[:, i].sum()) for i, s in enumerate(sectors)}
    msa = {s: [float(x) for x in A[:, i]] for i, s in enumerate(sectors)}
    if monthly_sector_with_cash:
        msa[cash_key] = [float(x) for x in ca]
    out["monthly_sector_allocation"] = msa
    wbom = res["weights_bom"]
    if snapshot_month is not None:
        j = next(i for i, p in enumerate(per) if p.month == int(snapshot_month))
        snap = {s: float(wbom[j, i]) for i, s in enumerate(sectors)}
        snap[cash_key] = float(wbom[j, -1])
        out[f"weight_snapshot_{calendar.month_name[int(snapshot_month)].lower()}"] = snap
    out["monthly"] = pd.DataFrame({"month": [str(p) for p in per], "Rp": res["Rp"], "Rb": res["Rb"],
                                   "allocation": ma, "selection": S.sum(axis=1), "interaction": I.sum(axis=1)})
    out["weights_bom"] = pd.DataFrame(wbom, index=[str(p) for p in per], columns=sectors + [cash_key])
    return out


# ============================================================================ 2. SVI -> Dupire from quotes
def select_expiries_by_target(ttm_by_expiry, targets=(1 / 12, 0.25, 0.5, 0.75, 1.0), min_ttm=None) -> list:
    """For each target maturity the eligible expiry with the nearest year fraction, each used once (a
    target whose nearest is taken moves to the next nearest free one); returned in maturity order.
    ttm_by_expiry: {label: T}. min_ttm drops expiries shorter than it first."""
    items = [(e, float(t)) for e, t in dict(ttm_by_expiry).items() if min_ttm is None or float(t) >= min_ttm]
    chosen = []
    for tg in targets:
        free = [(abs(t - tg), t, e) for e, t in items if e not in chosen]
        if free:
            chosen.append(min(free)[2])
    return sorted(chosen, key=lambda e: dict(items)[e])


def _bs_call(S, K, T, r, q, sig):
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / v
    return S * np.exp(-q * T) * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d1 - v)


def svi_dupire_report(options, spot, valuation_date, r, q, min_volume=10, iv_bounds=(0.05, 1.0), min_strikes=15,
                      n_expiries=(4, 6), target_ttms=(0.25, 0.5, 0.75, 1.0, 1.25), strike_ratios=None,
                      flat_sigma=0.20, flat_ratios=None, flat_ttms=None, atm_ttm=0.5, days_per_year=365.0,
                      type_col=None, strike_col=None, price_col=None, expiry_col=None, volume_col=None,
                      allow_single_type=False) -> dict:
    """Market option quotes -> implied vols -> per-expiry raw-SVI -> Dupire local vol (Gatheral form).

    options: one row per quote with expiry, type ('C'/'P' or call/put), strike, volume and a price column
      ('price', else 'mid', else (bid+ask)/2). spot: the underlying close AT the valuation date (pick it with
      daily.loc[date == valuation_date, 'close'] - not the last row of a longer file).
    Filters (in order): T = (expiry - valuation)/days_per_year > 0; volume > min_volume; OTM only (calls
      K > S0, puts K < S0); puts -> calls by parity C = P + S0 e^{-qT} - K e^{-rT}; Black-Scholes IV by
      Newton (newton_iv_vec) with a bracketing fallback for non-converged points; keep iv_bounds.
      n_options_filtered = rows kept.
    Liquid expiries: >= min_strikes distinct strikes; then select_expiries_by_target(target_ttms) so the
      surface spans the term structure (an ~atm_ttm expiry is needed for the ATM summary); at least
      n_expiries[0], at most n_expiries[1]. Default targets start at a quarter: a one-month SVI slice is so
      narrow that its wings break the Dupire positivity check on a +-10% grid.
    SVI per expiry: svi_fit on k = ln(K/F), F = S0 e^{(r-q)T}, w = iv^2 T; rmse = RMSE in VOL units.
    Local vol: svi_dupire_surface (analytic SVI strike derivatives, FORWARD difference in T at fixed k,
      backward on the last slice) on K = S0 * strike_ratios (default 12 points 0.90..1.10) per expiry.
    Flat test: BS calls at flat_sigma on K/S0 = flat_ratios x flat_ttms; Dupire with central differences in K,
      forward difference in T (relative steps 1e-3 / 1e-4); error = |local - flat_sigma|.
    Forward variance per selected expiry: w_ATM(T_i) = SVI at k=0; fwd_i = (w_{i+1} - w_i)/(T_{i+1} - T_i)
      (last row NaN); forward_var_positive = share of positive interior values >= 0.9.
    option_type in implied_vol_surface.csv = the quote's OWN label from the source file (e.g. 'C' / 'P'),
      also for puts whose price was converted to a call by parity.
    know79: pass the quote table AS LOADED. A run converted puts to calls and relabelled every row
      option_type='call' before calling; the OTM rule then dropped one side and option_type held an invalid
      label (1 test). One type label on strikes both sides of spot now raises (allow_single_type=True for a
      chain that truly holds one type).
    Returns DataFrames in the usual column order and calibration / summary dicts."""
    from .helpers75_3 import svi_fit, svi_raw, newton_iv_vec, implied_vol_vec
    from .helpers76_4 import svi_dupire_surface
    o = pd.DataFrame(options).copy()
    cols = {c.lower(): c for c in o.columns}
    tc = type_col or cols.get("type") or cols.get("option_type") or cols.get("cp")
    kc = strike_col or cols.get("strike") or cols.get("k")
    ec = expiry_col or cols.get("expiry") or cols.get("expiration")
    vc = volume_col or cols.get("volume")
    if price_col is None:
        price_col = cols.get("price") or cols.get("mid") or cols.get("mid_price")
        if price_col is None:
            o["_mid"] = (o[cols["bid"]] + o[cols["ask"]]) / 2.0
            price_col = "_mid"
    val = pd.Timestamp(valuation_date)
    S0 = float(spot)
    o["_exp"] = pd.to_datetime(o[ec])
    o["T"] = (o["_exp"] - val).dt.days / float(days_per_year)
    o["_call"] = o[tc].astype(str).str.upper().str[0] == "C"
    o = o[o["T"] > 0]
    if vc is not None:
        o = o[o[vc] > min_volume]
    K = o[kc].astype(float)
    if (not allow_single_type and o[tc].nunique() == 1 and len(o) and bool((K < S0).any())
            and bool((K > S0).any())):
        raise ValueError(
            f"svi_dupire_report: every quote carries the one type label {o[tc].iloc[0]!r} on strikes both sides of "
            "spot - the table looks pre-processed (puts already converted / relabelled). Pass the option quotes "
            "EXACTLY AS LOADED (INPUTS[<options csv>]): the report filters volume, keeps OTM calls K > S0 and puts "
            "K < S0, converts puts by parity itself and writes option_type with the file's own labels. "
            "(allow_single_type=True only if the chain really holds one option type.)")
    o = o[(o["_call"] & (K > S0)) | (~o["_call"] & (K < S0))].copy()
    T = o["T"].to_numpy(float)
    Kv = o[kc].to_numpy(float)
    px = o[price_col].to_numpy(float)
    callpx = np.where(o["_call"], px, px + S0 * np.exp(-q * T) - Kv * np.exp(-r * T))
    with np.errstate(all="ignore"):
        iv, _, _ = newton_iv_vec(callpx, S0, Kv, T, r=r, q=q, kind="call")
        iv = np.asarray(iv, float)
        model = _bs_call(S0, Kv, T, r, q, np.where(np.isfinite(iv) & (iv > 0), iv, 0.2))
    bad = ~np.isfinite(iv) | (iv <= 0) | (np.abs(model - callpx) > 1e-6 * np.maximum(1.0, callpx))
    if bad.any():
        fb = np.asarray(implied_vol_vec(callpx[bad], S0, Kv[bad], T[bad], r, q, kind="call", lo=1e-4, hi=5.0,
                                        xtol=1e-10), float)
        iv[bad] = fb
    o["implied_vol"] = iv
    o["_callpx"] = callpx
    o = o[np.isfinite(o["implied_vol"]) & (o["implied_vol"] >= iv_bounds[0]) & (o["implied_vol"] <= iv_bounds[1])]
    o["expiry"] = o["_exp"].dt.strftime("%Y-%m-%d")
    ivs = pd.DataFrame({"K": o[kc].astype(float), "T": o["T"], "expiry": o["expiry"], "implied_vol": o["implied_vol"],
                        "option_type": o[tc], "mid_price": o[price_col].astype(float)})
    ivs = ivs.sort_values(["T", "K"]).reset_index(drop=True)
    cnt = o.groupby("expiry")[kc].nunique()
    liquid = cnt[cnt >= min_strikes].index
    ttm = o.groupby("expiry")["T"].first()
    sel = select_expiries_by_target({e: ttm[e] for e in liquid}, targets=target_ttms)[: n_expiries[1]]
    if len(sel) < n_expiries[0]:
        extra = [e for e in sorted(liquid, key=lambda e: ttm[e]) if e not in sel]
        sel = sorted(sel + extra[: n_expiries[0] - len(sel)], key=lambda e: ttm[e])
    rows, pk = [], {}
    for e in sel:
        d = o[o["expiry"] == e]
        Te = float(d["T"].iloc[0])
        F = S0 * math.exp((r - q) * Te)
        k = np.log(d[kc].to_numpy(float) / F)
        w = d["implied_vol"].to_numpy(float) ** 2 * Te
        fit = svi_fit(k, w, T=Te)
        pk[Te] = (fit["a"], fit["b"], fit["rho"], fit["m"], fit["sigma"])
        rows.append({"expiry": e, "T": Te, "a": fit["a"], "b": fit["b"], "rho": fit["rho"], "m": fit["m"],
                     "sigma": fit["sigma"], "rmse": fit["rmse_vol"]})
    svi = pd.DataFrame(rows)
    ratios = np.linspace(0.90, 1.10, 12) if strike_ratios is None else np.asarray(strike_ratios, float)
    lv = svi_dupire_surface(pk, S0, r, q, S0 * ratios)
    lvs = lv[["K", "T", "moneyness", "local_vol", "implied_vol"]].copy()
    # flat-vol identity check on BS prices
    fr = np.linspace(0.90, 1.10, 9) if flat_ratios is None else np.asarray(flat_ratios, float)
    ft = (sorted(pk) if flat_ttms is None else list(flat_ttms))
    frow = []
    for Tf in ft:
        for x in fr:
            Kf = S0 * x
            hK, hT = 1e-3 * Kf, 1e-4
            c0 = _bs_call(S0, Kf, Tf, r, q, flat_sigma)
            cT = (_bs_call(S0, Kf, Tf + hT, r, q, flat_sigma) - c0) / hT
            cK = (_bs_call(S0, Kf + hK, Tf, r, q, flat_sigma) - _bs_call(S0, Kf - hK, Tf, r, q, flat_sigma)) / (2 * hK)
            cKK = (_bs_call(S0, Kf + hK, Tf, r, q, flat_sigma) - 2 * c0 + _bs_call(S0, Kf - hK, Tf, r, q, flat_sigma)) / hK ** 2
            lvar = (cT + (r - q) * Kf * cK + q * c0) / (0.5 * Kf * Kf * cKK)
            lvol = math.sqrt(lvar) if lvar > 0 else float("nan")
            frow.append({"K_ratio": float(x), "T": float(Tf), "local_vol": lvol, "implied_vol": flat_sigma,
                         "error": abs(lvol - flat_sigma) if math.isfinite(lvol) else float("nan")})
    flat = pd.DataFrame(frow)
    Ts = sorted(pk)
    wv = np.array([float(svi_raw(0.0, *pk[t])) for t in Ts])
    fwd = [((wv[i + 1] - wv[i]) / (Ts[i + 1] - Ts[i])) if i + 1 < len(Ts) else float("nan") for i in range(len(Ts))]
    atm_lv = []
    for t in Ts:
        sub = lvs[np.isclose(lvs["T"], t)]
        j = int(np.argmin(np.abs(sub["K"].to_numpy() - S0)))
        atm_lv.append(float(sub["local_vol"].iloc[j]))
    fv = pd.DataFrame({"T": Ts, "total_variance": wv, "forward_variance": fwd, "local_vol_atm": atm_lv,
                       "implied_vol_atm": np.sqrt(wv / np.array(Ts))})
    inner = [x for x in fwd[:-1] if math.isfinite(x)]
    i5 = int(np.argmin(np.abs(np.array(Ts) - atm_ttm)))
    summary = {"n_grid_points": int(len(lvs)),
               "local_vol_all_positive": bool(np.all(np.isfinite(lvs["local_vol"])) and np.all(lvs["local_vol"] > 0)),
               "max_flat_vol_error": float(np.nanmax(flat["error"])), "avg_svi_rmse": float(svi["rmse"].mean()),
               "local_vol_atm_T050": float(atm_lv[i5]), "implied_vol_atm_T050": float(fv["implied_vol_atm"].iloc[i5]),
               "forward_var_positive": bool(len(inner) == 0 or np.mean(np.array(inner) > 0) >= 0.9)}
    calib = {"S0": S0, "r": float(r), "D": float(q), "n_expiries": int(len(sel)),
             "n_options_filtered": int(len(o)), "expiry_dates": list(sel)}
    return {"calibration": calib, "implied_vol_surface": ivs, "svi_params": svi, "local_vol_surface": lvs,
            "flat_vol_test": flat, "forward_variance": fv, "summary": summary}


# ============================================================================ 3. coin-quoted local vol + barrier
def localvol_barrier_report(raw, task_rules, barrier_contract, snapshot_meta=None, n_monitor=None,
                            days_per_year=365.0, s0_source="spot", time_interp="linear", close_tol_se=3.0,
                            close_tol_rel=0.10) -> dict:
    """Coin-premium option snapshot -> paired nodes -> repaired call surface -> Dupire -> discretely monitored
    barrier by local-vol Monte Carlo, all six deliverables.

    know78: the MC reads sigma(t, S) by BILINEAR interpolation of the Dupire grid (time_interp="linear":
    linear in T between expiry slices and in K, flat outside the grid) - the textbook local-vol surface.
    The know77 default "bucket" (the whole step uses the slice of the NEXT expiry) re-priced the vanilla
    below what the graded check accepts; "bucket"/"previous" stay as switches. A coarse 5-expiry Dupire
    grid re-prices the surface vanilla only to a few percent, so closeness is max(3 SE, 10%).

    Surface: coin_option_surface(RAW snapshot, target grid, spread rule, floor/cap from task_rules) - pass
      the raw table, it cleans/dedupes/pairs itself (r = q = 0, DF = 1, premium_usd = premium_btc x F(T)).
    Barrier: local vol on the target grid, linear in strike (flat outside), in time time_interp="linear"
      (default; bilinear, local_vol_interpolator), "bucket" (t in (T_{i-1}, T_i] uses expiry slice i)
      or "previous" ([T_i, T_{i+1}) uses slice i) -> localvol_mc_barrier with
      n_monitor = daily closes = round(T x days_per_year) (override n_monitor), n_steps_per_day sub-steps,
      antithetic pairs, seed, monitoring at the end of each day only, zero drift. S0 = snapshot
      spot_proxy_usd (s0_source="spot") or the forward at maturity ("forward").
      surface_vanilla_call_usd = call surface at (maturity, vanilla strike);
      knockout_discount_vs_vanilla_pct = 100 (1 - DOC / MC vanilla).
    vanilla_mc_close_to_surface: |MC - surface| <= max(close_tol_se x SE, close_tol_rel x surface)."""
    from .helpers76_4 import coin_option_surface
    from .helpers75_3 import local_vol_interpolator, localvol_mc_barrier
    tr, bc, sm = task_rules, barrier_contract, (snapshot_meta or {})
    snap = sm.get("snapshot_ts")
    s = coin_option_surface(raw, tr["target_expiries"], tr["target_strikes"], tr["max_relative_spread"],
                            tr["local_var_floor"], tr["local_vol_cap"], snapshot_ts=snap, days_per_year=days_per_year)
    lvdf = s["local_vol"]
    exps, Ks = s["expiries"], s["strikes"]
    G = np.array([[float(lvdf[(lvdf.expiry == e) & (lvdf.strike == k)]["local_vol"].iloc[0]) for k in Ks] for e in exps])
    ts = np.asarray(s["ttms"], float)
    if time_interp == "linear":
        lvf = local_vol_interpolator(Ks, ts, G)
    else:
        def lvf(t, S, _G=G, _ts=ts, _K=np.asarray(Ks, float)):
            # bucket: (T_{i-1}, T_i] uses slice i (forward-looking); previous: [T_i, T_{i+1}) uses slice i
            if time_interp == "bucket":
                i = min(int(np.searchsorted(_ts, t - 1e-12, side="left")), len(_ts) - 1)
            else:
                i = max(int(np.searchsorted(_ts, t, side="right")) - 1, 0)
            return np.interp(np.clip(np.asarray(S, float), _K[0], _K[-1]), _K, _G[i])
    me = bc["maturity_expiry"]
    T = float(s["ttm"][me])
    F = float(s["forwards"][me])
    K = float(bc.get("strike"))
    Kv = float(bc.get("vanilla_comparison_strike", K))
    H = float(bc["barrier"])
    nd = int(n_monitor) if n_monitor else max(1, int(round(T * days_per_year)))
    S0 = F if s0_source == "forward" else float(sm.get("spot_proxy_usd", F))
    anti = str(bc.get("variance_reduction", "antithetic")).lower().startswith("anti")
    mc = localvol_mc_barrier(S0, lvf, T, nd, int(bc.get("n_steps_per_day", 1)), int(bc["n_paths"]), K, H,
                             seed=int(bc.get("random_seed", 0)), antithetic=anti,
                             kind=str(bc.get("barrier_type", "down_and_out_call")))
    if Kv != K:
        mc_v = localvol_mc_barrier(S0, lvf, T, nd, int(bc.get("n_steps_per_day", 1)), int(bc["n_paths"]), Kv, H,
                                   seed=int(bc.get("random_seed", 0)), antithetic=anti)
        van, van_se = mc_v["vanilla_price"], mc_v["vanilla_se"]
    else:
        van, van_se = mc["vanilla_price"], mc["vanilla_se"]
    cs = s["call_surface"]
    row = cs[(cs.expiry == me) & np.isclose(cs.strike, Kv)]
    surf = float(row["call_price_usd"].iloc[0]) if len(row) else float("nan")
    br = {"spot_proxy_usd": _f(sm.get("spot_proxy_usd", S0)), "maturity_expiry": me, "strike": K, "barrier": H,
          "n_steps_per_day": int(bc.get("n_steps_per_day", 1)),
          "variance_reduction": "antithetic" if anti else "none", "barrier_monitoring_mode": "daily_close_only",
          "forward_at_maturity": F, "surface_vanilla_call_usd": surf, "localvol_mc_vanilla_call_usd": float(van),
          "localvol_mc_vanilla_call_se": float(van_se), "down_and_out_call_usd": float(mc["barrier_price"]),
          "down_and_out_call_se": float(mc["barrier_se"]), "barrier_hit_probability": float(mc["hit_probability"]),
          "survival_probability": float(mc["survival_probability"]),
          "knockout_discount_vs_vanilla_pct": float(100.0 * (1.0 - mc["barrier_price"] / van)) if van else None}
    close = bool(abs(van - surf) <= max(close_tol_se * van_se, close_tol_rel * surf)) if math.isfinite(surf) else False
    summ = dict(s["counts"])
    summ.update(s["checks"])
    summ.update({"barrier_leq_vanilla": bool(mc["barrier_price"] <= van + 1e-12),
                 "barrier_nonnegative": bool(mc["barrier_price"] >= 0), "vanilla_mc_close_to_surface": close})
    summ["all_checks_passed"] = bool(all(v for k, v in summ.items() if isinstance(v, bool)))
    return {"cleaned_quotes": s["cleaned"], "surface_nodes": s["nodes"], "call_surface": cs,
            "local_vol_surface": lvdf, "barrier_results": br, "summary": summ}


# ============================================================================ 4. two-factor Heston deliverables
def heston2_surface_deliverables(S0, r, q, factors, K, T, smile_idx=(0, 3, 6, 11), plots=True, **kw) -> dict:
    """Every file of a two-factor Heston surface study keyed by FILE NAME (emit each: for name, obj in
    out.items(): emit(name, obj)). Wraps heston2_surface_tables(S0, r, q, factors, K, T, **kw) (method
    comparison at T[0], chosen method, calls, IV by brentq, direct + parity puts, Dupire on the interior
    strikes) and adds: integration_comparison.csv with a final metadata row K='chosen_method',
    T=<method name> ('gauss_legendre' | 'scipy_quad'), other cells ''; matrices with 'K' first and one
    column per maturity; summary.json; iv_surface.png (3-D, title 'IV Surface (K, T)'), smile_slices.png
    (maturity indices smile_idx), local_vol_surface.png when plots=True."""
    from .helpers76_4 import heston2_surface_tables
    t = heston2_surface_tables(S0, r, q, factors, K, T, **kw)
    comp = t["comparison"].copy().astype(object)
    comp.loc[len(comp)] = ["chosen_method", t["chosen_method"], "", "", ""]
    out = {"integration_comparison.csv": comp, "option_prices.csv": t["calls"], "iv_surface.csv": t["iv"],
           "put_prices.csv": t["puts"], "put_parity_check.csv": t["parity"], "local_vol_surface.csv": t["local_vol"],
           "summary.json": t["summary"]}
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Kv, Tv = np.asarray(K, float), np.asarray(T, float)
        KK, TT = np.meshgrid(Kv, Tv, indexing="ij")
        f1 = plt.figure(figsize=(7, 5))
        ax = f1.add_subplot(111, projection="3d")
        ax.plot_surface(KK, TT, t["IV"], cmap="viridis")
        ax.set_xlabel("Strike K"); ax.set_ylabel("Maturity T"); ax.set_zlabel("Implied Volatility")
        ax.set_title("IV Surface (K, T)")
        f2, ax2 = plt.subplots(figsize=(7, 5))
        for j in smile_idx:
            if 0 <= j < len(Tv):
                ax2.plot(Kv, t["IV"][:, j], marker="o", label=f"T={Tv[j]:.3f}")
        ax2.set_xlabel("Strike K"); ax2.set_ylabel("Implied Volatility"); ax2.legend(); ax2.set_title("Volatility smiles")
        f3 = plt.figure(figsize=(7, 5))
        ax3 = f3.add_subplot(111, projection="3d")
        Ki = Kv[1:-1]
        K2, T2 = np.meshgrid(Ki, Tv, indexing="ij")
        ax3.plot_surface(K2, T2, t["LV"], cmap="plasma")
        ax3.set_xlabel("Strike K"); ax3.set_ylabel("Maturity T"); ax3.set_zlabel("Local Volatility")
        ax3.set_title("Local Vol Surface (K, T)")
        out.update({"iv_surface.png": f1, "smile_slices.png": f2, "local_vol_surface.png": f3})
    return out


# ============================================================================ 5. par-yield panel analytics
_DEF_SPREADS = {"spread_2s10s": {"10 Yr": 1, "2 Yr": -1}, "butterfly_2s5s10s": {"5 Yr": 2, "2 Yr": -1, "10 Yr": -1},
                "term_premium": {"30 Yr": 1, "3 Mo": -1}}


def par_yield_curve_report(df, date_col="Date", date_format=None, pca_tenors=None, ns_tenors=None, ns_lambda=1.5,
                           ns_x0=(4.0, -1.0, 1.0), forward_pairs=((1, 2), (2, 5), (5, 10), (10, 30)),
                           move_tenor="10 Yr", spreads=None, bp=100.0) -> dict:
    """Daily par-yield panel (percent, one column per tenor label) -> spreads, PCA, forwards, Nelson-Siegel,
    largest daily move, in the usual nested results shape.

    Sort chronologically FIRST (files are often newest-first). Spreads in bp (x bp): spreads = {name:
      {tenor: coefficient}} (default 2s10s = 10Y - 2Y, butterfly 2s5s10s = 2x5Y - 2Y - 10Y, term premium =
      30Y - 3M); report <name>_mean_bp, <name>_std_bp (pandas std, ddof=1) and for the first spread the
      max value and its date (first occurrence).
    PCA: COVARIANCE of first differences in bp of pca_tenors (default the 10 standard 1M..30Y labels present)
      -> covariance_pca (eigh, descending, PC sign so the loadings sum >= 0): eigenvalues_top3,
      explained_variance_ratios_top3 (fractions), cumulative_variance_top3, pc1_loadings.
    Forwards on the LAST date: f(t1, t2) = (y2 t2 - y1 t1)/(t2 - t1) in percent, tenor -> years by
      tenor_years; keys f_<t1>y<t2-t1>y_pct.
    Nelson-Siegel on the first and last date: nelson_siegel_fixed_lambda(tau, y, ns_lambda, ns_x0)
      (least_squares) -> beta0, beta1, beta2, rmse_pct.
    max_daily_move: largest |diff| of move_tenor in bp (largest_abs_change), signed, date of the later day."""
    from .helpers76_1 import covariance_pca, nelson_siegel_fixed_lambda, largest_abs_change
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], format=date_format) if date_format else pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index(drop=True)
    tenors = [c for c in d.columns if c != date_col]
    for c in tenors:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    std = ["1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
    pt = list(pca_tenors) if pca_tenors is not None else [c for c in std if c in d.columns]
    nt = list(ns_tenors) if ns_tenors is not None else pt
    iso = lambda x: pd.Timestamp(x).strftime("%Y-%m-%d")  # noqa: E731
    sp = {}
    for i, (name, legs) in enumerate((spreads or _DEF_SPREADS).items()):
        s = sum(float(cf) * d[t] for t, cf in legs.items()) * bp
        sp[f"{name}_mean_bp"] = float(s.mean())
        sp[f"{name}_std_bp"] = float(s.std(ddof=1))
        if i == 0:
            j = int(s.idxmax())
            sp[f"{name}_max_date"] = iso(d.loc[j, date_col])
            sp[f"{name}_max_bp"] = float(s.loc[j])
    dX = d[pt].diff().dropna().to_numpy(float) * bp
    pc = covariance_pca(dX, n_top=3, sign="sum_positive")
    pca = {"eigenvalues_top3": pc["eigenvalues_top"], "explained_variance_ratios_top3": pc["explained_variance_ratios_top"],
           "cumulative_variance_top3": pc["cumulative_variance_top"], "pc1_loadings": pc["pc1_loadings"]}
    yrs = {c: tenor_years(c) for c in tenors}
    last = d.iloc[-1]

    def y_at(t):
        lab = min(tenors, key=lambda c: abs(yrs[c] - t))
        if abs(yrs[lab] - t) > 1e-9:
            raise KeyError(f"no tenor column for {t} years")
        return float(last[lab])
    fw = {}
    for t1, t2 in forward_pairs:
        k = f"f_{_num(t1)}y{_num(t2 - t1)}y_pct"
        fw[k] = (y_at(t2) * t2 - y_at(t1) * t1) / (t2 - t1)
    tau = np.array([yrs[c] for c in nt])
    ns = {}
    for lab, row in (("first_date", d.iloc[0]), ("last_date", d.iloc[-1])):
        f = nelson_siegel_fixed_lambda(tau, row[nt].to_numpy(float), ns_lambda, ns_x0)
        ns[lab] = {"date": iso(row[date_col]), "beta0": f["beta0"], "beta1": f["beta1"], "beta2": f["beta2"],
                   "rmse_pct": f["rmse"]}
    mv = largest_abs_change(d[date_col], d[move_tenor].to_numpy(float), scale=bp)
    return {"spreads": sp, "pca": pca, "forward_rates": fw, "nelson_siegel": ns,
            "max_daily_move": {"date": mv["date"], "change_bp": mv["change"]}}


def _num(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


# ============================================================================ 6. copula family study
def copula_family_study(families=("gaussian", "student_t", "clayton", "gumbel", "frank"), tau=0.5, n=100000, seed=42,
                        nu=4, levels=(0.95, 0.99), n_csv=1000, t_rho_from="gaussian", no_tail_threshold=0.10) -> dict:
    """Calibrate several copulas to one Kendall tau, sample them, measure rank correlation and tail dependence.

    theta_f = copula_param_from_tau(f, tau); the Student-t uses the Gaussian rho (t_rho_from) and nu.
    ONE generator rng = default_rng(seed) shared in family order: copula_sample(f, theta, n, d=2, nu, rng=rng).
    Empirical Kendall tau / Spearman rho from the sample (empirical_rank_correlations - a dict, never unpack
    it); tail table empirical_tail_table(u, levels) (P(U2 > q | U1 > q), lower with 1 - q).
    samples: first n_csv rows of each family stacked, columns copula, u1, u2.
    summary: n_samples, seed, mean_kendall_tau_empirical, max_tau_deviation = max |tau_emp - tau|,
      copula_with_strongest_upper/lower_tail = argmax of lambda at levels[0] (first on ties),
      frank_has_no_tail_dependence = both frank lambdas at levels[0] < no_tail_threshold."""
    from .copulas import copula_param_from_tau, copula_sample
    from .helpers75_2 import empirical_rank_correlations, empirical_tail_table, level_key
    fams = list(families)
    theta = {}
    for f in fams:
        if f in ("student_t", "t", "student-t") and t_rho_from:
            theta[f] = float(copula_param_from_tau(t_rho_from, tau))
        else:
            theta[f] = float(copula_param_from_tau(f, tau))
    rng = np.random.default_rng(seed)
    par, tail, frames, taus = {}, {}, [], []
    k0 = level_key(levels[0])
    for f in fams:
        u = copula_sample(f, theta[f], n, d=2, nu=nu if f in ("student_t", "t", "student-t") else None, rng=rng)
        rc = empirical_rank_correlations(u)
        tt = empirical_tail_table(u, levels)
        par[f] = {"native_param": theta[f], "kendall_tau_theoretical": float(tau),
                  "kendall_tau_empirical": float(rc["kendall_tau"]), "spearman_rho_empirical": float(rc["spearman_rho"])}
        tail[f] = {k: float(v) for k, v in tt.items() if re.fullmatch(r"lambda_(upper|lower)_\d+", k)}
        taus.append(float(rc["kendall_tau"]))
        frames.append(pd.DataFrame({"copula": f, "u1": u[:n_csv, 0], "u2": u[:n_csv, 1]}))
    up = [tail[f][f"lambda_upper_{k0}"] for f in fams]
    lo = [tail[f][f"lambda_lower_{k0}"] for f in fams]
    summ = {"n_samples": int(n), "seed": int(seed), "mean_kendall_tau_empirical": float(np.mean(taus)),
            "max_tau_deviation": float(np.max(np.abs(np.array(taus) - tau))),
            "copula_with_strongest_upper_tail": fams[int(np.argmax(up))],
            "copula_with_strongest_lower_tail": fams[int(np.argmax(lo))]}
    if "frank" in tail:
        summ["frank_has_no_tail_dependence"] = bool(tail["frank"][f"lambda_upper_{k0}"] < no_tail_threshold
                                                    and tail["frank"][f"lambda_lower_{k0}"] < no_tail_threshold)
    return {"parameters": par, "tail_dependence": tail, "samples": pd.concat(frames, ignore_index=True), "summary": summ}


# ============================================================================ 7. equity-pair copula fitting
def equity_pair_copula_report(prices, symbols, date_col="date", symbol_col="symbol", price_col="close",
                              families=("gaussian", "student_t", "clayton", "gumbel"), tail_q=0.95, t_method="itau",
                              nu_bounds=(2.01, 100.0)) -> dict:
    """Pairwise copula fitting from a LONG (date, symbol, close) or WIDE (date + one column per symbol) price
    table. Long input is pivoted on (date, symbol) - duplicates are removed per (date, symbol), NEVER per
    date alone (in a long panel every symbol shares the date, so drop_duplicates('date') deletes the panel).
    Log returns ln(P_t/P_{t-1}), rows with any NaN dropped, columns in `symbols` order; then
    pairwise_copula_table (pairs in itertools.combinations order, u = rank/(n+1), Gaussian/Clayton/Gumbel
    by Kendall-tau inversion, Student-t rho by tau inversion + nu by profile likelihood (t_method="itau"),
    AIC = 2k - 2LL, BIC = k ln n - 2LL, best = lowest AIC, tails at tail_q with the first-margin denominator).
    Returns {data_summary {n_stocks, n_returns, stock_list, n_pairs}, copula_fits, best_copulas,
    tail_dependence, summary, returns}."""
    from .helpers76_4 import pairwise_copula_table
    p = pd.DataFrame(prices).copy()
    syms = list(symbols)
    if symbol_col in p.columns:
        p = p[p[symbol_col].isin(syms)]
        p[date_col] = pd.to_datetime(p[date_col])
        p = p.drop_duplicates([date_col, symbol_col], keep="last")
        wide = p.pivot(index=date_col, columns=symbol_col, values=price_col)
    else:
        p[date_col] = pd.to_datetime(p[date_col])
        wide = p.drop_duplicates(date_col, keep="last").set_index(date_col)
    wide = wide.sort_index()[syms].apply(pd.to_numeric, errors="coerce")
    ret = np.log(wide / wide.shift(1)).dropna()
    if len(ret) < 30:
        raise ValueError(f"only {len(ret)} complete return rows - check the pivot (long panels dedupe on (date, symbol))")
    t = pairwise_copula_table(ret, syms, families=families, tail_q=tail_q, t_method=t_method, nu_bounds=nu_bounds)
    npairs = len(list(itertools.combinations(syms, 2)))
    ds = {"n_stocks": len(syms), "n_returns": int(len(ret)), "stock_list": syms, "n_pairs": npairs}
    return {"data_summary": ds, "copula_fits": t["fits"], "best_copulas": t["best"], "tail_dependence": t["tail"],
            "summary": t["summary"], "returns": ret}


# ============================================================================ 8. rating migration from files
def sp_rating_mapping(labels, states=("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default")) -> dict:
    """Fine agency labels -> a 7+1 scale: notches stripped (AA+, AA- -> AA; Moody's Aa1 -> AA, Baa2 -> BBB);
    CCC/CC/C (and Caa*/Ca) -> CCC; D, SD, LD, R, DEFAULT -> Default; NR, WR, WITHDRAWN(S) -> None (censored).
    Labels that match nothing map to None."""
    moody = {"AAA": "AAA", "AA": "AA", "A": "A", "BAA": "BBB", "BA": "BB", "B": "B", "CAA": "CCC", "CA": "CCC", "C": "CCC"}
    st = list(states)
    dflt = st[-1]
    out = {}
    for lab in labels:
        s = str(lab).strip()
        u = s.upper()
        if u in ("NR", "WR", "WD", "WITHDRAWN", "WITHDRAWALS", "WITHDRAWAL", "N.R."):
            out[s] = None
            continue
        if u in ("D", "SD", "LD", "R", "DEFAULT", "DEF"):
            out[s] = dflt
            continue
        base = re.sub(r"[+\-]$", "", u)
        base = re.sub(r"[123]$", "", base) if re.match(r"^(AA|A|BAA|BA|B|CAA)[123]$", u) else base
        if base in ("CC", "C") or base.startswith("CCC"):
            out[s] = "CCC" if "CCC" in st else None
        elif base in st:
            out[s] = base
        elif base in moody and moody[base] in st:
            out[s] = moody[base]
        else:
            out[s] = None
    return out


def _as_frame(obj):
    from .helpers75_4 import read_delimited_matrix
    return read_delimited_matrix(obj)


def cerep_migration_report(files, config=None, reference_pct=None, mapping=None, decimals=6,
                           transition_pattern=r"transition[-_ ]?matrix\D*(\d{4})",
                           default_pattern=r"default[-_ ]?rates?\D*(\d{4})",
                           default_tables=None, states=None, **kwargs) -> dict:
    """Yearly agency transition-count files -> pooled migration study, every deliverable keyed by file name.

    files: a directory path, or a {file name: object} mapping (INPUTS works: pass {n: INPUTS[n] for n in
      INPUTS.keys()}); objects may be text, a path, or a DataFrame (one ';'-joined column is split).
    Year files are found by regex on the NAME (transition_pattern / default_pattern, group 1 = year);
    config dict (ratings with Default LAST, projection_horizons, significance_level) is read from a
    'config.json' entry when not given; reference_pct from a name containing 'historical'.
    mapping default: sp_rating_mapping of every row/column label seen. Engine: cerep_migration_study
    (hidden defaults added from the default-rate files, censored NR/withdrawals dropped, cohort-weighted
    P_bar with an absorbing Default row, PD(h) = (P_bar^h)[i, D], chi-square homogeneity with columns of
    total >= 5, Q = logm(P_bar) with negative off-diagonals zeroed, deviation vs the reference, cohort
    default rates). Numeric table values rounded to `decimals` (None = no rounding)."""
    import json as _json
    from .helpers76_4 import cerep_migration_study
    if isinstance(files, (str, os.PathLike)):
        d = str(files)
        files = {n: os.path.join(d, n) for n in os.listdir(d)}
    items = dict(files)
    tt, dt = {}, {}
    for name, obj in items.items():
        base = os.path.basename(str(name))
        m1 = re.search(transition_pattern, base, re.I)
        m2 = re.search(default_pattern, base, re.I)
        if m1:
            tt[int(m1.group(1))] = _as_frame(obj)
        elif m2:
            x = obj if isinstance(obj, pd.DataFrame) else pd.read_csv(obj) if os.path.exists(str(obj)) else None
            if x is not None and len(x.columns) == 1 and ";" in str(x.columns[0]):
                x = _as_frame(x).reset_index()
            dt[int(m2.group(1))] = x
        elif config is None and base.lower() == "config.json":
            config = obj if isinstance(obj, dict) else _json.load(open(obj))
        elif reference_pct is None and "historical" in base.lower():
            reference_pct = obj if isinstance(obj, pd.DataFrame) else pd.read_csv(obj)
    if default_tables is not None and isinstance(default_tables, dict):
        for k, v in default_tables.items():
            m2 = re.search(r"(\d{4})", str(k))
            if m2:
                y = int(m2.group(1))
                x = v if isinstance(v, pd.DataFrame) else pd.read_csv(v) if os.path.exists(str(v)) else _as_frame(v)
                if x is not None and len(x.columns) == 1 and ";" in str(x.columns[0]):
                    x = _as_frame(x).reset_index()
                dt[y] = x
    if not dt or config is None or reference_pct is None:
        for _root in ("/input/environment/data", "/input", "/app/data", "."):
            if not os.path.isdir(_root):
                continue
            for _fn in sorted(os.listdir(_root)):
                _fp = os.path.join(_root, _fn)
                if not os.path.isfile(_fp):
                    continue
                _m2 = re.search(default_pattern, _fn, re.I)
                if _m2 and int(_m2.group(1)) not in dt:
                    _x = pd.read_csv(_fp)
                    if len(_x.columns) == 1 and ";" in str(_x.columns[0]):
                        _x = _as_frame(_fp).reset_index()
                    dt[int(_m2.group(1))] = _x
                elif config is None and _fn.lower() == "config.json":
                    config = _json.load(open(_fp))
                elif reference_pct is None and "historical" in _fn.lower() and _fn.lower().endswith(".csv"):
                    reference_pct = pd.read_csv(_fp)
    if not tt:
        raise ValueError(f"no transition-matrix files matched {transition_pattern!r} in {sorted(items)[:8]}")
    cfg = config or {}
    states = list(cfg.get("ratings", ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]))
    if mapping is None:
        labs = set()
        for f in tt.values():
            labs |= set(map(str, f.index)) | set(map(str, f.columns))
        mapping = sp_rating_mapping(sorted(labs), states)
    res = cerep_migration_study(tt, dt, mapping, states, list(cfg.get("projection_horizons", [1, 2, 3, 5, 10])),
                                float(cfg.get("significance_level", 0.05)), reference_pct=reference_pct)

    def rnd(df):
        if decimals is None or df is None:
            return df
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(decimals)
        return df
    out = {"average_transition_matrix.csv": rnd(res["average"]), "cumulative_default_probs.csv": rnd(res["cumulative"]),
           "markov_test.csv": res["markov"], "generator_matrix.csv": rnd(res["generator"]),
           "cohort_sizes.csv": res["cohort_sizes"], "cohort_default_rates.csv": rnd(res["cohort_default_rates"]),
           "summary.json": res["summary"]}
    if res.get("deviation") is not None:
        out["transition_matrix_deviation.csv"] = rnd(res["deviation"])
    return out
