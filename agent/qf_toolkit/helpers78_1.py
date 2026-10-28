"""know78 group-1 helpers (fixed income / rates / xccy): whole-task PATTERN bundles for the units whose
know77 programs ran out of requests while hand-coding a long desk workflow.

1. mtm_xccy_swap_bundle - collateralised mark-to-market cross-currency basis swap desk: quote-book cleaning,
   OIS discount + IBOR projection (deposit / FRA chain) bootstraps, FX forward nodes, CIP no-basis forwards
   and implied basis, MTM-resetting notional schedule with historical fixings, leg PVs, par basis, curve /
   FX risks and labelled stress scenarios, reconciliation flags and audit checkpoints.
2. fx_carry_hedge_bundle - G10 FX carry strategy with a stale quarterly forward-hedge overlay, dealer FX
   snapshot (via fx_snapshot_desk), 25-delta smile options (Garman-Kohlhagen, delta conventions,
   premium conversions, Greeks), strategy risk metrics and a rolling equity correlation.
   Building blocks: fx_carry_signals, fx_carry_hedge_backtest, fx_smile_option_book, return_risk_metrics.

Every function is general (no task names): conventions are parameters with the textbook default, numbers
a task fixes (dates, bumps, windows, top/bottom n) are REQUIRED arguments or come from the task's files.
Pure numpy/pandas/scipy. Self-tests: proven_work/fail72/selftest78_1.py.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================================
# shared small helpers
# =============================================================================================


def _iso(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _denom(day_count, default=360.0) -> float:
    if day_count is None:
        return float(default)
    if isinstance(day_count, (int, float)):
        return float(day_count)
    m = re.search(r"(360|365|366)", str(day_count))
    return float(m.group(1)) if m else float(default)


def tenor_months(label) -> float:
    """'1M' -> 1, '18M' -> 18, '1Y' -> 12, '2W' -> 0.5, 'SPOT'/'ON' -> 0, FRA '9x12' -> 12 (END month)."""
    s = str(label).strip().upper()
    m = re.fullmatch(r"(\d+)\s*X\s*(\d+)", s)
    if m:
        return float(m.group(2))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([DWMY])", s)
    if not m:
        return 0.0
    n, u = float(m.group(1)), m.group(2)
    return {"D": n / 30.0, "W": n * 7 / 30.0, "M": n, "Y": 12 * n}[u]


def _label_key(label) -> str:
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


def _bdays_between(a, b) -> int:
    a, b = pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()
    return int(np.busday_count(a.date(), b.date()))


def _add_bdays(d, n):
    return pd.Timestamp(np.busday_offset(pd.Timestamp(d).date(), int(n), roll="forward"))


# =============================================================================================
# 1. MTM cross-currency basis swap desk
# =============================================================================================

def _lnF_interp(spot_date, spot, dates, fwds):
    sd = pd.Timestamp(spot_date).normalize()
    x = np.array([0.0] + [(pd.Timestamp(d).normalize() - sd).days for d in dates], float)
    y = np.log(np.array([float(spot)] + [float(f) for f in fwds], float))
    o = np.argsort(x)
    x, y = x[o], y[o]

    def f(d):
        return float(np.exp(np.interp((pd.Timestamp(d).normalize() - sd).days, x, y)))
    return f


def mtm_xccy_swap_bundle(market_quotes, libor_fixings, trade_schedule, trade, *, scenarios=(),
                       quote_key=("curve", "instrument_type", "tenor", "start_date", "end_date", "ccy_or_pair"),
                       ois_pay_months=None, pv01_bp=1.0, fx_delta_pct=1.0, reset_cashflow="borrower",
                       accrued_fx="spot", decimals=None, twist_short_bp=None, twist_long_bp=None,
                       twist_pivot_months=None, return_market=False, event_order="leg") -> dict:
    """-> {file name: object} for a collateralised MTM cross-currency basis swap desk (11 deliverables).

    Inputs (parsed INPUTS objects or paths): market_quotes [quote_time, curve, instrument_type, tenor,
    start_date, end_date, ccy_or_pair, quote_unit, quote_value, quote_adjustment_bps]; libor_fixings
    [fixing_date, index_name, rate_pct]; trade_schedule [period_id, accrual_start, accrual_end,
    payment_date, reset_date, fixing_date, ...]; trade dict (valuation_date, spot_date, fx_pair 'BBBQQQ'
    quoted Q per B, collateral_currency, <ccy>_day_count, <coll>_notional, inception_spot_fx,
    contract_spread_bps, receive_leg / pay_leg labels starting with the leg currency).
    Conventions (textbook defaults, each a parameter):
      * quote book: normalize_quote_book (latest quote_time per quote_key; rate_pct /100, pips /10^4 or
        /100 for JPY pairs); the FRA quote_adjustment_bps is ADDED for the projection curve only.
      * curves: bootstrap_df_curve anchored at the quotes' common start (spot) date, DF = 1 there, linear in
        -ln DF on calendar days (= 'linear in g(T) = -ln DF' / 'linear in ln Q'); OIS = deposits + OIS swap
        quotes; ois_pay_months=None (know79 default, no fixed-leg frequency stated): every OIS quote is ONE
        simple-rate period start -> end, DF(end) = DF(start)/(1 + R tau); ois_pay_months=12: par swap with an
        annual fixed leg rolled back from maturity (front stub); projection = deposit + FRA
        chain; tau = days / denominator of the currency's day count. year_frac = days(spot, pillar)/basis.
        DF used for discounting = DF(d) / DF(valuation date) (= forward DFs for a rolled valuation).
      * FX: nodes F_i = spot + increment_i at the forward end dates, ln F linear in calendar days from the
        spot date (spot is the first node), flat beyond the last node -> every broken date.
      * CIP no-basis forward F_cip(T) = S [DF_B(T)/DF_B(s)] / [DF_Q(T)/DF_Q(s)] (B = base, Q = quote ccy
        = collateral); implied basis b = -ln(F_quoted / F_cip) / tau_Q(s, T) in bp (negative when the
        quoted forward is above CIP); 'basis held fixed' rebuilds F_i = S' F_cip'(T_i)/S exp(-b_i tau_i).
      * swap: collateral-currency leg has a constant notional (<coll>_notional); the other (MTM) leg starts
        at N_0 = coll_notional / inception_spot_fx and RESETS at each later reset date to
        N_k = coll_notional / F(reset_date_k); each reset is a separate MTM-leg 'mtm_reset' cash flow.
        reset_cashflow="borrower" (default, know78): the holder RECEIVING the MTM leg is paid the notional
        increase at each reset, signed = +(N_k - N_{k-1}) (pay leg: the opposite); with it a receive-MTM /
        pay-collateral trade has NEGATIVE fx delta and USD-OIS PV01, POSITIVE MTM-ccy OIS PV01, and par
        basis above the contractual spread when PV < 0 - the risk-sign pattern such desk tasks check.
        "lender" = the opposite sign +(N_{k-1} - N_k). Final principal exchange on the last payment date (receive leg +,
        pay leg -). Coupons: fixing_date <= valuation date -> historical fixing (libor_fixings, index
        whose name starts with the currency), later -> projected forward (Q(s)/Q(e) - 1)/tau over the
        accrual period; contractual spread on the leg whose label contains 'SPREAD' (else the MTM leg);
        coupon_rate column = all-in rate paid. Cash flows strictly before the valuation date are excluded.
        PV: collateral leg sum signed x DF_coll(pay); MTM leg sum signed x F(pay) x DF_coll(pay).
        Row order (know79 event_order="leg"): payment date, then the MTM leg (coupon, mtm_reset, principal),
        then the collateral leg (coupon, principal); event_order="type": both coupons, reset, principals.
      * accrued (current period): signed notional x rate x days(accrual start, valuation)/basis, the MTM
        leg converted at spot (accrued_fx="spot") or at F(payment) ("forward").
      * par basis: all-in spread on the spread leg that sets trade PV to zero (PV is linear in it), bp.
      * risks: pv01_<ccy>_<curve> = PV(quotes of that curve + pv01_bp) - PV (OIS bumps keep the basis
        fixed, so FX forwards move through CIP); fx_delta_<coll>_per_1pct = PV(spot and nodes x 1.01) - PV.
    scenarios: sequence of (label, spec) with spec one of
        {'type': 'ois', 'ccy': 'USD', 'bp': 5}            parallel OIS bump, basis held fixed
        {'type': 'basis', 'bp': -2}                       add bp to every basis node (widen = negative)
        {'type': 'basis_leg', 'bp': -2}                   only the MTM leg revalued, other leg at base PV
        {'type': 'fx', 'mult': 1.01, 'nodes': True}       spot and all tenor nodes x mult
        {'type': 'fx', 'mult': 1.01, 'nodes': False}      spot x mult, tenor nodes unchanged (re-interpolate)
        {'type': 'twist', 'short_bp': 5, 'long_bp': -5, 'pivot_months': 12}   OIS + projection pillars
        {'type': 'roll', 'date': 'YYYY-MM-DD'}            revalue on that date: earlier cash flows dropped,
            fixings on/before it historical, forward DFs of the frozen curves, FX rebuilt from the rolled
            spot (base forward to the new spot date) and the frozen basis term structure.
      Plain LABEL strings are parsed by stress_scenarios_from_labels (numbers read from the label:
       'usd_ois_up_5bp', 'basis_widen_2bp', 'gbp_leg_basis_widen_2bp', 'spot_up_1pct',
       'spot_realized_1d_up_1pct', 'roll_down_to_YYYY_MM_DD'); 'curve_twist' takes twist_short_bp,
       twist_long_bp, twist_pivot_months from the task text.
    Returns {cleaned_quotes.csv, <ccy>_curve.csv per OIS currency, basis_curve.csv, fx_forward_curve.csv,
    cashflow_schedule.csv, results.json, stress_results.csv, reconciliation.json, solution.json
    {'intermediates': {name: {'value': x}}}, summary.json} - every value emit-ready; '_market' (curves, fx
    functions) only with return_market=True."""
    from .helpers75_4 import normalize_quote_book
    from .helpers77_4 import bootstrap_df_curve

    rd = lambda o: pd.read_csv(o) if isinstance(o, (str, bytes)) or hasattr(o, "__fspath__") else o  # noqa: E731
    mq, fx_tab, sch = rd(market_quotes).copy(), rd(libor_fixings).copy(), rd(trade_schedule).copy()
    if not isinstance(trade, dict):
        import json as _json
        trade = _json.load(open(trade))
    T = {str(k).lower(): v for k, v in trade.items()}
    val0 = pd.Timestamp(T["valuation_date"]).normalize()
    pair = str(T.get("fx_pair", "")).upper().replace("/", "")
    B_ccy, Q_ccy = pair[:3], pair[3:6]
    coll = str(T.get("collateral_currency", Q_ccy)).upper()
    if coll != Q_ccy:
        raise NotImplementedError("collateral currency must be the quote (terms) currency of fx_pair")
    mtm = B_ccy
    pipm = 100.0 if "JPY" in pair else 1e4
    keys = [c for c in quote_key if c in mq.columns]
    nq = normalize_quote_book(mq, keys, pip_multiplier=pipm)
    cleaned, full = nq["cleaned"], nq["full"]
    dc = {c: T.get(f"{c.lower()}_day_count", "ACT/365" if c in ("GBP", "AUD", "NZD", "CAD") else "ACT/360")
          for c in (B_ccy, Q_ccy)}
    basis_days = {c: _denom(v) for c, v in dc.items()}
    adjc = "quote_adjustment_bps" if "quote_adjustment_bps" in full.columns else None

    # ---- curve quote sets
    is_fx = full["instrument_type"].astype(str).str.lower().isin(["spot", "forward", "fx_forward", "outright"])
    curves = {}
    for lab, g in full[~is_fx].groupby("curve", sort=False):
        u = str(lab).upper()
        ccy = str(g["ccy_or_pair"].iloc[0]).upper()[:3] if "ccy_or_pair" in g else u[:3]
        kind = "ois" if "OIS" in u else "proj"
        rows = []
        for _, r in g.iterrows():
            adj = float(r[adjc]) if adjc and pd.notna(r[adjc]) else 0.0
            rows.append({"instrument_type": str(r["instrument_type"]).lower(), "start_date": r["start_date"],
                         "end_date": r["end_date"], "rate": float(r["normalized_quote"]), "adj_bps": adj,
                         "tenor": r["tenor"]})
        curves[(ccy, kind)] = {"label": str(lab), "rows": rows,
                               "anchor": min(pd.Timestamp(x["start_date"]) for x in rows)}
    fxr = full[is_fx]
    srow = fxr[fxr["instrument_type"].astype(str).str.lower() == "spot"]
    S0 = float(srow["normalized_quote"].iloc[0])
    sd0 = pd.Timestamp(T.get("spot_date", srow["start_date"].iloc[0] if "start_date" in srow else val0)).normalize()
    frows = fxr[fxr["instrument_type"].astype(str).str.lower() != "spot"].copy()
    frows["_e"] = pd.to_datetime(frows["end_date"])
    frows = frows.sort_values("_e")
    node_dates = list(frows["_e"])
    node_tenors = list(frows["tenor"])
    node_F = [S0 + float(v) for v in frows["normalized_quote"]]
    node_pips = [float(v) * pipm for v in frows["normalized_quote"]]

    def build(shift=None):
        out = {}
        for key, c in curves.items():
            rows = []
            for r in c["rows"]:
                rr = dict(r)
                if shift is not None:
                    rr["rate"] = r["rate"] + shift(key, r) * 1e-4
                rows.append(rr)
            out[key] = bootstrap_df_curve(c["anchor"], rows, dc.get(key[0], "ACT/360"),
                                          pay_months=(ois_pay_months or 1200) if key[1] == "ois" else 12)
        return out

    def cip(cv, S, sd, d):
        b, q = cv[(B_ccy, "ois")]["df"], cv[(Q_ccy, "ois")]["df"]
        return S * (b(d) / b(sd)) / (q(d) / q(sd))

    def tauQ(sd, d):
        return (pd.Timestamp(d) - pd.Timestamp(sd)).days / basis_days[Q_ccy]

    base_cv = build()
    basis_b = [-math.log(F / cip(base_cv, S0, sd0, d)) / tauQ(sd0, d) for F, d in zip(node_F, node_dates)]

    def market(cv, S=S0, nodes=None, basis=None):
        if nodes is None:
            bb = basis_b if basis is None else basis
            nodes = [cip(cv, S, sd0, d) * math.exp(-b * tauQ(sd0, d)) for d, b in zip(node_dates, bb)]
        return {"cv": cv, "S": S, "sd": sd0, "nodes": list(nodes), "fx": _lnF_interp(sd0, S, node_dates, nodes)}

    base_mkt = market(base_cv, nodes=node_F)

    # ---- fixings
    fx_tab.columns = [str(c).strip().lower() for c in fx_tab.columns]
    fcol = "rate_pct" if "rate_pct" in fx_tab.columns else [c for c in fx_tab.columns if "rate" in c][0]
    fscale = 0.01 if fcol.endswith("pct") or fx_tab[fcol].abs().max() > 0.5 else 1.0
    fx_tab["_d"] = pd.to_datetime(fx_tab["fixing_date"]).dt.normalize()

    def fixing(ccy, d):
        m = fx_tab["index_name"].astype(str).str.upper().str.startswith(ccy) & (fx_tab["_d"] == pd.Timestamp(d).normalize())
        if not m.any():
            return None
        return float(fx_tab.loc[m, fcol].iloc[0]) * fscale

    sch.columns = [str(c).strip().lower() for c in sch.columns]
    for c in ("accrual_start", "accrual_end", "payment_date", "reset_date", "fixing_date"):
        sch[c] = pd.to_datetime(sch[c]).dt.normalize()
    sch = sch.sort_values("accrual_start").reset_index(drop=True)
    Ncoll = float(T.get(f"{coll.lower()}_notional", T.get("notional")))
    N0 = Ncoll / float(T["inception_spot_fx"])
    rec = str(T.get("receive_leg", mtm)).upper()
    sign = {mtm: 1.0 if rec.startswith(mtm) else -1.0}
    sign[coll] = -sign[mtm]
    spread_leg = mtm
    for lab in (T.get("receive_leg", ""), T.get("pay_leg", "")):
        if "SPREAD" in str(lab).upper():
            spread_leg = str(lab).upper()[:3]
    s0 = float(T.get("contract_spread_bps", 0.0)) * 1e-4
    rsign = -1.0 if reset_cashflow == "borrower" else 1.0
    proj_key = {c: (c, "proj") for c in (mtm, coll)}

    def value(mkt, val=val0, spread=s0, past_fx=None, only_leg=None):
        cv, fxf = mkt["cv"], mkt["fx"]
        pfx = past_fx or fxf
        dcoll = cv[(coll, "ois")]["df"]
        dref = dcoll(val)
        rows, notes = [], []
        Nprev = None
        cur = {}
        for k, p in sch.iterrows():
            R = p["reset_date"]
            if k == 0:
                Nk = N0
            else:
                Nk = Ncoll / (pfx(R) if R < val else fxf(R))
            notes.append((R, Nk))
            for ccy in (mtm, coll):
                fixd = p["fixing_date"]
                h = fixing(ccy, fixd) if fixd <= val else None
                if h is not None:
                    L, rt = h, "historical_fixing"
                else:
                    q = cv[proj_key[ccy]]["df"]
                    tau_p = (p["accrual_end"] - p["accrual_start"]).days / basis_days[ccy]
                    L, rt = (q(p["accrual_start"]) / q(p["accrual_end"]) - 1.0) / tau_p, "projected"
                spr = spread if ccy == spread_leg else 0.0
                yf = (p["accrual_end"] - p["accrual_start"]).days / basis_days[ccy]
                Nleg = Nk if ccy == mtm else Ncoll
                cpn = L + spr
                if p["accrual_start"] <= val < p["accrual_end"]:
                    cur[ccy] = {"rate": cpn, "fixing": L, "notional": Nleg, "start": p["accrual_start"],
                                "payment": p["payment_date"], "rate_type": rt}
                if p["payment_date"] < val:
                    continue
                rows.append({"leg": ccy, "cashflow_type": "coupon", "reset_date": R, "fixing_date": fixd,
                             "payment_date": p["payment_date"], "accrual_start": p["accrual_start"],
                             "accrual_end": p["accrual_end"], "year_frac": yf, "notional": Nleg, "rate_type": rt,
                             "coupon_rate": cpn, "cashflow_ccy": ccy, "signed_cashflow_ccy": sign[ccy] * Nleg * cpn * yf,
                             "_order": 0 if ccy == mtm else 1, "_spread_leg": ccy == spread_leg, "_ann": sign[ccy] * Nleg * yf})
            if k > 0 and R >= val:
                rows.append({"leg": mtm, "cashflow_type": "mtm_reset", "reset_date": R, "fixing_date": p["fixing_date"],
                             "payment_date": R, "accrual_start": R, "accrual_end": R, "year_frac": 0.0, "notional": Nk,
                             "rate_type": "fx_reset", "coupon_rate": 0.0, "cashflow_ccy": mtm,
                             "signed_cashflow_ccy": rsign * sign[mtm] * (Nprev - Nk), "_order": 2,
                             "_spread_leg": False, "_ann": 0.0, "_prev": Nprev})
            Nprev = Nk
        last = sch.iloc[-1]
        if last["payment_date"] >= val:
            for j, (ccy, Nleg) in enumerate(((mtm, Nprev), (coll, Ncoll))):
                rows.append({"leg": ccy, "cashflow_type": "principal", "reset_date": last["reset_date"],
                             "fixing_date": last["fixing_date"], "payment_date": last["payment_date"],
                             "accrual_start": last["accrual_start"], "accrual_end": last["accrual_end"], "year_frac": 0.0,
                             "notional": Nleg, "rate_type": "principal", "coupon_rate": 0.0, "cashflow_ccy": ccy,
                             "signed_cashflow_ccy": sign[ccy] * Nleg, "_order": 3 + j, "_spread_leg": False, "_ann": 0.0})
        for r in rows:
            r["fx_forward"] = fxf(r["payment_date"]) if r["leg"] == mtm else 1.0
            r["discount_factor_usd"] = dcoll(r["payment_date"]) / dref
            r["pv_usd"] = r["signed_cashflow_ccy"] * r["fx_forward"] * r["discount_factor_usd"]
        if event_order == "leg":
            # know79: per payment date the MTM (reset) leg's events first - coupon, mtm_reset, principal - then
            # the constant-notional leg's coupon and principal (each leg's events stay together)
            _tr = {"coupon": 0, "mtm_reset": 1, "principal": 2}
            rows.sort(key=lambda r: (r["payment_date"], 0 if r["leg"] == mtm else 1, _tr.get(r["cashflow_type"], 3)))
        else:   # "type": coupons of both legs, then the reset, then the principals
            rows.sort(key=lambda r: (r["payment_date"], r["_order"]))
        legpv = {c: float(sum(r["pv_usd"] for r in rows if r["leg"] == c)) for c in (mtm, coll)}
        ann = float(sum(r["_ann"] * r["fx_forward"] * r["discount_factor_usd"] for r in rows if r["_spread_leg"]))
        return {"rows": rows, "leg_pv": legpv, "pv": legpv[mtm] + legpv[coll], "annuity": ann, "current": cur,
                "notionals": notes}

    base = value(base_mkt)
    pv0 = base["pv"]

    def bump_curve(key, bp, fx_follow=True):
        cv = build(lambda kk, r: bp if kk == key else 0.0)
        return market(cv) if fx_follow and key[1] == "ois" else {**market(cv, nodes=node_F)}

    risks = {}
    for key, c in curves.items():
        nm = f"pv01_{key[0].lower()}_{'ois' if key[1] == 'ois' else _label_key(c['label']).replace(key[0].lower(), '', 1)}"
        risks[nm] = value(bump_curve(key, pv01_bp))["pv"] - pv0
    m = 1.0 + fx_delta_pct / 100.0
    risks[f"fx_delta_{coll.lower()}_per_{('%g' % fx_delta_pct).replace('.', 'p')}pct"] = \
        value(market(base_cv, S=S0 * m, nodes=[F * m for F in node_F]))["pv"] - pv0

    # ---- stress scenarios
    stress = []
    for item in scenarios:
        lab, spec = (item, None) if isinstance(item, str) else item
        if spec is None:
            spec = stress_scenarios_from_labels([lab], twist_short_bp, twist_long_bp, twist_pivot_months)[0][1]
        t = spec["type"]
        if t == "ois":
            pv = value(bump_curve((str(spec["ccy"]).upper(), "ois"), float(spec["bp"])))["pv"]
        elif t in ("basis", "basis_leg"):
            mk = market(base_cv, basis=[b + float(spec["bp"]) * 1e-4 for b in basis_b])
            v = value(mk)
            pv = v["pv"] if t == "basis" else v["leg_pv"][mtm] + base["leg_pv"][coll]
        elif t == "fx":
            mm = float(spec.get("mult", 1.0 + float(spec.get("pct", 0.0)) / 100.0))
            nodes = [F * mm for F in node_F] if spec.get("nodes", True) else node_F
            pv = value(market(base_cv, S=S0 * mm, nodes=nodes))["pv"]
        elif t == "twist":
            piv = float(spec.get("pivot_months", 12))
            cv = build(lambda kk, r: float(spec["short_bp"]) if tenor_months(r["tenor"]) <= piv + 1e-9 else float(spec["long_bp"]))
            pv = value(market(cv))["pv"]
        elif t == "roll":
            rdt = pd.Timestamp(spec["date"]).normalize()
            sd_r = _add_bdays(rdt, _bdays_between(val0, sd0))
            S_r = base_mkt["fx"](sd_r)
            ndates = [sd_r + (d - sd0) for d in node_dates]
            nodes = [cip(base_cv, S_r, sd_r, nd) * math.exp(-b * tauQ(sd_r, nd)) for nd, b in zip(ndates, basis_b)]
            mk = {"cv": base_cv, "S": S_r, "sd": sd_r, "nodes": nodes, "fx": _lnF_interp(sd_r, S_r, ndates, nodes)}
            pv = value(mk, val=rdt, past_fx=base_mkt["fx"])["pv"]
        else:
            raise ValueError(f"unknown scenario type {t!r}")
        stress.append({"scenario": lab, "trade_pv_usd": float(pv), "pnl_usd": float(pv - pv0)})

    # ---- deliverables
    rows = base["rows"]
    cols = ["leg", "cashflow_type", "reset_date", "fixing_date", "payment_date", "accrual_start", "accrual_end",
            "year_frac", "notional", "rate_type", "coupon_rate", "cashflow_ccy", "signed_cashflow_ccy", "fx_forward",
            "discount_factor_usd", "pv_usd"]
    cf = pd.DataFrame([{c: (_iso(r[c]) if isinstance(r[c], pd.Timestamp) else r[c]) for c in cols} for r in rows])
    curve_files = {}
    for (ccy, kind), c in curves.items():
        if kind != "ois":
            continue
        tab = base_cv[(ccy, kind)]["table"]("OIS")
        tab["tenor"] = [str(x) for x in tab["tenor"]]
        curve_files[f"{ccy.lower()}_curve.csv"] = tab[["curve_type", "tenor", "pillar_date", "year_frac", "discount_factor",
                                                       "zero_rate", "forward_rate"]]
    fcip = [cip(base_cv, S0, sd0, d) for d in node_dates]
    bc = pd.DataFrame({"tenor": node_tenors, "maturity_date": [_iso(d) for d in node_dates],
                       "year_frac": [tauQ(sd0, d) for d in node_dates], "spot_fx": S0, "quoted_forward_fx": node_F,
                       "cip_forward_no_basis": fcip, "basis_bps": [b * 1e4 for b in basis_b]})
    ff = pd.DataFrame({"tenor": node_tenors, "maturity_date": [_iso(d) for d in node_dates], "spot_fx": S0,
                       "forward_fx": [base_mkt["fx"](d) for d in node_dates], "forward_points": node_pips,
                       "implied_basis_bps": [-math.log(base_mkt["fx"](d) / F) / tauQ(sd0, d) * 1e4
                                             for d, F in zip(node_dates, fcip)]})
    cur = base["current"]
    resets = [r for r in rows if r["cashflow_type"] == "mtm_reset"]
    par = (s0 - pv0 / base["annuity"]) * 1e4 if base["annuity"] else float("nan")
    accr = {}
    for ccy in (mtm, coll):
        c = cur.get(ccy)
        if c is None:
            accr[ccy] = 0.0
            continue
        a = sign[ccy] * c["notional"] * c["rate"] * (val0 - c["start"]).days / basis_days[ccy]
        if ccy != coll:
            a *= S0 if accrued_fx == "spot" else base_mkt["fx"](c["payment"])
        accr[ccy] = float(a)
    cl, ml = coll.lower(), mtm.lower()
    results = {f"trade_pv_{cl}": pv0, f"{cl}_leg_pv_{cl}": base["leg_pv"][coll], f"{ml}_leg_pv_{cl}": base["leg_pv"][mtm],
               "par_basis_bps": par, f"next_reset_notional_{ml}": resets[0]["notional"] if resets else None,
               f"next_coupon_rate_{cl}": cur.get(coll, {}).get("rate"), f"next_coupon_rate_{ml}": cur.get(mtm, {}).get("rate"),
               f"accrued_{cl}": accr[coll], f"accrued_{ml}_{cl}": accr[mtm]}
    order = [f"pv01_{cl}_ois", f"pv01_{ml}_ois"]
    results.update({k: risks[k] for k in order if k in risks})
    results.update({k: v for k, v in risks.items() if k not in results and k.startswith("pv01_") and k.split("_")[1] == cl})
    results.update({k: v for k, v in risks.items() if k not in results and k.startswith("pv01_")})
    results.update({k: v for k, v in risks.items() if k not in results})
    results.update({"num_clean_quotes": int(len(cleaned)), "num_future_resets": int(sum(1 for r in resets if r["reset_date"] > val0))})

    # reconciliation
    def reprices(key):
        cdf = base_cv[key]["df"]
        errs = []
        for r in curves[key]["rows"]:
            s, e = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
            tau = (e - s).days / basis_days[key[0]]
            k_in = r["rate"] + r["adj_bps"] * 1e-4
            if r["instrument_type"] == "swap":
                ds, d = [], e
                while d > s:
                    ds.append(d)
                    d = d - pd.DateOffset(months=(ois_pay_months or 1200))
                ds = sorted(ds)
                pv_ = [s] + ds[:-1]
                ann = sum((b_ - a_).days / basis_days[key[0]] * cdf(b_) for a_, b_ in zip(pv_, ds))
                k_m = (cdf(s) - cdf(e)) / ann
            else:
                k_m = (cdf(s) / cdf(e) - 1.0) / tau
            errs.append(abs(k_m - k_in))
        return bool(max(errs) < 1e-10)
    rep = {key: reprices(key) for key in curves}
    fwd_err = max(abs(base_mkt["fx"](d) - F) for d, F in zip(node_dates, node_F)) * pipm
    bas_err = float(np.max(np.abs(ff["implied_basis_bps"].to_numpy() - bc["basis_bps"].to_numpy())))
    first_bd = rows[0]["fx_forward"] if rows and rows[0]["leg"] == mtm else base_mkt["fx"](rows[0]["payment_date"])
    final_pay = max(r["payment_date"] for r in rows)
    recon = {"cleaned_quote_count": int(len(cleaned)), "max_forward_node_abs_error_pips": float(fwd_err),
             "max_basis_node_abs_error_bps": bas_err, "first_broken_date_forward": float(first_bd),
             "final_payment_forward": float(base_mkt["fx"](final_pay))}
    for ccy in (coll, mtm):
        recon[f"{ccy.lower()}_curve_reprices_inputs"] = bool(rep.get((ccy, "ois"), True))
    recon["fx_forward_nodes_reprice"] = bool(fwd_err < 1e-6)
    recon["basis_curve_matches_fx_forwards"] = bool(bas_err < 1e-9)
    recon["all_reconciliation_checks_passed"] = bool(all(v for k, v in recon.items() if isinstance(v, bool)))

    # intermediates (every pillar / FRA / node / schedule date under generic names)
    inter = {}
    for (ccy, kind), c in curves.items():
        cdf = base_cv[(ccy, kind)]["df"]
        for r in c["rows"]:
            tl = _label_key(r["tenor"])
            if kind == "ois":
                inter[f"{ccy.lower()}_df_{tl}"] = cdf(r["end_date"])
            elif r["instrument_type"] == "fra":
                inter[f"{ccy.lower()}_fra_{tl}"] = r["rate"]
                inter[f"{ccy.lower()}_fra_{tl}_effective"] = r["rate"] + r["adj_bps"] * 1e-4
    for tnr, F, b in zip(node_tenors, node_F, basis_b):
        inter[f"fx_forward_{_label_key(tnr)}"] = F
        inter[f"implied_basis_{_label_key(tnr)}_bps"] = b * 1e4
    inter[f"derived_initial_{ml}_notional"] = N0
    for d in sorted(set(sch["payment_date"]) | set(sch["reset_date"])):
        if d >= val0:
            inter[f"broken_date_fx_{d.strftime('%Y_%m_%d')}"] = base_mkt["fx"](d)
    for ccy in (coll, mtm):
        if ccy in cur:
            inter[f"current_{ccy.lower()}_coupon"] = cur[ccy]["rate"]
    if resets:
        inter[f"next_reset_notional_{ml}"] = resets[0]["notional"]
        inter[f"final_reset_notional_{ml}"] = resets[-1]["notional"]
    inter.update({f"{cl}_leg_pv_{cl}": results[f"{cl}_leg_pv_{cl}"], f"{ml}_leg_pv_{cl}": results[f"{ml}_leg_pv_{cl}"],
                  f"trade_pv_{cl}": pv0, "par_basis_bps": par})

    # summary flags
    hist_ok = all(cur[c]["rate_type"] == "historical_fixing" for c in cur) if cur else False
    seq_ok = True
    prevN = None
    for (R, Nk), k in zip(base["notionals"], range(len(base["notionals"]))):
        if k > 0 and R > val0 and abs(Nk - Ncoll / base_mkt["fx"](R)) > 1e-6 * Ncoll:
            seq_ok = False
        prevN = Nk
    for r in resets:
        if abs(r["signed_cashflow_ccy"] - rsign * sign[mtm] * (r["_prev"] - r["notional"])) > 1e-9 * Ncoll:
            seq_ok = False
    seq_ok = bool(seq_ok and len(resets) == results["num_future_resets"] and len({round(r["notional"], 6) for r in resets}) > 1)
    proj_signs = []
    for (ccy, kind), c in curves.items():
        if kind == "proj":
            nm = f"pv01_{ccy.lower()}_{_label_key(c['label']).replace(ccy.lower(), '', 1)}"
            proj_signs.append(np.sign(risks[nm]) == sign[ccy])
    summary = {"curves_reprice_inputs": bool(all(rep.values())),
               "fx_forwards_consistent": bool(recon["fx_forward_nodes_reprice"] and recon["basis_curve_matches_fx_forwards"]),
               "past_cashflows_excluded": bool(all(r["payment_date"] >= val0 for r in rows)),
               "fixed_coupon_uses_historical_fixing": bool(hist_ok),
               "mtm_reset_sequence_consistent": seq_ok,
               "risk_signs_consistent": bool(all(proj_signs) and len(proj_signs) > 0)}
    summary["all_checks_passed"] = bool(all(summary.values()) and recon["all_reconciliation_checks_passed"])

    def rnd(x):
        if decimals is None:
            return x
        if isinstance(x, pd.DataFrame):
            x = x.copy()
            for c in x.columns:
                if pd.api.types.is_float_dtype(x[c]):
                    x[c] = x[c].round(decimals)
            return x
        return x
    out = {"cleaned_quotes.csv": cleaned[[c for c in ("quote_time", *quote_key, "normalized_unit", "normalized_quote")
                                          if c in cleaned.columns]].reset_index(drop=True)}
    out.update({k: rnd(v) for k, v in curve_files.items()})
    out.update({"basis_curve.csv": rnd(bc), "fx_forward_curve.csv": rnd(ff), "cashflow_schedule.csv": rnd(cf),
                "results.json": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in results.items()},
                "stress_results.csv": pd.DataFrame(stress, columns=["scenario", "trade_pv_usd", "pnl_usd"]),
                "reconciliation.json": recon,
                "solution.json": {"intermediates": {k: {"value": float(v)} for k, v in inter.items()}},
                "summary.json": summary})
    if return_market:
        out["_market"] = {"curves": base_cv, "fx": base_mkt["fx"], "basis": dict(zip(node_tenors, basis_b)), "value": value,
                          "base": base}
    return out


def stress_scenarios_from_labels(labels, twist_short_bp=None, twist_long_bp=None, twist_pivot_months=None) -> list:
    """Scenario specs for mtm_xccy_swap_bundle from desk labels (numbers are read from the LABEL itself):
    '<ccy>_ois_up_<n>bp' / '_down_' -> ois bump; 'basis_widen_<n>bp' -> basis -n (widen = more negative),
    'basis_tighten_<n>bp' -> +n; '<leg>_leg_basis_widen_<n>bp' -> leg-only; 'spot_up_<n>pct' -> spot and
    nodes x (1 + n/100); 'spot_realized_..._up_<n>pct' -> spot only; 'roll_down_to_YYYY_MM_DD' -> roll;
    'curve_twist' needs twist_short_bp, twist_long_bp, twist_pivot_months from the task text.
    Returns [(label, spec)] in the given order."""
    out = []
    for lab in labels:
        s = str(lab).lower()
        num = re.search(r"(\d+(?:p\d+|\.\d+)?)\s*(bp|pct)", s)
        n = float(num.group(1).replace("p", ".")) if num else None
        sg = -1.0 if "_down_" in s else 1.0
        if "roll" in s:
            m = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", s)
            out.append((lab, {"type": "roll", "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}"}))
        elif "twist" in s:
            if None in (twist_short_bp, twist_long_bp, twist_pivot_months):
                raise ValueError("curve_twist: pass twist_short_bp, twist_long_bp, twist_pivot_months from the task")
            out.append((lab, {"type": "twist", "short_bp": twist_short_bp, "long_bp": twist_long_bp,
                              "pivot_months": twist_pivot_months}))
        elif "basis" in s:
            bp = -n if "widen" in s else n
            out.append((lab, {"type": "basis_leg" if "_leg_" in s else "basis", "bp": bp}))
        elif "ois" in s:
            out.append((lab, {"type": "ois", "ccy": s[:3].upper(), "bp": sg * n}))
        elif "spot" in s or "fx" in s:
            mult = 1.0 + (n if "_down_" not in s else -n) / 100.0
            out.append((lab, {"type": "fx", "mult": mult, "nodes": not ("realized" in s or "realised" in s)}))
        else:
            raise ValueError(f"cannot parse scenario label {lab!r}")
    return out


# =============================================================================================
# 2. G10 FX carry strategy, stale quarterly forward-hedge overlay, smile options, risk
# =============================================================================================

def _pair_of(ccy, pairs, usd="USD"):
    """Spot column / convention pair for a currency vs USD: 'EURUSD' or 'USDJPY' (whichever exists)."""
    for p in pairs:
        p_ = str(p).upper()
        if p_ in (ccy + usd, usd + ccy):
            return p_
    raise KeyError(f"no {ccy}/{usd} pair among {list(pairs)[:12]}")


def _usd_value(series, pair, ccy):
    """USD value of one unit of ccy from a spot series quoted as pair."""
    return series if pair.startswith(ccy) else 1.0 / series


def fx_carry_signals(deposit_rates, dates, currencies, usd="USD", rates_in_percent=True) -> pd.DataFrame:
    """Daily carry signal per foreign currency = deposit rate(ccy) - deposit rate(USD) (DECIMAL), the
    latest monthly observation dated on/before each day carried forward (as-of join, no lookahead).
    deposit_rates: DataFrame [date, USD, EUR, ...]; dates: the daily calendar (e.g. spot-rate dates).
    Returns DataFrame [date ('YYYY-MM-DD'), <ccy>...]."""
    dr = deposit_rates.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    dr = dr.sort_values("date").set_index("date")
    sc = 0.01 if rates_in_percent else 1.0
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))
    x = dr.reindex(dr.index.union(idx)).ffill().reindex(idx)
    out = pd.DataFrame({c: (x[c] - x[usd]).to_numpy() * sc for c in currencies}, index=idx)
    out.insert(0, "date", [_iso(d) for d in idx])
    return out.reset_index(drop=True)


def fx_carry_hedge_backtest(spot_rates, deposit_rates, currencies, pair_conventions, day_counts, *, top_n, bottom_n,
                            start_date, end_date, signal_lag_days, hedge_signal_months, hedge_lag_days,
                            hedge_tenor_months, holiday_calendar=None, usd="USD", include_carry=True,
                            periods_per_year=252, deposit_tenor="3M", first_hedge="pre_start",
                            window_basis="tenor", carry_accrual="per_period") -> dict:
    """Monthly long/short carry portfolio + a SEPARATE stale forward-hedge book on the long side.

    Calendar = the spot-rate dates. Signal dates = last trading day of every month; basket at a signal
    date = top_n currencies by carry (long) and bottom_n (short), carry = fx_carry_signals at that date;
    the basket is ACTIVE from signal_lag_days trading days after the signal date until the next basket
    becomes active (positions use only information on the signal date). Currency return (USD investor):
    v = USD value of 1 unit (S for CCYUSD pairs, 1/S for USDCCY), r_spot_t = v_t/v_{t-1} - 1, plus the
    carry accrual (r_ccy - r_usd)_{t-1}/periods_per_year when include_carry. long_leg = mean of long
    currency returns, short_leg = -mean of short currency returns, unhedged = long_leg + short_leg.
    Hedge rolls: signal dates whose month is in hedge_signal_months; hedge basket = that date's LONG
    basket, traded hedge_lag_days trading days later (trade_date = effective_date) and kept unchanged
    until the next roll even when the monthly basket changes (stale book). Per roll and currency:
    spot_date = trade_date + pair spot lag business days, delivery = spot + hedge_tenor_months (following
    business day), window_days = delivery - spot; spot_locked = spot on trade_date; forward_locked =
    S (1 + r_terms t_terms)/(1 + r_base t_base) (deposit rates as of trade_date, t = window_days /
    day-count denominator); hedge_quote_return = locked return of SELLING the currency forward in USD
    value terms (F/S - 1 for CCYUSD, S/F - 1 for USDCCY), pct = x100, bps = x1e4, daily_quote_return_bps =
    bps / window_days (per calendar day), cumulative_cost_pct = running sum per currency.
    Only rolls whose trade date lies in [start_date, end_date] are booked. first_hedge="start" (know79; the
    default of fx_carry_hedge_bundle): no hedge signal before start_date - the book is opened with signal_date
    = the first trading day >= start_date (its long basket by carry that day), then rolls at the configured
    month-ends after it; first_hedge="pre_start" (this function's default): a roll-month signal before the
    window whose trade falls inside it is booked. Overlay on day t while a hedge is active = mean over hedge currencies of (-r_spot_t +
    daily_quote_return x calendar days since the previous trading day / 1e4); 0 before the first roll.
    fwd_hedged = unhedged + overlay.
    know80 window_basis: "tenor" (this function's default) = the description above (window_days = delivery -
    spot calendar days, amortised per calendar day). "active" (the default of fx_carry_hedge_bundle) = the
    HEDGE WINDOW is the stretch the book is actually held: window_days = number of daily return rows from the
    roll's effective_date (inclusive) to the next roll's effective_date (exclusive; the last roll runs to
    end_date inclusive), daily_quote_return_bps = hedge_quote_return_bps / window_days, and the overlay adds
    that daily quote return once per active row (no calendar-gap factor), so the amortised quote returns of a
    roll sum exactly to its locked hedge quote return. The forward itself is still priced over spot ->
    delivery (t = (delivery - spot) / day-count denominator) under both bases.
    know81 carry_accrual (conventions table id carry_accrual): "per_period" (this function's default) =
    differential_{t-1} / periods_per_year on every row; "home_calendar" = differential_{t-1} x calendar days
    since the previous row / the home (usd) currency's money-market basis from day_counts (one excess rate of a
    home-currency investor, ISDA ACT/360 for USD); "leg_calendar" = (r_ccy/B_ccy - r_usd/B_usd)_{t-1} x calendar
    days (each leg on its own day count).
    Returns {carry_signals, portfolio_returns, roll_costs (DataFrames, ISO dates), currency_returns}."""
    if window_basis not in ("tenor", "active"):
        raise ValueError("window_basis must be 'tenor' or 'active'")
    if carry_accrual not in ("per_period", "home_calendar", "leg_calendar"):
        raise ValueError("carry_accrual must be 'per_period', 'home_calendar' or 'leg_calendar'")
    from .helpers76_4 import _unwrap, _spot, _tenor_date
    sp = spot_rates.copy()
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.sort_values("date").reset_index(drop=True)
    conv = _unwrap(pair_conventions, "pairs") if pair_conventions is not None else {}
    pairs = {c: _pair_of(c, [k for k in sp.columns if k != "date"], usd) for c in currencies}
    dates = list(sp["date"])
    sig_all = fx_carry_signals(deposit_rates, dates, currencies, usd)
    sig_all.index = pd.DatetimeIndex(dates)
    dr = deposit_rates.copy()
    dr["date"] = pd.to_datetime(dr["date"])
    dr = dr.sort_values("date").set_index("date")
    rates_daily = dr.reindex(dr.index.union(pd.DatetimeIndex(dates))).ffill().reindex(pd.DatetimeIndex(dates)) / 100.0
    V = pd.DataFrame({c: _usd_value(sp.set_index("date")[pairs[c]], pairs[c], c) for c in currencies})
    r_spot = (V / V.shift(1) - 1.0).reset_index(drop=True)
    carry_acc = ((sig_all[list(currencies)].shift(1) / periods_per_year).reset_index(drop=True) if include_carry
                 else 0.0 * r_spot)
    if include_carry and carry_accrual != "per_period":
        # know81: interest accrues over the calendar days between rows on the rate known the day before
        dcn0 = {str(k).upper(): v for k, v in (day_counts or {}).items() if not str(k).startswith("_")}
        gap = pd.Series(pd.DatetimeIndex(dates)).diff().dt.days.to_numpy(float)
        if carry_accrual == "home_calendar":
            per_day = sig_all[list(currencies)].shift(1) / _denom(dcn0.get(str(usd).upper()))
        else:
            per_day = pd.DataFrame({c: rates_daily[c].shift(1) / _denom(dcn0.get(c.upper()))
                                    - rates_daily[usd].shift(1) / _denom(dcn0.get(str(usd).upper()))
                                    for c in currencies}, index=rates_daily.index)
        carry_acc = per_day.reset_index(drop=True).mul(gap, axis=0)
    r_tot = r_spot + carry_acc
    dser = pd.Series(dates)
    month_end = dser.groupby(dser.dt.to_period("M")).max().tolist()
    pos = {d: i for i, d in enumerate(dates)}
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    baskets = []
    for s in month_end:
        i = pos[s] + int(signal_lag_days)
        if i >= len(dates):
            continue
        c = sig_all.loc[s, list(currencies)].astype(float).sort_values(ascending=False, kind="mergesort")
        baskets.append({"signal_date": s, "effective": dates[i], "long": list(c.index[:top_n]),
                        "short": list(c.index[::-1][:bottom_n])})
    hol = {}
    if holiday_calendar:
        hraw = _unwrap(holiday_calendar, "holidays")
        hol = {str(k).upper(): {str(h)[:10] for h in v} for k, v in hraw.items() if isinstance(v, (list, tuple))}
    dcn = {str(k).upper(): v for k, v in (day_counts or {}).items() if not str(k).startswith("_")}
    rolls = []
    hmonths = set(int(m) for m in hedge_signal_months)
    if first_hedge == "start":
        # know79: no signal before the strategy window - the hedge book is OPENED on its first trading day
        # (signal = that day, trade = signal + hedge lag) and then rolls on the configured month-ends inside it
        d0 = next((d for d in dates if d >= start), None)
        hsig = []
        if d0 is not None:
            c0 = sig_all.loc[d0, list(currencies)].astype(float).sort_values(ascending=False, kind="mergesort")
            hsig.append({"signal_date": d0, "long": list(c0.index[:top_n])})
        hsig += [b for b in baskets if b["signal_date"].month in hmonths and d0 is not None and b["signal_date"] > d0]
    elif first_hedge == "pre_start":
        hsig = [b for b in baskets if b["signal_date"].month in hmonths]
    else:
        raise ValueError("first_hedge must be 'start' or 'pre_start'")
    for b in hsig:
        j = pos[b["signal_date"]] + int(hedge_lag_days)
        if j >= len(dates):
            continue
        td = dates[j]
        if td < start or td > end:   # the hedge book lives inside the strategy window
            continue
        for c in b["long"]:
            p = pairs[c]
            cc = conv.get(p, {})
            base, terms = str(cc.get("base", p[:3])).upper(), str(cc.get("terms", p[3:])).upper()
            lag = int(cc.get("spot_lag_business_days", cc.get("spot_lag", 2)))
            sdt = _spot(td, lag, hol, [base, terms])
            dl = _tenor_date(sdt, f"{int(hedge_tenor_months)}M", hol, [base, terms])
            w = (dl - sdt).days
            S = float(sp.loc[pos[td], p])
            rb, rt = float(rates_daily.loc[td, base]), float(rates_daily.loc[td, terms])
            tb, tt = w / _denom(dcn.get(base)), w / _denom(dcn.get(terms))
            F = S * (1 + rt * tt) / (1 + rb * tb)
            q = F / S - 1.0 if base == c else S / F - 1.0
            rolls.append({"signal_date": b["signal_date"], "effective_date": td, "pair": p, "currency": c,
                          "trade_date": td, "spot_date": sdt, "delivery_date": dl, "window_days": int(w),
                          "spot_locked": S, "forward_locked": F, "hedge_quote_return_pct": q * 100.0,
                          "hedge_quote_return_bps": q * 1e4, "daily_quote_return_bps": q * 1e4 / w})
    if window_basis == "active" and rolls:
        # know80: hedge window = daily rows the book is held (effective date .. next roll's effective date)
        effs = sorted({x["effective_date"] for x in rolls})
        inwin = [d for d in dates if start <= d <= end]
        for x in rolls:
            i = effs.index(x["effective_date"])
            nxt = effs[i + 1] if i + 1 < len(effs) else None
            n_act = sum(1 for d in inwin if d >= x["effective_date"] and (nxt is None or d < nxt))
            x["window_days"] = int(n_act)
            x["daily_quote_return_bps"] = x["hedge_quote_return_bps"] / n_act if n_act else float("nan")
    rc = pd.DataFrame(rolls)
    if len(rc):
        rc["cumulative_cost_pct"] = rc.groupby("currency")["hedge_quote_return_pct"].cumsum()
    roll_dates = sorted(set(rc["effective_date"])) if len(rc) else []
    out_rows = []
    bi, hi = -1, -1
    prev_d = None
    for k, d in enumerate(dates):
        while bi + 1 < len(baskets) and baskets[bi + 1]["effective"] <= d:
            bi += 1
        while hi + 1 < len(roll_dates) and roll_dates[hi + 1] <= d:
            hi += 1
        if d < start or d > end or bi < 0 or k == 0:
            prev_d = d
            continue
        b = baskets[bi]
        ll = float(r_tot.loc[k, b["long"]].mean())
        sl = -float(r_tot.loc[k, b["short"]].mean())
        if hi >= 0:
            h = rc[rc["effective_date"] == roll_dates[hi]]
            gap = (d - prev_d).days if window_basis == "tenor" else 1
            ov =float(np.mean([-float(r_spot.loc[k, c]) + float(dq) * gap / 1e4
                                for c, dq in zip(h["currency"], h["daily_quote_return_bps"])]))
            hs, hc = _iso(h["signal_date"].iloc[0]), ",".join(h["currency"])
        else:
            ov, hs, hc = 0.0, None, None
        out_rows.append({"date": _iso(d), "signal_date": _iso(b["signal_date"]), "active_hedge_signal_date": hs,
                         "long_currencies": ",".join(b["long"]), "short_currencies": ",".join(b["short"]),
                         "active_hedge_currencies": hc, "unhedged_return": ll + sl, "hedge_overlay_return": ov,
                         "fwd_hedged_return": ll + sl + ov, "long_leg": ll, "short_leg": sl})
        prev_d = d
    pr = pd.DataFrame(out_rows)
    if len(rc):
        for c in ("signal_date", "effective_date", "trade_date", "spot_date", "delivery_date"):
            rc[c] = [_iso(x) for x in rc[c]]
    cs = sig_all.reset_index(drop=True)
    cs = cs[(pd.to_datetime(cs["date"]) >= start) & (pd.to_datetime(cs["date"]) <= end)].reset_index(drop=True)
    cols = ["signal_date", "effective_date", "pair", "currency", "trade_date", "spot_date", "delivery_date", "window_days",
            "spot_locked", "forward_locked", "hedge_quote_return_pct", "hedge_quote_return_bps", "daily_quote_return_bps",
            "cumulative_cost_pct"]
    return {"carry_signals": cs, "portfolio_returns": pr, "roll_costs": rc[cols] if len(rc) else pd.DataFrame(columns=cols),
            "currency_returns": r_tot}


def fx_smile_option_book(options, days_basis=365.0, hedge_amount="delta") -> dict:
    """25-delta smile -> strike -> Garman-Kohlhagen price, premium conversions and Greeks per quote.

    options: list of dicts (quote_id, pair, quote_type '25d_call'/'25d_put', delta_convention
    'spot_unadjusted' | 'forward_unadjusted' | 'spot_premium_adjusted' | 'forward_premium_adjusted',
    premium_currency, spot, trade_date, expiry_date, domestic_rate_cc, foreign_rate_cc (continuous),
    atm_vol, rr_25d, bf_25d, target_abs_delta, notional_base). Wings: call = ATM + BF + RR/2, put =
    ATM + BF - RR/2; selected vol by quote_type. T = days(trade, expiry)/days_basis. Strike:
    fx_strike_from_delta(S, T, rd, rf, vol, target, kind, convention). F = S e^{(rd - rf)T}; prices
    terms ccy per 1 base; terms_pips = selected price, base_pips = price / S, terms_pct = 100 price/K,
    base_pct = 100 price/S; pcp_error = C - P - (S e^{-rf T} - K e^{-rd T}). Greeks (spot, GK):
    d1 = (ln(S/K) + (rd - rf + v^2/2)T)/(v sqrt T); delta_pips = e^{-rf T} N(d1) (call) / -e^{-rf T}
    N(-d1) (put); delta_premium_included = delta_pips - price/S; gamma = e^{-rf T} phi(d1)/(S v sqrt T);
    vega = S e^{-rf T} phi(d1) sqrt T (per 1.00 vol), vega_1pct = vega/100; spot hedge amounts =
    delta x notional_base (pips delta and premium-included delta).
    know80 hedge_amount: "delta" (this function's default) = +delta x notional (the option's base-ccy
    exposure); "offset" (the default of fx_carry_hedge_bundle) = the SPOT HEDGE TRADE that neutralises a long
    option: -delta x notional_base (sell base against a long call, buy base against a long put), for both the
    pips delta and the premium-included delta.
    Returns {smile, prices, greeks} lists of dicts in input order."""
    if hedge_amount not in ("delta", "offset"):
        raise ValueError("hedge_amount must be 'delta' or 'offset'")
    hsgn = 1.0 if hedge_amount == "delta" else -1.0
    from .helpers75_4 import fx_smile_25d, fx_strike_from_delta
    from .options import garman_kohlhagen_price
    sm, pr, gk = [], [], []
    for o in options:
        S, rd, rf = float(o["spot"]), float(o["domestic_rate_cc"]), float(o["foreign_rate_cc"])
        days = (pd.Timestamp(o["expiry_date"]) - pd.Timestamp(o["trade_date"])).days
        T = days / days_basis
        vc, vp = fx_smile_25d(float(o["atm_vol"]), float(o["rr_25d"]), float(o["bf_25d"]))
        kind = "call" if "call" in str(o["quote_type"]).lower() else "put"
        vol = vc if kind == "call" else vp
        tgt = float(o["target_abs_delta"])
        K = fx_strike_from_delta(S, T, rd, rf, vol, tgt, kind, o["delta_convention"])
        C = float(garman_kohlhagen_price(S, K, T, rd, rf, vol, "call"))
        P = float(garman_kohlhagen_price(S, K, T, rd, rf, vol, "put"))
        px = C if kind == "call" else P
        F = S * math.exp((rd - rf) * T)
        head = {k: o.get(k) for k in ("quote_id", "pair", "quote_type", "delta_convention", "premium_currency")}
        sm.append({**head, "atm_vol": float(o["atm_vol"]), "rr_25d": float(o["rr_25d"]), "bf_25d": float(o["bf_25d"]),
                   "vol_25d_call": vc, "vol_25d_put": vp, "selected_vol": vol,
                   "target_delta": tgt if kind == "call" else -tgt, "derived_strike": K})
        pr.append({**head, "spot": S, "strike": K, "forward": F, "T_years": T, "days_to_expiry": int(days),
                   "selected_vol": vol, "call_price": C, "put_price": P, "selected_option_price": px,
                   "terms_pips": px, "base_pips": px / S, "terms_pct": 100.0 * px / K, "base_pct": 100.0 * px / S,
                   "pcp_error": C - P - (S * math.exp(-rf * T) - K * math.exp(-rd * T))})
        v = vol * math.sqrt(T)
        d1 = (math.log(S / K) + (rd - rf + 0.5 * vol * vol) * T) / v
        d2 = d1 - v
        dpips = math.exp(-rf * T) * (stats.norm.cdf(d1) if kind == "call" else -stats.norm.cdf(-d1))
        dpi = dpips - px / S
        gam = math.exp(-rf * T) * stats.norm.pdf(d1) / (S * v)
        veg = S * math.exp(-rf * T) * stats.norm.pdf(d1) * math.sqrt(T)
        N = float(o.get("notional_base", 1.0))
        gk.append({"quote_id": o.get("quote_id"), "pair": o.get("pair"), "quote_type": o.get("quote_type"),
                   "delta_pips": dpips, "delta_premium_included": dpi, "gamma": gam, "vega": veg, "vega_1pct": veg / 100.0,
                   "d1": d1, "d2": d2, "spot_hedge_amount_pips_delta": hsgn * dpips * N,
                   "spot_hedge_amount_pi_delta": hsgn * dpi * N})
    return {"smile": sm, "prices": pr, "greeks": gk}


def return_risk_metrics(returns, *, periods_per_year, var_level, ddof=1) -> dict:
    """Daily DECIMAL return series -> {ann_return = mean A, ann_vol = std(ddof) sqrt A, sharpe = ratio,
    max_drawdown = min(W/cummax W - 1) on W = cumprod(1 + r) (<= 0), var_<q>pct = q-quantile of returns
    (historical, a RETURN, negative for a loss), cvar_<q>pct = mean of returns <= that quantile,
    n_obs = non-missing count}; q = 100 (1 - var_level)."""
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy(float)
    A = float(periods_per_year)
    lv = float(var_level) if var_level < 1 else float(var_level) / 100.0
    q = 1.0 - lv
    tag = ("%g" % round(100 * q, 6)).replace(".", "p")
    mu, sd = r.mean() * A, r.std(ddof=ddof) * math.sqrt(A)
    W = np.cumprod(1.0 + r)
    var = float(np.quantile(r, q))
    return {"ann_return": float(mu), "ann_vol": float(sd), "sharpe": float(mu / sd) if sd > 0 else float("nan"),
            "max_drawdown": float(np.min(W / np.maximum.accumulate(W) - 1.0)), f"var_{tag}pct": var,
            f"cvar_{tag}pct": float(r[r <= var].mean()), "n_obs": int(len(r))}


def fx_carry_hedge_bundle(spot_rates, deposit_rates, spx, spot_bidoffer, deposit_bidoffer, pair_conventions,
                        holiday_calendar, market_forwards, ndf_quotes, option_params, day_counts, params, *,
                        crosses, spot_date_pairs=None, usd="USD", include_carry=True, first_hedge="start",
                        swap_points_units="pips", window_basis="active", hedge_amount="offset",
                        carry_accrual=None) -> dict:
    """ONE call -> the 13 deliverables of a dealer-style G10 FX carry / hedge / option / risk study.

    Inputs are the parsed task files (DataFrames / dicts). Snapshot: fx_snapshot_desk (crosses through
    USD, joint-calendar spot dates, CIP forwards, implied yields, swap rolls with swap_points in quoted points
    (swap_points_units="pips", know79), NDF). Strategy (first_hedge="start", know79: the hedge book opens on
    the first strategy day, no signal before start_date):
    fx_carry_hedge_backtest with params['carry_trade'] (top_n, bottom_n, start_date, end_date,
    signal_lag_business_days) and params['forward_hedge'] (roll_tenor_months, roll_signal_months,
    signal_lag_business_days); foreign currencies = the deposit-rate columns other than USD. Options:
    fx_smile_option_book(option_params['options']). Risk: return_risk_metrics(ddof=1) with
    params['risk_metrics'] (annualization_factor, var_confidence) on unhedged, fwd_hedged and the
    simplified option hedge = unhedged - daily cost, daily cost = mean premium fraction (base_pct/100)
    / (mean T_years x annualization) (n_obs = the unhedged observation set). Equity correlation:
    rolling_window_days correlation of unhedged returns with SPX close-to-close returns on common
    dates (rows where it is defined). know80 defaults: window_basis="active" (roll window_days = daily rows
    the hedge book is held, effective date to the next roll; the quote return is amortised per held row) and
    hedge_amount="offset" (spot hedge amounts = the offsetting spot trade, -delta x notional).
    know81 carry_accrual=None: the conventions table decides from the task's words
    (helpers81_1.carry_accrual_from_text; e.g. 'for a US-based asset manager' -> "home_calendar"); silent text
    -> "per_period". Pass a value to override.
    Returns {file name: object}."""
    from .helpers76_4 import fx_snapshot_desk, _unwrap
    if carry_accrual is None:
        from .helpers81_1 import carry_accrual_from_text
        carry_accrual = carry_accrual_from_text()[0]
    conv = _unwrap(pair_conventions, "pairs")
    sdp = list(spot_date_pairs) if spot_date_pairs is not None else list(conv)
    snap = fx_snapshot_desk(spot_bidoffer, deposit_bidoffer, pair_conventions, holiday_calendar, market_forwards,
                            day_counts, ndf_quotes, crosses=tuple(crosses), spot_date_pairs=tuple(sdp),
                            swap_points_units=swap_points_units)
    ct, fh, rm = params.get("carry_trade", {}), params.get("forward_hedge", {}), params.get("risk_metrics", {})
    ccys = [c for c in deposit_rates.columns if c not in ("date", usd)]
    bt = fx_carry_hedge_backtest(spot_rates, deposit_rates, ccys, pair_conventions, day_counts,
                                 top_n=int(ct["top_n"]), bottom_n=int(ct["bottom_n"]), start_date=ct["start_date"],
                                 end_date=ct["end_date"], signal_lag_days=int(ct["signal_lag_business_days"]),
                                 hedge_signal_months=fh["roll_signal_months"],
                                 hedge_lag_days=int(fh["signal_lag_business_days"]),
                                 hedge_tenor_months=int(fh["roll_tenor_months"]), holiday_calendar=holiday_calendar,
                                 usd=usd, include_carry=include_carry,
                                 periods_per_year=int(rm["annualization_factor"]), first_hedge=first_hedge,
                                 window_basis=window_basis, carry_accrual=carry_accrual)
    ob = fx_smile_option_book(option_params["options"] if isinstance(option_params, dict) else option_params,
                              hedge_amount=hedge_amount)
    A = int(rm["annualization_factor"])
    lvl = float(rm["var_confidence"])
    pr = bt["portfolio_returns"]
    cost = float(np.mean([p["base_pct"] / 100.0 for p in ob["prices"]]))
    life = float(np.mean([p["T_years"] for p in ob["prices"]]))
    opt_ret = pr["unhedged_return"] - cost / (life * A)
    risk = {"unhedged": return_risk_metrics(pr["unhedged_return"], periods_per_year=A, var_level=lvl),
            "fwd_hedged": return_risk_metrics(pr["fwd_hedged_return"], periods_per_year=A, var_level=lvl),
            "option_hedged": return_risk_metrics(opt_ret, periods_per_year=A, var_level=lvl)}
    w = int(rm["rolling_window_days"])
    eq = spx.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").set_index("date")
    eqr = eq[[c for c in eq.columns if c != "date"][0]].pct_change()
    u = pr.set_index(pd.to_datetime(pr["date"]))["unhedged_return"]
    j = pd.concat([u, eqr], axis=1, join="inner").dropna()
    rc = j.iloc[:, 0].rolling(w).corr(j.iloc[:, 1]).dropna()
    eqc = pd.DataFrame({"date": [_iso(d) for d in rc.index], f"rolling_corr_{w}d": rc.to_numpy()})
    return {"cross_rates.json": snap["cross_rates"], "date_diagnostics.json": snap["date_diagnostics"],
            "forward_rates.json": snap["forward_rates"], "implied_yields.json": snap["implied_yields"],
            "ndf_analysis.json": snap["ndf_analysis"], "carry_signals.csv": bt["carry_signals"],
            "roll_costs.csv": bt["roll_costs"], "portfolio_returns.csv": pr,
            "smile_reconstruction.json": ob["smile"], "option_prices.json": ob["prices"], "greeks.json": ob["greeks"],
            "risk_metrics.json": risk, "equity_correlation.csv": eqc}
