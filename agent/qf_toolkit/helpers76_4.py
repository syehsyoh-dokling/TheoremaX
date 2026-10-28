"""know76 group-4 helpers: whole-step functions for the decisions that swung or crashed in the
know75 full run - FX desk snapshot (crosses, settlement dates, CIP forwards, implied yields, NDF),
an FX forward book (CIP curve, synthetic crosses, MTM, attribution, net delta), tolerant curve
selection and log-linear FX forward nodes for xccy books, SVI -> Dupire surfaces, a BTC
(coin-premium) option surface pipeline, two-factor Heston tables, a Bollinger backtest with
risk sizing and a trailing stop, pairwise copula selection, return-panel cleaning against a
trading calendar, a CreditMetrics default simulation, and a CEREP migration study.

Every function is general (no task names), pure numpy/pandas/scipy, and states its formula and
conventions in the docstring. Numbers a task fixes (windows, fees, seeds, horizons, levels) are
REQUIRED arguments - pass the task's own values. Self-tests: proven_work/fail72/selftest76_4.py.
"""
from __future__ import annotations

import ast
import itertools
import math
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats


# =============================================================================================
# helpers shared by the FX functions
# =============================================================================================

def _basis_days(dc) -> float:
    """'ACT/360' -> 360, 'ACT/365', 'ACT/365F', 'A365' -> 365, a number -> that number."""
    if isinstance(dc, (int, float)):
        return float(dc)
    m = re.search(r"(360|365|366)", str(dc))
    if not m:
        raise ValueError(f"day count {dc!r} has no 360/365 denominator")
    return float(m.group(1))


def _pair_ccys(p):
    s = str(p).upper().replace("-", "/").replace(" ", "").replace("_", "")
    if "/" in s:
        a, b = s.split("/")[:2]
    else:
        a, b = s[:3], s[3:6]
    return a, b


def _two_way(v):
    """(bid, offer) from [bid, offer], (bid, offer, mid), {'bid','ask'|'offer'} or a number."""
    if isinstance(v, dict):
        lo = {str(k).lower(): x for k, x in v.items()}
        b = lo.get("bid")
        a = lo.get("offer", lo.get("ask"))
        if b is None or a is None:
            m = lo.get("mid")
            if m is None:
                raise ValueError(f"quote {v!r} has no bid/offer(ask)/mid")
            return float(m), float(m)
        return float(b), float(a)
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        vals = [float(x) for x in list(v)]
        if len(vals) >= 2:
            return vals[0], vals[1]
        return vals[0], vals[0]
    return float(v), float(v)


def _quote_book(obj) -> dict:
    """{PAIR: (bid, offer)} from a dict that may also hold '_comment'/'snapshot_date' strings,
    a DataFrame with pair/bid/ask(offer) columns, or a list of records."""
    book = {}
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower(): c for c in obj.columns}
        pc = cols.get("pair") or cols.get("ccy_pair") or cols.get("symbol")
        bc = cols.get("bid")
        ac = cols.get("offer") or cols.get("ask")
        for _, r in obj.iterrows():
            book[str(r[pc]).upper().replace("/", "")] = (float(r[bc]), float(r[ac]))
        return book
    if isinstance(obj, list):
        for r in obj:
            lo = {str(k).lower(): x for k, x in r.items()}
            book[str(lo["pair"]).upper().replace("/", "")] = _two_way(lo)
        return book
    for k, v in obj.items():
        if str(k).startswith("_") or isinstance(v, str):
            continue
        if re.fullmatch(r"[A-Za-z]{3}/?[A-Za-z]{3}", str(k)):
            book[str(k).upper().replace("/", "")] = _two_way(v)
    return book


def _unwrap(d, key):
    """d[key] when present (nested config such as pair_conventions['pairs']), else d itself."""
    if isinstance(d, dict) and key in d and isinstance(d[key], (dict, list)):
        return d[key]
    return d


def _bday(d, hol, ccys):
    if d.weekday() >= 5:
        return False
    iso = d.strftime("%Y-%m-%d")
    return not any(iso in hol.get(c, set()) for c in ccys)


def _spot(trade, lag, hol, ccys):
    d = pd.Timestamp(trade).normalize()
    k = 0
    while k < int(lag):
        d += pd.Timedelta(days=1)
        if _bday(d, hol, ccys):
            k += 1
    return d


def _tenor_date(spot, tenor, hol, ccys):
    m = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", str(tenor).upper())
    if not m:
        raise ValueError(f"tenor {tenor!r} not like 6W / 3M / 1Y")
    n, u = int(m.group(1)), m.group(2)
    d = pd.Timestamp(spot).normalize()
    if u == "D":
        d += pd.Timedelta(days=n)
    elif u == "W":
        d += pd.Timedelta(days=7 * n)
    else:
        d += pd.DateOffset(months=n if u == "M" else 12 * n)
    while not _bday(d, hol, ccys):
        d += pd.Timedelta(days=1)
    return d


def _iso(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


# =============================================================================================
# 1. FX desk snapshot: crosses, settlement dates, CIP forwards, implied yields, NDF
# =============================================================================================

def fx_snapshot_desk(spot_bidoffer, deposit_bidoffer, pair_conventions, holiday_calendar,
                     market_forwards=None, day_counts=None, ndf_quotes=None, crosses=(),
                     spot_date_pairs=(), trade_date=None, swap_points_units="quote") -> dict:
    """Dealer snapshot from the RAW parsed JSON inputs (top-level '_comment' / 'snapshot_date'
    strings are skipped; nested blocks 'pairs', 'holidays', 'rates', 'deliverable_forwards',
    'broken_date_forwards', 'roll_cases', 'cases' are found by name).

    Calendar: a business day is a weekday that is not a holiday of EITHER pair currency; spot =
    trade date + spot_lag business days counted on that joint calendar; tenor dates add calendar
    weeks/months to SPOT then roll FOLLOWING; days_from_spot = delivery - spot.
    Day counts: t_c = days / denominator(day_counts[c]) (ACT/360 -> 360, ACT/365 -> 365).
    Crosses X/Y through USD (dealer rule): bid = XUSD_bid * USDY_bid, offer = XUSD_offer *
    USDY_offer, an inverted leg 1/(USD/X) swaps sides (bid = 1/offer); mid = product of LEG MIDS.
    CIP mid forward: F = S_mid (1 + r_terms t_terms)/(1 + r_base t_base), rates = mid of the
    [bid, offer] deposit quotes in percent / 100; points = F - S, points_pips = points x
    point_multiplier. Broken dates: deposit rate linear in days between tenor dates, flat outside;
    the case's own delivery_date and spot_mid are used when given.
    Implied foreign yields (spot MID, USD deposits at the tenor): USDXXX bid uses forward_bid with
    USD offer, offer uses forward_offer with USD bid; XXXUSD bid uses forward_offer with USD bid,
    offer uses forward_bid with USD offer; reported in percent.
    FX swap rolls: one side on both legs, chosen by the far leg (short_base_forward -> bid,
    long_base_forward -> offer); outright = spot_side + points_side / multiplier; swap_points =
    far - near outright (swap_points_units="quote", default) or far_points - near_points on that side in
    the case's own POINT units (swap_points_units="pips", know79: the market quotes swap points in pips;
    fx_carry_hedge_bundle passes it).
    NDF (case year fractions, spot mid): cip_bid = S (1 + r_loc_bid t_loc)/(1 + r_usd_offer t_usd),
    cip_offer = S (1 + r_loc_offer t_loc)/(1 + r_usd_bid t_usd); implied local rate from the MARKET
    forward with the USDXXX pairing; basis_pips = (market - cip) x point_multiplier per side.
    Returns {cross_rates, date_diagnostics, forward_rates, implied_yields, ndf_analysis} (plain
    dicts/lists, dates 'YYYY-MM-DD', ready for json)."""
    quotes = _quote_book(spot_bidoffer)
    conv = _unwrap(pair_conventions, "pairs")
    hol_raw = _unwrap(holiday_calendar or {}, "holidays")
    hol = {str(c).upper(): {str(h)[:10] for h in v} for c, v in hol_raw.items()
           if not str(c).startswith("_") and isinstance(v, (list, tuple))}
    dep_rates = _unwrap(deposit_bidoffer, "rates")
    tenor_months = deposit_bidoffer.get("tenor_months") if isinstance(deposit_bidoffer, dict) else None
    dc = {str(k).upper(): v for k, v in (day_counts or {}).items() if not str(k).startswith("_")}
    mf = market_forwards or {}
    if trade_date is None:
        for src in (spot_bidoffer, mf, deposit_bidoffer):
            if isinstance(src, dict) and src.get("snapshot_date"):
                trade_date = src["snapshot_date"]
                break
    trade = pd.Timestamp(trade_date).normalize()

    def pconv(p):
        c = conv[p]
        base = str(c.get("base", p[:3])).upper()
        terms = str(c.get("terms", c.get("quote", p[3:6]))).upper()
        lag = int(c.get("spot_lag_business_days", c.get("spot_lag", 2)))
        mult = float(c.get("point_multiplier", c.get("pip_multiplier", 10000)))
        return base, terms, lag, mult

    def tfrac(days, ccy):
        return days / _basis_days(dc.get(ccy, "ACT/360"))

    def mid(p):
        b, a = quotes[p]
        return 0.5 * (b + a)

    tenors = list(tenor_months) if tenor_months else [t for t in next(iter(dep_rates.values())) if not str(t).startswith("_")]

    def dep(ccy, tenor):
        b, a = _two_way(dep_rates[ccy][tenor])
        return b / 100.0, a / 100.0

    def dep_mid_at(ccy, days, spot, ccys):
        xs, ys = [], []
        for t in tenors:
            d = (_tenor_date(spot, t, hol, ccys) - spot).days
            b, a = dep(ccy, t)
            xs.append(d)
            ys.append(0.5 * (b + a))
        o = np.argsort(xs)
        return float(np.interp(days, np.asarray(xs)[o], np.asarray(ys)[o]))

    # --- crosses
    cross_out = {}
    for x in crosses:
        b, a, m = _cross(x, quotes)
        cross_out[str(x).upper().replace("/", "")] = {"bid": b, "offer": a, "mid": m}

    # --- spot dates
    spot_dates = {}
    for p in spot_date_pairs:
        base, terms, lag, _ = pconv(p)
        spot_dates[p] = {"trade_date": _iso(trade), "spot_lag_business_days": lag,
                         "spot_date": _iso(_spot(trade, lag, hol, [base, terms]))}

    # --- broken dates
    broken = {}
    for cid, case in (mf.get("broken_date_forwards") or {}).items():
        if not isinstance(case, dict):
            continue
        p = str(case.get("pair", cid.split("_")[0])).upper()
        base, terms, lag, _ = pconv(p)
        sp = _spot(trade, lag, hol, [base, terms])
        if case.get("delivery_date"):
            dl = pd.Timestamp(case["delivery_date"]).normalize()
        else:
            dl = _tenor_date(sp, cid.split("_")[-1], hol, [base, terms])
        days = (dl - sp).days
        tb, tt = tfrac(days, base), tfrac(days, terms)
        S = float(case.get("spot_mid", mid(p)))
        rb = dep_mid_at(base, days, sp, [base, terms])
        rt = dep_mid_at(terms, days, sp, [base, terms])
        broken[cid] = {"spot_date": _iso(sp), "delivery_date": _iso(dl), "days_from_spot": int(days),
                       "t_base": tb, "t_terms": tt,
                       "cip_forward_mid": S * (1 + rt * tt) / (1 + rb * tb)}

    # --- roll cases
    rolls = []
    for case in (mf.get("roll_cases") or []):
        pos = str(case.get("position", "")).lower()
        side = "bid" if ("short" in pos or "sell" in pos) else "offer"
        mult = float(case.get("multiplier", case.get("point_multiplier", 10000)))
        s = float(case[f"spot_{side}"])
        near = s + float(case[f"near_points_{side}"]) / mult
        far = s + float(case[f"far_points_{side}"]) / mult
        # know79: swap_points_units="pips" reports the swap points in the case's own POINT units (far_points -
        # near_points on the chosen side), the market convention for an FX swap; "quote" = far - near outright
        sp_pts = (far - near) if swap_points_units == "quote" else (
            float(case[f"far_points_{side}"]) - float(case[f"near_points_{side}"]))
        rolls.append({"case_id": case.get("case_id"), "pair": case.get("pair"), "side_used": side,
                      "near_outright": near, "far_outright": far, "swap_points": sp_pts})

    # --- forward rates for every pair in the conventions
    fwd = {}
    for p in conv:
        if p not in quotes:
            continue
        base, terms, lag, mult = pconv(p)
        sp = _spot(trade, lag, hol, [base, terms])
        S = mid(p)
        fwd[p] = {}
        for t in tenors:
            dl = _tenor_date(sp, t, hol, [base, terms])
            days = (dl - sp).days
            tb, tt = tfrac(days, base), tfrac(days, terms)
            rb = 0.5 * sum(dep(base, t))
            rt = 0.5 * sum(dep(terms, t))
            F = S * (1 + rt * tt) / (1 + rb * tb)
            fwd[p][t] = {"trade_date": _iso(trade), "spot_date": _iso(sp), "delivery_date": _iso(dl),
                         "forward_mid": F, "points": F - S, "points_pips": (F - S) * mult,
                         "days_from_spot": int(days), "t_base": tb, "t_terms": tt, "spot_mid": S,
                         "r_base_mid_pct": rb * 100.0, "r_terms_mid_pct": rt * 100.0}

    # --- implied yields from deliverable market forwards
    iy = {}
    for p, block in (mf.get("deliverable_forwards") or {}).items():
        if not isinstance(block, dict):
            continue
        base, terms, lag, _ = pconv(p)
        usd_is_base = base == "USD"
        foreign = terms if usd_is_base else base
        sp = _spot(trade, lag, hol, [base, terms])
        S = float(block.get("spot_mid", mid(p) if p in quotes else np.nan))
        iy[p] = {}
        for t, q in block.items():
            if not isinstance(q, dict):
                continue
            dl = pd.Timestamp(q["delivery_date"]).normalize() if q.get("delivery_date") else _tenor_date(sp, t, hol, [base, terms])
            days = (dl - sp).days
            t_usd, t_f = tfrac(days, "USD"), tfrac(days, foreign)
            ub, uo = dep("USD", t)
            fb, fo = float(q["forward_bid"]), float(q["forward_offer"])
            if usd_is_base:
                rb_ = ((fb / S) * (1 + uo * t_usd) - 1) / t_f
                ro_ = ((fo / S) * (1 + ub * t_usd) - 1) / t_f
            else:
                rb_ = ((S / fo) * (1 + ub * t_usd) - 1) / t_f
                ro_ = ((S / fb) * (1 + uo * t_usd) - 1) / t_f
            iy[p][t] = {"spot_date": _iso(sp), "delivery_date": _iso(dl),
                        "implied_foreign_rate_bid_pct": rb_ * 100.0,
                        "implied_foreign_rate_offer_pct": ro_ * 100.0, "foreign_currency": foreign}

    # --- NDF
    ndf = {}
    cases = (ndf_quotes or {}).get("cases", []) if isinstance(ndf_quotes, dict) else (ndf_quotes or [])
    for c in cases:
        p = str(c["pair"]).upper()
        S = float(c["spot_mid"])
        tu, tl = float(c["t_usd"]), float(c["t_local"])
        ub, uo = c["usd_rate_bid_pct"] / 100.0, c["usd_rate_offer_pct"] / 100.0
        lb, lo_ = c["local_rate_bid_pct"] / 100.0, c["local_rate_offer_pct"] / 100.0
        mult = float(c.get("point_multiplier", 10000))
        cb = S * (1 + lb * tl) / (1 + uo * tu)
        co = S * (1 + lo_ * tl) / (1 + ub * tu)
        mb, mo = float(c["market_forward_bid"]), float(c["market_forward_offer"])
        ilb = ((mb / S) * (1 + uo * tu) - 1) / tl
        ilo = ((mo / S) * (1 + ub * tu) - 1) / tl
        ndf[p] = {"trade_date": c.get("trade_date"), "fixing_date": c.get("fixing_date"),
                  "settlement_date": c.get("settlement_date"), "local_currency": c.get("local_currency"),
                  "cip_forward_bid": cb, "cip_forward_offer": co,
                  "market_forward_bid": mb, "market_forward_offer": mo,
                  "implied_local_rate_bid_pct": ilb * 100.0, "implied_local_rate_offer_pct": ilo * 100.0,
                  "market_basis_bid_pips": (mb - cb) * mult, "market_basis_offer_pips": (mo - co) * mult}

    return {"cross_rates": cross_out,
            "date_diagnostics": {"spot_dates": spot_dates, "broken_dates": broken, "roll_cases": rolls},
            "forward_rates": fwd, "implied_yields": iy, "ndf_analysis": ndf}


def _cross(target, book):
    """(bid, offer, mid) of X/Y through USD from {PAIR: (bid, offer)} (PAIR 'EURUSD' or 'EUR/USD')."""
    bk = {}
    for k, v in book.items():
        bk[_pair_ccys(k)] = _two_way(v)
    x, y = _pair_ccys(target)
    if (x, y) in bk:
        b, a = bk[(x, y)]
        return b, a, 0.5 * (b + a)

    def leg(base, quote):
        if base == quote:
            return 1.0, 1.0, 1.0
        if (base, quote) in bk:
            b, a = bk[(base, quote)]
            return b, a, 0.5 * (b + a)
        if (quote, base) in bk:
            b, a = bk[(quote, base)]
            return 1.0 / a, 1.0 / b, 1.0 / (0.5 * (b + a))
        raise KeyError(f"no quote for {base}/{quote} or {quote}/{base}; have {sorted(bk)}")

    b1, a1, m1 = leg(x, "USD")
    b2, a2, m2 = leg("USD", y)
    return b1 * b2, a1 * a2, m1 * m2


# =============================================================================================
# 2. FX forward book: CIP curve, crosses, MTM, attribution, net delta
# =============================================================================================

def fx_forward_book(spot, deposits, portfolio, historical_spots, tenor_days, crosses,
                    pip_multipliers=None, round_forward=6, round_points=2, round_cross=4,
                    round_money=2, day_counts=None, rates_in_percent=None) -> dict:
    """Valuation pipeline for a book of outright FX forwards (results.json-shaped dict).

    spot: DataFrame[pair 'BASE/QUOTE', bid, ask (, pip_multiplier, base_ccy, quote_ccy)]; rows with
    bid >= ask are STALE and dropped first. deposits: DataFrame[currency, tenor_1m/tenor_3m/...
    (PERCENT), day_count]. tenor_days {'1M': days, ...}. crosses: list like ['GBP/JPY', ...].
    Year fraction per currency t_c = days / denominator(day_count_c).
    Forward (CIP, simple): F_side = S_side (1 + r_q t_q)/(1 + r_b t_b) for side bid/ask/mid
    (S_mid = (bid+ask)/2; one deposit rate per tenor), rounded to `round_forward`; points =
    (F - S) x pip_multiplier per side, rounded to `round_points`.
    Crosses through the USD legs (dealer rule, inverted legs swap sides, mid = product of leg
    mids), rounded to `round_cross`; a cross that is itself a quoted pair uses that pair. Forward
    crosses: the same rule on each leg's UNROUNDED forward bid/ask/mid at the tenor, rounded once
    to `round_cross`; a cross trade is marked against that ROUNDED cross forward (MTM uses it).
    cip_deviations_bps[pair] = (F_mid_from_rounded - F_cip_unrounded)/F_cip x 1e4 at 3M (0 when the
    forward is the CIP forward itself).
    MTM: sign = +1 buy_base / -1 sell_base; current_forward_mid = (cross) forward mid at the trade
    tenor; pnl_quote = sign N (F - K); pnl_usd: quote USD as is, base USD -> / spot mid of USD/QUOTE,
    other quote ccy -> x spot mid of QUOTE/USD (or / USD/QUOTE).
    Attribution (USD, all from the same spot mids): spot_delta = sign N (S_now - S_then); carry =
    sign N (F - S_now); residual = total - spot_delta - carry (= sign N (S_then - K)); historical
    cross spots from the historical USD legs (mid products/ratios).
    Net delta (USD): base leg + sign N in base ccy, quote leg - sign N F in quote ccy, each converted
    at spot mid; summed per currency.
    Returns dict{forward_rates, forward_points, cross_rates, forward_cross_rates_3M,
    cip_deviations_bps, portfolio_mtm, pnl_attribution, net_delta_usd}."""
    # know77: tolerant inputs - the model pre-divided the deposit table by 100 and dropped day_count
    # (KeyError 'day_count'); pass the RAW csv frames, but a decimal table / missing day_count still works:
    # rates_in_percent=None -> percent when max |rate| > 1 (else decimal); day_count column, else
    # day_counts={ccy: 'ACT/360'|...}, else the money-market default (ACT/365 GBP/AUD/NZD/CAD/..., ACT/360 rest).
    sp = spot.copy()
    sp.columns = [str(c).strip().lower() for c in sp.columns]
    sp = sp[sp["bid"] < sp["ask"]].copy()
    if "pair" not in sp.columns and {"base_ccy", "quote_ccy"} <= set(sp.columns):
        sp["pair"] = sp["base_ccy"].astype(str) + "/" + sp["quote_ccy"].astype(str)
    sp["pair"] = ["%s/%s" % _pair_ccys(p) for p in sp["pair"].astype(str).str.strip().str.upper()]
    sp = sp.drop_duplicates("pair", keep="last")
    dp = deposits.copy()
    dp.columns = [str(c).strip().lower() for c in dp.columns]
    ccol = "currency" if "currency" in dp.columns else ("ccy" if "ccy" in dp.columns else dp.columns[0])
    dp[ccol] = dp[ccol].astype(str).str.upper()
    dp = dp.set_index(ccol)
    pm = {"%s/%s" % _pair_ccys(k): float(v) for k, v in (pip_multipliers or {}).items()}

    def pip(p):
        if p in pm:
            return pm[p]
        if "pip_multiplier" in sp.columns and p in set(sp["pair"]):
            return float(sp.loc[sp["pair"] == p, "pip_multiplier"].iloc[0])
        return 100.0 if "JPY" in p else 10000.0

    def _rcol(tenor):
        t = str(tenor).lower()
        for c in (f"tenor_{t}", t, f"rate_{t}", f"{t}_rate", f"deposit_{t}"):
            if c in dp.columns:
                return c
        raise KeyError(f"no deposit column for tenor {tenor!r}; columns {list(dp.columns)}")

    rcols = [_rcol(t) for t in tenor_days]
    pct = rates_in_percent
    if pct is None:
        pct = bool(np.nanmax(np.abs(dp[rcols].to_numpy(float))) > 1.0)
    rscale = 100.0 if pct else 1.0

    def rate(ccy, tenor):
        return float(dp.loc[ccy, _rcol(tenor)]) / rscale

    from .helpers77_4 import money_market_basis

    def tf(ccy, tenor):
        if "day_count" in dp.columns and pd.notna(dp.loc[ccy, "day_count"]):
            return float(tenor_days[tenor]) / _basis_days(dp.loc[ccy, "day_count"])
        return float(tenor_days[tenor]) / money_market_basis(ccy, day_counts)

    book = {}
    for _, r in sp.iterrows():
        book[r["pair"]] = (float(r["bid"]), float(r["ask"]))
    S_mid = {p: 0.5 * (b + a) for p, (b, a) in book.items()}

    fwd, pts, fwd_raw = {}, {}, {}
    for p, (b, a) in book.items():
        base, quote = _pair_ccys(p)
        fwd[p], pts[p], fwd_raw[p] = {}, {}, {}
        for t in tenor_days:
            g = (1 + rate(quote, t) * tf(quote, t)) / (1 + rate(base, t) * tf(base, t))
            raw = {"bid": b * g, "ask": a * g, "mid": S_mid[p] * g}
            fwd_raw[p][t] = raw
            fwd[p][t] = {k: round(v, round_forward) for k, v in raw.items()}
            s_side = {"bid": b, "ask": a, "mid": S_mid[p]}
            pts[p][t] = {k: round((fwd[p][t][k] - s_side[k]) * pip(p), round_points) for k in raw}

    def cross_from(target, quotes_by_pair):
        q = {k: (v["bid"], v["ask"]) for k, v in quotes_by_pair.items()}
        x, y = _pair_ccys(target)
        key = f"{x}/{y}"
        if key in quotes_by_pair:
            v = quotes_by_pair[key]
            return v["bid"], v["ask"], v["mid"]
        mids = {k: v["mid"] for k, v in quotes_by_pair.items()}
        b, a, _ = _cross(target, q)
        # mid = product of leg mids (inverted leg: 1 / mid)
        def legmid(u, v):
            if u == v:
                return 1.0
            if f"{u}/{v}" in mids:
                return mids[f"{u}/{v}"]
            return 1.0 / mids[f"{v}/{u}"]
        return b, a, legmid(x, "USD") * legmid("USD", y)

    spot_q = {p: {"bid": b, "ask": a, "mid": S_mid[p]} for p, (b, a) in book.items()}
    cross_out = {}
    for c in crosses:
        b, a, m = cross_from(c, spot_q)
        cross_out[c] = {"bid": round(b, round_cross), "ask": round(a, round_cross), "mid": round(m, round_cross)}

    def fwd_any(pair, tenor):
        """quoted pair: its rounded forward; cross: legs' UNROUNDED forwards, rounded once."""
        if pair in fwd:
            return fwd[pair][tenor]
        q = {p: fwd_raw[p][tenor] for p in fwd_raw}
        b, a, m = cross_from(pair, q)
        return {"bid": round(b, round_cross), "ask": round(a, round_cross), "mid": round(m, round_cross)}

    fx3 = {}
    t3 = "3M" if "3M" in tenor_days else list(tenor_days)[0]
    for c in crosses:
        v = fwd_any(c, t3)
        fx3[c] = {t3: {k: round(x, round_cross) for k, x in v.items()}}

    cip = {}
    for p in fwd:
        cip[p] = (fwd[p][t3]["mid"] - fwd_raw[p][t3]["mid"]) / fwd_raw[p][t3]["mid"] * 1e4

    def spot_mid_any(pair, mids):
        x, y = _pair_ccys(pair)
        if f"{x}/{y}" in mids:
            return mids[f"{x}/{y}"]
        def legmid(u, v):
            if u == v:
                return 1.0
            if f"{u}/{v}" in mids:
                return mids[f"{u}/{v}"]
            return 1.0 / mids[f"{v}/{u}"]
        return legmid(x, "USD") * legmid("USD", y)

    def to_usd(amount, ccy):
        if ccy == "USD":
            return amount
        return amount * spot_mid_any(f"{ccy}/USD", S_mid)

    hist = historical_spots.copy()
    hist.columns = [str(c).strip().lower() for c in hist.columns]
    hmid = {"%s/%s" % _pair_ccys(str(r["pair"]).strip().upper()): float(r["mid"]) for _, r in hist.iterrows()}

    pf = portfolio.copy()
    pf.columns = [str(c).strip().lower() for c in pf.columns]
    mtm, attr = {}, {}
    expo = {}
    for _, r in pf.iterrows():
        tid = str(r["trade_id"])
        pair = "%s/%s" % _pair_ccys(str(r["pair"]).strip().upper())
        d = str(r["direction"]).lower().replace("_", "")
        sign = 1.0 if d.startswith("buy") or d.startswith("long") else -1.0
        N = float(r["notional_base"])
        K = float(r["trade_rate"])
        F = float(fwd_any(pair, str(r["tenor"]).upper())["mid"])
        base, quote = _pair_ccys(pair)
        pq = sign * N * (F - K)
        pu = to_usd(pq, quote)
        mtm[tid] = {"current_forward_mid": F, "pnl_quote_ccy": round(pq, round_money),
                    "pnl_usd": round(pu, round_money)}
        s_now = spot_mid_any(pair, S_mid)
        s_then = spot_mid_any(pair, hmid)
        sd = to_usd(sign * N * (s_now - s_then), quote)
        ca = to_usd(sign * N * (F - s_now), quote)
        attr[tid] = {"spot_delta_usd": round(sd, round_money), "carry_usd": round(ca, round_money),
                     "residual_usd": round(pu - sd - ca, round_money), "total_usd": round(pu, round_money)}
        expo[base] = expo.get(base, 0.0) + to_usd(sign * N, base)
        expo[quote] = expo.get(quote, 0.0) + to_usd(-sign * N * F, quote)
    return {"forward_rates": fwd, "forward_points": pts, "cross_rates": cross_out,
            "forward_cross_rates_3M": fx3, "cip_deviations_bps": cip, "portfolio_mtm": mtm,
            "pnl_attribution": attr, "net_delta_usd": {k: round(v, round_money) for k, v in expo.items()}}


# =============================================================================================
# 3. Bollinger mean reversion with risk sizing, fees and a trailing stop
# =============================================================================================

def bollinger_risk_stop_backtest(dates, close, open_, window, k, initial_capital, flat_fee,
                                 commission, risk_fraction, trail_stop, ddof=1, ticker=None,
                                 periods_per_year=252, peak_init="signal") -> dict:
    """Long-only Bollinger mean reversion, signal on close t, execution at open t+1.

    Bands: mid = SMA_window(close), upper/lower = mid +- k * rolling std(ddof).
    BUY signal (flat): close_t < lower_t. Shares = min(floor(risk_fraction * cash /
    (open_{t+1} - lower_t)) [only if that distance > 0, else 0], floor((cash - flat_fee) /
    (open_{t+1} (1 + commission)))); 0 shares -> the signal is consumed, no trade.
    Buy cost = q P (1 + commission) + flat_fee; sell revenue = q P (1 - commission) - flat_fee;
    pnl = revenue - buy cost. While long: peak = max adj close since entry (peak_init="signal":
    starts at the SIGNAL day's close; "exec": at the execution day's close); EXIT signal on close t:
    close_t > upper_t ('bollinger', has priority) else close_t < peak (1 - trail_stop)
    ('stop_loss'); the peak is updated AFTER the exit checks of the day. A day's open first
    executes the pending order, then its close is checked (an exit on the buy-execution day is
    possible). Still long after the last bar: sold at the last CLOSE ('forced_close').
    Value_t = cash + q close_t; r_0 = 0, r_t = V_t/V_{t-1} - 1; cum = cumprod(1 + r).
    Metrics: total_return = V_end/initial - 1, annualized_return = mean(r) ppy, annualized_volatility
    = std(r, ddof=1) sqrt(ppy), sharpe = ret/vol, max_drawdown = max((runmax - cum)/runmax) (> 0),
    calmar = ret/mdd, num_trades = BUY count, win_rate = share of round trips with pnl > 0,
    total_pnl = sum pnl, num_trading_days = n.
    Returns {trades: DataFrame[ticker, type BUY/SELL, signal_date, exec_date, price, shares, pnl
    (NaN on BUY)], analysis: list of dicts (trade_num, entry_date, exit_date, entry_price,
    exit_price, pnl, exit_reason, intra_trade_max_drawdown over close from entry to exit day
    inclusive), daily: DataFrame[date, portfolio_value, daily_return, cumulative_return],
    metrics: dict, bands: DataFrame[mid, upper, lower]}."""
    d = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in list(dates)]
    c = np.asarray(close, float)
    o = np.asarray(open_, float)
    n = len(c)
    s = pd.Series(c)
    mid = s.rolling(window).mean().to_numpy()
    sd = s.rolling(window).std(ddof=ddof).to_numpy()
    up, lo = mid + k * sd, mid - k * sd
    cash, q, entry_px, entry_i, peak = float(initial_capital), 0, 0.0, -1, 0.0
    buy_cost = 0.0
    pending, p_reason, p_sig = None, None, -1
    trades, analysis, values = [], [], np.zeros(n)

    def close_trade(i, px, reason, sig_date):
        nonlocal cash, q, buy_cost
        rev = q * px * (1 - commission) - flat_fee
        cash += rev
        pnl = rev - buy_cost
        trades.append({"type": "SELL", "signal_date": sig_date, "exec_date": d[i], "price": px,
                       "shares": q, "pnl": pnl})
        seg = c[entry_i:i + 1]
        rm = np.maximum.accumulate(seg)
        analysis.append({"trade_num": len(analysis) + 1, "entry_date": d[entry_i], "exit_date": d[i],
                         "entry_price": entry_px, "exit_price": px, "pnl": pnl, "exit_reason": reason,
                         "intra_trade_max_drawdown": float(np.max((rm - seg) / rm)) if len(seg) else 0.0})
        q, buy_cost = 0, 0.0

    for t in range(n):
        if pending == "BUY":
            px = o[t]
            dist = px - lo[p_sig]
            by_risk = int(math.floor(risk_fraction * cash / dist)) if dist > 0 else 0
            by_cash = int(math.floor((cash - flat_fee) / (px * (1 + commission)))) if cash > flat_fee else 0
            qty = max(min(by_risk, by_cash), 0)
            if qty > 0:
                buy_cost = qty * px * (1 + commission) + flat_fee
                cash -= buy_cost
                q, entry_px, entry_i = qty, px, t
                peak = c[p_sig] if peak_init == "signal" else c[t]
                trades.append({"type": "BUY", "signal_date": d[p_sig], "exec_date": d[t], "price": px,
                               "shares": qty, "pnl": np.nan})
            pending = None
        elif pending == "SELL":
            close_trade(t, o[t], p_reason, d[p_sig])
            pending = None
        if q == 0:
            if np.isfinite(lo[t]) and c[t] < lo[t]:
                pending, p_sig = "BUY", t
        else:
            if np.isfinite(up[t]) and c[t] > up[t]:
                pending, p_reason, p_sig = "SELL", "bollinger", t
            elif c[t] < peak * (1 - trail_stop):
                pending, p_reason, p_sig = "SELL", "stop_loss", t
            peak = max(peak, c[t])
        values[t] = cash + q * c[t]
    if q > 0:
        close_trade(n - 1, c[-1], "forced_close", d[-1])
        values[-1] = cash
    r = np.zeros(n)
    r[1:] = values[1:] / values[:-1] - 1
    cum = np.cumprod(1 + r)
    rm = np.maximum.accumulate(cum)
    mdd = float(np.max((rm - cum) / rm))
    ar = float(r.mean() * periods_per_year)
    av = float(r.std(ddof=1) * math.sqrt(periods_per_year))
    tr = pd.DataFrame(trades, columns=["type", "signal_date", "exec_date", "price", "shares", "pnl"])
    tr.insert(0, "ticker", ticker)
    sells = tr[tr["type"] == "SELL"]
    nb = int((tr["type"] == "BUY").sum())
    metrics = {"total_return": float(values[-1] / initial_capital - 1), "final_capital": float(values[-1]),
               "annualized_return": ar, "annualized_volatility": av,
               "sharpe_ratio": ar / av if av > 0 else float("nan"), "max_drawdown": mdd,
               "calmar_ratio": ar / mdd if mdd > 0 else float("nan"), "num_trades": nb,
               "win_rate": float((sells["pnl"] > 0).sum()) / nb if nb else 0.0,
               "total_pnl": float(sells["pnl"].sum()), "num_trading_days": int(n)}
    daily = pd.DataFrame({"date": d, "portfolio_value": values, "daily_return": r, "cumulative_return": cum})
    return {"trades": tr, "analysis": analysis, "daily": daily, "metrics": metrics,
            "bands": pd.DataFrame({"mid": mid, "upper": up, "lower": lo})}


# =============================================================================================
# 4. pairwise copula selection
# =============================================================================================

def pairwise_copula_table(returns, symbols, families=("gaussian", "student_t", "clayton", "gumbel"),
                          tail_q=0.95, t_method="itau", nu_bounds=(2.01, 100.0)) -> dict:
    """Fit every family to every symbol pair and pick the best by AIC.

    returns: wide DataFrame (one column per symbol, NaN rows already dropped); pairs =
    itertools.combinations(symbols, 2) in the given order; u = rank/(n+1) per pair.
    Gaussian/Clayton/Gumbel: theta by Kendall-tau inversion; Student-t: t_method="itau" -> rho by
    tau inversion and nu by profile likelihood (the textbook 'itau' recipe), "mle" -> joint
    (rho, nu) maximum likelihood. Clayton/Gumbel with tau <= 0 -> independence (LL 0). k = 1
    parameter (t: 2), AIC = 2k - 2LL, BIC = k ln n - 2LL.
    Tail dependence: lambda_U = #(u1 > q and u2 > q)/#(u1 > q), lambda_L = #(u1 <= 1-q and
    u2 <= 1-q)/#(u1 <= 1-q) (denominator = FIRST margin).
    most_common_best_copula = the MODE of the per-pair best families (ties -> family order), NOT
    the most frequent row label in the 4-per-pair fit table (every family has the same count).
    Returns {fits DataFrame[stock1, stock2, copula, param1, param2, loglik, aic, bic], best
    {"S1-S2": family}, tail DataFrame[stock1, stock2, lambda_upper, lambda_lower, kendall_tau],
    summary {n_pairs, most_common_best_copula, mean_kendall_tau, has_tail_dependence (> 0.1)}}."""
    from . import copulas as _cop
    rows, tails, best = [], [], {}
    # know77: a long panel deduplicated on 'date' alone left 0-3 return rows and fit_copula died with
    # "gaussian rho must be in (-1,1)" ten times - say what is wrong instead
    nr = len(returns)
    if nr < 30 or any(s not in getattr(returns, "columns", []) for s in symbols):
        raise ValueError(f"pairwise_copula_table got {nr} return rows / columns {list(getattr(returns, 'columns', []))[:8]}: "
                         "build WIDE log returns (one column per symbol, NaN rows dropped); from a long "
                         "(date, symbol, close) table call equity_pair_copula_report(prices, symbols) - never "
                         "drop_duplicates('date') on a long panel")
    for s1, s2 in itertools.combinations(list(symbols), 2):
        x = np.column_stack([np.asarray(returns[s1], float), np.asarray(returns[s2], float)])
        u = _cop.pseudo_observations(x)
        tau = float(_cop.empirical_kendall_tau(u))
        fits = []
        for fam in families:
            meth = t_method if fam in ("t", "student_t", "student-t") else "itau"
            r = _cop.fit_copula(fam, u, method=meth, nu_bounds=nu_bounds)
            fits.append({"stock1": s1, "stock2": s2, "copula": fam, "param1": float(r["theta"]),
                         "param2": float(r["nu"]) if r.get("nu") is not None else np.nan,
                         "loglik": float(r["loglik"]), "aic": float(r["aic"]), "bic": float(r["bic"])})
        rows.extend(fits)
        best[f"{s1}-{s2}"] = min(fits, key=lambda z: z["aic"])["copula"]
        a, b = u[:, 0], u[:, 1]
        up = float(np.sum((a > tail_q) & (b > tail_q)) / max(np.sum(a > tail_q), 1))
        lw = float(np.sum((a <= 1 - tail_q) & (b <= 1 - tail_q)) / max(np.sum(a <= 1 - tail_q), 1))
        tails.append({"stock1": s1, "stock2": s2, "lambda_upper": up, "lambda_lower": lw, "kendall_tau": tau})
    vals = list(best.values())
    counts = {f: vals.count(f) for f in families}
    mode = max(families, key=lambda f: counts[f])
    tail = pd.DataFrame(tails)
    summary = {"n_pairs": len(best), "most_common_best_copula": mode,
               "mean_kendall_tau": float(tail["kendall_tau"].mean()),
               "has_tail_dependence": bool(((tail["lambda_upper"] > 0.1) | (tail["lambda_lower"] > 0.1)).any())}
    return {"fits": pd.DataFrame(rows), "best": best, "tail": tail, "summary": summary}


# =============================================================================================
# 5. return-panel cleaning against a trading calendar
# =============================================================================================

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%y", "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y", "%Y%m%d")


def iso_date_strings(values) -> pd.Series:
    """Any date-like column -> 'YYYY-MM-DD' strings: datetime64 / Timestamps (as INPUTS may have
    parsed them), or text in mixed formats ('2023-03-15', '03/15/2023', '15-Mar-23', with stray
    spaces) tried format by format, then pandas 'mixed'. Unparseable -> NaN."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d")
    txt = s.astype(str).str.strip()
    out = pd.Series([np.nan] * len(s), dtype=object)
    for fmt in _DATE_FORMATS:
        m = out.isna()
        if not m.any():
            break
        p = pd.to_datetime(txt[m], format=fmt, errors="coerce")
        ok = p.notna()
        out.loc[p[ok].index] = p[ok].dt.strftime("%Y-%m-%d")
    m = out.isna()
    if m.any():
        p = pd.to_datetime(txt[m], format="mixed", errors="coerce")
        ok = p.notna()
        out.loc[p[ok].index] = p[ok].dt.strftime("%Y-%m-%d")
    return out


def clean_returns_with_calendar(raw, calendar, asset_cols=None, outlier_abs=None,
                                date_col="date") -> dict:
    """Canonical cleaning pipeline (each count = rows removed at that step, from rows still present):
    1 strip header whitespace (n_rows_raw = input rows); 2 dates -> 'YYYY-MM-DD' (iso_date_strings);
    3 drop dates not in the calendar (calendar dates normalised the same way - a datetime64
    calendar compared to strings matches NOTHING, the know75 failure); 4 duplicate dates: keep the
    LAST occurrence in file order; 5 outliers: any |r| > outlier_abs (NaN is not an outlier;
    None skips); 6 listwise NaN deletion; 7 sort by date.
    Returns {clean DataFrame (date str + asset_cols), report {n_rows_raw, n_non_trading_days_removed,
    n_duplicates_removed, n_outliers_removed, n_missing_rows_removed, n_rows_after_cleaning}}."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    n0 = len(df)
    cols = list(asset_cols) if asset_cols is not None else [c for c in df.columns if c != date_col]
    df[date_col] = iso_date_strings(df[date_col]).to_numpy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    cal = calendar[calendar.columns[0]] if isinstance(calendar, pd.DataFrame) else pd.Series(calendar)
    cal_set = set(iso_date_strings(cal).dropna())
    a = df[df[date_col].isin(cal_set)]
    b = a.drop_duplicates(subset=date_col, keep="last")
    if outlier_abs is not None:
        out_mask = (b[cols].abs() > outlier_abs).any(axis=1)
        c_ = b[~out_mask]
    else:
        c_ = b
    e = c_.dropna(subset=cols)
    clean = e.sort_values(date_col, kind="mergesort").reset_index(drop=True)[[date_col] + cols]
    report = {"n_rows_raw": int(n0), "n_non_trading_days_removed": int(n0 - len(a)),
              "n_duplicates_removed": int(len(a) - len(b)), "n_outliers_removed": int(len(b) - len(c_)),
              "n_missing_rows_removed": int(len(c_) - len(e)), "n_rows_after_cleaning": int(len(clean))}
    from ._rdict import ResultDict
    return ResultDict({"clean": clean, "report": report, "cleaning_report": report, **report})


# =============================================================================================
# 6. CreditMetrics one-factor default / migration simulation
# =============================================================================================

def creditmetrics_default_sim(P, initial_counts, rho, n_sims, seed, lgd, face=1.0,
                              rng="legacy", levels=(0.95, 0.99, 0.999), es_tail="geq") -> dict:
    """-> dict with FLAT var_95/var_99/var_999/es_95/es_99, mean_loss, std_loss, by_rating, *_rate (var/es are DICTS keyed by level, never unpack them).

    One-factor Gaussian CreditMetrics: Z_i = sqrt(rho) M + sqrt(1-rho) e_i; new rating from
    thresholds cumulated from the WORST state (default iff Z < Phi^-1(p_default)).
    P: transition matrix in PROBABILITIES (percent rows / 100), rows = initial ratings in the
    order of initial_counts (best first), last column = Default. Draw order: M (n_sims) FIRST,
    then e (n_sims x n_bonds). rng="legacy": np.random.seed(seed); np.random.randn(...) - the
    stream task scripts that say 'random seed N' are usually generated with (the 99% VaR of a
    discrete loss count can move by a whole default between streams); rng="default":
    np.random.default_rng(seed).standard_normal.
    Loss = lgd x face x #defaults. VaR_a = np.quantile(L, a) (linear); ES_a = mean(L >= VaR_a)
    ("geq") or mean(L > VaR_a) ("gt").
    Returns {losses, new_idx [n_sims x n_bonds], initial_idx, var {a: v}, es {a: v}, mean_loss,
    std_loss (ddof=1), by_rating DataFrame[n_bonds, mean_loss, prob_default],
    upgrade_rate, downgrade_rate (worse but not default), default_rate}."""
    P = np.asarray(P, float)
    m = P.shape[1]
    init = np.repeat(np.arange(len(initial_counts)), np.asarray(initial_counts, int))
    if rng == "legacy":
        np.random.seed(int(seed))
        M = np.random.randn(int(n_sims))
        e = np.random.randn(int(n_sims), len(init))
    else:
        g = np.random.default_rng(int(seed))
        M = g.standard_normal(int(n_sims))
        e = g.standard_normal((int(n_sims), len(init)))
    Z = math.sqrt(rho) * M[:, None] + math.sqrt(1 - rho) * e
    cumw = np.cumsum(P[:, ::-1], axis=1)[:, :-1]
    thr = stats.norm.ppf(np.clip(cumw, 0.0, 1.0))
    new = np.empty(Z.shape, dtype=np.int16)
    for r in np.unique(init):
        cols = np.flatnonzero(init == r)
        kk = np.searchsorted(thr[r], Z[:, cols], side="right")
        new[:, cols] = (m - 1 - kk)
    D = new == m - 1
    L = lgd * face * D.sum(axis=1)
    var, es = {}, {}
    for a in levels:
        v = float(np.quantile(L, a))
        sel = L >= v if es_tail == "geq" else L > v
        var[a], es[a] = v, float(L[sel].mean()) if sel.any() else v
    by = []
    for r in range(len(initial_counts)):
        cm = init == r
        by.append({"n_bonds": int(cm.sum()), "mean_loss": float((lgd * face * D[:, cm].sum(axis=1)).mean()),
                   "prob_default": float(D[:, cm].mean())})
    # know77: flat keys too - the model unpacked the var dict as a tuple (got its KEYS) and read es[0]
    from .helpers75_2 import level_key
    flat = {f"var_{level_key(a)}": var[a] for a in var}
    flat.update({f"es_{level_key(a)}": es[a] for a in es})
    return {**flat, "losses": L, "new_idx": new, "initial_idx": init, "var": var, "es": es,
            "mean_loss": float(L.mean()), "std_loss": float(L.std(ddof=1)), "by_rating": pd.DataFrame(by),
            "upgrade_rate": float((new < init[None, :]).mean()),
            "downgrade_rate": float(((new > init[None, :]) & ~D).mean()), "default_rate": float(D.mean())}


# =============================================================================================
# 7. CEREP rating-migration study
# =============================================================================================

def cerep_migration_study(transition_tables, default_tables=None, mapping=None, states=None, horizons=None,
                          alpha=None, reference_pct=None, min_col_total=5, years=None, decimals=6) -> dict:
    """Pooled migration analysis of yearly S&P/CEREP count tables.

    transition_tables {year: DataFrame} (index = BOP fine rating, columns = EOP fine ratings incl.
    NR/Withdrawals; read ';' files with read_delimited_matrix); default_tables {year: DataFrame
    [rating, n_defaults]}; mapping {fine: target or None} (None = censored, e.g. NR, Withdrawals);
    states target order with Default LAST.
    Count matrix C_t: collapse rows and columns through mapping (censored columns dropped, censored
    rows dropped). Hidden defaults per FINE BOP rating = max(true n_defaults - visible defaults in
    that fine row (EOP columns mapping to Default), 0), added to the collapsed row's Default cell.
    The Default row of C_t is kept as counted (it does not enter P_bar's Default row).
    Cohort P_t = C_t / row sums (empty row -> identity). P_bar = sum_t C_t / sum_t row sums, Default
    row forced to e_Default. PD_i(h) = (P_bar^h)[i, Default]. Markov test per non-default row i:
    table = rows C_t[i, :] over cohorts, keep columns with total >= min_col_total (and rows with a
    positive total), chi2 = sum (O - E)^2/E with E = row total x column total / grand total,
    dof = (r-1)(c-1), p = chi2.sf; reject when p < alpha (dof 0 -> stat 0, p 1, not rejected).
    Generator: Q = real(logm(P_bar)); negative off-diagonals -> 0, diagonal = -sum of off-diagonals;
    Default row 0. Deviation (if reference_pct DataFrame with rating index or 'rating' column and
    an NR column): ref = drop NR, /100, renormalise rows; Delta = P_bar - ref on non-default rows.
    Cohort default rate_t = sum_i C_t[i, Default] / sum_i sum_k C_t[i, k] over non-default i.
    Returns dict of DataFrames {average (rating + states), cumulative (rating, horizon_h...),
    markov (rating, chi2_stat, df, p_value, reject_H0), generator, cohort_sizes (year + non-default
    states), deviation, cohort_default_rates (year, total_issuers, total_defaults, default_rate,
    <R>_defaults)}, summary dict (keys of the usual summary.json), counts {year: C_t}, P_bar, Q.
    know78: inputs are normalised first (the model passed LISTS -> 'truth value of a DataFrame is
    ambiguous' x7): transition/default tables may be {year: obj}, {file name: obj} (year = the 4-digit
    number in the name), [(year, obj)] or a plain list (years= gives the labels, else 0..n-1); each obj
    may be a parsed DataFrame (one ';'-joined column is split), text or a path. mapping None ->
    sp_rating_mapping of every label seen; states None -> AAA..CCC, Default. The result ALSO carries the
    deliverables keyed by file name (values rounded to `decimals`) and by stem
    ('average_transition_matrix', 'summary', ...): for name, obj in res['files'].items(): emit(name, obj)."""
    if horizons is None or alpha is None:
        raise ValueError("pass horizons= (config projection_horizons) and alpha= (config significance_level)")
    return _cerep_study_core(*_cerep_normalise(transition_tables, default_tables, mapping, states, years),
                             horizons, alpha, reference_pct=_cerep_reference(reference_pct),
                             min_col_total=min_col_total, decimals=decimals)


def _cerep_year_dict(obj, years=None, parse=None):
    if obj is None:
        return {}
    if isinstance(obj, pd.DataFrame):
        obj = [obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            m = re.search(r"(19|20)\d{2}", str(k))
            key = int(k) if isinstance(k, (int, np.integer)) else (int(m.group(0)) if m else k)
            out[key] = parse(v) if parse else v
        return out
    items = list(obj)
    if items and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in items):
        return _cerep_year_dict(dict(items), parse=parse)
    labels = list(years) if years is not None else list(range(len(items)))
    return {int(y) if str(y).isdigit() else y: (parse(v) if parse else v) for y, v in zip(labels, items)}


def _cerep_default_frame(x):
    from .helpers75_4 import read_delimited_matrix
    if isinstance(x, pd.DataFrame) and len(x.columns) >= 2 and "rating" in [str(c).strip().lower() for c in x.columns]:
        return x
    f = read_delimited_matrix(x)
    f = f.reset_index()
    f.columns = ["rating"] + list(f.columns[1:])
    return f


def _cerep_normalise(transition_tables, default_tables, mapping, states, years):
    from .helpers75_4 import read_delimited_matrix
    from .helpers77_3 import sp_rating_mapping
    tt = _cerep_year_dict(transition_tables, years, read_delimited_matrix)
    dt = _cerep_year_dict(default_tables, years, _cerep_default_frame)
    S = list(states) if states is not None else ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
    labs = set()
    for f in tt.values():
        labs |= set(map(str, f.index)) | set(map(str, f.columns))
    for f in dt.values():
        if isinstance(f, pd.DataFrame) and "rating" in f.columns:
            labs |= set(map(str, f["rating"]))
    auto = sp_rating_mapping(sorted(labs), S)
    # a supplied mapping wins where it has the label; labels it does not mention (CC, C, SD, D, R ... when the
    # caller mapped only the target scale) are mapped by the agency rules instead of being silently censored
    mapping = auto if mapping is None else {**auto, **{str(k).strip(): v for k, v in dict(mapping).items()}}
    return tt, dt, mapping, S


def _cerep_reference(ref):
    if ref is None:
        return None
    from .helpers75_4 import read_delimited_matrix
    if isinstance(ref, pd.DataFrame) and "rating" in ref.columns:
        return ref
    f = read_delimited_matrix(ref)
    f.index.name = "rating"
    return f.reset_index()


def _cerep_study_core(transition_tables, default_tables, mapping, states, horizons, alpha, reference_pct=None,
                      min_col_total=5, decimals=6):
    from scipy.linalg import logm
    S = list(states)
    nd = S[:-1]
    dflt = S[-1]
    years = sorted(transition_tables)
    counts = {}
    for y in years:
        raw = transition_tables[y].copy()
        raw.index = [str(i).strip() for i in raw.index]
        raw.columns = [str(c).strip() for c in raw.columns]
        C = pd.DataFrame(0.0, index=S, columns=S)
        for fr in raw.index:
            tr = mapping.get(fr)
            if tr is None:
                continue
            for fc in raw.columns:
                tc = mapping.get(fc)
                if tc is None:
                    continue
                C.loc[tr, tc] += float(raw.loc[fr, fc])
        dt = default_tables.get(y)
        if dt is not None:
            dd = dt.copy()
            dd.columns = [str(c).strip().lower() for c in dd.columns]
            vis_cols = [fc for fc in raw.columns if mapping.get(fc) == dflt]
            for _, r in dd.iterrows():
                fr = str(r["rating"]).strip()
                tr = mapping.get(fr)
                if tr is None or tr == dflt:
                    continue
                vis = float(raw.loc[fr, vis_cols].sum()) if fr in raw.index else 0.0
                C.loc[tr, dflt] += max(float(r["n_defaults"]) - vis, 0.0)
        counts[y] = C
    tot = sum(counts.values())
    rs = tot.sum(axis=1)
    Pb = tot.div(rs.replace(0, np.nan), axis=0).fillna(0.0)
    for s_ in S:
        if rs[s_] == 0:
            Pb.loc[s_] = 0.0
            Pb.loc[s_, s_] = 1.0
    Pb.loc[dflt] = 0.0
    Pb.loc[dflt, dflt] = 1.0
    Pm = Pb.to_numpy()
    cum = {f"horizon_{h}": [np.linalg.matrix_power(Pm, int(h))[S.index(r), -1] for r in nd] for h in horizons}
    cumdf = pd.DataFrame({"rating": nd, **cum})
    mk = []
    for r in nd:
        T = np.array([counts[y].loc[r].to_numpy() for y in years])
        T = T[:, T.sum(axis=0) >= min_col_total]
        T = T[T.sum(axis=1) > 0]
        if T.shape[0] < 2 or T.shape[1] < 2:
            mk.append({"rating": r, "chi2_stat": 0.0, "df": 0, "p_value": 1.0, "reject_H0": False})
            continue
        E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
        st = float(np.sum((T - E) ** 2 / E))
        dof = (T.shape[0] - 1) * (T.shape[1] - 1)
        pv = float(stats.chi2.sf(st, dof))
        mk.append({"rating": r, "chi2_stat": st, "df": int(dof), "p_value": pv, "reject_H0": bool(pv < alpha)})
    mkdf = pd.DataFrame(mk)
    Q = np.real(logm(Pm))
    n = len(S)
    for i in range(n):
        off = [j for j in range(n) if j != i]
        Q[i, off] = np.clip(Q[i, off], 0.0, None)
        Q[i, i] = -Q[i, off].sum()
    Q[-1, :] = 0.0
    Qdf = pd.DataFrame(Q, index=S, columns=S)
    sizes = pd.DataFrame([{"year": y, **{r: float(counts[y].loc[r].sum()) for r in nd}} for y in years])
    cdr = []
    for y in years:
        C = counts[y]
        ti = float(C.loc[nd].to_numpy().sum())
        td = float(C.loc[nd, dflt].sum())
        cdr.append({"year": y, "total_issuers": ti, "total_defaults": td,
                    "default_rate": td / ti if ti else 0.0, **{f"{r}_defaults": float(C.loc[r, dflt]) for r in nd}})
    cdrdf = pd.DataFrame(cdr)
    dev = None
    if reference_pct is not None:
        ref = reference_pct.copy()
        if "rating" in ref.columns:
            ref = ref.set_index("rating")
        ref.index = [str(i).strip() for i in ref.index]
        ref = ref[[c for c in ref.columns if str(c).strip().upper() != "NR"]] / 100.0
        ref = ref.div(ref.sum(axis=1), axis=0)
        dev = Pb.loc[nd, S] - ref.loc[nd, S]
    avg_df = Pb.reset_index().rename(columns={"index": "rating"})
    gen_df = Qdf.reset_index().rename(columns={"index": "rating"})
    summary = {"total_transitions": int(round(float(sizes[nd].to_numpy().sum()))), "n_cohorts": len(years),
               "avg_default_rate_1y": float(np.mean(cumdf.get("horizon_1", cumdf.iloc[:, 1]))),
               "avg_default_rate_5y": float(np.mean(cumdf["horizon_5"])) if "horizon_5" in cumdf else None,
               "n_markov_rejections": int(mkdf["reject_H0"].sum()),
               "generator_max_offdiag": float(np.max(Q[~np.eye(n, dtype=bool)])),
               "generator_min_diag": float(np.min(np.diag(Q)))}
    h1 = cumdf.set_index("rating").iloc[:, 0]
    ig = [r for r in nd if r in ("AAA", "AA", "A", "BBB")]
    sg = [r for r in nd if r not in ig]
    summary["investment_grade_1y_pd"] = float(h1[ig].mean()) if ig else None
    summary["speculative_grade_1y_pd"] = float(h1[sg].mean()) if sg else None
    if dev is not None:
        a = dev.abs().to_numpy()
        i, j = np.unravel_index(np.argmax(a), a.shape)
        summary.update({"max_absolute_deviation": float(a[i, j]),
                        "max_deviation_cell": f"{dev.index[i]}->{dev.columns[j]}",
                        "mean_absolute_deviation": float(a.mean())})
    pk = cdrdf.loc[cdrdf["default_rate"].idxmax()]
    lw = cdrdf.loc[cdrdf["default_rate"].idxmin()]
    summary.update({"peak_default_year": int(pk["year"]), "peak_default_rate": float(pk["default_rate"]),
                    "lowest_default_year": int(lw["year"]), "lowest_default_rate": float(lw["default_rate"])})
    dev_df = None if dev is None else dev.reset_index().rename(columns={"index": "rating"})
    for col in nd:   # counts are whole issuers
        if np.allclose(sizes[col], np.round(sizes[col])):
            sizes[col] = sizes[col].round().astype(int)
    for col in ["total_issuers", "total_defaults"] + [f"{r}_defaults" for r in nd]:
        if col in cdrdf and np.allclose(cdrdf[col], np.round(cdrdf[col])):
            cdrdf[col] = cdrdf[col].round().astype(int)

    def rnd(df):
        if decimals is None or df is None:
            return df
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(decimals)
        return df
    files = {"average_transition_matrix.csv": rnd(avg_df), "cumulative_default_probs.csv": rnd(cumdf),
             "markov_test.csv": mkdf, "generator_matrix.csv": rnd(gen_df), "cohort_sizes.csv": sizes,
             "cohort_default_rates.csv": rnd(cdrdf), "summary.json": summary}
    if dev_df is not None:
        files["transition_matrix_deviation.csv"] = rnd(dev_df)
    out = {"average": avg_df, "cumulative": cumdf, "markov": mkdf, "generator": gen_df,
           "cohort_sizes": sizes, "deviation": dev_df,
           "cohort_default_rates": cdrdf, "summary": summary, "counts": counts, "P_bar": Pb, "Q": Qdf,
           "files": files}
    out.update(files)
    out.update({k.rsplit(".", 1)[0]: v for k, v in files.items() if k.rsplit(".", 1)[0] not in out})
    return out


# =============================================================================================
# 8. coin-premium (Deribit-style) option snapshot -> call surface -> Dupire local vol
# =============================================================================================

def _b76_call(F, K, T, sig):
    v = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return F * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - v)


def _b76_iv(C, F, K, T):
    lo_, hi_ = 1e-6, 5.0
    intrinsic = max(F - K, 0.0)
    if not (C > intrinsic and C < F):
        return float("nan")
    return float(optimize.brentq(lambda s: _b76_call(F, K, T, s) - C, lo_, hi_, xtol=1e-12))


def coin_option_surface(raw, target_expiries, target_strikes, max_relative_spread, local_var_floor,
                        local_vol_cap, snapshot_ts=None, days_per_year=365.0, dedupe="latest",
                        repair=True) -> dict:
    """Coin-premium snapshot -> repaired call surface -> Dupire (4 surface tables; with a barrier contract use localvol_barrier_report).

    know79: a run built the four surface files with this helper, then hand-fed its local_vol TABLE to
    localvol_mc_barrier and crashed 3x - barrier_results.json and summary.json were stubs (5 tests).
    localvol_barrier_report(raw, task_rules, barrier_contract, snapshot_meta) calls this helper AND the MC
    and returns all six deliverables in contract keys.

    raw: DataFrame[snapshot_ts, creation_timestamp, instrument_name 'BTC-24APR26-70000-C',
    expiration_timestamp (ms), option_type call/put, strike, bid_price_btc, ask_price_btc,
    mid_price_btc, ...]. Conventions: r = q = 0, DF = 1.
    Cleaning (counted in this order): duplicates = older rows of an instrument_name (keep the LATEST
    creation_timestamp); crossed = bid > ask; wide = (ask - bid)/mid > max_relative_spread with
    mid > 0. Clean one-sided rows stay in `cleaned` but only call+put pairs at the same
    (expiry label, strike) build the surface.
    ttm = (expiration_timestamp - snapshot_ts)/(1000*86400*days_per_year) from the exact ms stamps.
    Forward per expiry: the paired strike with minimal |c_btc - p_btc| gives F = K/(1 - (c - p))
    (coin parity c - p = (F - K)/F); premium_usd = premium_btc x F for EVERY node of that expiry.
    OTM side: K < F -> 'put_parity', synthetic call = P_usd + F - K; K >= F -> 'call_direct' (C_usd).
    log_moneyness = ln(K/F); total_variance = Black-76 implied vol^2 x ttm of the synthetic call.
    Call surface on the target grid: scipy PchipInterpolator in strike of the synthetic calls per
    expiry; repair (repair=True): clip to [max(F-K, 0), F], then running minimum in strike
    (non-increasing), then the lower convex hull through the points (convex).
    Local vol (target grid, rows = expiries by ttm): dC/dT = np.gradient(C[:, j], ttm,
    edge_order=2), dC/dK = np.gradient(C[i, :], K, edge_order=2), d2C/dK2 = np.gradient(dC/dK, K,
    edge_order=2); local_var = 2 dC/dT/(K^2 d2C/dK2), non-finite or < floor -> floor; local_vol =
    min(sqrt(local_var), cap).
    Returns {cleaned, nodes, call_surface, local_vol (DataFrames in the usual column order),
    forwards {expiry: F}, ttm {expiry: T}, counts {n_raw_rows, n_clean_rows, n_paired_nodes,
    n_surface_nodes, n_localvol_nodes, duplicates_removed, crossed_rows_removed,
    wide_spread_rows_removed, unmatched_rows_removed}, checks {call_surface_monotone_in_strike,
    call_surface_convex_in_strike, local_vol_positive}}."""
    from scipy.interpolate import PchipInterpolator
    df = raw.copy()
    n_raw = len(df)
    # know77: INPUTS[...] parses ms epoch columns into datetime64 ("Cannot cast DatetimeArray to dtype
    # float64") - turn them back into integer milliseconds before any arithmetic
    for _c in ("snapshot_ts", "creation_timestamp", "expiration_timestamp"):
        if _c in df.columns and pd.api.types.is_datetime64_any_dtype(df[_c]):
            _s = df[_c]
            if getattr(_s.dt, "tz", None) is not None:
                _s = _s.dt.tz_convert("UTC").dt.tz_localize(None)
            df[_c] = (_s - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    if snapshot_ts is not None and not isinstance(snapshot_ts, (int, float, np.integer, np.floating)):
        snapshot_ts = (pd.Timestamp(snapshot_ts).tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    # know77: the model passed a column SUBSET (KeyError 'creation_timestamp' / 'expiration_timestamp'
    # x6) - pass the RAW snapshot; without the stamps say so instead of a bare KeyError
    need = [c for c in ("instrument_name", "bid_price_btc", "ask_price_btc", "mid_price_btc") if c not in df.columns]
    if need:
        raise ValueError(f"coin_option_surface needs the RAW snapshot table; missing columns {need}")
    if "expiration_timestamp" not in df.columns and "ttm_years" not in df.columns:
        raise ValueError("coin_option_surface needs the RAW snapshot (expiration_timestamp in ms for the exact "
                         "ACT/365 maturities) - pass INPUTS[...] unmodified; it cleans, dedupes and pairs itself")
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["creation_timestamp", "_ord"] if "creation_timestamp" in df.columns else ["_ord"],
                        kind="mergesort")
    dd = df.drop_duplicates("instrument_name", keep="last" if dedupe == "latest" else "first")
    n_dup = len(df) - len(dd)
    cr = dd[~(dd["bid_price_btc"] > dd["ask_price_btc"])]
    n_cross = len(dd) - len(cr)
    midc = cr["mid_price_btc"].astype(float)
    rel = (cr["ask_price_btc"] - cr["bid_price_btc"]) / midc.where(midc > 0)
    wide = (midc > 0) & (rel > max_relative_spread)
    cl = cr[~wide].sort_values("_ord").drop(columns="_ord").copy()
    n_wide = int(wide.sum())
    parts = cl["instrument_name"].astype(str).str.split("-")
    cl["expiry"] = parts.str[1]
    cl["strike"] = cl["strike"].astype(float) if "strike" in cl else parts.str[2].astype(float)
    ot = cl["option_type"].astype(str).str.lower() if "option_type" in cl else parts.str[3]
    cl["option_type"] = np.where(ot.str.startswith("c"), "call", "put")
    if "expiration_timestamp" in cl.columns:
        snap = float(snapshot_ts if snapshot_ts is not None else cl["snapshot_ts"].iloc[0])
        cl["ttm_years"] = (cl["expiration_timestamp"].astype(float) - snap) / (1000.0 * 86400.0 * days_per_year)
    calls = cl[cl["option_type"] == "call"].set_index(["expiry", "strike"])
    puts = cl[cl["option_type"] == "put"].set_index(["expiry", "strike"])
    keys = calls.index.intersection(puts.index)
    paired_rows = 2 * len(keys)
    n_unmatched = len(cl) - paired_rows
    rows = []
    fwd, ttm = {}, {}
    for e in list(dict.fromkeys(k[0] for k in keys)):
        ks = sorted(k[1] for k in keys if k[0] == e)
        c = np.array([float(calls.loc[(e, k), "mid_price_btc"]) for k in ks])
        p = np.array([float(puts.loc[(e, k), "mid_price_btc"]) for k in ks])
        j = int(np.argmin(np.abs(c - p)))
        F = ks[j] / (1.0 - (c[j] - p[j]))
        T = float(calls.loc[(e, ks[0]), "ttm_years"])
        fwd[e], ttm[e] = F, T
        for K, cb, pb in zip(ks, c, p):
            cu, pu = cb * F, pb * F
            side = "put_parity" if K < F else "call_direct"
            syn = pu + F - K if side == "put_parity" else cu
            iv = _b76_iv(syn, F, K, T)
            rows.append({"expiry": e, "ttm_years": T, "strike": K, "forward_usd": F, "call_mid_usd": cu,
                         "put_mid_usd": pu, "otm_side": side, "synthetic_call_usd": syn,
                         "log_moneyness": math.log(K / F), "total_variance": iv * iv * T})
    nodes = pd.DataFrame(rows).sort_values(["ttm_years", "strike"]).reset_index(drop=True)
    Ks = np.asarray(sorted(float(k) for k in target_strikes))
    exps = sorted([e for e in target_expiries if e in fwd], key=lambda e: ttm[e])
    surf = []
    Cm = np.zeros((len(exps), len(Ks)))
    for i, e in enumerate(exps):
        sub = nodes[nodes["expiry"] == e].sort_values("strike")
        f = PchipInterpolator(sub["strike"].to_numpy(), sub["synthetic_call_usd"].to_numpy(), extrapolate=True)
        cv = f(Ks)
        if repair:
            F = fwd[e]
            cv = np.clip(cv, np.maximum(F - Ks, 0.0), F)
            cv = np.minimum.accumulate(cv)
            cv = _lower_convex(Ks, cv)
        Cm[i] = cv
        for K, v in zip(Ks, cv):
            surf.append({"expiry": e, "ttm_years": ttm[e], "strike": K, "call_price_usd": float(v)})
    t = np.array([ttm[e] for e in exps])
    eo_t = 2 if len(t) > 2 else 1
    dT = np.gradient(Cm, t, axis=0, edge_order=eo_t)
    dK = np.gradient(Cm, Ks, axis=1, edge_order=2)
    dKK = np.gradient(dK, Ks, axis=1, edge_order=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lvar = 2.0 * dT / (Ks[None, :] ** 2 * dKK)
    lvar = np.where(np.isfinite(lvar) & (lvar >= local_var_floor), lvar, local_var_floor)
    lv = np.minimum(np.sqrt(lvar), local_vol_cap)
    lrows = []
    for i, e in enumerate(exps):
        for j, K in enumerate(Ks):
            lrows.append({"expiry": e, "ttm_years": ttm[e], "strike": K, "dC_dT": dT[i, j], "dC_dK": dK[i, j],
                          "d2C_dK2": dKK[i, j], "local_vol": lv[i, j]})
    dif = np.diff(Cm, axis=1)
    sl = dif / np.diff(Ks)[None, :]
    checks = {"call_surface_monotone_in_strike": bool(np.all(dif <= 1e-9)),
              "call_surface_convex_in_strike": bool(np.all(np.diff(sl, axis=1) >= -1e-9)),
              "local_vol_positive": bool(np.all(lv > 0))}
    cols = ["instrument_name", "expiry", "option_type", "strike", "bid_price_btc", "ask_price_btc",
            "mid_price_btc", "mark_iv_pct", "underlying_price_usd", "open_interest", "volume_usd", "creation_timestamp"]
    cleaned = cl[[c for c in cols if c in cl.columns] + [c for c in cl.columns if c not in cols]]
    counts = {"n_raw_rows": int(n_raw), "n_clean_rows": int(len(cl)), "n_paired_nodes": int(len(nodes)),
              "n_surface_nodes": int(len(surf)), "n_localvol_nodes": int(len(lrows)),
              "duplicates_removed": int(n_dup), "crossed_rows_removed": int(n_cross),
              "wide_spread_rows_removed": n_wide, "unmatched_rows_removed": int(n_unmatched)}
    return {"cleaned": cleaned, "nodes": nodes, "call_surface": pd.DataFrame(surf),
            "local_vol": pd.DataFrame(lrows), "forwards": fwd, "ttm": ttm, "counts": counts, "checks": checks,
            "C": Cm, "strikes": Ks, "ttms": t, "expiries": exps}


def _lower_convex(x, y):
    """Greatest convex minorant through the points (x ascending), evaluated at x."""
    pts = []
    for xi, yi in zip(x, y):
        while len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            if (y2 - y1) * (xi - x1) >= (yi - y1) * (x2 - x1):
                pts.pop()
            else:
                break
        pts.append((xi, yi))
    px, py = zip(*pts)
    return np.interp(x, px, py)


# =============================================================================================
# 9. SVI slices -> Dupire local vol (Gatheral total-variance form)
# =============================================================================================

def select_liquid_expiries(df, expiry_col, strike_col, min_strikes, n_min, n_max, volume_col=None):
    """Expiries with at least `min_strikes` distinct strikes (after the caller's OTM/volume
    filters), ranked by total volume (else strike count), then the top n_max kept in DATE order;
    raises if fewer than n_min qualify. Returns a sorted list of expiry labels."""
    g = df.groupby(expiry_col)
    cnt = g[strike_col].nunique()
    ok = cnt[cnt >= min_strikes]
    if len(ok) < n_min:
        raise ValueError(f"only {len(ok)} expiries have >= {min_strikes} strikes; lower the filter")
    score = g[volume_col].sum().loc[ok.index] if volume_col else ok
    keep = score.sort_values(ascending=False).index[:n_max]
    return sorted(keep)


def svi_dupire_surface(params_by_T, S0, r, q, strike_grid, ttm_grid=None) -> pd.DataFrame:
    """Local vol from raw-SVI slices {T: (a, b, rho, m, sigma)} (w(k) = total implied variance,
    k = ln(K/F_T), F_T = S0 e^{(r-q)T}).
    Strike derivatives ANALYTIC from SVI (w_k, w_kk); time derivative by FORWARD difference at
    fixed k between consecutive slices (w(k, T_{i+1}) - w(k, T_i))/(T_{i+1} - T_i), the last slice
    uses the backward difference. Gatheral (2006) eq. 1.10 (Dupire in total variance, carry
    included through F_T):
        sigma_loc^2 = w_T / [1 - (k/w) w_k + 0.25(-0.25 - 1/w + k^2/w^2) w_k^2 + 0.5 w_kk].
    Flat surface (w = s^2 T) returns s exactly. ttm_grid: slice maturities to report (default
    all). Returns DataFrame[K, T, moneyness (K/S0), log_moneyness, local_vol, implied_vol,
    local_var, w_T, denominator] (local_vol NaN where local_var <= 0)."""
    from .helpers75_3 import svi_raw, svi_derivatives
    Ts = sorted(float(t) for t in params_by_T)
    pk = {float(t): tuple(v) for t, v in params_by_T.items()}
    rows = []
    for i, T in enumerate(Ts if ttm_grid is None else [t for t in Ts if t in set(map(float, ttm_grid))]):
        i = Ts.index(T)
        F = S0 * math.exp((r - q) * T)
        K = np.asarray(strike_grid, float)
        k = np.log(K / F)
        w, wk, wkk = svi_derivatives(k, *pk[T])
        if len(Ts) == 1:
            wT = w / T
        elif i + 1 < len(Ts):
            T2 = Ts[i + 1]
            wT = (svi_raw(k, *pk[T2]) - w) / (T2 - T)
        else:
            T0 = Ts[i - 1]
            wT = (w - svi_raw(k, *pk[T0])) / (T - T0)
        den = 1 - k / w * wk + 0.25 * (-0.25 - 1 / w + k * k / (w * w)) * wk ** 2 + 0.5 * wkk
        lvar = wT / den
        for j in range(len(K)):
            rows.append({"K": K[j], "T": T, "moneyness": K[j] / S0, "log_moneyness": k[j],
                         "local_vol": math.sqrt(lvar[j]) if lvar[j] > 0 else float("nan"),
                         "implied_vol": math.sqrt(max(w[j], 0) / T), "local_var": lvar[j],
                         "w_T": wT[j], "denominator": den[j]})
    return pd.DataFrame(rows)


# =============================================================================================
# 10. two-factor Heston tables (prices, IV, puts, parity, Dupire)
# =============================================================================================

def heston2_surface_tables(S0, r, q, factors, K, T, tol=1e-5, n_nodes=2000, gl_upper=100.0,
                           quad_bounds=(1e-8, 50.0), quad_limit=200, iv_bounds=(1e-4, 5.0),
                           iv_xtol=1e-8, den_min=1e-15, var_floor=1e-10, kk_scheme="three_point") -> dict:
    """Every table of a two-factor Heston (Christoffersen-Heston-Jacobs) surface study.

    factors: list of dicts {kappa, theta, sigma, rho, v0} (aliases nu0/v_0/var0, xi/eta for sigma)
    or (kappa, theta, sigma, rho, v0) tuples. K, T: 1-d grids (ascending).
    1. Method comparison at T[0]: price_quad = heston2_price(method='quad', bounds quad_bounds,
       limit), price_gl = Gauss-Legendre n_nodes on [0, gl_upper]. One pairwise difference cannot
       rank two methods, so: chosen = 'gauss_legendre' when every |diff| < tol; otherwise the
       method whose prices are arbitrage-consistent (>= 0, non-increasing in K), Gauss-Legendre
       preferred when both are (the truncated quad integral is the one that breaks deep OTM).
    2. Calls C[K_i, T_j] with the chosen method; IV by brentq on BS in iv_bounds (xtol).
    3. Direct puts (CZ put formula) and parity puts C - S0 e^{-qT} + K e^{-rT}; abs diff.
    4. Dupire on C(K, T): sigma^2 = (dC/dT + (r - q) K dC/dK + q C) / (0.5 K^2 d2C/dK2)
       (q = 0: (C_T + r K C_K)/(0.5 K^2 C_KK)); C_T and C_K by np.gradient edge_order=1 (central
       inside, one-sided at the ends); C_KK by fd_second_derivative(kk_scheme): default 'three_point'
       = (C_{i+1} - 2 C_i + C_{i-1})/h^2, the central second difference, undefined at K[0] / K[-1]
       (hence the interior strikes); 'gradient2' = np.gradient twice (2h stencil - know78: it bent the
       interior-maturity local vols too low). NaN where the denominator < den_min, else local_var
       floored at var_floor; reported on the INTERIOR strikes K[1:-1] x all T.
    Matrix tables have rows = strikes ('K' first column) and one column per maturity (header =
    the T value). Returns {comparison DataFrame[K, T, price_quad, price_gl, abs_diff] (+ the caller
    appends the metadata row), chosen_method, calls, iv, puts, parity DataFrame[K, T, put_direct,
    put_parity, abs_diff], local_vol (matrices as DataFrames), C, IV, P, LV arrays, summary dict
    (chosen_method, iv_min, iv_max, iv_mean, atm_iv_T0p5, skew_T0p083, skew_T1p0, num_nan,
    put_parity_max_diff, local_vol_atm_T0p5, local_vol_range)}."""
    from .helpers75_3 import heston2_price, implied_vol_vec
    fs = [_heston_factor(f) for f in factors]
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    pq = heston2_price(K, T[0], S0, r, q, fs, kind="call", method="quad", quad_lower=quad_bounds[0],
                       quad_upper=quad_bounds[1], quad_limit=quad_limit)
    pg = heston2_price(K, T[0], S0, r, q, fs, kind="call", method="gauss_legendre", n_nodes=n_nodes, upper=gl_upper)
    diff = np.abs(np.asarray(pq) - np.asarray(pg))
    if np.all(diff < tol):
        chosen = "gauss_legendre"
    else:
        def sane(p):
            p = np.asarray(p)
            return bool(np.all(p >= -1e-10) and np.all(np.diff(p) <= 1e-10))
        chosen = "gauss_legendre" if sane(pg) or not sane(pq) else "scipy_quad"
    meth = "gauss_legendre" if chosen == "gauss_legendre" else "quad"
    kw = dict(n_nodes=n_nodes, upper=gl_upper) if meth == "gauss_legendre" else dict(
        quad_lower=quad_bounds[0], quad_upper=quad_bounds[1], quad_limit=quad_limit)
    C = np.column_stack([heston2_price(K, t, S0, r, q, fs, kind="call", method=meth, **kw) for t in T])
    P = np.column_stack([heston2_price(K, t, S0, r, q, fs, kind="put", method=meth, **kw) for t in T])
    IV = np.asarray(implied_vol_vec(C, S0, K[:, None], T[None, :], r, q, kind="call", lo=iv_bounds[0],
                                    hi=iv_bounds[1], xtol=iv_xtol), float)
    Ppar = C - S0 * np.exp(-q * T)[None, :] + K[:, None] * np.exp(-r * T)[None, :]
    pdiff = np.abs(P - Ppar)
    dT = np.gradient(C, T, axis=1, edge_order=1)
    dK = np.gradient(C, K, axis=0, edge_order=1)
    from .helpers79_2 import fd_second_derivative
    dKK = fd_second_derivative(C, K, axis=0, scheme=kk_scheme)
    num = dT + (r - q) * K[:, None] * dK + q * C
    den = 0.5 * K[:, None] ** 2 * dKK
    with np.errstate(divide="ignore", invalid="ignore"):
        lvar = num / den
    LV = np.where(den < den_min, np.nan, np.sqrt(np.maximum(lvar, var_floor)))[1:-1, :]

    def mat(A, ks):
        d = pd.DataFrame(A, columns=[float(t) for t in T])
        d.insert(0, "K", ks)
        return d
    it = int(np.argmin(np.abs(T - 0.5)))
    ia = int(np.argmin(np.abs(K - S0)))
    Ki = K[1:-1]
    ia2 = int(np.argmin(np.abs(Ki - S0)))
    summ = {"chosen_method": chosen, "iv_min": float(np.nanmin(IV)), "iv_max": float(np.nanmax(IV)),
            "iv_mean": float(np.nanmean(IV)), "atm_iv_T0p5": float(IV[ia, it]),
            "skew_T0p083": float(IV[0, 0] - IV[-1, 0]), "skew_T1p0": float(IV[0, -1] - IV[-1, -1]),
            "num_nan": int(np.isnan(IV).sum()), "put_parity_max_diff": float(pdiff.max()),
            "local_vol_atm_T0p5": float(LV[ia2, it]),
            "local_vol_range": [float(np.nanmin(LV)), float(np.nanmax(LV))]}
    comp = pd.DataFrame({"K": K, "T": T[0], "price_quad": pq, "price_gl": pg, "abs_diff": diff})
    parity = pd.DataFrame({"K": np.repeat(K, len(T)), "T": np.tile(T, len(K)), "put_direct": P.ravel(),
                           "put_parity": Ppar.ravel(), "abs_diff": pdiff.ravel()})
    return {"comparison": comp, "chosen_method": chosen, "calls": mat(C, K), "iv": mat(IV, K),
            "puts": mat(P, K), "parity": parity, "local_vol": mat(LV, Ki), "C": C, "IV": IV, "P": P,
            "LV": LV, "summary": summ}


def _heston_factor(f):
    if isinstance(f, dict):
        lo = {str(k).lower(): v for k, v in f.items()}

        def g(*names):
            for n in names:
                if n in lo:
                    return float(lo[n])
            raise KeyError(f"factor {f!r} lacks one of {names}")
        return (g("kappa"), g("theta"), g("sigma", "xi", "eta", "vol_of_vol"), g("rho"),
                g("v0", "nu0", "v_0", "nu_0", "var0", "variance0"))
    return tuple(float(x) for x in f)


# =============================================================================================
# 11. xccy quote books: tolerant curve selection and log-linear FX forward nodes
# =============================================================================================

def select_curve_quotes(df, curve, instrument_type=None, curve_col="curve", type_col="instrument_type"):
    """Rows of a quote book for a curve label matched IGNORING case, '_', '-', spaces
    ('USD_LIBOR_3M' finds 'USD_LIBOR3M'); instrument_type the same way (None = all types).
    Raises KeyError listing the real labels when nothing matches (never returns an empty frame
    silently - an empty frame later surfaces as StopIteration/IndexError far from the cause)."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
    m = df[curve_col].map(norm) == norm(curve)
    if instrument_type is not None:
        m &= df[type_col].map(norm) == norm(instrument_type)
    out = df[m]
    if out.empty:
        raise KeyError(f"no rows for curve={curve!r}, instrument_type={instrument_type!r}; curves: "
                       f"{sorted(df[curve_col].astype(str).unique())}; types: {sorted(df[type_col].astype(str).unique())}")
    return out.copy()


def fx_forward_nodes(spot, spot_date, maturities, points, pip_divisor):
    """Quoted outright nodes F_i = spot + points_i / pip_divisor (points in pips), with the spot
    itself as the node at spot_date. Returns a function f(date) -> forward by LINEAR interpolation
    of ln F in calendar days from spot_date (spot -> first node covers broken dates inside the
    first tenor), flat ln F beyond the last node, and the node table DataFrame[maturity_date,
    days, forward_fx, forward_points_pips]."""
    sd = pd.Timestamp(spot_date).normalize()
    mats = [pd.Timestamp(m).normalize() for m in maturities]
    F = [float(spot) + float(p) / float(pip_divisor) for p in points]
    order = np.argsort([(m - sd).days for m in mats])
    days = np.array([0] + [(mats[i] - sd).days for i in order], float)
    lnF = np.log(np.array([float(spot)] + [F[i] for i in order]))

    def f(d):
        x = (pd.Timestamp(d).normalize() - sd).days
        return float(np.exp(np.interp(x, days, lnF)))
    tab = pd.DataFrame({"maturity_date": [mats[i] for i in order], "days": days[1:],
                        "forward_fx": [F[i] for i in order], "forward_points_pips": [float(points[i]) for i in order]})
    return f, tab
