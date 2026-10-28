"""know75 group-4 helpers: execution/TCA, intraday volume profiles, OFI-PCA signals, FX desk dates
and crosses, xccy quote books, crossover backtests, discrete delta hedging, MC Greeks, compound
option grids, par bootstrapping with zero interpolation, rating-migration counting, and a Polars
0.x -> 1.x source rewriter.

Every function is general (no task names), pure numpy/pandas/scipy, and states its formula and
conventions in the docstring. Numbers a task fixes (window, half-life, bumps, horizons) are
REQUIRED arguments - pass the task's own values. Self-tests: proven_work/fail72/selftest75_4.py.

API lines (one per function):
     epoch_ms_int(x) -> int64 epoch ms from datetime64 / Timestamp / int (undoes INPUTS auto-parsing)
     first_quote_at_or_after(times_ms, t_ms) -> row position of the first quote with time >= t (-1 if none)
     participation_bucket_fills(quotes, trades, start_ms, end_ms, bucket_ms, target_qty, participation_cap, max_child_qty, side="buy", realized_horizon_ms=0) -> dict{fills DataFrame, metrics, intermediates}
     write_fixed_csv(df, path, decimals=6) -> floats '%.6f', ints plain, None/NaN/'' empty, bools true/false
     volume_share_profile(window_shares, model, half_life=None, winsor_pct=None) -> profile summing to 1; rows OLDEST->NEWEST, ewma lag 0 = LAST row
     share_r2(realized, predicted) -> 1 - SSE/SST (zero SST: 1.0 if equal else 0.0)
     walk_forward_profile_scores(shares, window, models, half_life=None, winsor_pct=None) -> dict{model: daily R2 array}, test day i uses rows i-window..i-1
     largest_remainder_allocation(quantity, weights) -> int array summing to quantity, ties to the EARLIER bar
     nan_safe_ic(pred, actual) -> (spearman, pearson) over pairs where BOTH are finite
     rolling_pc1_ols_forecast(X, y, window, sign_ref_col=0) -> dict{pc1, pred} arrays (NaN where not computed); train rows t-window..t-1, predict row t
     fx_spot_date(trade_date, lag_days, holidays_by_ccy, ccys) | fx_add_tenor(spot_date, tenor, holidays_by_ccy, ccys) "6W"/"3M"/"1Y", following
     fx_swap_roll(position, spot_bid, spot_offer, near_pts_bid, near_pts_offer, far_pts_bid, far_pts_offer, multiplier) -> dict{side_used, near_outright, far_outright, swap_points, swap_points_pips}
     implied_foreign_yield(spot_mid, fwd_bid, fwd_offer, usd_bid, usd_offer, t_usd, t_foreign, usd_is_base) -> (bid, offer) DECIMAL
     fx_smile_25d(atm, rr, bf) -> (vol_25d_call, vol_25d_put) | fx_strike_from_delta(S, T, rd, rf, sigma, delta, kind, convention) GK strike
     cross_via_usd(target, quotes, mid="product") -> (bid, ask, mid); quotes {pair: (bid, ask)}
     normalize_quote_book(df, key_cols, time_col="quote_time", unit_col="quote_unit", value_col="quote_value", pip_multiplier=1e4) -> dict{cleaned, full}
     loglinear_interp(times, values, t) -> exp(linear interp of ln value) (DFs: linear in -ln DF; FX: linear in ln F), flat beyond the ends
     crossover_backtest(close, open_, fast, slow, dates=None, initial_capital=..., periods_per_year=252, ticker=None) -> dict{trades, daily, metrics}; golden/death CROSS events, next-open fills, forced close at last close
     escrowed_dividend_delta_hedge(S, iv, K, r, maturity_days, days_per_year, freq, div_days, div_amounts, buy_rate, sell_rate) -> dict{t0, terminal, simulation}
     mc_greeks_suite(S0, K, r, q, sigma, T, n_paths, seed, n_avg, h_spot, h_vol, h_time, h_rate) -> dict{prices, greeks[option][greek][method]={value,se}, variance_ratio}
     bs_delta_vega_surface(spots, vols, K, T, r, q=0.0) -> DataFrame[S0, sigma, delta_call_eu, delta_put_eu, vega_call_eu, vega_put_eu] spot-major
     geske_compound_grid(S0, K1_list, moneyness, T1, T2, r, sigma, q=0.0, kinds=("call_on_call","put_on_put"), n_paths=..., seed=None) -> DataFrame[type "call-on-call", K1, K2, K2_moneyness, T1, T2, S_star, compound_price, mc_price, mc_std_err]
     mc_agreement(price, mc_price, se, z=3.0) -> bool array |price - mc| <= z se
     bootstrap_par_linear_zero(maturities, par_rates, freq) -> dict{dfs, zeros, zero_fn}; coupon dates between maturities by LINEAR ZERO interpolation (solved)
     forward_from_zero(zero_fn, T, tenor=1.0, first_maturity=None) -> (z(T)T - z(T-tenor)(T-tenor))/tenor
     read_delimited_matrix(path_or_text) -> DataFrame, delimiter sniffed (';' ',' tab '|'), first column as index
     collapse_rating_counts(raw, mapping, states) -> DataFrame counts on the target scale (mapping value None = censored, dropped)
     row_stochastic(C, empty="identity") | chi2_homogeneity(table, min_col_total=5) -> (stat, dof, p) | generator_logm_clip(P) -> valid generator
     polars_0x_to_1x_source(src, new_name=None) -> (new_src, todo list) | check_python_source(src) -> raises SyntaxError with the line
"""
from __future__ import annotations

import ast
import csv
import io
import math
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

#: the API lines above, for appending to qf_toolkit.API (toolkit_hints.block only shows names
#: that appear in API): API += "\n[know75 group 4]\n" + helpers75_4.API_LINES
API_LINES = (__doc__ or "").split("API lines (one per function):", 1)[-1].strip("\n").rstrip('"\n ')


# =============================================================================================
# 1. execution / TCA
# =============================================================================================

def epoch_ms_int(x):
    """Integer epoch milliseconds (UTC) from datetime64 / Timestamp / tz-aware stamps / ints.

    The agent's INPUTS loader converts integer columns whose name says 'time' into datetime64;
    comparing those with integer *_ms parameters raises 'Invalid comparison between
    dtype=datetime64[ms] and int'. Call this on every time column first. Ints pass through.
    Series in -> int64 Series (same index); array/list -> int64 ndarray; scalar -> int.
    """
    if isinstance(x, pd.Series):
        s = x
        if pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            return s.astype("datetime64[ms]").astype("int64")
        return pd.to_numeric(s).astype("int64")
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return int(ts.value // 1_000_000)
    if np.isscalar(x):
        return int(x)
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ms]").astype("int64")
    return arr.astype("int64")


def first_quote_at_or_after(times_ms, t_ms) -> int:
    """Position of the FIRST quote whose time >= t_ms in an ascending time array (searchsorted
    'left'); -1 when every quote is earlier. For the 'prevailing' quote (last at or before t)
    use searchsorted(t, 'right') - 1 instead."""
    arr = np.asarray(epoch_ms_int(np.asarray(times_ms)))
    i = int(np.searchsorted(arr, int(t_ms), side="left"))
    return i if i < arr.size else -1


def participation_bucket_fills(quotes, trades, start_ms, end_ms, bucket_ms, target_qty,
                               participation_cap, max_child_qty, side="buy",
                               realized_horizon_ms=0, *, quote_time_col="transaction_time",
                               quote_event_col="event_time", quote_id_col="update_id",
                               bid_col="best_bid_price", ask_col="best_ask_price",
                               trade_time_col="transact_time", trade_price_col="price",
                               trade_qty_col="quantity", trade_id_col="agg_trade_id",
                               header=None) -> dict:
    """Participation-capped (POV) bucket schedule reconstructed from trades and top-of-book quotes.

    Buckets b = 0..B-1, B = ceil((end - start)/bucket_ms), window [start + b*dt, start + (b+1)*dt)
    (half-open on trade time). Eligible trades: bucket_start <= trade_time < bucket_end.
      participation_cap_qty = cap * eligible_trade_qty
      child = min(remaining_before, participation_cap_qty, max_child_qty)  (0 when no trades)
      fill at the touch of the FIRST quote with quote_time >= bucket_start (buy: ask, sell: bid)
      remaining_after = remaining_before - fill
    Arrival quote = first quote at/after start_ms; future quote = first at/after end_ms + horizon.
    s = +1 buy / -1 sell, P = avg fill, M = arrival mid, M_f = future mid:
      implementation_shortfall_bps = s 1e4 (P - M)/M ; effective_spread_bps = s 2e4 (P - M)/M
      realized_spread_bps = s 2e4 (P - M_f)/M ; temporary_impact_bps = s 1e4 (M_f - M)/M
      market_vwap = sum(p q)/sum(q) over eligible trades in [start, end).
    bucket_count_filled = buckets with fill_qty > 0; bucket_count_empty = eligible count == 0.
    Empty / zero-child rows keep zeros for counts/quantities and '' for ids, times and prices.
    All times are int64 epoch ms (datetime64 inputs are converted back).

    Returns dict:
      fills: DataFrame in the conventional column order (bucket_index ... is_empty_bucket)
      metrics: ordered dict (header first, then start_time_ms ... bucket_count_empty); None where
               a ratio has a zero denominator
      intermediates: counts/volumes/notional and the quote-link timestamps.
    """
    s = 1 if str(side).lower() in ("buy", "b", "long", "1", "+1") else -1
    q = quotes.copy()
    t = trades.copy()
    # every datetime64 column goes back to int64 ms - the loader also parses ids whose NAME
    # contains 'date'/'time' (update_id holds 'date') when their magnitude looks like epoch ms
    for frame in (q, t):
        for col_name in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[col_name]):
                frame[col_name] = epoch_ms_int(frame[col_name])
    q[quote_time_col] = epoch_ms_int(q[quote_time_col])
    t[trade_time_col] = epoch_ms_int(t[trade_time_col])
    # know78: many quotes share one millisecond; the exchange sequence id orders them. Sorting on time
    # alone kept whatever order the caller left (a default quicksort scrambles ties), so 'the first
    # quote at or after the bucket start' picked a later same-ms update_id. Sort on (time, id).
    from .helpers78_3 import event_order
    q = event_order(q, quote_time_col, quote_id_col)
    t = event_order(t, trade_time_col, trade_id_col)
    start_ms, end_ms, bucket_ms = int(start_ms), int(end_ms), int(bucket_ms)
    qt = q[quote_time_col].to_numpy()

    def quote_at(ts):
        i = first_quote_at_or_after(qt, ts)
        if i < 0:
            return None
        row = q.iloc[i]
        return {"id": int(row[quote_id_col]), "bid": float(row[bid_col]), "ask": float(row[ask_col]),
                "tt": int(row[quote_time_col]),
                "et": int(row[quote_event_col]) if quote_event_col in q.columns else int(row[quote_time_col])}

    n_b = int(math.ceil((end_ms - start_ms) / bucket_ms))
    tt = t[trade_time_col].to_numpy()
    in_win = (tt >= start_ms) & (tt < end_ms)
    tw = t[in_win]
    remaining = float(target_qty)
    rows = []
    executed, notional_fill = 0.0, 0.0
    fill_quote_times = []
    for b in range(n_b):
        b0 = start_ms + b * bucket_ms
        b1 = b0 + bucket_ms
        m = (tw[trade_time_col] >= b0) & (tw[trade_time_col] < b1)
        e = tw[m]
        cnt = int(len(e))
        vol = float(e[trade_qty_col].sum()) if cnt else 0.0
        notl = float((e[trade_price_col] * e[trade_qty_col]).sum()) if cnt else 0.0
        cap_qty = float(participation_cap) * vol
        child = min(remaining, cap_qty, float(max_child_qty)) if cnt else 0.0
        child = max(child, 0.0)
        row = {"bucket_index": b, "bucket_start_ms": b0, "bucket_end_ms": b1,
               "eligible_trade_window_start_ms": b0, "eligible_trade_window_end_ms": b1,
               "eligible_trade_count": cnt, "eligible_trade_qty": vol, "eligible_trade_notional": notl,
               "eligible_first_trade_id": int(e[trade_id_col].iloc[0]) if cnt else "",
               "eligible_last_trade_id": int(e[trade_id_col].iloc[-1]) if cnt else "",
               "eligible_first_trade_time_ms": int(e[trade_time_col].iloc[0]) if cnt else "",
               "eligible_last_trade_time_ms": int(e[trade_time_col].iloc[-1]) if cnt else "",
               "participation_cap_qty": cap_qty, "max_child_qty": float(max_child_qty),
               "child_order_qty": child, "fill_qty": 0.0, "fill_price": "", "reference_bid": "",
               "reference_ask": "", "fill_quote_update_id": "", "fill_quote_transaction_time_ms": "",
               "fill_quote_event_time_ms": ""}
        if child > 0:
            qq = quote_at(b0)
            if qq is None:
                raise ValueError(f"no quote at or after bucket start {b0}")
            px = qq["ask"] if s > 0 else qq["bid"]
            row.update({"fill_qty": child, "fill_price": px, "reference_bid": qq["bid"],
                        "reference_ask": qq["ask"], "fill_quote_update_id": qq["id"],
                        "fill_quote_transaction_time_ms": qq["tt"], "fill_quote_event_time_ms": qq["et"]})
            remaining -= child
            if remaining < 1e-12:
                remaining = 0.0
            executed += child
            notional_fill += child * px
            fill_quote_times.append(qq["tt"])
        row["remaining_qty"] = remaining
        row["is_empty_bucket"] = cnt == 0
        rows.append(row)
    fills = pd.DataFrame(rows)
    arr, fut = quote_at(start_ms), quote_at(end_ms + int(realized_horizon_ms))
    if arr is None or fut is None:
        raise ValueError("no arrival or future reference quote at or after the reference time")
    m0 = 0.5 * (arr["bid"] + arr["ask"])
    mf = 0.5 * (fut["bid"] + fut["ask"])
    avg = notional_fill / executed if executed > 0 else None
    mvol = float(tw[trade_qty_col].sum())
    mvwap = float((tw[trade_price_col] * tw[trade_qty_col]).sum()) / mvol if mvol > 0 else None

    def bps(k, a, ref):
        return None if a is None else s * k * (a - ref) / m0

    metrics = dict(header or {})
    metrics.update({
        "start_time_ms": start_ms, "end_time_ms": end_ms, "bucket_ms": bucket_ms,
        "target_qty": float(target_qty), "executed_qty": executed,
        "unfilled_qty": float(target_qty) - executed,
        "fill_rate": executed / float(target_qty) if float(target_qty) else None,
        "arrival_mid": m0, "arrival_quote_update_id": arr["id"],
        "arrival_quote_transaction_time_ms": arr["tt"], "arrival_quote_event_time_ms": arr["et"],
        "future_reference_time_ms": end_ms + int(realized_horizon_ms), "future_mid": mf,
        "future_quote_update_id": fut["id"], "future_quote_transaction_time_ms": fut["tt"],
        "future_quote_event_time_ms": fut["et"], "avg_fill_price": avg, "market_vwap": mvwap,
        "implementation_shortfall_bps": bps(1e4, avg, m0), "effective_spread_bps": bps(2e4, avg, m0),
        "realized_spread_bps": bps(2e4, avg, mf), "temporary_impact_bps": s * 1e4 * (mf - m0) / m0,
        "market_trade_count": int(len(tw)), "market_volume": mvol,
        "bucket_count_total": n_b, "bucket_count_filled": int((fills["fill_qty"] > 0).sum()),
        "bucket_count_empty": int((fills["eligible_trade_count"] == 0).sum()),
    })
    inter = {
        "bucket_count_total": n_b, "bucket_count_filled": metrics["bucket_count_filled"],
        "bucket_count_empty": metrics["bucket_count_empty"],
        "eligible_trade_count": int(fills["eligible_trade_count"].sum()),
        "eligible_trade_volume": float(fills["eligible_trade_qty"].sum()),
        "eligible_trade_notional": float(fills["eligible_trade_notional"].sum()),
        "first_fill_quote_transaction_time_ms": fill_quote_times[0] if fill_quote_times else None,
        "last_fill_quote_transaction_time_ms": fill_quote_times[-1] if fill_quote_times else None,
        "arrival_quote_transaction_time_ms": arr["tt"], "arrival_quote_event_time_ms": arr["et"],
        "future_quote_transaction_time_ms": fut["tt"], "future_quote_event_time_ms": fut["et"],
    }
    # know76: the model indexed result['executed_qty'], result['bucket_count_total'] and
    # result['bucket_4'] directly (KeyError x7); every metric / intermediate and a 'bucket_<i>'
    # row dict are now ALSO top-level keys. fills / metrics / intermediates are unchanged.
    out = {"fills": fills, "metrics": metrics, "intermediates": inter}
    for _k, _v in list(metrics.items()) + list(inter.items()):
        out.setdefault(_k, _v)
    for _i, _row in enumerate(rows):
        out[f"bucket_{_i}"] = dict(_row)
    return out


def write_fixed_csv(df, path, decimals=6):
    """Write a CSV by hand: Python/numpy floats as fixed '%.{decimals}f', ints unchanged,
    bools lowercase true/false, None/NaN/'' as empty fields; column order as in df.
    Returns the path."""
    def cell(v):
        if v is None or (isinstance(v, str) and v == ""):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return "true" if bool(v) else "false"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.{decimals}f}"
        return str(v)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(df.columns))
        for rec in df.itertuples(index=False, name=None):
            w.writerow([cell(v) for v in rec])
    return path


# =============================================================================================
# 2. intraday volume profiles and schedules
# =============================================================================================

def volume_share_profile(window_shares, model, half_life=None, winsor_pct=None):
    """Full-day volume-share profile from a window of daily share vectors -> 1-D array, sum 1.

    window_shares: (W, T) rows in CHRONOLOGICAL order (oldest first, most recent LAST).
    model: 'mean' | 'median' | 'ewma' | 'winsorized' (aliases historical_mean_profile ...).
      ewma: lag L = 0 for the LAST row (most recent, weight 1) ... W-1 for the first;
            w_L = exp(-ln 2 * L / half_life); shat = sum w_L s_L / sum w_L  (half_life required)
      winsorized: per bar (column) clip the W values to [P_lo, P_hi] (numpy linear percentiles,
            winsor_pct=(lo, hi) in percent, required) then mean.
    The result is renormalised to sum to 1.
    """
    S = np.asarray(window_shares, dtype=float)
    m = str(model).lower().replace("historical_", "").replace("_profile", "").replace("_mean", "")
    if m in ("mean",):
        p = S.mean(axis=0)
    elif m == "median":
        p = np.median(S, axis=0)
    elif m == "ewma":
        if half_life is None:
            raise ValueError("ewma needs half_life")
        W = S.shape[0]
        lags = np.arange(W)[::-1]          # row i (chronological) has lag W-1-i
        w = np.exp(-np.log(2.0) * lags / float(half_life))
        p = (w[:, None] * S).sum(axis=0) / w.sum()
    elif m in ("winsorized", "winsorised", "winsor"):
        if winsor_pct is None:
            raise ValueError("winsorized needs winsor_pct=(lo, hi) in percent")
        lo = np.percentile(S, winsor_pct[0], axis=0)
        hi = np.percentile(S, winsor_pct[1], axis=0)
        p = np.clip(S, lo, hi).mean(axis=0)
    else:
        raise ValueError(f"unknown profile model {model!r}")
    return p / p.sum()


def share_r2(realized, predicted) -> float:
    """R^2 = 1 - sum (s - shat)^2 / sum (s - mean s)^2 on share vectors; zero denominator ->
    1.0 if predicted equals realized exactly, else 0.0."""
    s = np.asarray(realized, dtype=float)
    p = np.asarray(predicted, dtype=float)
    sst = float(((s - s.mean()) ** 2).sum())
    if sst == 0.0:
        return 1.0 if np.array_equal(s, p) else 0.0
    return 1.0 - float(((s - p) ** 2).sum()) / sst


def walk_forward_profile_scores(shares, window, models, half_life=None, winsor_pct=None) -> dict:
    """Rolling walk-forward: for every day index i >= window (0-based, i.e. from the
    (window+1)-th complete day) estimate each profile on rows i-window..i-1 and score it on row i
    with share_r2. Returns {model: np.array of daily R2} (same order for every model)."""
    S = np.asarray(shares, dtype=float)
    out = {m: [] for m in models}
    for i in range(int(window), S.shape[0]):
        est = S[i - int(window):i]
        for m in models:
            out[m].append(share_r2(S[i], volume_share_profile(est, m, half_life, winsor_pct)))
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def largest_remainder_allocation(quantity, weights) -> np.ndarray:
    """Integer allocation of `quantity` proportional to weights (renormalised to sum 1):
    floor(Q w), then the leftover units go one each to the largest fractional remainders; ties
    go to the EARLIER position. Sum equals quantity exactly."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = int(quantity) * w
    base = np.floor(raw).astype(np.int64)
    left = int(quantity) - int(base.sum())
    frac = raw - base
    order = sorted(range(len(w)), key=lambda i: (-frac[i], i))
    for i in order[:left]:
        base[i] += 1
    return base


# =============================================================================================
# 3. OFI / PCA rolling signal
# =============================================================================================

def nan_safe_ic(pred, actual):
    """(spearman, pearson) correlation over the pairs where BOTH values are finite (a forward
    return is NaN on the last row; np.corrcoef with a NaN returns NaN -> JSON null)."""
    a = np.asarray(pred, dtype=float)
    b = np.asarray(actual, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return float("nan"), float("nan")
    return float(stats.spearmanr(a[m], b[m])[0]), float(stats.pearsonr(a[m], b[m])[0])


def rolling_pc1_ols_forecast(X, y, window, sign_ref_col=0) -> dict:
    """Walk-forward PCA(1) + OLS forecast without look-ahead.

    For each t >= window: train rows t-window..t-1 (skipped when any y in them is NaN);
    z = (X - mean)/std (ddof=0, zero std -> 1); PC1 = first right-singular vector of the
    centred z (as sklearn PCA(n_components=1)); sign flipped so corr(PC1_train, X_train[:, ref]) >= 0;
    OLS y_train = a + b PC1_train; test row t standardised with the TRAIN mean/std:
    pc1[t] = (z_t - mean(z_train)) . v * sign, pred[t] = a + b pc1[t].
    Returns dict{pc1, pred, beta (n, 2)} as length-n arrays with NaN where not computed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    pc1 = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    beta = np.full((n, 2), np.nan)
    W = int(window)
    for t in range(W, n):
        Xt, yt = X[t - W:t], y[t - W:t]
        if not np.all(np.isfinite(yt)):
            continue
        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = (Xt - mu) / sd
        zc = Z.mean(axis=0)
        _, _, vt = np.linalg.svd(Z - zc, full_matrices=False)
        v = vt[0]
        score = (Z - zc) @ v
        c = np.corrcoef(score, Xt[:, sign_ref_col])[0, 1]
        sgn = -1.0 if (np.isfinite(c) and c < 0) else 1.0
        score = score * sgn
        A = np.column_stack([np.ones(W), score])
        coef = np.linalg.lstsq(A, yt, rcond=None)[0]
        zt = (X[t] - mu) / sd
        pc1[t] = float((zt - zc) @ v) * sgn
        pred[t] = coef[0] + coef[1] * pc1[t]
        beta[t] = coef
    return {"pc1": pc1, "pred": pred, "beta": beta}


# =============================================================================================
# 4. FX desk: dates, rolls, implied yields, smile strikes, crosses
# =============================================================================================

def _is_bday(d, holidays_by_ccy, ccys):
    if d.weekday() >= 5:
        return False
    iso = d.strftime("%Y-%m-%d")
    for c in ccys:
        if iso in set(str(h)[:10] for h in (holidays_by_ccy or {}).get(c, [])):
            return False
    return True


def _following(d, holidays_by_ccy, ccys):
    while not _is_bday(d, holidays_by_ccy, ccys):
        d = d + pd.Timedelta(days=1)
    return d


def fx_spot_date(trade_date, lag_days, holidays_by_ccy, ccys) -> pd.Timestamp:
    """Spot date = trade date + `lag_days` business days, a business day being a weekday that
    is not a holiday of ANY currency in `ccys` (holidays_by_ccy {ccy: ['YYYY-MM-DD', ...]}).
    USD/CAD style T+1 pairs simply pass lag_days=1."""
    d = pd.Timestamp(trade_date).normalize()
    k = 0
    while k < int(lag_days):
        d = d + pd.Timedelta(days=1)
        if _is_bday(d, holidays_by_ccy, ccys):
            k += 1
    return d


def fx_add_tenor(spot_date, tenor, holidays_by_ccy, ccys) -> pd.Timestamp:
    """Delivery date for a tenor from the spot date: 'nD' days, 'nW' weeks (7n calendar days),
    'nM' months, 'nY' years (calendar months, day clipped to month end), then FOLLOWING
    business-day adjustment on the joint calendar of `ccys`."""
    m = re.fullmatch(r"\s*(\d+)\s*([DWMY])\s*", str(tenor).upper())
    if not m:
        raise ValueError(f"tenor {tenor!r} not like 6W / 3M / 1Y")
    n, u = int(m.group(1)), m.group(2)
    d = pd.Timestamp(spot_date).normalize()
    if u == "D":
        d = d + pd.Timedelta(days=n)
    elif u == "W":
        d = d + pd.Timedelta(days=7 * n)
    else:
        d = d + pd.DateOffset(months=n if u == "M" else 12 * n)
    return _following(d, holidays_by_ccy, ccys)


def fx_swap_roll(position, spot_bid, spot_offer, near_pts_bid, near_pts_offer, far_pts_bid,
                 far_pts_offer, multiplier) -> dict:
    """FX swap roll priced on ONE side chosen by the far leg: far leg short base forward
    ('short_base_forward', 'sell') -> bid on both legs; long base forward -> offer on both.
    outright = spot_side + points_side / multiplier; swap_points = far - near in QUOTE units,
    swap_points_pips = far_points - near_points (pips)."""
    p = str(position).lower()
    side = "bid" if ("short" in p or "sell" in p) else "offer"
    if side == "bid":
        s, n, f = spot_bid, near_pts_bid, far_pts_bid
    else:
        s, n, f = spot_offer, near_pts_offer, far_pts_offer
    near = float(s) + float(n) / float(multiplier)
    far = float(s) + float(f) / float(multiplier)
    return {"side_used": side, "near_outright": near, "far_outright": far,
            "swap_points": far - near, "swap_points_pips": float(f) - float(n)}


def implied_foreign_yield(spot_mid, fwd_bid, fwd_offer, usd_bid, usd_offer, t_usd, t_foreign,
                          usd_is_base) -> tuple:
    """Implied foreign money-market yields (simple, DECIMAL) from a two-way outright, spot MID.

    USD/XXX (usd_is_base=True):  F = S (1 + r_f t_f)/(1 + r_usd t_usd)
        r_f = ((F/S)(1 + r_usd t_usd) - 1)/t_f; bid uses fwd_bid with USD OFFER, offer uses
        fwd_offer with USD BID.
    XXX/USD (usd_is_base=False): F = S (1 + r_usd t_usd)/(1 + r_f t_f)
        r_f = ((S/F)(1 + r_usd t_usd) - 1)/t_f; bid uses fwd_offer with USD BID, offer uses
        fwd_bid with USD OFFER.
    Rates in decimals (percent/100); multiply by 100 for *_pct fields."""
    S = float(spot_mid)
    if usd_is_base:
        bid = ((fwd_bid / S) * (1 + usd_offer * t_usd) - 1) / t_foreign
        off = ((fwd_offer / S) * (1 + usd_bid * t_usd) - 1) / t_foreign
    else:
        bid = ((S / fwd_offer) * (1 + usd_bid * t_usd) - 1) / t_foreign
        off = ((S / fwd_bid) * (1 + usd_offer * t_usd) - 1) / t_foreign
    return float(bid), float(off)


def fx_smile_25d(atm, rr, bf) -> tuple:
    """Broker-quote 25-delta wing vols: call = ATM + BF + RR/2, put = ATM + BF - RR/2
    (RR = call vol - put vol, BF = mean wing vol - ATM)."""
    return float(atm + bf + 0.5 * rr), float(atm + bf - 0.5 * rr)


def _gk_delta(S, K, T, rd, rf, sigma, kind, convention):
    F = S * math.exp((rd - rf) * T)
    v = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    w = 1.0 if kind == "call" else -1.0
    conv = convention.lower()
    pa = ("premium" in conv) or conv.endswith("_pa")
    fwd = conv.startswith("forward") or conv.startswith("fwd")
    if pa:
        core = w * (K / F) * stats.norm.cdf(w * d2)
    else:
        core = w * stats.norm.cdf(w * d1)
    return core if fwd else math.exp(-rf * T) * core


def fx_strike_from_delta(S, T, rd, rf, sigma, delta, kind, convention) -> float:
    """Garman-Kohlhagen strike for a target delta (sign ignored; |delta| matched).

    convention: 'spot_unadjusted' e^{-rf T} w N(w d1) | 'forward_unadjusted' w N(w d1) |
    'spot_premium_adjusted' e^{-rf T} w (K/F) N(w d2) | 'forward_premium_adjusted' w (K/F) N(w d2),
    F = S e^{(rd - rf) T}, rates continuous. Root searched on the OTM side of F (call K > F,
    put K < F), where premium-adjusted call delta is monotone."""
    kind = "call" if str(kind).lower().startswith("c") else "put"
    F = S * math.exp((rd - rf) * T)
    span = math.exp(10 * sigma * math.sqrt(T))
    target = abs(float(delta))
    g = lambda K: abs(_gk_delta(S, K, T, rd, rf, sigma, kind, convention)) - target  # noqa: E731
    lo, hi = (F, F * span) if kind == "call" else (F / span, F)
    if g(lo) * g(hi) > 0:
        lo, hi = F / span, F * span
    return float(optimize.brentq(g, lo, hi, xtol=1e-14, maxiter=500))


def _pair(p):
    s = str(p).upper().replace("-", "/").replace(" ", "")
    if "/" in s:
        a, b = s.split("/")
    else:
        a, b = s[:3], s[3:]
    return a, b


def cross_via_usd(target, quotes, mid="product") -> tuple:
    """Two-way cross X/Y through USD from {pair: (bid, ask)} quotes (any orientation, 'EUR/USD'
    or 'EURUSD').  X/USD leg: direct, or 1/(USD/X) with bid/ask swapped (bid = 1/ask);
    USD/Y leg: direct, or 1/(Y/USD) swapped.  bid = XUSD_bid * USDY_bid,
    ask = XUSD_ask * USDY_ask (the dealer rule: sell X -> buy USD at bid, sell USD -> buy Y).
    mid='product': XUSD_mid * USDY_mid with each leg mid = (bid+ask)/2 of the QUOTED pair
    (inverted when the leg is inverted); mid='average': (bid + ask)/2 of the cross.
    Returns (bid, ask, mid).
    know76: quote values may be (bid, ask), [bid, ask, mid], {'bid','ask'|'offer'[,'mid']} dicts,
    or `quotes` may be a DataFrame with pair/bid/ask(offer) columns; '_comment'-style string
    entries are skipped (the know75 run crashed with float('bid') unpacking a dict)."""
    if isinstance(quotes, pd.DataFrame):
        cols = {str(c).lower(): c for c in quotes.columns}
        pc, bc = cols.get("pair"), cols.get("bid")
        ac = cols.get("ask") or cols.get("offer")
        quotes = {r[pc]: (r[bc], r[ac]) for _, r in quotes.iterrows()}
    book = {}
    for k, v in quotes.items():
        if isinstance(v, str) or str(k).startswith("_"):
            continue
        if isinstance(v, dict):
            lo = {str(x).lower(): y for x, y in v.items()}
            b, a = lo["bid"], lo.get("ask", lo.get("offer"))
        else:
            b, a = list(v)[:2]
        book[_pair(k)] = (float(b), float(a))
    x, y = _pair(target)

    def leg(base, quote):
        if base == quote:
            return 1.0, 1.0, 1.0
        if (base, quote) in book:
            b, a = book[(base, quote)]
            return b, a, 0.5 * (b + a)
        if (quote, base) in book:
            b, a = book[(quote, base)]
            return 1.0 / a, 1.0 / b, 1.0 / (0.5 * (b + a))
        raise KeyError(f"no quote for {base}/{quote} or {quote}/{base}; have {sorted(book)}")

    b1, a1, m1 = leg(x, "USD")
    b2, a2, m2 = leg("USD", y)
    bid, ask = b1 * b2, a1 * a2
    return bid, ask, (m1 * m2 if mid == "product" else 0.5 * (bid + ask))


# =============================================================================================
# 5. xccy quote book and log-linear interpolation
# =============================================================================================

_UNIT_MAP = {
    "rate_pct": ("rate_decimal", 1e-2), "pct": ("rate_decimal", 1e-2), "percent": ("rate_decimal", 1e-2),
    "rate_bps": ("rate_decimal", 1e-4), "bps": ("rate_decimal", 1e-4),
    "rate_decimal": ("rate_decimal", 1.0), "decimal": ("rate_decimal", 1.0),
    "spot_rate": ("fx_rate", 1.0), "fx_rate": ("fx_rate", 1.0), "spot": ("fx_rate", 1.0),
    "outright": ("fx_rate", 1.0),
}


def normalize_quote_book(df, key_cols, time_col="quote_time", unit_col="quote_unit",
                         value_col="quote_value", pip_multiplier=1e4) -> dict:
    """Deduplicate and normalise a mixed quote snapshot.

    1. keep the LATEST time_col per unique key (key_cols, e.g. curve, instrument_type, tenor,
       start_date, end_date, ccy_or_pair); times compared as UTC instants, original text kept;
    2. EXACT unit mapping (never substring tests - 'spot_rate' contains 'rate'):
       rate_pct -> rate_decimal (/100), rate_bps -> rate_decimal (/1e4), rate_decimal as is,
       forward_points_pips / pips -> fx_forward_increment (/pip_multiplier),
       spot_rate / fx_rate / outright -> fx_rate (as is); any other unit raises.
    Returns {"cleaned": DataFrame[time_col, *key_cols, normalized_unit, normalized_quote]
    sorted by curve, instrument_type, end_date, tenor when present, "full": cleaned plus every
    original column (adjustments such as quote_adjustment_bps stay separate, NOT baked in)}.
    know78: a CLEAN book (no time / unit column, e.g. tenor_label, maturity_date, par_rate) is not an
    error - a time_col / unit_col that is None or absent skips that step (rows keep file order, last
    duplicate wins; unit taken as rate_decimal), value_col None/absent -> the single numeric column
    among (quote_value, par_rate, rate, quote, value, mid), key_cols absent are ignored. Such files
    need no normalisation at all: pass them straight to the bootstrap helper."""
    d = df.copy()
    key_cols = [c for c in (key_cols or []) if c in d.columns]
    if value_col is None or value_col not in d.columns:
        cand = [c for c in ("quote_value", "par_rate", "rate", "quote", "value", "mid") if c in d.columns]
        if not cand:
            num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            cand = num[-1:]
        if not cand:
            raise KeyError(f"no quote value column among {list(d.columns)}")
        value_col = cand[0]
    if time_col is not None and time_col in d.columns:
        d["_t"] = pd.to_datetime(d[time_col], utc=True)
        d = d.sort_values("_t", kind="mergesort")
    else:
        d["_t"] = np.arange(len(d))
        time_col = "_row"
        d[time_col] = np.arange(len(d))
    if key_cols:
        d = d.groupby(list(key_cols), dropna=False, sort=False).tail(1).copy()
    if unit_col is None or unit_col not in d.columns:
        unit_col = "_unit"
        d[unit_col] = "rate_decimal"
    units, vals = [], []
    for u, v in zip(d[unit_col].astype(str).str.strip().str.lower(), d[value_col].astype(float)):
        if u in _UNIT_MAP:
            nu, f = _UNIT_MAP[u]
        elif u in ("forward_points_pips", "pips", "points_pips", "forward_points"):
            nu, f = "fx_forward_increment", 1.0 / float(pip_multiplier)
        else:
            raise ValueError(f"unknown quote unit {u!r}")
        units.append(nu)
        vals.append(v * f)
    d["normalized_unit"] = units
    d["normalized_quote"] = vals
    order = [c for c in ("curve", "instrument_type", "end_date", "tenor") if c in d.columns]
    if order:
        d = d.sort_values(order, kind="mergesort")
    d = d.drop(columns="_t").reset_index(drop=True)
    cleaned = d[[time_col, *[c for c in key_cols if c != time_col], "normalized_unit", "normalized_quote"]]
    return _QuoteBook({"cleaned": cleaned.reset_index(drop=True), "full": d}, key_cols)


class _QuoteBook(dict):
    """normalize_quote_book result; know78: book['10Y'] (a key-column value such as a tenor label) returns
    that row's normalized_quote instead of KeyError (a know77 program indexed the result by tenor)."""

    def __init__(self, data, key_cols):
        super().__init__(data)
        self._keys = list(key_cols)

    def __missing__(self, key):
        full = dict.__getitem__(self, "full")
        for c in self._keys:
            m = full[c].astype(str).str.strip() == str(key).strip()
            if int(m.sum()) == 1:
                return float(full.loc[m, "normalized_quote"].iloc[0])
        raise KeyError(f"{key!r}: keys are 'cleaned', 'full' or one {self._keys} value")


def loglinear_interp(times, values, t):
    """exp of the linear interpolation of ln(values) in time, flat (in ln) beyond the end nodes.
    Discount factors: 'linear in -ln DF' (= linear in ln DF, i.e. piecewise-flat forwards);
    FX forwards: 'linear in ln F'; projection 'linear in ln Q'. Put the spot/DF=1 node at the
    front yourself when it belongs to the curve. Scalar in -> float, array in -> ndarray."""
    x = np.asarray(times, dtype=float)
    lv = np.log(np.asarray(values, dtype=float))
    o = np.argsort(x)
    out = np.exp(np.interp(np.asarray(t, dtype=float), x[o], lv[o]))
    return float(out) if np.ndim(out) == 0 else out


# =============================================================================================
# 6. crossover backtest (event-based)
# =============================================================================================

def crossover_backtest(close, open_, fast, slow, dates=None, initial_capital=100000.0,
                       periods_per_year=252, ticker=None) -> dict:
    """Long-only MA CROSSOVER backtest, next-bar-open fills; fast/slow = MA arrays OR window ints (SMA); trades already carry 'ticker'.

    fast/slow: the two averages already computed by the caller (SMA rolling(window).mean() or
    EMA ewm(span, adjust=False).mean()), or plain ints = SMA windows on close; NaN bars never
    signal. dates (str / Timestamp / datetime64) come back as 'YYYY-MM-DD' strings. When ticker is
    given the trades frame ALREADY has a 'ticker' first column - do not insert it again.
      golden cross at i: fast[i-1] <= slow[i-1] and fast[i] > slow[i]   (BUY signal when flat)
      death cross at i:  fast[i-1] >= slow[i-1] and fast[i] < slow[i]   (SELL signal when long)
    A signal on bar i executes at open_[i+1]; buy shares = floor(cash / open); sell all.
    Signals on the last bar cannot execute; a position still open after the last bar is closed
    at close[-1] (forced close: signal_date = exec_date = last date, type SELL).
    Portfolio value = cash + shares * close; daily_return[0] = 0, then pct change;
    cumulative_return = cumprod(1 + r) starting at 1.0.
    Metrics: total_return = final/initial - 1; annualized_return = mean(r) * ppy;
    annualized_volatility = std(r, ddof=1) sqrt(ppy); sharpe = ann_ret/ann_vol;
    max_drawdown = max((runmax - cum)/runmax) POSITIVE; calmar = ann_ret/max_drawdown;
    num_trades = BUY count; win_rate = SELLs with pnl > 0 / num_trades;
    total_pnl = sum of SELL pnl = shares (sell price - buy price).
    Returns {trades: DataFrame[(ticker,) type BUY/SELL, signal_date, exec_date, price, shares,
    pnl (NaN on BUY)], daily: DataFrame[date, portfolio_value, daily_return, cumulative_return],
    metrics: dict incl. num_trading_days}.
    """
    c = np.asarray(close, dtype=float)
    o = np.asarray(open_, dtype=float)
    n = c.size
    # know76: fast=50, slow=200 (window LENGTHS) crashed with 'invalid index to scalar variable';
    # a scalar is now read as a window and the SMA close.rolling(window).mean() is built here.
    if np.ndim(fast) == 0:
        fast = pd.Series(c).rolling(int(fast)).mean().to_numpy()
    if np.ndim(slow) == 0:
        slow = pd.Series(c).rolling(int(slow)).mean().to_numpy()
    f = np.asarray(fast, dtype=float)
    s = np.asarray(slow, dtype=float)
    if dates is not None:
        import datetime as _dt
        dates = [pd.Timestamp(d).strftime("%Y-%m-%d") if isinstance(d, (pd.Timestamp, np.datetime64, _dt.date)) else d
                 for d in list(dates)]
    idx = list(dates) if dates is not None else list(range(n))
    ok = np.isfinite(f) & np.isfinite(s)
    golden = np.zeros(n, bool)
    death = np.zeros(n, bool)
    golden[1:] = ok[1:] & ok[:-1] & (f[:-1] <= s[:-1]) & (f[1:] > s[1:])
    death[1:] = ok[1:] & ok[:-1] & (f[:-1] >= s[:-1]) & (f[1:] < s[1:])
    cash, shares, entry = float(initial_capital), 0, 0.0
    pending = None
    trades, values = [], []
    for i in range(n):
        if pending is not None:
            kind, sdate = pending
            px = o[i]
            if kind == "BUY":
                qty = int(math.floor(cash / px))
                if qty > 0:
                    cash -= qty * px
                    shares, entry = qty, px
                    trades.append({"type": "BUY", "signal_date": sdate, "exec_date": idx[i],
                                   "price": px, "shares": qty, "pnl": np.nan})
            elif shares > 0:
                cash += shares * px
                trades.append({"type": "SELL", "signal_date": sdate, "exec_date": idx[i],
                               "price": px, "shares": shares, "pnl": shares * (px - entry)})
                shares, entry = 0, 0.0
            pending = None
        if i < n - 1:
            if shares == 0 and golden[i]:
                pending = ("BUY", idx[i])
            elif shares > 0 and death[i]:
                pending = ("SELL", idx[i])
        values.append(cash + shares * c[i])
    if shares > 0:
        px = c[-1]
        cash += shares * px
        trades.append({"type": "SELL", "signal_date": idx[-1], "exec_date": idx[-1], "price": px,
                       "shares": shares, "pnl": shares * (px - entry)})
        shares = 0
        values[-1] = cash
    v = np.asarray(values, dtype=float)
    r = np.zeros(n)
    r[1:] = v[1:] / v[:-1] - 1.0
    cum = np.cumprod(1.0 + r)
    runmax = np.maximum.accumulate(cum)
    mdd = float(np.max((runmax - cum) / runmax))
    ann_ret = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year))
    tr = pd.DataFrame(trades, columns=["type", "signal_date", "exec_date", "price", "shares", "pnl"])
    if ticker is not None:
        tr.insert(0, "ticker", ticker)
    buys = int((tr["type"] == "BUY").sum())
    sells = tr[tr["type"] == "SELL"]
    metrics = {
        "total_return": float(v[-1] / initial_capital - 1.0), "final_capital": float(v[-1]),
        "annualized_return": ann_ret, "annualized_volatility": ann_vol,
        "sharpe_ratio": ann_ret / ann_vol if ann_vol > 0 else float("nan"),
        "max_drawdown": mdd, "calmar_ratio": ann_ret / mdd if mdd > 0 else float("nan"),
        "num_trades": buys, "win_rate": float((sells["pnl"] > 0).sum()) / buys if buys else 0.0,
        "total_pnl": float(sells["pnl"].sum()), "num_trading_days": int(n),
    }
    daily = pd.DataFrame({"date": idx, "portfolio_value": v, "daily_return": r, "cumulative_return": cum})
    return {"trades": tr, "daily": daily, "metrics": metrics}


# =============================================================================================
# 7. discrete delta hedge with escrowed dividends and asymmetric costs
# =============================================================================================

def _bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2), float(stats.norm.cdf(d1))


def escrowed_dividend_delta_hedge(S, iv, K, r, maturity_days, days_per_year, freq, div_days,
                                  div_amounts, buy_rate, sell_rate) -> dict:
    """Short one European call at day 0 and delta-hedge it (escrowed-dividend Black-Scholes).

    Day t (0..maturity_days): PV_divs(t) = sum_{ex > t} D e^{-r (ex - t)/days_per_year},
    S_eff = S[t] - PV_divs(t), T_rem = (maturity_days - t)/days_per_year, price/delta =
    BS call on S_eff with iv[t] (delta = N(d1), the derivative w.r.t. the OBSERVED spot).
    Day 0: cash = price - delta S0 - buy_rate |delta S0|; shares = delta.
    Each day t >= 1: (1) cash *= e^{r/days_per_year}; (2) dividend D on day ex: cash += shares D;
    (3) if t % freq == 0 and t < maturity_days: trade to the new delta, cash -= dN S_t and
    cash -= rate |dN| S_t (buy_rate when dN > 0, else sell_rate).
    Expiry: sell all shares (cash += shares S_T - sell_rate shares S_T), pay max(S_T - K, 0).
    Rates are DECIMALS (bps/1e4). Every fee is both deducted from cash and summed in
    total_transaction_cost. Returns {t0{price, delta, S_eff, pv_divs}, terminal{final_stock,
    call_payoff}, simulation{freq_days, n_rebalances, total_transaction_cost,
    total_dividends_received, hedging_pnl}}."""
    S = np.asarray(S, dtype=float)
    iv = np.asarray(iv, dtype=float)
    N = int(maturity_days)
    divs = [(int(d), float(a)) for d, a in zip(div_days, div_amounts)]
    dpy = float(days_per_year)

    def pv_divs(t):
        return float(sum(a * math.exp(-r * (d - t) / dpy) for d, a in divs if d > t))

    def price_delta(t):
        pv = pv_divs(t)
        p, dl = _bs_call(S[t] - pv, K, (N - t) / dpy, r, iv[t])
        return p, dl, S[t] - pv, pv

    p0, d0, se0, pv0 = price_delta(0)
    payoff = max(S[N] - K, 0.0)

    def _run_one(f_int):
        fee = buy_rate * abs(d0 * S[0])
        cash = p0 - d0 * S[0] - fee
        tc, shares, divs_rx, n_reb = fee, d0, 0.0, 0
        for t in range(1, N + 1):
            cash *= math.exp(r / dpy)
            for d, a in divs:
                if d == t:
                    cash += shares * a
                    divs_rx += shares * a
            if t % f_int == 0 and t < N:
                _, dl, _, _ = price_delta(t)
                dn = dl - shares
                fee = (buy_rate if dn > 0 else sell_rate) * abs(dn) * S[t]
                cash -= dn * S[t] + fee
                tc += fee
                shares = dl
                n_reb += 1
        fee = sell_rate * abs(shares) * S[N]
        cash += shares * S[N] - fee
        tc += fee
        cash -= payoff
        return {"freq_days": f_int, "n_rebalances": n_reb,
                "total_transaction_cost": tc, "total_dividends_received": divs_rx,
                "hedging_pnl": cash}

    if isinstance(freq, (list, tuple, np.ndarray)):
        sims = [_run_one(int(f)) for f in freq]
        return {"t0": {"price": p0, "delta": d0, "S_eff": se0, "pv_divs": pv0},
                "terminal": {"final_stock": float(S[N]), "call_payoff": payoff},
                "simulations": sims, "simulation": sims[0] if sims else {}}
    sim_single = _run_one(int(freq))
    return {"t0": {"price": p0, "delta": d0, "S_eff": se0, "pv_divs": pv0},
            "terminal": {"final_stock": float(S[N]), "call_payoff": payoff},
            "simulation": sim_single, "simulations": [sim_single]}


# =============================================================================================
# 8. Monte Carlo Greeks (FD with common random numbers, pathwise, likelihood ratio)
# =============================================================================================

def _bs(S, K, T, r, q, sig, kind):
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    if kind == "call":
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)


def bs_delta_vega_surface(spots, vols, K, T, r, q=0.0) -> pd.DataFrame:
    """Closed-form Black-Scholes delta (call e^{-qT}N(d1), put e^{-qT}(N(d1)-1)) and vega
    (S e^{-qT} phi(d1) sqrt T, per 1.00 vol, same for call and put) on a spot x vol grid,
    rows spot-major (for each spot, every vol)."""
    rows = []
    for S in spots:
        for sg in vols:
            v = sg * math.sqrt(T)
            d1 = (math.log(S / K) + (r - q + 0.5 * sg * sg) * T) / v
            dc = math.exp(-q * T) * stats.norm.cdf(d1)
            vg = S * math.exp(-q * T) * stats.norm.pdf(d1) * math.sqrt(T)
            rows.append({"S0": float(S), "sigma": float(sg), "delta_call_eu": dc,
                         "delta_put_eu": dc - math.exp(-q * T), "vega_call_eu": vg, "vega_put_eu": vg})
    return pd.DataFrame(rows)


def mc_greeks_suite(S0, K, r, q, sigma, T, n_paths, seed, n_avg, h_spot, h_vol, h_time, h_rate,
                    z_eu=None, z_asian=None) -> dict:
    """European and arithmetic-Asian (n_avg equally spaced dates t_j = jT/n) prices and Greeks by
    three Monte Carlo methods on ONE set of normals (common random numbers).

    Normals: RandomState(seed).standard_normal(N) for Europeans, then (N, n_avg) for Asians
    (or pass z_eu / z_asian). S_T = S0 exp((r-q-sigma^2/2)T + sigma sqrt(T) Z), D = e^{-rT}.
    FD (central; theta backward): delta (V(S+h)-V(S-h))/2h, gamma (V(S+h)-2V+V(S-h))/h^2,
      vega (V(s+h)-V(s-h))/2h, rho (V(r+h)-V(r-h))/2h (discount bumped too),
      theta -(V(T-h)-V(T))/h (date grid rebuilt for T-h, same normals). SE = std of the per-path
      FD estimator (ddof=1)/sqrt(N). Units are the literal formulas (vega per 1.00 vol, theta
      per year when h is in years) - rescale outside if the task asks.
    Pathwise (European call; put with 1{S_T<K} and sign flipped): delta D 1 S_T/S0;
      vega D 1 S_T (sqrt(T) Z - sigma T); rho D 1 K T; theta = D[-r payoff + 1 S_T (r-q-sigma^2/2
      + sigma Z/(2 sqrt T))] = dV/dT, the SAME quantity the written FD formula -(V(T-h)-V(T))/h
      estimates (calendar BS theta dV/dt is its negative - flip both if the task wants that sign);
      gamma None. Asian: delta D 1{A>K} A/S0, vega D 1 mean_j S_j (W_j -
      sigma t_j); gamma/theta/rho None.
    LR (weights x D payoff, per-step increments Z_j, dt = T/n; n = 1 for Europeans):
      delta Z_1/(S0 sigma sqrt dt); gamma (Z_1^2 - 1 - Z_1 sigma sqrt dt)/(S0 sigma sqrt dt)^2;
      vega sum_j[(Z_j^2-1)/sigma - Z_j sqrt dt]; rho sum_j Z_j sqrt dt/sigma - T; theta None.
    Returns {prices{option: {mc_price, mc_se, bs_price (Europeans)}}, greeks{option: {greek:
    {method: {value, se}}}} with None for unavailable, variance_ratio{pw_variance, lr_variance,
    pw_se, lr_se} for the European call delta}.
    """
    N = int(n_paths)
    rs = np.random.RandomState(seed)
    Ze = np.asarray(z_eu, dtype=float) if z_eu is not None else rs.standard_normal(N)
    Za = np.asarray(z_asian, dtype=float) if z_asian is not None else rs.standard_normal((N, int(n_avg)))
    n = Za.shape[1]

    def eu_paths(s0, sg, rr, TT):
        return s0 * np.exp((rr - q - 0.5 * sg * sg) * TT + sg * np.sqrt(TT) * Ze)

    def as_paths(s0, sg, rr, TT):
        dt = TT / n
        tj = dt * np.arange(1, n + 1)[None, :]
        Wj = np.cumsum(Za, axis=1) * np.sqrt(dt)
        return s0 * np.exp((rr - q - 0.5 * sg * sg) * tj + sg * Wj), tj, Wj

    def pay(opt, s0=S0, sg=sigma, rr=r, TT=T):
        if opt.startswith("european"):
            x = eu_paths(s0, sg, rr, TT)
        else:
            x = as_paths(s0, sg, rr, TT)[0].mean(axis=1)
        p = np.maximum(x - K, 0.0) if opt.endswith("call") else np.maximum(K - x, 0.0)
        return np.exp(-rr * TT) * p

    def est(x):
        x = np.asarray(x, dtype=float)
        return {"value": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(x.size))}

    na = {"value": None, "se": None}
    opts = ("european_call", "european_put", "asian_call", "asian_put")
    prices, greeks = {}, {}
    D = math.exp(-r * T)
    ST = eu_paths(S0, sigma, r, T)
    SA, tj, Wj = as_paths(S0, sigma, r, T)
    A = SA.mean(axis=1)
    dt_a = T / n
    pw_delta_call = None
    lr_delta_call = None
    for opt in opts:
        base = pay(opt)
        pr = {"mc_price": float(base.mean()), "mc_se": float(base.std(ddof=1) / math.sqrt(N))}
        call = opt.endswith("call")
        if opt.startswith("european"):
            pr["bs_price"] = float(_bs(S0, K, T, r, q, sigma, "call" if call else "put"))
        prices[opt] = pr
        up, dn = pay(opt, s0=S0 + h_spot), pay(opt, s0=S0 - h_spot)
        fd = {"delta": est((up - dn) / (2 * h_spot)),
              "gamma": est((up - 2 * base + dn) / h_spot ** 2),
              "vega": est((pay(opt, sg=sigma + h_vol) - pay(opt, sg=sigma - h_vol)) / (2 * h_vol)),
              "theta": est(-(pay(opt, TT=T - h_time) - base) / h_time),
              "rho": est((pay(opt, rr=r + h_rate) - pay(opt, rr=r - h_rate)) / (2 * h_rate))}
        w = 1.0 if call else -1.0
        if opt.startswith("european"):
            ind = (ST > K) if call else (ST < K)
            payoff = np.maximum(w * (ST - K), 0.0)
            pw = {"delta": est(w * D * ind * ST / S0), "gamma": na,
                  "vega": est(w * D * ind * ST * (math.sqrt(T) * Ze - sigma * T)),
                  "theta": est(D * (-r * payoff + w * ind * ST * ((r - q - 0.5 * sigma ** 2)
                                                               + sigma * Ze / (2 * math.sqrt(T))))),
                  "rho": est(w * D * ind * K * T)}
            Z1, Zs, dtl = Ze, Ze[:, None], T
        else:
            ind = (A > K) if call else (A < K)
            payoff = np.maximum(w * (A - K), 0.0)
            pw = {"delta": est(w * D * ind * A / S0), "gamma": na,
                  "vega": est(w * D * ind * (SA * (Wj - sigma * tj)).mean(axis=1)),
                  "theta": na, "rho": na}
            Z1, Zs, dtl = Za[:, 0], Za, dt_a
        sq = sigma * math.sqrt(dtl)
        dpay = D * payoff
        lr = {"delta": est(dpay * Z1 / (S0 * sq)),
              "gamma": est(dpay * (Z1 ** 2 - 1 - Z1 * sq) / (S0 * sq) ** 2),
              "vega": est(dpay * ((Zs ** 2 - 1) / sigma - Zs * math.sqrt(dtl)).sum(axis=1)),
              "theta": na,
              "rho": est(dpay * ((Zs * math.sqrt(dtl) / sigma).sum(axis=1) - T))}
        if opt == "european_call":
            pw_delta_call = D * ind * ST / S0
            lr_delta_call = dpay * Z1 / (S0 * sq)
        greeks[opt] = {g: {"fd": fd[g], "pathwise": pw[g], "lr": lr[g]}
                       for g in ("delta", "gamma", "vega", "theta", "rho")}
    pv, lv = float(np.var(pw_delta_call, ddof=1)), float(np.var(lr_delta_call, ddof=1))
    return {"prices": prices, "greeks": greeks,
            "variance_ratio": {"pw_variance": pv, "lr_variance": lv,
                               "pw_se": math.sqrt(pv / N), "lr_se": math.sqrt(lv / N)}}


# =============================================================================================
# 9. compound option grid (Geske) with MC check
# =============================================================================================

_KIND_LABEL = {"call_on_call": "call-on-call", "put_on_call": "put-on-call",
               "call_on_put": "call-on-put", "put_on_put": "put-on-put"}


def geske_compound_grid(S0, K1_list, moneyness, T1, T2, r, sigma, q=0.0,
                        kinds=("call_on_call", "put_on_put"), n_paths=300000, seed=None,
                        mc_q=None) -> pd.DataFrame:
    """Grid of Geske (1979) compound prices with the T1 exercise boundary and a Monte Carlo check.

    For each kind, outer strike K1 and inner strike K2 = S0 * m: S_star solves
    BS(S*, K2, T2 - T1, inner) = K1 (geske_critical_price), compound_price = Geske closed form
    (compound_option_price), MC: S_T1 = S0 exp((r - mc_q - sigma^2/2)T1 + sigma sqrt(T1) Z) with
    np.random.RandomState(seed) re-seeded per row, inner valued by BS at T1, payoff discounted
    e^{-r T1}; mc_q defaults to q. type labels use hyphens: 'call-on-call', 'put-on-put'.
    Columns: type, K1, K2, K2_moneyness, T1, T2, S_star, compound_price, mc_price, mc_std_err."""
    from .compound import compound_option_price, geske_critical_price
    mq = q if mc_q is None else mc_q
    rows = []
    for kind in kinds:
        outer, inner = kind.split("_on_")
        for K1 in K1_list:
            for m in moneyness:
                K2 = S0 * m
                s_star = geske_critical_price(K1, K2, T1, T2, r, sigma, q=q, inner=inner)
                price = compound_option_price(S0, K1, K2, T1, T2, r, sigma, q=q, kind=kind)
                z = np.random.RandomState(seed).standard_normal(int(n_paths))
                st1 = S0 * np.exp((r - mq - 0.5 * sigma ** 2) * T1 + sigma * math.sqrt(T1) * z)
                val = _bs(st1, K2, T2 - T1, r, q, sigma, inner)
                pay = np.maximum(val - K1, 0.0) if outer == "call" else np.maximum(K1 - val, 0.0)
                disc = math.exp(-r * T1) * pay
                rows.append({"type": _KIND_LABEL[kind], "K1": K1, "K2": K2, "K2_moneyness": m,
                             "T1": T1, "T2": T2, "S_star": s_star, "compound_price": price,
                             "mc_price": float(disc.mean()),
                             "mc_std_err": float(disc.std(ddof=1) / math.sqrt(disc.size))})
    return pd.DataFrame(rows, columns=["type", "K1", "K2", "K2_moneyness", "T1", "T2", "S_star",
                                       "compound_price", "mc_price", "mc_std_err"])


def mc_agreement(price, mc_price, se, z=3.0):
    """Statistical agreement of a closed form with a Monte Carlo estimate:
    |price - mc_price| <= z * se (z = 3: three standard errors). Arrays in -> bool array."""
    return np.abs(np.asarray(price, float) - np.asarray(mc_price, float)) <= z * np.asarray(se, float)


# =============================================================================================
# 10. par bootstrap with linear zero interpolation between maturities
# =============================================================================================

def bootstrap_par_linear_zero(maturities, par_rates, freq) -> dict:
    """Bootstrap a continuously-compounded zero curve from par coupon rates.

    Par bond n (coupon c_n paid every 1/freq, maturity T_n):
        1 = (c_n/freq) sum_{k=1}^{T_n freq - 1} df(t_k) + (1 + c_n/freq) df(T_n).
    Coupon dates strictly between T_{n-1} and T_n use z(t) LINEAR between z(T_{n-1}) and the
    unknown z(T_n) (flat z(T_1) before the first maturity), df = e^{-z t}; z(T_n) is solved
    exactly (brentq). With no interior dates this is the closed form
    df(T_n) = (1 - c/freq sum df)/(1 + c/freq). zero_fn(t) interpolates z linearly (flat outside).
    Returns {dfs, zeros (arrays at the maturities), zero_fn}."""
    T = [float(x) for x in maturities]
    c = [float(x) for x in par_rates]
    f = int(freq)
    nodes_t, nodes_z = [], []

    def z_at(t, zn, Tn):
        if not nodes_t:
            return zn
        if t <= nodes_t[0]:
            return nodes_z[0]
        Tp, zp = nodes_t[-1], nodes_z[-1]
        if t <= Tp:
            return float(np.interp(t, nodes_t, nodes_z))
        return zp + (zn - zp) * (t - Tp) / (Tn - Tp)

    for Tn, cn in zip(T, c):
        k = int(round(Tn * f))
        times = [j / f for j in range(1, k + 1)]

        def resid(zn):
            pv = sum(cn / f * math.exp(-z_at(t, zn, Tn) * t) for t in times[:-1])
            return pv + (1 + cn / f) * math.exp(-zn * Tn) - 1.0

        zn = optimize.brentq(resid, -0.5, 1.0, xtol=1e-15, maxiter=500)
        nodes_t.append(Tn)
        nodes_z.append(zn)
    zt, zz = np.asarray(nodes_t), np.asarray(nodes_z)

    def zero_fn(t):
        out = np.interp(np.asarray(t, dtype=float), zt, zz)
        return float(out) if np.ndim(out) == 0 else out

    return {"dfs": np.exp(-zz * zt), "zeros": zz, "zero_fn": zero_fn}


def forward_from_zero(zero_fn, T, tenor=1.0, first_maturity=None) -> float:
    """Continuously-compounded forward over [T - tenor, T]: (z(T) T - z(T-tenor)(T-tenor))/tenor,
    with z from zero_fn (linear interpolation). At the first maturity (T <= first_maturity or
    T - tenor <= 0) it is z(T)."""
    T = float(T)
    if (first_maturity is not None and T <= float(first_maturity)) or T - tenor <= 0:
        return float(zero_fn(T))
    return float((zero_fn(T) * T - zero_fn(T - tenor) * (T - tenor)) / tenor)


# =============================================================================================
# 11. rating migration counting
# =============================================================================================

def read_delimited_matrix(path_or_text) -> pd.DataFrame:
    """Read a CSV whose delimiter may be ';', ',', tab or '|' (sniffed from the header line);
    the first column becomes the index (e.g. a 'BOP\\EOP' rating label), values numeric.
    Use when INPUTS gave ONE column named like 'A;B;C' (KeyError: 'BOP\\EOP;AAA;...').
    know77: also accepts the DataFrame INPUTS already loaded (one ';'-joined column is split; a parsed
    frame gets its first column as the index) - 'DataFrame' object has no attribute 'splitlines' x3."""
    import os
    if isinstance(path_or_text, pd.DataFrame):
        f = path_or_text
        if len(f.columns) == 1 and any(s in str(f.columns[0]) for s in (";", "\t", "|")):
            text = "\n".join([str(f.columns[0])] + [str(v) for v in f.iloc[:, 0].tolist()])
            return read_delimited_matrix(text + "\n")
        df = f.set_index(f.columns[0]) if not pd.api.types.is_numeric_dtype(f[f.columns[0]]) else f.copy()
        df.index = df.index.astype(str).str.strip()
        df.columns = [str(c).strip() for c in df.columns]
        return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if isinstance(path_or_text, bytes):
        path_or_text = path_or_text.decode("utf-8-sig")
    path_or_text = os.fspath(path_or_text) if isinstance(path_or_text, os.PathLike) else path_or_text
    text = path_or_text
    if "\n" not in str(path_or_text):
        with open(path_or_text, encoding="utf-8-sig") as fh:
            text = fh.read()
    head = text.splitlines()[0]
    sep = max([";", ",", "\t", "|"], key=head.count)
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str).str.strip()
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def collapse_rating_counts(raw, mapping, states) -> pd.DataFrame:
    """Collapse a fine-grained count matrix (rows = start rating, columns = end rating) onto a
    target scale. mapping {fine: target or None}; None (e.g. NR, Withdrawals) = CENSORED - that
    row/column is dropped, not redistributed; labels missing from mapping map to themselves.
    Returns a DataFrame indexed and columned by `states` (zeros where nothing maps)."""
    out = pd.DataFrame(0.0, index=list(states), columns=list(states))
    for r_lab, row in raw.iterrows():
        tr = mapping.get(str(r_lab).strip(), str(r_lab).strip())
        if tr is None or tr not in out.index:
            continue
        for c_lab, v in row.items():
            tc = mapping.get(str(c_lab).strip(), str(c_lab).strip())
            if tc is None or tc not in out.columns:
                continue
            out.loc[tr, tc] += float(v) if np.isfinite(v) else 0.0
    return out


def row_stochastic(C, empty="identity"):
    """Row-normalise a count matrix: P_ij = C_ij / sum_k C_ik; rows with zero total become the
    identity row (empty='identity') or stay zero (empty='zero'). Keeps DataFrame labels."""
    is_df = isinstance(C, pd.DataFrame)
    M = np.asarray(C, dtype=float)
    tot = M.sum(axis=1, keepdims=True)
    P = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
    if empty == "identity":
        for i in np.where(tot.ravel() == 0)[0]:
            P[i, i] = 1.0
    return pd.DataFrame(P, index=C.index, columns=C.columns) if is_df else P


def chi2_homogeneity(table, min_col_total=5):
    """Chi-squared homogeneity test on a (groups x categories) count table -> (stat, dof, p).
    Columns whose total over all groups is < min_col_total (Cochran's rule of five) are dropped
    first, then all-zero rows; E_ij = row_i col_j / grand; stat = sum (O - E)^2 / E;
    dof = (rows - 1)(cols - 1). Fewer than 2 rows or columns left -> (0.0, 0, 1.0)."""
    M = np.asarray(table, dtype=float)
    M = M[:, M.sum(axis=0) >= min_col_total]
    M = M[M.sum(axis=1) > 0]
    if M.shape[0] < 2 or M.shape[1] < 2:
        return 0.0, 0, 1.0
    E = M.sum(axis=1, keepdims=True) * M.sum(axis=0, keepdims=True) / M.sum()
    stat = float(((M - E) ** 2 / E).sum())
    dof = (M.shape[0] - 1) * (M.shape[1] - 1)
    return stat, int(dof), float(stats.chi2.sf(stat, dof))


def generator_logm_clip(P):
    """Generator Q = real(logm(P)); negative off-diagonals set to 0, diagonal reset to minus the
    off-diagonal row sum (rows sum to 0); absorbing rows (P_ii = 1) are all zero.
    Keeps DataFrame labels."""
    from scipy.linalg import logm
    is_df = isinstance(P, pd.DataFrame)
    M = np.asarray(P, dtype=float)
    Q = np.real(logm(M))
    n = Q.shape[0]
    for i in range(n):
        if np.isclose(M[i, i], 1.0):
            Q[i, :] = 0.0
            continue
        off = np.delete(np.arange(n), i)
        Q[i, off] = np.clip(Q[i, off], 0.0, None)
        Q[i, i] = -Q[i, off].sum()
    return pd.DataFrame(Q, index=P.index, columns=P.columns) if is_df else Q


# =============================================================================================
# 12. Polars 0.x -> 1.x source rewriting
# =============================================================================================

_PL_RULES = [
    (r"\.groupby_dynamic\(", ".group_by_dynamic("),
    (r"\.groupby_rolling\(", ".rolling("),
    (r"\.groupby\(", ".group_by("),
    (r"\.with_column\(", ".with_columns("),
    (r"\.with_row_count\(", ".with_row_index("),
    (r"\.cumsum\(", ".cum_sum("), (r"\.cumprod\(", ".cum_prod("), (r"\.cummax\(", ".cum_max("),
    (r"\.cummin\(", ".cum_min("), (r"\.cumcount\(", ".cum_count("),
    (r"\.list\.lengths\(", ".list.len("), (r"\.arr\.lengths\(", ".list.len("),
    (r"\.str\.lengths\(", ".str.len_bytes("), (r"\.str\.n_chars\(", ".str.len_chars("),
    (r"\.str\.strip\(", ".str.strip_chars("), (r"\.str\.lstrip\(", ".str.strip_chars_start("),
    (r"\.str\.rstrip\(", ".str.strip_chars_end("),
    (r"\.str\.ljust\(\s*width\s*=", ".str.pad_end(length="), (r"\.str\.rjust\(\s*width\s*=", ".str.pad_start(length="),
    (r"\.str\.ljust\(", ".str.pad_end("), (r"\.str\.rjust\(", ".str.pad_start("),
    (r"\bfillchar\s*=", "fill_char="),
    (r"\.take_every\(", ".gather_every("), (r"\.take\(", ".gather("),
    (r"\.shift_and_fill\(\s*([^,()]+?)\s*,\s*fill_value\s*=", r".shift(\1, fill_value="),
    (r"\.is_first\(\)", ".is_first_distinct()"), (r"\.is_last\(\)", ".is_last_distinct()"),
    (r"\.is_not\(\)", ".not_()"),
    (r"\.keep_name\(\)", ".name.keep()"), (r"(?<!\.name)\.suffix\(", ".name.suffix("),
    (r"(?<!\.name)\.prefix\(", ".name.prefix("),
    (r"\.map_dict\(", ".replace_strict("),
    (r"\.apply\(\s*lambda", ".map_elements(lambda"),
    (r"\.frame_equal\(", ".equals("), (r"\.series_equal\(", ".equals("),
    (r"reverse\s*=", "descending="), (r"\bdtypes\s*=", "schema_overrides="),
    (r"min_periods\s*=", "min_samples="),
    (r"how\s*=\s*([\"'])outer\1", 'how="full", coalesce=True'),
    (r"how\s*=\s*([\"'])outer_coalesce\1", 'how="full", coalesce=True'),
]


def polars_0x_to_1x_source(src, new_name=None, *, fetch_style="head", null_keys_match=True, join_order=True):
    """Mechanical Polars 0.x -> 1.x rewrite of SOURCE TEXT (the deliverable is a .py file, not a
    function object) -> (new_src, todo). Applied: groupby -> group_by (dynamic/rolling too),
    with_column(s), with_row_count -> with_row_index, cum* -> cum_*, list.lengths -> list.len,
    str.strip/ljust/rjust -> strip_chars/pad_end/pad_start, take/take_every -> gather/
    gather_every, shift_and_fill -> shift(fill_value=), is_first -> is_first_distinct,
    is_not -> not_, keep_name/suffix/prefix -> name.*, map_dict -> replace_strict,
    reverse= -> descending=, dtypes= -> schema_overrides=, how='outer' -> how='full',
    coalesce=True (0.x coalesced keys), pl.count() -> pl.len() (alias kept by the caller),
    <ns>.any(a, b)/all(a, b) -> any_horizontal/all_horizontal, melt -> unpivot(index=, on=),
    pivot(columns=) -> pivot(on=), group_by_dynamic(by=) -> group_by=, LazyFrame.fetch(n) noted.
    `new_name` renames the first def. todo lists what still needs a hand translation: any
    .apply( / .map( with a Python lambda (write native expressions; map_elements raises
    PolarsInefficientMapWarning), fetch(), and unchanged default column names (count -> len).
    know79 (outputs PRESERVED, not just renamed - helpers79_4): fetch_style='head' turns .fetch(n) into
    .head(n).collect() (first n rows of the RESULT; 'n_rows' = the old scan-level sample, n source rows);
    null_keys_match=True adds nulls_equal=True to joins (0.19 matched null keys, 0.20+ does not);
    join_order=True adds maintain_order='left' / 'left_right' (0.x kept the left row order)."""
    from .helpers79_4 import polars_fetch_to_head, polars_join_semantics_0x
    out = src
    todo = []
    for pat, rep in _PL_RULES:
        out = re.sub(pat, rep, out)
    ns = r"(\b(?:polars|pl))"
    out = re.sub(ns + r"\.count\(\)", r"\1.len()", out)
    out = _horizontal(out, "any")
    out = _horizontal(out, "all")
    for lazy in set(re.findall(r"(\b\w+)\s*=\s*(?:polars|pl)\.scan_\w+\(", out)):
        out = re.sub(rf"\b{lazy}\.columns\b", f"{lazy}.collect_schema().names()", out)
    out = re.sub(r"\.melt\(\s*id_vars\s*=", ".unpivot(index=", out)
    out = re.sub(r"(\.unpivot\([^)]*?)value_vars\s*=", r"\1on=", out)
    out = re.sub(r"(\.pivot\([^)]*?)\bcolumns\s*=", r"\1on=", out)
    out = re.sub(r"(\.group_by_dynamic\([^)]*?)\bby\s*=", r"\1group_by=", out)
    if new_name:
        out = re.sub(r"def\s+\w+\(", f"def {new_name}(", out, count=1)
    try:
        from .helpers79_3 import instruction_text
        _m = re.search(r"polars==(\d+\.\d+\.\d+)", instruction_text())
        if _m:
            out = re.sub(r"\b(?:polars|pl)\.__version__\b", repr(_m.group(1)), out)
    except Exception:  # noqa: BLE001
        pass
    # know76: translate the common lambdas to native expressions and fetch(n) to n_rows=n
    out = _pl_native_lambdas(out)
    out = polars_fetch_to_head(out) if fetch_style == "head" else _pl_fetch_to_nrows(out)
    if null_keys_match or join_order:
        out = polars_join_semantics_0x(out, null_keys_match=null_keys_match, maintain_order=join_order)
    for i, line in enumerate(out.splitlines(), 1):
        if re.search(r"\.(apply|map|map_elements)\(\s*lambda", line):
            todo.append(f"line {i}: Python lambda (now .map_elements, runs but warns "
                        "PolarsInefficientMapWarning/MapWithoutReturnDtypeWarning) - replace with a native "
                        "expression, e.g. (-(pl.col(c)**2)).abs(), (pl.col(a)-pl.col(b)).abs()/pl.col(a)*100, "
                        "(pl.col(c).max()-pl.col(c).min()).cast(pl.Float64)")
        if ".fetch(" in line:
            todo.append(f"line {i}: LazyFrame.fetch(n) - use scan_csv(..., n_rows=n) ... .collect()")
        if re.search(r"\b(?:polars|pl)\.len\(\)", line) and "alias" not in line:
            todo.append(f"line {i}: pl.len() names its column 'len' (0.x 'count') - alias if the old name was written")
    return MigratedSource(out, todo)


class MigratedSource(str):
    """The migrated SOURCE TEXT (a real str: write it, compile it, emit it) that ALSO unpacks as
    (source, todo): `new, todo = polars_0x_to_1x_source(src)` and `new = polars_0x_to_1x_source(src)`
    both work. know75 wrote str((src, todo)) - a tuple repr - as the .py deliverable."""

    def __new__(cls, text, todo=()):
        obj = super().__new__(cls, text)
        obj.todo = list(todo)
        return obj

    def __iter__(self):
        return iter((str.__str__(self), self.todo))

    def __len__(self):
        return str.__len__(self)


def _pl_expr_from_lambda(lam, receiver_src, is_struct):
    """Native Polars expression text for `receiver.apply(lambda v: body)` or None if the body uses
    anything other than arithmetic, abs(), float()/int() and Series methods. Element lambdas on
    pl.col(c) -> body with v := pl.col(c); struct lambdas v['a'] -> pl.col('a'); abs(e) -> (e).abs();
    float(e) -> (e).cast(Float64)."""
    arg = lam.args.args[0].arg if lam.args.args else None
    if arg is None:
        return None
    ns = receiver_src.split(".")[0] if receiver_src.split(".")[0] in ("polars", "pl") else "polars"
    recv = ast.parse(receiver_src, mode="eval").body
    ok = [True]

    class T(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == arg and is_struct:
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    return ast.parse(f"{ns}.col({key.value!r})", mode="eval").body
            ok[0] = ok[0] and not (isinstance(node.value, ast.Name) and node.value.id == arg)
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == arg:
                if is_struct:
                    ok[0] = False
                    return node
                return recv
            if node.id not in ("abs", "float", "int"):
                ok[0] = False
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1:
                return ast.Call(func=ast.Attribute(value=node.args[0], attr="abs", ctx=ast.Load()), args=[], keywords=[])
            if isinstance(node.func, ast.Name) and node.func.id in ("float", "int") and len(node.args) == 1:
                dt = "Float64" if node.func.id == "float" else "Int64"
                return ast.Call(func=ast.Attribute(value=node.args[0], attr="cast", ctx=ast.Load()),
                                args=[ast.parse(f"{ns}.{dt}", mode="eval").body], keywords=[])
            if isinstance(node.func, ast.Name):
                ok[0] = False
            return node

    body = T().visit(ast.parse(ast.unparse(lam.body), mode="eval").body)
    if not ok[0]:
        return None
    return "(" + ast.unparse(ast.fix_missing_locations(body)) + ")"


def _pl_native_lambdas(src):
    import ast as _ast
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return src
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    pos = lambda l, c: offs[l - 1] + len(lines[l - 1].encode("utf-8")[:c].decode("utf-8", "ignore"))  # noqa: E731
    edits = []
    for node in _ast.walk(tree):
        if (isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute)
                and node.func.attr in ("map_elements", "apply", "map") and len(node.args) == 1
                and isinstance(node.args[0], _ast.Lambda) and not node.keywords):
            rv = node.func.value
            rsrc = _ast.get_source_segment(src, rv)
            if rsrc is None:
                continue
            is_struct = (isinstance(rv, _ast.Call) and isinstance(rv.func, _ast.Attribute)
                         and rv.func.attr == "struct")
            is_col = "col(" in rsrc or is_struct
            if not is_col:
                continue
            expr = _pl_expr_from_lambda(node.args[0], " ".join(rsrc.split()), is_struct)
            if expr is None:
                continue
            edits.append((pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset), expr))
    for a, b, e in sorted(edits, reverse=True):
        src = src[:a] + e + src[b:]
    return src


def _pl_fetch_to_nrows(src):
    """`<ns>.scan_csv(ARGS) ... .fetch(N)` inside one expression -> scan_csv(ARGS, n_rows=N) ...
    .collect() (fetch read only the first N source rows before the rest of the query)."""
    pat = re.compile(r"(\b(?:polars|pl)\.scan_csv\()([^()]*)\)((?:(?!\bscan_csv\().){0,800}?)\.fetch\(\s*([^()]+?)\s*\)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}{m.group(2)}, n_rows={m.group(4)}){m.group(3)}.collect()", src)


def _horizontal(src, fn):
    """polars.any(a, b) / pl.all(a, b) with 2+ top-level arguments -> any_horizontal/all_horizontal
    (1.x any()/all() are column reductions)."""
    out, pos = [], 0
    pat = re.compile(rf"\b(polars|pl)\.{fn}\(")
    while True:
        m = pat.search(src, pos)
        if not m:
            out.append(src[pos:])
            return "".join(out)
        depth, commas, j = 1, 0, m.end()
        quote = None
        while j < len(src) and depth:
            ch = src[j]
            if quote:
                quote = None if ch == quote else quote
            elif ch in "\"'":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 1:
                commas += 1
            j += 1
        name = f"{fn}_horizontal" if commas >= 1 else fn
        out.append(src[pos:m.start()] + f"{m.group(1)}.{name}(")
        pos = m.end()


def check_python_source(src):
    """compile() the source; raises SyntaxError naming the line, returns True when it parses."""
    compile(src, "<migrated>", "exec")
    return True
