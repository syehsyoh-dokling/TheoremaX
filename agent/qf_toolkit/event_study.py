"""Event-study abnormal returns and test statistics with every convention as a keyword.

Why it exists: re-derived event-study tests came out wrong (the Corrado rank statistic ~3.7x too
small: s(K) taken over events instead of over DAYS of the cross-sectional mean rank). Each function
follows the ORIGINAL paper's definition, cited in its docstring.

    from qf_toolkit import event_study as es
    mm  = es.market_model(r_est, m_est)                       # alpha, beta, sigma (ddof=2), L, ...
    ar  = es.abnormal_returns(r_evt, m_evt, mm["alpha"], mm["beta"])
    ps  = es.patell_standardized(ar, m_evt, mm)              # SAR per day, SCAR over the window
    z   = es.patell_test(sar0_list, L_list); t = es.bmp_test(scar_list)
    t_kp = es.kolari_pynnonen_adjust(t, es.average_residual_correlation(resid_series_list), n)
    rk  = es.corrado_rank_test(est_resid_list, event_ar_list, window=(-1, 1))
    out = es.event_study_panel(events, returns, market, est_window=(-250, -11), event_window=(-5, 5))

Conventions: returns are per-period DECIMAL returns (simple or log, as the task states). Relative
days are POSITIONS in the trading-day index (not calendar days). `window=(a, b)` is INCLUSIVE; it
refers to relative-day LABELS when the input is a pandas Series with an integer index, otherwise to
array POSITIONS 0..n-1.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .stats import ols

__all__ = ["market_model", "abnormal_returns", "car", "patell_standardized", "patell_test",
           "bmp_test", "kolari_pynnonen_adjust", "average_residual_correlation",
           "corrado_rank_test", "corrado_zivney_sign_test", "sign_test", "cross_sectional_t",
           "crude_dependence_t", "event_study_panel"]


# ----------------------------------------------------------------------------- helpers
def _is_int_series(x) -> bool:
    return isinstance(x, pd.Series) and pd.api.types.is_integer_dtype(x.index)


def _window_mask(x, window) -> np.ndarray:
    """Boolean mask of the elements of x inside the INCLUSIVE window (labels or positions)."""
    n = len(x)
    if window is None:
        return np.ones(n, dtype=bool)
    lo, hi = window
    pos = np.asarray(x.index) if _is_int_series(x) else np.arange(n)
    return (pos >= lo) & (pos <= hi)


def _values(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _pval(z) -> float:
    return float(2 * norm.sf(abs(z))) if np.isfinite(z) else float("nan")


# ----------------------------------------------------------------------------- market model
def market_model(stock_ret_est, market_ret_est) -> dict:
    """OLS market model R_it = a_i + b_i R_mt + e_it on the estimation window (MacKinlay 1997).

    Inputs are arrays (same length) or pandas Series (aligned on their index, inner join).
    Pairs with a non-finite value are dropped. Returns dict:
      alpha, beta, resid (Series if Series input, index kept), sigma = sqrt(SSR/(L-2)) (ddof=2),
      L = number of estimation observations used, mkt_mean = mean R_m over the estimation window,
      mkt_ss = V_m = sum (R_mt - mkt_mean)^2, r2.
    """
    if isinstance(stock_ret_est, pd.Series) and isinstance(market_ret_est, pd.Series):
        df = pd.concat([stock_ret_est.astype(float), market_ret_est.astype(float)],
                       axis=1, join="inner")
        df.columns = ["r", "m"]
        index = df.index
        y, x = df["r"].to_numpy(), df["m"].to_numpy()
    else:
        y, x = _values(stock_ret_est), _values(market_ret_est)
        if len(y) != len(x):
            raise ValueError(f"stock ({len(y)}) and market ({len(x)}) lengths differ")
        index = None
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    L = int(len(y))
    if L < 3:
        raise ValueError(f"market_model needs >= 3 finite observations, got {L}")
    fit = ols(y, x, add_const=True)
    resid = fit["resid"]
    sigma = math.sqrt(float(resid @ resid) / (L - 2))
    mkt_mean = float(x.mean())
    resid_out = pd.Series(resid, index=index[ok]) if index is not None else resid
    return {"alpha": float(fit["beta"][0]), "beta": float(fit["beta"][1]), "resid": resid_out,
            "sigma": sigma, "L": L, "mkt_mean": mkt_mean,
            "mkt_ss": float(((x - mkt_mean) ** 2).sum()), "r2": float(fit["r2"])}


def abnormal_returns(stock_ret, market_ret, alpha, beta):
    """AR_t = R_t - (alpha + beta * R_mt). Series in -> Series out (index of stock_ret)."""
    if isinstance(stock_ret, pd.Series):
        m = market_ret.reindex(stock_ret.index) if isinstance(market_ret, pd.Series) \
            else pd.Series(_values(market_ret), index=stock_ret.index)
        return stock_ret.astype(float) - (alpha + beta * m.astype(float))
    return _values(stock_ret) - (alpha + beta * _values(market_ret))


def car(ar, window=None) -> float:
    """Cumulative abnormal return: sum of AR over the INCLUSIVE window (NaN if any AR is NaN)."""
    v = _values(ar)[_window_mask(ar, window)]
    return float(v.sum()) if len(v) else float("nan")


# ----------------------------------------------------------------------------- Patell / BMP / KP
def patell_standardized(ar_event, market_ret_event, mm, *, window=None, scar_var="sum_c") -> dict:
    """Patell (1976) standardized abnormal returns and the standardized CAR over `window`.

    Day-wise (Patell 1976, eq. for the prediction error of an OLS forecast):
        C_t   = 1 + 1/L + (R_mt - Rbar_m)^2 / V_m        (Rbar_m, V_m from the ESTIMATION window)
        SAR_t = AR_t / (sigma * sqrt(C_t))               sigma = sqrt(SSR/(L-2)); SAR_t ~ t(L-2)
    Window statistic (`scar_var`):
      "sum_c"  (default) SCAR = CAR / (sigma * sqrt(sum_{t in W} C_t)) - the variance of the window
               sum taken as the SUM OF THE DAY-WISE prediction-error variances (Patell 1976 cumulative;
               the Boehmer-Musumeci-Poulsen 1991 standardized CAR). Ignores the covariance between days.
      "full"   exact variance of the sum of OLS prediction errors, INCLUDING the covariance created by
               the common estimation error in (alpha, beta) (Mikkelson-Partch 1988; Campbell-Lo-
               MacKinlay 1997 ch.4 gamma'V gamma):
               Var(CAR) = sigma^2 [ W + W^2/L + (sum_{t in W}(R_mt - Rbar_m))^2 / V_m ].
      "sum_sar" SCAR = sum_{t in W} SAR_t / sqrt(W)  (Patell's cumulated SAR; the theory-card form).
      "iid"    SCAR = CAR / (sigma * sqrt(W))       (no prediction-error correction).
    Returns dict: sar (same type as ar_event), C, car, scar, var_car (None for "sum_sar"), W.
    """
    ar_v = _values(ar_event)
    m_v = _values(market_ret_event)
    if len(ar_v) != len(m_v):
        raise ValueError("ar_event and market_ret_event lengths differ")
    L, sigma, mbar, vm = mm["L"], mm["sigma"], mm["mkt_mean"], mm["mkt_ss"]
    C = 1.0 + 1.0 / L + (m_v - mbar) ** 2 / vm
    sar_v = ar_v / (sigma * np.sqrt(C))
    mask = _window_mask(ar_event, window)
    W = int(mask.sum())
    car_w = float(ar_v[mask].sum())
    if scar_var == "sum_c":
        var_car = sigma ** 2 * float(C[mask].sum())
        scar = car_w / math.sqrt(var_car)
    elif scar_var == "full":
        dev = float((m_v[mask] - mbar).sum())
        var_car = sigma ** 2 * (W + W ** 2 / L + dev ** 2 / vm)
        scar = car_w / math.sqrt(var_car)
    elif scar_var == "sum_sar":
        var_car = None
        scar = float(sar_v[mask].sum()) / math.sqrt(W)
    elif scar_var == "iid":
        var_car = sigma ** 2 * W
        scar = car_w / math.sqrt(var_car)
    else:
        raise ValueError('scar_var must be "sum_c", "full", "sum_sar" or "iid"')
    sar = pd.Series(sar_v, index=ar_event.index) if isinstance(ar_event, pd.Series) else sar_v
    return {"sar": sar, "C": C, "car": car_w, "scar": float(scar), "var_car": var_car, "W": W}


def patell_test(sars_or_scars, L_list) -> float:
    """Patell (1976) Z = sum_i SAR_i / sqrt( sum_i (L_i-2)/(L_i-4) ).

    SAR_i (one day) or SCAR_i (window, any `scar_var` of patell_standardized) per event;
    (L_i-2)/(L_i-4) is the variance of a t(L_i-2) variable. L_list: per-event estimation lengths
    (a scalar is broadcast). Requires every L_i > 4. Returns float, ~N(0,1) under H0.
    """
    s = _values(sars_or_scars)
    L = np.broadcast_to(np.asarray(L_list, dtype=float), s.shape)
    ok = np.isfinite(s)
    s, L = s[ok], L[ok]
    if np.any(L <= 4):
        raise ValueError("patell_test needs every L_i > 4")
    return float(s.sum() / math.sqrt(((L - 2) / (L - 4)).sum()))


def bmp_test(scars) -> float:
    """Boehmer, Musumeci & Poulsen (1991) standardized cross-sectional test.

    t = mean(SCAR) * sqrt(N) / std(SCAR, ddof=1), with SCAR_i the Patell-standardized (C)AR of
    event i (use SAR_i0 for the event day). Non-finite values are dropped. Returns float.
    """
    s = _values(scars)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 2:
        return float("nan")
    return float(s.mean() * math.sqrt(n) / s.std(ddof=1))


def kolari_pynnonen_adjust(t_bmp, rho_bar, n, *, form="kp2010") -> float:
    """Cross-correlation adjustment of the BMP t (Kolari & Pynnonen 2010, RFS 23(11), eq. 17).

    form="kp2010" (default, the published adjusted BMP):
        t_adj = t_bmp * sqrt( (1 - rho_bar) / (1 + (n - 1) * rho_bar) )
    form="simple": t_adj = t_bmp / sqrt(1 + (n - 1) * rho_bar)  (variance-inflation only, no
        (1 - rho_bar) numerator; use ONLY when a task defines the scaling this way).
    rho_bar = average pairwise correlation of the events' estimation-window residuals
    (average_residual_correlation); n = number of events. rho_bar = 0 -> t_bmp unchanged.
    """
    denom = 1.0 + (n - 1) * rho_bar
    if denom <= 0:
        raise ValueError("1 + (n-1)*rho_bar must be > 0")
    if form == "kp2010":
        return float(t_bmp * math.sqrt((1.0 - rho_bar) / denom))
    if form == "simple":
        return float(t_bmp / math.sqrt(denom))
    raise ValueError('form must be "kp2010" or "simple"')


def average_residual_correlation(resid_series_list, min_overlap=10, *, return_pairs=False):
    """Mean pairwise Pearson correlation of estimation residuals on COMMON calendar dates.

    resid_series_list: pandas Series with a DatetimeIndex (e.g. market_model(...)["resid"] from
    Series inputs). Each pair is aligned on the intersection of dates (non-finite dropped); pairs
    with fewer than `min_overlap` common dates (or zero variance) are SKIPPED. Returns the mean over
    the kept pairs (nan if none); with return_pairs=True returns (mean, list of (i, j, rho, n)).
    """
    # 30 Sep: accepting bare arrays (aligned by position) produced a silently WRONG rho_bar - the
    # correlation must pair residuals on the same CALENDAR dates. Refuse with the fix spelled out.
    series = []
    for s in resid_series_list:
        if not isinstance(s, pd.Series) or not isinstance(s.index, pd.DatetimeIndex):
            raise TypeError("average_residual_correlation needs each event's estimation residuals as a "
                            "pandas Series indexed by CALENDAR DATE (pd.DatetimeIndex), e.g. "
                            "pd.Series(resid, index=est_dates); pairs are aligned on common dates")
        series.append(s.astype(float).dropna())
    pairs = []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            a, b = series[i].align(series[j], join="inner")
            ok = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
            n = int(ok.sum())
            if n < min_overlap:
                continue
            x, y = a.to_numpy()[ok], b.to_numpy()[ok]
            if x.std() == 0 or y.std() == 0:
                continue
            pairs.append((i, j, float(np.corrcoef(x, y)[0, 1]), n))
    mean = float(np.mean([p[2] for p in pairs])) if pairs else float("nan")
    return (mean, pairs) if return_pairs else mean


# ----------------------------------------------------------------------------- rank / sign tests
def _align_events(est_list, evt_list, event_day_index, est_gap):
    """Per-event Series indexed by RELATIVE DAY: estimation values + event-window values."""
    if len(est_list) != len(evt_list):
        raise ValueError("est and event lists have different numbers of events")
    out = []
    for est, evt in zip(est_list, evt_list):
        if _is_int_series(evt):
            e = evt.astype(float)
        else:
            v = _values(evt)
            d0 = event_day_index
            if d0 is None:
                if len(v) == 1:
                    d0 = 0
                elif len(v) % 2 == 1:
                    d0 = len(v) // 2
                else:
                    raise ValueError("even-length event window: pass event_day_index "
                                     "(position of day 0) or integer-indexed Series")
            e = pd.Series(v, index=np.arange(len(v)) - d0)
        if _is_int_series(est):
            s = est.astype(float)
        else:
            v = _values(est)
            end = int(e.index.min()) - int(est_gap) - 1
            s = pd.Series(v, index=np.arange(end - len(v) + 1, end + 1))
        if len(set(s.index) & set(e.index)):
            # 30 Sep: a residual Series with a default 0..L-1 index is not relative-day labelled -
            # place the estimation values positionally before the event window instead of failing
            v = _values(est)
            end = int(e.index.min()) - int(est_gap) - 1
            s = pd.Series(v, index=np.arange(end - len(v) + 1, end + 1))
        out.append((s, e))
    return out


def _cross_section(per_event, transform, denominator, sd_days):
    cols = {}
    evt_days = set()
    est_days = set()
    for i, (s, e) in enumerate(per_event):
        comb = pd.concat([s, e]).sort_index()
        cols[i] = transform(comb)
        evt_days |= set(e.index)
        est_days |= set(s.index)
    frame = pd.DataFrame(cols).sort_index()
    return frame, sorted(evt_days), sorted(est_days)


def _days_denominator(frame, used_days, denominator):
    if denominator == "observed":
        return len(used_days)
    if denominator == "span":
        return int(max(used_days) - min(used_days) + 1)
    raise ValueError('denominator must be "observed" or "span"')


def corrado_rank_test(est_resids_list, event_ars_list, event_day_index=None, window=None, *,
                      est_gap=0, rank_scale="standardized", denominator="observed",
                      sd_days="all") -> dict:
    """Corrado (1989) nonparametric rank test; Corrado & Zivney (1992) standardized ranks.

    Per event i, its estimation residuals and event-window ARs are pooled into one series of
    T_i = L_i + W_i observations and ranked JOINTLY (average ranks for ties; NaN skipped) -> K_it.
      rank_scale="raw" (Corrado 1989; needs equal T_i = T): deviations D_it = K_it - (T+1)/2.
      rank_scale="standardized" (default; Corrado-Zivney 1992, valid for unequal T_i):
          D_it = K_it/(T_i+1) - 1/2. With equal T this gives EXACTLY the raw statistic (scale cancels).
    Dbar_t = mean over events observed on day t;
      s = sqrt( (1/D) sum_t Dbar_t^2 ) over the days t of the combined period
          (sd_days="all", Corrado; sd_days="estimation" uses estimation days only),
          D = number of days with >= 1 observation (denominator="observed") or the full span of
          relative days from first to last incl. gap days, which contribute 0 (denominator="span").
      z_day0   = Dbar_0 / s                                   (Corrado 1989 T0)
      z_window = sum_{t in W} Dbar_t / (sqrt(|W|) * s)        (cumulative rank statistic; this is
          also the Campbell & Wasley 1993 multi-day form sqrt(|W|)*(mean_{t in W} Dbar_t)/s).
    Inputs: lists of arrays or Series. Integer-indexed Series = relative-day labels (day 0 = 0).
    Arrays: event array element j is day j - event_day_index (default: centre of an odd-length
    window, 0 for length 1); estimation array ends `est_gap` days before the event window's first day.
    window = (a, b) INCLUSIVE relative days (default: the whole event window).
    Returns dict(z_day0, z_window, p_day0, p_window, s_K, n_events, n_days, mean_dev) where s_K is in
    the units of D_it (multiply by T+1 for the raw-rank s(K) when all T_i are equal).
    """
    per_event = _align_events(est_resids_list, event_ars_list, event_day_index, est_gap)
    Ts = [int(pd.concat([s, e]).notna().sum()) for s, e in per_event]
    if rank_scale == "raw":
        if len(set(Ts)) != 1:
            raise ValueError('rank_scale="raw" needs equal T_i; use "standardized"')

        def transform(x):
            r = pd.Series(np.nan, index=x.index)
            ok = x.notna().to_numpy()
            r[ok] = rankdata(x[ok].to_numpy(), method="average") - (ok.sum() + 1) / 2.0
            return r
    elif rank_scale == "standardized":
        def transform(x):
            r = pd.Series(np.nan, index=x.index)
            ok = x.notna().to_numpy()
            r[ok] = rankdata(x[ok].to_numpy(), method="average") / (ok.sum() + 1.0) - 0.5
            return r
    else:
        raise ValueError('rank_scale must be "standardized" or "raw"')
    frame, evt_days, est_days = _cross_section(per_event, transform, denominator, sd_days)
    dbar = frame.mean(axis=1, skipna=True).dropna()
    if sd_days == "all":
        used = list(dbar.index)
    elif sd_days == "estimation":
        used = [d for d in dbar.index if d in set(est_days)]
    else:
        raise ValueError('sd_days must be "all" or "estimation"')
    D = _days_denominator(frame, used, denominator)
    s = math.sqrt(float((dbar.loc[used] ** 2).sum()) / D)
    z0 = float(dbar.get(0, np.nan) / s)
    if window is None:
        wdays = [d for d in evt_days if d in dbar.index]
    else:
        wdays = [d for d in dbar.index if window[0] <= d <= window[1]]
    zw = float(dbar.loc[wdays].sum() / (math.sqrt(len(wdays)) * s)) if wdays else float("nan")
    return {"z_day0": z0, "z_window": zw, "p_day0": _pval(z0), "p_window": _pval(zw), "s_K": s,
            "n_events": len(per_event), "n_days": D, "mean_dev": dbar}


def corrado_zivney_sign_test(est_resids_list, event_ars_list, event_day_index=None, window=None, *,
                             est_gap=0, denominator="observed") -> dict:
    """Corrado & Zivney (1992) sign test on signs relative to each event's MEDIAN.

    S_it = sign(A_it - median_i) over the pooled estimation + event series of event i;
    G_t = N_t^{-1/2} sum_i S_it; s = sqrt( (1/D) sum_t G_t^2 ) over the combined days;
    z_day0 = G_0 / s; z_window = sum_{t in W} G_t / (sqrt(|W|) s). Same inputs/alignment as
    corrado_rank_test. Returns dict(z_day0, z_window, p_day0, p_window, s, n_events).
    """
    per_event = _align_events(est_resids_list, event_ars_list, event_day_index, est_gap)

    def transform(x):
        return np.sign(x - x.median())
    frame, evt_days, _ = _cross_section(per_event, transform, denominator, "all")
    nt = frame.notna().sum(axis=1)
    g = (frame.sum(axis=1, skipna=True) / np.sqrt(nt.where(nt > 0))).dropna()
    D = _days_denominator(frame, list(g.index), denominator)
    s = math.sqrt(float((g ** 2).sum()) / D)
    z0 = float(g.get(0, np.nan) / s)
    wdays = [d for d in (evt_days if window is None else g.index)
             if d in g.index and (window is None or window[0] <= d <= window[1])]
    zw = float(g.loc[wdays].sum() / (math.sqrt(len(wdays)) * s)) if wdays else float("nan")
    return {"z_day0": z0, "z_window": zw, "p_day0": _pval(z0), "p_window": _pval(zw), "s": s,
            "n_events": len(per_event)}


def sign_test(cars, *, method="binomial", est_resids_list=None, p_hat=None) -> dict:
    """Sign test on event (C)ARs; a zero counts as NOT positive.

    method="binomial": Z = (w - N/2) / sqrt(N/4), w = #positive (p = 0.5 under H0).
    method="generalized" (Cowan 1992 generalized sign test):
        p_hat = (1/N) sum_i (1/L_i) sum_{t in est} 1[AR_it > 0]  (from est_resids_list, or pass p_hat)
        Z = (w - N p_hat) / sqrt(N p_hat (1 - p_hat)).
    Returns dict(z, p_value, n_pos, n, p_hat).
    """
    c = _values(cars)
    c = c[np.isfinite(c)]
    n, w = len(c), int((c > 0).sum())
    if method == "binomial":
        p = 0.5
    elif method == "generalized":
        if p_hat is None:
            if est_resids_list is None:
                raise ValueError("generalized sign test needs est_resids_list or p_hat")
            fr = []
            for e in est_resids_list:
                v = _values(e)
                v = v[np.isfinite(v)]
                fr.append((v > 0).mean())
            p_hat = float(np.mean(fr))
        p = float(p_hat)
    else:
        raise ValueError('method must be "binomial" or "generalized"')
    z = (w - n * p) / math.sqrt(n * p * (1 - p)) if n else float("nan")
    return {"z": float(z), "p_value": _pval(z), "n_pos": w, "n": n, "p_hat": p}


def cross_sectional_t(cars, ddof=1) -> float:
    """Plain cross-sectional t = mean(CAR) / (std(CAR, ddof) / sqrt(N)) (Brown-Warner 1985)."""
    c = _values(cars)
    c = c[np.isfinite(c)]
    return float(c.mean() / (c.std(ddof=ddof) / math.sqrt(len(c)))) if len(c) > 1 else float("nan")


def crude_dependence_t(est_resids_list, event_ars_list, event_day_index=None, window=None, *,
                       est_gap=0, ddof=1) -> dict:
    """Brown & Warner (1980, 1985) crude dependence adjustment (time-series of the portfolio AR).

    AAR_t = mean over events of AR_it on relative day t. S = std over ESTIMATION days of AAR_t
    (ddof=1: divide by (#days - 1)). t_day0 = AAR_0 / S; t_window = CAAR_W / (sqrt(|W|) S).
    Same inputs/alignment as corrado_rank_test. Returns dict(t_day0, t_window, s_aar, aar0, caar).
    """
    per_event = _align_events(est_resids_list, event_ars_list, event_day_index, est_gap)
    est = pd.DataFrame({i: s for i, (s, _) in enumerate(per_event)}).mean(axis=1).dropna()
    evt = pd.DataFrame({i: e for i, (_, e) in enumerate(per_event)}).mean(axis=1).dropna()
    s = float(est.std(ddof=ddof))
    wd = [d for d in evt.index if window is None or window[0] <= d <= window[1]]
    caar = float(evt.loc[wd].sum())
    aar0 = float(evt.get(0, np.nan))
    return {"t_day0": aar0 / s, "t_window": caar / (math.sqrt(len(wd)) * s) if wd else float("nan"),
            "s_aar": s, "aar0": aar0, "caar": caar}


# ----------------------------------------------------------------------------- panel helper
def _find(frame, name):
    try:
        from .io import col
        return col(frame, name)
    except Exception:  # pragma: no cover - fall back to the exact name
        if name in frame.columns:
            return name
        raise KeyError(f"column {name!r} not in {list(frame.columns)}")


def event_study_panel(events, returns, market, *, est_window=(-250, -11), event_window=(-5, 5),
                      car_window=None, id_col="id", date_col="event_date", ret_date_col="date",
                      ret_col="ret", market_col="market", min_est_obs=60,
                      require_full_event_window=True, scar_var="sum_c", kp_form="kp2010",
                      min_overlap=10, corrado_denominator="observed") -> dict:
    """Market-model event study over many events with every statistic in this module.

    events  : DataFrame with `id_col` (security id) and `date_col` (event date).
    returns : WIDE DataFrame (DatetimeIndex x security ids) or LONG DataFrame with
              [ret_date_col, id_col, ret_col].
    market  : Series indexed by date, or DataFrame with [ret_date_col, market_col].
    Calendar = the market's trading dates (sorted, non-NaN). Day 0 = first trading date >= event
    date (non-trading date -> next trading day). Relative days are positions in that calendar.
    est_window / event_window / car_window: INCLUSIVE relative days (car_window default = event_window;
    the CORRADO/sign/crude window statistics use car_window too). Estimation uses the days with
    finite stock and market returns; events with < min_est_obs, out-of-sample windows or (with
    require_full_event_window) any missing event-window return are skipped and listed.
    Returns dict:
      params : DataFrame per event [event, id, event_date, day0_date, alpha, beta, sigma, L, r2,
               car, scar, sar0, ar0]
      ar     : long DataFrame [event, id, rel_day, date, ret, market_ret, ar, sar]
      aar, caar : Series by rel_day (mean AR over events; cumulative over the event window)
      stats  : dict with n_events, caar (car_window), patell_z_day0/_window, bmp_t_day0/_window,
               rho_bar, kp_t_day0/_window, corrado (dict), cz_sign (dict), sign_binomial,
               sign_generalized, cs_t_day0/_window, crude (dict)
      skipped: DataFrame [id, event_date, reason]
    """
    car_window = tuple(event_window) if car_window is None else tuple(car_window)
    # --- market series and calendar
    if isinstance(market, pd.DataFrame):
        mkt = market.set_index(_find(market, ret_date_col))[_find(market, market_col)]
    else:
        mkt = market
    mkt = mkt.astype(float).copy()
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt[~mkt.index.duplicated()].dropna().sort_index()
    cal = mkt.index
    # --- wide returns on the calendar
    if isinstance(returns, pd.DataFrame) and {_safe(returns, ret_date_col), _safe(returns, id_col),
                                              _safe(returns, ret_col)} <= set(returns.columns):
        wide = returns.pivot_table(index=_find(returns, ret_date_col), columns=_find(returns, id_col),
                                   values=_find(returns, ret_col), aggfunc="first")
    else:
        wide = returns
    wide = wide.copy()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.reindex(cal).astype(float)
    ev = events.copy()
    idc, dc = _find(ev, id_col), _find(ev, date_col)
    ev[dc] = pd.to_datetime(ev[dc])

    rows, ar_rows, skipped = [], [], []
    est_list, evt_list, resid_list, sar0s, scars, Ls, cars_, ar0s = [], [], [], [], [], [], [], []
    for _, e in ev.iterrows():
        sid, edate = e[idc], e[dc]
        if sid not in wide.columns:
            skipped.append((sid, edate, "no returns for id"))
            continue
        p0 = int(cal.searchsorted(edate, side="left"))
        lo = p0 + min(est_window[0], event_window[0])
        hi = p0 + max(est_window[1], event_window[1])
        if p0 >= len(cal) or lo < 0 or hi >= len(cal):
            skipped.append((sid, edate, "window outside data"))
            continue
        r = wide[sid]
        est_pos = np.arange(p0 + est_window[0], p0 + est_window[1] + 1)
        r_est, m_est = r.iloc[est_pos], mkt.iloc[est_pos]
        okm = np.isfinite(r_est.to_numpy()) & np.isfinite(m_est.to_numpy())
        if okm.sum() < max(min_est_obs, 5):
            skipped.append((sid, edate, f"only {int(okm.sum())} estimation obs"))
            continue
        evt_pos = np.arange(p0 + event_window[0], p0 + event_window[1] + 1)
        r_evt, m_evt = r.iloc[evt_pos], mkt.iloc[evt_pos]
        if require_full_event_window and not np.isfinite(r_evt.to_numpy()).all():
            skipped.append((sid, edate, "missing event-window return"))
            continue
        mm = market_model(r_est, m_est)
        k = len(rows)
        rel_evt = evt_pos - p0
        ar = pd.Series(abnormal_returns(r_evt.to_numpy(), m_evt.to_numpy(), mm["alpha"], mm["beta"]),
                       index=rel_evt)
        ps = patell_standardized(ar, m_evt.to_numpy(), mm, window=car_window, scar_var=scar_var)
        est_rel = pd.Series(mm["resid"].to_numpy(),
                            index=np.asarray(cal.get_indexer(mm["resid"].index)) - p0)
        est_list.append(est_rel)
        evt_list.append(ar)
        resid_list.append(mm["resid"])
        sar0 = float(ps["sar"].get(0, np.nan))
        sar0s.append(sar0)
        scars.append(ps["scar"])
        Ls.append(mm["L"])
        cars_.append(ps["car"])
        ar0s.append(float(ar.get(0, np.nan)))
        rows.append({"event": k, "id": sid, "event_date": edate, "day0_date": cal[p0],
                     "alpha": mm["alpha"], "beta": mm["beta"], "sigma": mm["sigma"], "L": mm["L"],
                     "r2": mm["r2"], "car": ps["car"], "scar": ps["scar"], "sar0": sar0,
                     "ar0": ar0s[-1]})
        for j, d in enumerate(rel_evt):
            ar_rows.append({"event": k, "id": sid, "rel_day": int(d), "date": cal[evt_pos[j]],
                            "ret": float(r_evt.iloc[j]), "market_ret": float(m_evt.iloc[j]),
                            "ar": float(ar.iloc[j]), "sar": float(ps["sar"].iloc[j])})
    params = pd.DataFrame(rows, columns=["event", "id", "event_date", "day0_date", "alpha", "beta",
                                         "sigma", "L", "r2", "car", "scar", "sar0", "ar0"])
    ar_tab = pd.DataFrame(ar_rows, columns=["event", "id", "rel_day", "date", "ret", "market_ret",
                                            "ar", "sar"])
    skipped = pd.DataFrame(skipped, columns=["id", "event_date", "reason"])
    n = len(params)
    stats = {"n_events": n}
    if n == 0:
        return {"params": params, "ar": ar_tab, "aar": pd.Series(dtype=float),
                "caar": pd.Series(dtype=float), "stats": stats, "skipped": skipped}
    aar = ar_tab.groupby("rel_day")["ar"].mean()
    caar_s = aar.cumsum()
    stats["caar"] = float(np.mean(cars_))
    stats["aar0"] = float(np.nanmean(ar0s))
    stats["patell_z_day0"] = patell_test(sar0s, Ls)
    stats["patell_z_window"] = patell_test(scars, Ls)
    stats["bmp_t_day0"] = bmp_test(sar0s)
    stats["bmp_t_window"] = bmp_test(scars)
    rho = average_residual_correlation(resid_list, min_overlap=min_overlap) if n > 1 else float("nan")
    stats["rho_bar"] = rho
    rho_use = 0.0 if not np.isfinite(rho) else rho
    stats["kp_t_day0"] = kolari_pynnonen_adjust(stats["bmp_t_day0"], rho_use, n, form=kp_form)
    stats["kp_t_window"] = kolari_pynnonen_adjust(stats["bmp_t_window"], rho_use, n, form=kp_form)
    cr = corrado_rank_test(est_list, evt_list, window=car_window, denominator=corrado_denominator)
    cr.pop("mean_dev", None)
    stats["corrado"] = cr
    stats["cz_sign"] = corrado_zivney_sign_test(est_list, evt_list, window=car_window,
                                                denominator=corrado_denominator)
    stats["sign_binomial"] = sign_test(cars_, method="binomial")
    stats["sign_generalized"] = sign_test(cars_, method="generalized", est_resids_list=est_list)
    stats["cs_t_day0"] = cross_sectional_t(ar0s)
    stats["cs_t_window"] = cross_sectional_t(cars_)
    stats["crude"] = crude_dependence_t(est_list, evt_list, window=car_window)
    return {"params": params, "ar": ar_tab, "aar": aar, "caar": caar_s, "stats": stats,
            "skipped": skipped}


def _safe(frame, name):
    try:
        return _find(frame, name)
    except KeyError:
        return object()  # never a column
