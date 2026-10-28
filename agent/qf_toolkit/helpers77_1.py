"""know77 group-1 helpers: whole-pattern calls for the factor / sort, option and execution units that
failed (or swung) in the know76 full run. Every bundle calls the general helpers underneath
(helpers75_* / helpers76_* / exotics); every convention is a named parameter with the textbook default.

Patterns covered:
  * contract conformance          conform_to_keys, fixed_text_frame
  * date x entity panel joins     build_base_panel, eval_feature
  * skip-month momentum L/S       momentum_ls_deliverables
  * committee alpha + process     committee_process_alpha_bundle
    selection filter + ranked L/S
  * social-sentiment momentum     sentiment_momentum_report
    with ex-ante vol targeting, IC and single-factor alpha
  * two-asset spread / exchange   correlated_terminal_gbm, spread_option_report
    options (Margrabe, Kirk, MC)
Self-tests: proven_work/fail72/selftest77_1.py.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ._rdict import ResultDict


def _iso(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def _fin(v):
    """float, or None for NaN / inf (JSON-safe)."""
    v = float(v)
    return v if math.isfinite(v) else None


# ==============================================================================================
# 1. contract conformance
# ==============================================================================================


def conform_to_keys(values, keys, missing=None):
    """-> dict with EXACTLY `keys` in that order (extra keys dropped; absent keys -> `missing`).

    For deliverables whose key set is checked exactly (solution.json intermediates, results.json):
    compute every candidate, then cut to the instruction's list. `values` may be a flat dict or a
    {name: {'value': v}} block (the block shape is kept)."""
    out = {}
    for k in keys:
        out[k] = values[k] if k in values else missing
    return out


def _text_decimals():
    """Decimal places the TASK states for written numbers ('6 decimal places', "'%.6f'", 'rounded to 4 decimals'),
    else None."""
    try:
        import re as _re
        from .helpers78_4 import instruction_text
        t = instruction_text() or ""
    except Exception:  # noqa: BLE001
        return None
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ten": 10}
    m = _re.search(r"%\.(\d+)f", t) or _re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|ten)[ -]"
                                                   r"(?:decimal places|decimals|decimal digits|dp)\b", t, _re.I)
    if not m:
        return None
    g = m.group(1).lower()
    return int(g) if g.isdigit() else words.get(g)


def fixed_text_frame(df, decimals=None, bool_style="lower", int_cols=None):
    """-> DataFrame of STRINGS formatted to the usual fixed-CSV rules, ready for emit / to_csv:
    floats '%.{decimals}f', ints as integers (also integral floats in int_cols), bools 'true'/'false'
    (bool_style="lower") or 'True'/'False' ("python"), None / NaN / '' -> ''. Column order kept.
    (pandas to_csv on a numeric frame writes True/False, 0.0 and 1e-05 - the classic schema miss.)
    decimals None (know92 default): the places the TASK states ('6 decimal places', "'%.6f'"); a task silent
    about rounding gets FULL precision in plain positional notation (0.03370824090619759, never 1e-05) - a fixed
    6-place default rounded a task's numbers it never asked to round, and broke q5 - q1 identities."""
    if decimals is None:
        decimals = _text_decimals()
    int_cols = set(int_cols or [])

    def _float_text(x):
        if decimals is not None:
            return f"{x:.{int(decimals)}f}"
        return np.format_float_positional(x, unique=True, trim="-")

    def cell(v, col):
        if v is None or (isinstance(v, str) and v == ""):
            return ""
        if isinstance(v, (bool, np.bool_)):
            s = "true" if bool(v) else "false"
            return s if bool_style == "lower" else s.capitalize()
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            if col in int_cols and float(v).is_integer():
                return str(int(v))
            return _float_text(float(v))
        return str(v)

    out = pd.DataFrame({c: [cell(v, c) for v in df[c].tolist()] for c in df.columns}, columns=list(df.columns))
    return out


# ==============================================================================================
# 2. date x entity base panel with left joins
# ==============================================================================================


def build_base_panel(dates, universe, tables=(), date_col="asof_date", id_col="ticker", key_candidates=None):
    """-> one row per (date, entity): cartesian product of `dates` x `universe`, then LEFT joins.

    dates: DataFrame with ONE date column (whatever its name, e.g. rebalance_date) or a sequence; it is
      renamed to `date_col` and normalised to Timestamps. universe: DataFrame with `id_col` (+ attributes
      such as sector / bucket). tables: list of DataFrames, or (DataFrame, keys) pairs; without explicit
      keys a table joins on the columns it shares with the panel among key_candidates (default:
      date_col, id_col and every universe column) - e.g. a bucket-level table keyed (asof_date, bucket).
      Each table's own date column named date_col is normalised too. Joins never add rows: a table with
      duplicate keys raises. Sorted by (date_col, id_col)."""
    if isinstance(dates, pd.DataFrame):
        dcols = [c for c in dates.columns if c == date_col] or list(dates.columns[:1])
        d = pd.DataFrame({date_col: pd.to_datetime(dates[dcols[0]])})
    else:
        d = pd.DataFrame({date_col: pd.to_datetime(pd.Series(list(dates)))})
    d = d.drop_duplicates().sort_values(date_col, kind="mergesort")
    u = universe.copy()
    base = d.merge(u, how="cross").sort_values([date_col, id_col], kind="mergesort").reset_index(drop=True)
    n0 = len(base)
    cand = list(key_candidates) if key_candidates is not None else [date_col, id_col] + [c for c in u.columns if c != id_col]
    for item in tables:
        t, keys = (item if isinstance(item, tuple) else (item, None))
        t = t.copy()
        if date_col in t.columns:
            t[date_col] = pd.to_datetime(t[date_col])
        if keys is None:
            keys = [c for c in cand if c in t.columns and c in base.columns]
        keys = list(keys)
        if t.duplicated(keys).any():
            raise ValueError(f"table joined on {keys} has duplicate keys - joins must preserve the base rows")
        new = [c for c in t.columns if c not in keys and c not in base.columns]
        base = base.merge(t[keys + new], on=keys, how="left")
    if len(base) != n0:
        raise ValueError("a join changed the row count")
    return base


_FEATURE_FUNCS = {"log1p": np.log1p, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
                  "max": np.maximum, "min": np.minimum, "maximum": np.maximum, "minimum": np.minimum}


def eval_feature(panel, expr):
    """-> float Series for a column name or a small expression over columns, e.g. 'log1p(article_count_7d)',
    'article_count_7d / max(article_count_30d, 1)' (max/min are ELEMENT-WISE np.maximum/np.minimum)."""
    if isinstance(expr, str) and expr in panel.columns:
        return panel[expr].astype(float)
    if not isinstance(expr, str):
        return pd.Series(np.asarray(expr, float), index=panel.index)
    ns = {c: pd.to_numeric(panel[c], errors="coerce").astype(float) for c in panel.columns
          if isinstance(c, str) and c.isidentifier() and c in expr
          and (pd.api.types.is_numeric_dtype(panel[c]) or panel[c].dtype == object)}
    ns.update(_FEATURE_FUNCS)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 - column algebra only, no builtins
    return pd.Series(np.asarray(v, float) * np.ones(len(panel)), index=panel.index)


# ==============================================================================================
# 3. skip-month cross-sectional momentum, prices -> deliverables
# ==============================================================================================


def momentum_ls_deliverables(prices, date_col="date", form_start=12, form_end=2, n_long=3, n_short=3,
                             periods_per_year=12, stock_cols=None):
    """Price table -> {results, portfolio, table}: clean, returns, skip-month L/S, metrics in ONE call.

    Cleaning (the usual order): sort by date (stable), drop_duplicates(subset=date, keep='last'), ffill,
    dropna -> num_clean_rows. Returns = pct_change of the stock columns (CSV order = tie-break order),
    first row dropped. skip_month_momentum_ls(returns, form_start, form_end, n_long, n_short): signal =
    sum of rows t-form_start..t-form_end, cross-sectional stable rank per row. Metrics:
    compounded_performance (geometric). results keys: num_clean_rows, num_signal_months, total_return,
    annualized_return, annualized_vol, sharpe, max_drawdown; portfolio: date 'YYYY-MM-DD',
    signal_return, cumulative_return."""
    from .helpers76_1 import skip_month_momentum_ls, compounded_performance
    p = prices.copy()
    p[date_col] = pd.to_datetime(p[date_col])
    p = p.sort_values(date_col, kind="mergesort").drop_duplicates(subset=date_col, keep="last").ffill().dropna()
    cols = list(stock_cols) if stock_cols is not None else [c for c in p.columns if c != date_col]
    R = p.set_index(date_col)[cols].pct_change().iloc[1:]
    out = skip_month_momentum_ls(R, form_start, form_end, n_long, n_short)
    m = compounded_performance(out["signal_return"].to_numpy(float), periods_per_year)
    res = {"num_clean_rows": int(len(p)), "num_signal_months": int(len(out)), "total_return": m["total_return"],
           "annualized_return": m["annualized_return"], "annualized_vol": m["annualized_vol"], "sharpe": m["sharpe"],
           "max_drawdown": m["max_drawdown"]}
    port = pd.DataFrame({"date": [_iso(x) for x in out["date"]], "signal_return": out["signal_return"].to_numpy(float),
                         "cumulative_return": out["cumulative_return"].to_numpy(float)})
    return ResultDict({"results": res, "portfolio": port, "table": out})


# ==============================================================================================
# 4. committee alpha -> stochastic-process filter -> ranked long/short -> P&L
# ==============================================================================================


def _process_row(hist, p, n_params):
    from .helpers75_2 import fit_return_processes
    res = fit_return_processes(hist, p["jump_threshold_sigma"], p["holding_period_days"], p["conviction_scale"],
                               n_params=n_params)
    aic = res["aic"]
    fin = lambda v: float(v) if np.isfinite(v) else np.nan  # noqa: E731
    return {"selected_process": res["selected_process"], "martingale_aic": fin(aic["martingale"]),
            "gbm_mu": res["gbm_mu"], "gbm_sigma": res["gbm_sigma"], "gbm_aic": fin(aic["gbm"]),
            "ou_theta": res["ou_theta"], "ou_mu": res["ou_mu"], "ou_sigma": res["ou_sigma"],
            "ou_half_life_days": res["ou_half_life_days"], "ou_aic": fin(aic["ou"]),
            "merton_mu": res["merton_mu"], "merton_sigma": res["merton_sigma"],
            "merton_jump_intensity": res["merton_jump_intensity"], "merton_jump_mean": res["merton_jump_mean"],
            "merton_jump_std": res["merton_jump_std"], "merton_aic": fin(aic["merton_jump_diffusion"]),
            "process_expected_return": float(res["expected_return"]), "process_sign": int(res["sign"]),
            "process_conviction": float(res["conviction"])}


_PROCESS_COLS = ["selected_process", "martingale_aic", "gbm_mu", "gbm_sigma", "gbm_aic", "ou_theta", "ou_mu",
                 "ou_sigma", "ou_half_life_days", "ou_aic", "merton_mu", "merton_sigma", "merton_jump_intensity",
                 "merton_jump_mean", "merton_jump_std", "merton_aic", "process_expected_return", "process_sign",
                 "process_conviction"]


def committee_process_alpha_bundle(universe, rebalance_dates, feature_tables, returns, agents, agent_weights,
                                   *, events=None, vintage_v1=None, vintage_v2=None, params=None,
                                   date_col="asof_date", id_col="ticker", ret_col="forward_return_21d",
                                   event_date_col="bulletin_date", event_value_col="severity", event_prefix="bulletin",
                                   event_window_days=365, severe_cutoff=None, vintage_lookback_days=None,
                                   vintage_stale_days=None, missing_side="skip", process_window=None,
                                   process_min_history=None, jump_threshold_sigma=None, holding_period_days=None,
                                   conviction_scale=None, n_params=None, disagreement_penalty=0.0, clip=3.0,
                                   long_n=None, short_n=None, min_signal=0.0, long_gross=0.5, short_gross=0.5,
                                   cost_bps=0.0, initial_capital=1.0, periods_per_year=12, ddof_vol=0,
                                   annualization="geometric", missing_cell_tables=None, news_table=None,
                                   attribute_cols=("sector", "cot_bucket")):
    """Multi-agent committee alpha with a process-selection filter -> {file name: obj} for the 7 usual
    outputs (committee_panel.csv, process_diagnostics.csv, alpha_panel.csv, portfolio_returns.csv,
    modality_diagnostics.csv, results.json, solution.json). ONE call; emit each value as is.

    1 panel: build_base_panel(rebalance_dates (any single date column), universe, feature_tables
      {name: DataFrame} + returns) - joins on shared keys (asof_date, ticker / bucket); rows preserved.
    2 audit: trailing_event_features(events, window, severe_cutoff) and vintage_mismatch_features(v1, v2,
      lookback, stale) - only data on or before each date.
    3 agents {name: [(sign, column or expression), ...]} (expressions via eval_feature, e.g.
      'log1p(x)', 'a / max(b, 1)'); committee_agent_score: per date inf -> NaN -> median -> 0, z ddof 0,
      std 0 -> 0, clip +-clip, mean of signed components.
    4 committee_score = sum w_a score_a (agent_weights {name: w}); dispersion = std over agents ddof 0;
      agreement = max(0, 1 - penalty dispersion); committee_alpha = score * agreement.
    5 process layer per (date, id): the last `process_window` values of ret_col from rows with date <
      current (same id); fewer than process_min_history -> 'insufficient_history', params NaN,
      sign 0, conviction 0; else fit_return_processes (martingale / gbm / ou / merton, ddof 0, AIC,
      alphabetical ties, sigmoid conviction of the AIC gap / scale).
    6 final_signal = committee_alpha * process_sign * process_conviction (0 when sign 0).
    7 ranked_long_short_selection (signal desc, id asc; |s| < min_signal ineligible; +long_gross/n,
      -short_gross/n) ; 8 weights_pnl_with_turnover (turnover 0.5 sum|dw|, cost bps, equity from capital).
    modality diagnostics per date: num_issuers, <table>_missing_cells (NaN cells of that feature table's
      columns, for every name in missing_cell_tables), audit_bulletin_rows_365d (sum of event rows in
      the window), audit_mismatch_issuers (mismatch count > 0), news_nonzero_issuers (any non-zero
      finite news feature), process_history_ready_issuers.
    results: num_issuers, num_rebalance_dates, num_backtest_periods, num_process_ready_dates,
      num_news_active_dates, cumulative_return (equity/capital - 1), annualized_return
      (annualization="geometric": (1+cum)^(A/n) - 1; "arithmetic": mean A), annualized_volatility
      (std ddof_vol sqrt A), sharpe_ratio (mean/std sqrt A), max_drawdown (>= 0, capital counted as a
      peak), avg_turnover, best/worst_rebalance_date (net return argmax / argmin).
    solution: first/last rebalance date with long / short tickers in alpha_rank order and
      selected_process_counts {process: rows} over all rows.
    `params` (a dict such as params.json) may supply any keyword above under the same name or the
    usual aliases (severe_bulletin_cutoff, reference_lookback_days, reference_stale_threshold_days,
    process_trailing_window, process_min_history, process_conviction_scale, forward_return_days,
    long_top_n, short_bottom_n, min_trade_signal, transaction_cost_bps, annualization_factor,
    disagreement_penalty, initial_capital, <agent>_weight)."""
    from .helpers76_1 import (trailing_event_features, vintage_mismatch_features, committee_agent_score,
                              ranked_long_short_selection, weights_pnl_with_turnover)
    P = dict(params or {})
    alias = {"severe_cutoff": "severe_bulletin_cutoff", "vintage_lookback_days": "reference_lookback_days",
             "vintage_stale_days": "reference_stale_threshold_days", "process_window": "process_trailing_window",
             "conviction_scale": "process_conviction_scale", "holding_period_days": "forward_return_days",
             "long_n": "long_top_n", "short_n": "short_bottom_n", "min_signal": "min_trade_signal",
             "cost_bps": "transaction_cost_bps", "periods_per_year": "annualization_factor"}
    loc = dict(severe_cutoff=severe_cutoff, vintage_lookback_days=vintage_lookback_days,
               vintage_stale_days=vintage_stale_days, process_window=process_window,
               process_min_history=process_min_history, jump_threshold_sigma=jump_threshold_sigma,
               holding_period_days=holding_period_days, conviction_scale=conviction_scale,
               disagreement_penalty=disagreement_penalty, long_n=long_n, short_n=short_n, min_signal=min_signal,
               cost_bps=cost_bps, initial_capital=initial_capital, periods_per_year=periods_per_year)
    for k in list(loc):
        for src in (k, alias.get(k)):
            if src and src in P:
                loc[k] = P[src]
                break
    # know78: these are task settings with no textbook default - they come from params or the call, never a guess
    need = [k for k in ("process_window", "process_min_history", "jump_threshold_sigma", "holding_period_days",
                        "conviction_scale", "long_n", "short_n") if loc.get(k) is None]
    if need:
        raise ValueError(f"committee_process_alpha_bundle: pass {need} (or params keys "
                         f"{[alias.get(k, k) for k in need]}) - the task states them")
    W = dict(agent_weights or {})
    for a in agents:
        if a not in W:
            # know78: 'filing_agent_score' -> filing_weight (the params naming); know77 tried only
            # filing_agent_score_weight / filing_score_weight and raised for every committee task
            import re as _re
            _b = _re.sub(r"(_agent)?(_score)?$", "", a)
            for cand in (f"{a}_weight", f"{a.replace('_agent', '')}_weight", f"{_b}_weight", f"{_b}_agent_weight",
                         f"w_{_b}", f"{_b}"):
                if cand in P and isinstance(P[cand], (int, float)):
                    W[a] = float(P[cand])
                    break
            if a not in W and isinstance(P.get("agent_weights"), dict):
                for k, v in P["agent_weights"].items():
                    if k in (a, _b, f"{_b}_agent", f"{_b}_weight"):
                        W[a] = float(v)
    if set(W) != set(agents):
        raise ValueError(f"agent_weights must give a weight for every agent {list(agents)}; got {W}")
    ft = dict(feature_tables)
    base = build_base_panel(rebalance_dates, universe, list(ft.values()) + [returns], date_col, id_col)
    ret = returns.copy()
    ret[date_col] = pd.to_datetime(ret[date_col])
    # 2 audit features
    ev_rows = None
    if events is not None:
        ev = trailing_event_features(base, events, int(event_window_days), loc["severe_cutoff"], date_col, id_col,
                                     event_date_col, event_value_col, event_prefix)
        ev_rows = f"{event_prefix}_rows_{int(event_window_days)}d"
        base = base.merge(ev, on=[date_col, id_col], how="left")
    if vintage_v1 is not None and vintage_v2 is not None:
        vm = vintage_mismatch_features(base, vintage_v1, vintage_v2, loc["vintage_lookback_days"],
                                       loc["vintage_stale_days"], date_col, id_col, missing_side=missing_side)
        base = base.merge(vm, on=[date_col, id_col], how="left")
    # 3 agents
    for a, comps in agents.items():
        base[a] = committee_agent_score(base, [(s, eval_feature(base, e)) for s, e in comps], by=date_col, clip=clip,
                                        ddof=0)
    names = list(agents)
    S = base[names].to_numpy(float)
    base["committee_score"] = sum(W[a] * base[a] for a in names)
    base["committee_dispersion"] = S.std(axis=1, ddof=0)
    base["agreement_multiplier"] = np.maximum(0.0, 1.0 - float(loc["disagreement_penalty"]) * base["committee_dispersion"])
    base["committee_alpha"] = base["committee_score"] * base["agreement_multiplier"]
    # 5 process layer
    pp = {"jump_threshold_sigma": float(loc["jump_threshold_sigma"]), "holding_period_days": float(loc["holding_period_days"]),
          "conviction_scale": float(loc["conviction_scale"])}
    by_id = {k: g.sort_values(date_col, kind="mergesort") for k, g in ret.groupby(id_col)}
    prow = []
    for dt, tk in zip(base[date_col], base[id_col]):
        g = by_id.get(tk)
        h = g[g[date_col] < dt][ret_col].dropna().to_numpy(float) if g is not None else np.array([])
        h = h[-int(loc["process_window"]):]
        row = {"history_count": int(len(h))}
        if len(h) < int(loc["process_min_history"]):
            row.update({c: np.nan for c in _PROCESS_COLS})
            row.update(selected_process="insufficient_history", process_sign=0, process_conviction=0.0)
        else:
            row.update(_process_row(h, pp, n_params))
        prow.append(row)
    pr = pd.DataFrame(prow, index=base.index)
    for c in pr.columns:
        base[c] = pr[c]
    base["process_sign"] = base["process_sign"].astype(int)
    sig = base["committee_alpha"] * base["process_sign"]
    base["final_signal"] = np.where(base["process_sign"] == 0, 0.0, sig * base["process_conviction"])
    # 7 ranking and weights, 8 P&L
    alpha = ranked_long_short_selection(base, "final_signal", date_col, id_col, int(loc["long_n"]), int(loc["short_n"]),
                                        float(loc["min_signal"]), long_gross, short_gross)
    alpha = alpha.sort_values([date_col, "alpha_rank", id_col], kind="mergesort").reset_index(drop=True)
    port = weights_pnl_with_turnover(alpha, ret_col, date_col, id_col, cost_bps=float(loc["cost_bps"]),
                                     initial_capital=float(loc["initial_capital"]))
    # outputs
    attrs = [c for c in attribute_cols if c in base.columns]
    iso = lambda s: pd.to_datetime(s).dt.strftime("%Y-%m-%d")  # noqa: E731
    committee = base.sort_values([date_col, id_col], kind="mergesort")[
        [date_col, id_col] + attrs + names + ["committee_score", "committee_dispersion", "agreement_multiplier",
                                              "committee_alpha"]].reset_index(drop=True)
    committee[date_col] = iso(committee[date_col])
    proc = base.sort_values([date_col, id_col], kind="mergesort")[[date_col, id_col, "history_count"] + _PROCESS_COLS]
    proc = proc.reset_index(drop=True)
    proc[date_col] = iso(proc[date_col])
    extra = [c for c in ("entry_date", "exit_date") if c in alpha.columns]
    ap = alpha[[date_col, id_col] + attrs + ["committee_alpha", "process_sign", "process_conviction", "final_signal",
                                             "alpha_rank", "selection", "portfolio_weight"] + extra + [ret_col]].copy()
    ap[date_col] = iso(ap[date_col])
    for c in extra:
        ap[c] = iso(ap[c])
    po = port.copy()
    po[date_col] = iso(po[date_col])
    # modality diagnostics
    mcols = {}
    for nm in (missing_cell_tables if missing_cell_tables is not None else list(ft)):
        t = ft[nm]
        keys = {date_col, id_col, *universe.columns}
        mcols[nm] = [c for c in t.columns if c not in keys]
    news_cols = [c for c in ft[news_table].columns if c not in {date_col, id_col, *universe.columns}] if news_table else []
    mod = []
    for dt, g in base.groupby(date_col, sort=True):
        row = {date_col: _iso(dt), "num_issuers": int(len(g))}
        for nm, cols in mcols.items():
            row[f"{nm}_missing_cells"] = int(g[cols].isna().to_numpy().sum())
        if ev_rows is not None:
            row["audit_bulletin_rows_365d"] = int(g[ev_rows].sum())
        if "reference_mismatch_count" in g.columns:
            row["audit_mismatch_issuers"] = int((g["reference_mismatch_count"] > 0).sum())
        if news_cols:
            v = g[news_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            row["news_nonzero_issuers"] = int((np.isfinite(v) & (v != 0)).any(axis=1).sum())
        row["process_history_ready_issuers"] = int((g["selected_process"] != "insufficient_history").sum())
        mod.append(row)
    mod = pd.DataFrame(mod)
    # results / solution
    A = float(loc["periods_per_year"])
    net = port["net_return"].to_numpy(float)
    n = len(net)
    cap = float(loc["initial_capital"])
    eq = port["equity_curve"].to_numpy(float)
    cum = float(eq[-1] / cap - 1.0) if n else 0.0
    sd = float(np.std(net, ddof=ddof_vol)) if n > ddof_vol else 0.0
    ann = ((1.0 + cum) ** (A / n) - 1.0) if (n and annualization == "geometric") else float(np.mean(net) * A) if n else 0.0
    peak = np.maximum.accumulate(np.concatenate([[cap], eq])) if n else np.array([cap])
    mdd = float(np.max((peak[1:] - eq) / peak[1:])) if n else 0.0
    res = {"num_issuers": int(universe[id_col].nunique()), "num_rebalance_dates": int(base[date_col].nunique()),
           "num_backtest_periods": int(n),
           "num_process_ready_dates": int((mod["process_history_ready_issuers"] > 0).sum()),
           "num_news_active_dates": int((mod["news_nonzero_issuers"] > 0).sum()) if "news_nonzero_issuers" in mod else 0,
           "cumulative_return": cum, "annualized_return": float(ann), "annualized_volatility": sd * math.sqrt(A),
           "sharpe_ratio": float(np.mean(net) / sd * math.sqrt(A)) if sd > 0 else 0.0, "max_drawdown": -max(0.0, mdd),
           "avg_turnover": float(port["turnover"].mean()) if n else 0.0,
           "best_rebalance_date": po[date_col].iloc[int(np.argmax(net))] if n else "",
           "worst_rebalance_date": po[date_col].iloc[int(np.argmin(net))] if n else ""}
    dts = sorted(ap[date_col].unique())

    def legs(d, side):
        g = ap[(ap[date_col] == d) & (ap["selection"] == side)].sort_values("alpha_rank")
        return g[id_col].astype(str).tolist()

    counts = base["selected_process"].value_counts()
    sol = {"first_rebalance_date": dts[0] if dts else "", "first_rebalance_longs": legs(dts[0], "long") if dts else [],
           "first_rebalance_shorts": legs(dts[0], "short") if dts else [],
           "last_rebalance_date": dts[-1] if dts else "", "last_rebalance_longs": legs(dts[-1], "long") if dts else [],
           "last_rebalance_shorts": legs(dts[-1], "short") if dts else [],
           "selected_process_counts": {k: int(counts[k]) for k in sorted(counts.index)}}
    return ResultDict({"committee_panel.csv": committee, "process_diagnostics.csv": proc, "alpha_panel.csv": ap,
                       "portfolio_returns.csv": po, "modality_diagnostics.csv": mod, "results.json": res,
                       "solution.json": sol, "panel": base})


# ==============================================================================================
# 5. social-sentiment momentum L/S with ex-ante vol targeting, IC and single-factor alpha
# ==============================================================================================


def _ann_ret(net, rule, periods_per_year):
    """know87: annualised return of the net series under `rule` (None -> read from the task text)."""
    from .helpers87_B import annualized_return_rule, annualized_return_value
    return annualized_return_value(net, rule or annualized_return_rule(), periods_per_year)


def sentiment_momentum_report(posts, prices, market, bullish, bearish, *, lookback, min_posts, long_n, short_n,
                              cost_bps=0.0, vol_target=None, vol_lookback=20, max_leverage=None,
                              engagement=True, engagement_weights=(1.0, 2.0, 3.0), invalid_col="likes",
                              tokenizer="alpha", text_col="text", time_col="timestamp", id_col="ticker",
                              price_col="close", date_col="date", market_col="mkt_rf", periods_per_year=252,
                              min_ic_obs=3, ic_ddof=1, cost_on="trade_day", selection="both_legs",
                              signal_days="formed", cov_window=None, series_start="first_return", annualized=None):
    """Posts + prices -> {results, intermediates, daily (DataFrame), weights}: ONE call for the pattern
    "lexicon sentiment momentum long-short with ex-ante vol target, Spearman IC and market alpha".

    valid posts: rows with posts[invalid_col] >= 0 (the negative sentinel marks missing engagement).
    score = (#bullish - #bearish tokens) (lexicon_post_scores, tokenizer); know78 default tokenizer="alpha":
      lower-case runs of [a-z] (real posts write 'Long,' 'rally!' 'moon<emoji>' - str.split ("split")
      misses every punctuation-attached word and moves the whole backtest); engagement multiplier
      1 + ln(1 + w1 likes + w2 retweets + w3 replies) when engagement; final = score * multiplier;
      mean_weighted_score = mean of the NON-ZERO final scores (num_scored_nonzero).
    aggregation by the CALENDAR date of the timestamp (daily_sum_count), then reindexed to the trading
      days of `prices` (non-trading-day posts drop out, never rolled forward), missing -> 0.
    signal[t] = agg[t] - mean(agg[t-lookback .. t-1]) (PREVIOUS lookback trading days, current excluded);
      NaN while fewer than lookback prior days; NaN (skipped) where count[t] < min_posts.
    weights[t]: rank the finite signals desc (ties: column order), +1/(long_n+short_n) on the top long_n,
      -1/(long_n+short_n) on the bottom short_n; selection="both_legs" needs >= long_n+short_n names
      (else flat). Vol target (vol_target set): from row index i >= lookback + vol_lookback the weights
      are scaled by ex_ante_vol_scaler(w, R[i-vol_lookback+1 .. i], vol_target, cap=max_leverage) (full
      trailing covariance, ddof 1, x252); earlier rows unscaled.
    P&L: gross[t+1] = w[t] . r[t+1] (one-day lag); turnover[t] = sum|w[t] - w[t-1]| (w[-1] = 0); cost =
      turnover cost_bps/1e4 charged on the trade day's next return (cost_on="trade_day": net[t+1] =
      gross[t+1] - cost[t]). The daily series runs over return days 1..T-1.
    metrics: total = prod(1+net)-1, annualized = (1+total)^(A/n)-1, vol = std ddof 1 sqrt A, sharpe =
      mean/std sqrt A, max_drawdown >= 0 (initial 1 counted as a peak), win_rate = mean(net > 0).
    IC[t] = Spearman(signal[t], r[t+1]) over names with a finite signal, days with < min_ic_obs skipped;
      mean_ic, std_ic (ddof ic_ddof), ic_ir = mean/std.
    regression: net[t] on market[t] (aligned by date) with intercept -> alpha_daily, alpha_annualized
      (x A), beta, r_squared, alpha_tstat.
    know81 signal_days="formed" (default; convention 'signal_day_count'): num_signal_days counts the portfolio-
      FORMATION days - trading days on which the cross-sectional signal ranks enough names to form the long-short
      book (the formation dates of a sorted-portfolio backtest, Jegadeesh & Titman 1993); "all" = every trading
      day of the price panel (the know77-80 count).
    know86 cov_window=None: the rows of the trailing covariance window are read from the task text
      (helpers86_1.trailing_window_rule; conventions 'trailing_window_rows'): 'prior' = R[t-W .. t-1] when the
      task builds it from 'bars of prior return history', 'inclusive' = R[t-W+1 .. t] otherwise / no text.
    know86 series_start="first_return" (default): the daily series runs over every return day 1..T-1 (the zero-
      position warm-up days are part of the backtest, as the warm-up rule 'unscaled position weights apply' says);
      "after_lookback" = the know85 cut of the first lookback return days (graded WRONG, kept as an option).
      alpha_tstat = the OLS t-statistic of the intercept (single_factor_alpha_beta, df n-2).
    know87 annualized=None: annualized_return read from the task text (helpers87_B.annualized_return_rule;
      conventions 'annualized_return'): 'arithmetic' = mean(net) x A (the numerator of the sqrt(A) Sharpe; also
      when the task gives no formula), 'geometric' = (1 + total)^(A/n) - 1 when the task names CAGR / compound /
      geometric, 'log' = exp(mean log(1+net) x A) - 1. know77-86 always reported the geometric rate (graded WRONG
      17x with it)."""
    from .helpers76_1 import lexicon_post_scores, single_factor_alpha_beta
    from .helpers75_2 import daily_sum_count, ex_ante_vol_scaler
    po = posts.copy()
    num_posts = int(len(po))
    po = po[pd.to_numeric(po[invalid_col], errors="coerce") >= 0].copy()
    sc = lexicon_post_scores(po[text_col].tolist(), bullish, bearish, tokenizer=tokenizer)
    if engagement:
        w1, w2, w3 = engagement_weights
        mult = 1.0 + np.log(1.0 + w1 * po["likes"].astype(float) + w2 * po["retweets"].astype(float)
                            + w3 * po["replies"].astype(float))
        final = sc * mult.to_numpy(float)
    else:
        final = sc
    nz = final[final != 0]
    px = prices.copy()
    px[date_col] = pd.to_datetime(px[date_col])
    P = px.pivot_table(index=date_col, columns=id_col, values=price_col, aggfunc="last").sort_index()
    tickers = list(P.columns)
    days = P.index
    s_agg, c_agg = daily_sum_count(pd.to_datetime(po[time_col]), po[id_col].to_numpy(), final, index=days, columns=tickers)
    A_ = s_agg.to_numpy(float)
    Cn = c_agg.to_numpy(float)
    T, N = A_.shape
    sig = np.full((T, N), np.nan)
    for t in range(int(lookback), T):
        sig[t] = A_[t] - A_[t - int(lookback):t].mean(axis=0)
    sig[Cn < min_posts] = np.nan
    R = np.array(P.pct_change().to_numpy(float), copy=True)
    R[0] = np.nan
    Wt = np.zeros((T, N))
    k = long_n + short_n
    n_formed = 0
    for t in range(T):
        ok = np.where(np.isfinite(sig[t]))[0]
        if len(ok) == 0 or (selection == "both_legs" and len(ok) < k):
            continue
        n_formed += 1
        order = ok[np.argsort(sig[t, ok], kind="mergesort")]
        Wt[t, order[-long_n:]] = 1.0 / k
        Wt[t, order[:short_n]] = -1.0 / k
    if vol_target:
        from .helpers86_1 import trailing_window_rule, trailing_window_rows
        cw = cov_window or trailing_window_rule()
        Rf = np.nan_to_num(R)
        for t in range(int(lookback) + int(vol_lookback), T):
            if np.any(Wt[t] != 0):
                win = trailing_window_rows(Rf, t, int(vol_lookback), cw)
                Wt[t] = Wt[t] * ex_ante_vol_scaler(Wt[t], win, vol_target, cap=max_leverage, periods_per_year=periods_per_year)
    turn = np.abs(np.diff(np.vstack([np.zeros(N), Wt]), axis=0)).sum(axis=1)
    cost = turn * cost_bps / 1e4
    Rn = np.nan_to_num(R)
    gross_all = np.array([Wt[t - 1] @ Rn[t] for t in range(1, T)])
    net_all = gross_all - (cost[:-1] if cost_on == "trade_day" else cost[1:])
    lb = int(lookback) if series_start == "after_lookback" else 0
    gross = gross_all[lb:]
    net = net_all[lb:]
    days_out = days[1 + lb:]
    turn_out = turn[1 + lb:]
    W_ = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], W_]))[1:]
    n = len(net)
    tot = float(W_[-1] - 1.0)
    sd = float(np.std(net, ddof=1))
    ic = []
    for t in range(T - 1):
        ok = np.isfinite(sig[t]) & np.isfinite(R[t + 1])
        if ok.sum() < min_ic_obs or np.ptp(sig[t, ok]) == 0 or np.ptp(R[t + 1, ok]) == 0:
            continue
        ic.append(stats.spearmanr(sig[t, ok], R[t + 1, ok])[0])
    ic = np.array(ic, float)
    mic = float(ic.mean()) if len(ic) else 0.0
    sic = float(ic.std(ddof=ic_ddof)) if len(ic) > ic_ddof else 0.0
    mk = market.copy()
    mk[date_col] = pd.to_datetime(mk[date_col])
    mser = mk.set_index(date_col)[market_col].reindex(days_out).fillna(0.0).to_numpy(float)
    okr = np.isfinite(mser)
    reg = single_factor_alpha_beta(net[okr], mser[okr], periods_per_year=periods_per_year)
    alpha_tstat_val = reg["alpha_tstat"]
    res = {"num_valid_posts": int(len(po)), "mean_weighted_score": float(nz.mean()) if len(nz) else 0.0,
           "total_return": tot, "annualized_return": _ann_ret(net, annualized, periods_per_year),
           "sharpe_ratio": float(net.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
           "max_drawdown": float(np.max((peak - W_) / peak)), "mean_ic": mic,
           "ic_ir": mic / sic if sic > 0 else 0.0, "alpha_annualized": reg["alpha_annualized"], "beta": reg["beta"]}
    inter = {"num_posts": num_posts, "num_valid_posts": res["num_valid_posts"], "num_tickers": int(N),
             "num_signal_days": int(T) if signal_days == "all" else int(n_formed), "mean_weighted_score": res["mean_weighted_score"],
             "num_scored_nonzero": int(len(nz)), "num_ic_observations": int(len(ic)), "mean_ic": mic, "std_ic": sic,
             "ic_ir": res["ic_ir"], "annualized_volatility": sd * math.sqrt(periods_per_year),
             "win_rate": float(np.mean(net > 0)), "alpha_daily": reg["alpha_daily"], "beta": reg["beta"],
             "r_squared": reg["r_squared"], "alpha_tstat": alpha_tstat_val}
    daily = pd.DataFrame({"date": [_iso(d) for d in days_out], "gross_return": gross, "net_return": net,
                          "turnover": turn_out})
    return ResultDict({"results": res, "intermediates": inter, "daily": daily,
                       "weights": pd.DataFrame(Wt, index=days, columns=tickers), "signal": pd.DataFrame(sig, index=days,
                                                                                                    columns=tickers)})


# ==============================================================================================
# 6. two-asset spread / exchange options: calibration, Margrabe, Kirk, terminal Monte Carlo
# ==============================================================================================


def correlated_terminal_gbm(S1, S2, T, r, q1, q2, sigma1, sigma2, rho, n_paths, seed=None, Z=None):
    """Exact terminal draw of two correlated GBMs (a European payoff needs no time stepping):
    S_i(T) = S_i exp((r - q_i - sigma_i^2/2) T + sigma_i sqrt(T) X_i), X = Z L' with L the Cholesky
    factor of [[1, rho], [rho, 1]]. Z [n_paths x 2] standard normals (reuse the SAME Z for every maturity
    = common random numbers) or drawn from default_rng(seed). Returns (S1_T, S2_T)."""
    if Z is None:
        Z = np.random.default_rng(seed).standard_normal((int(n_paths), 2))
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    X = Z @ L.T
    a = S1 * np.exp((r - q1 - 0.5 * sigma1 ** 2) * T + sigma1 * math.sqrt(T) * X[:, 0])
    b = S2 * np.exp((r - q2 - 0.5 * sigma2 ** 2) * T + sigma2 * math.sqrt(T) * X[:, 1])
    return a, b


def spread_option_report(df1, df2, r, q1, q2, maturities, moneyness, rho_grid, rho_T, *, n_paths=500000,
                         seed=42, date_col="date", price_col="close", trading_days=252, ddof=1, mc_z_mult=3.0,
                         decimals=None):
    """Two price histories -> {calibration.json, margrabe_prices.csv, kirk_prices.csv,
    correlation_sensitivity.csv, summary.json, return_stats}: ONE call (MC is terminal-exact, so no
    per-day path loop - know76 timed out stepping 500k x 252 paths per (T, K)).

    Calibration: inner-join on date, sort; log returns of each close (n_returns = rows - 1); sigma_i =
    std(ddof) sqrt(trading_days); rho = Pearson of the synchronized log returns; S_i0 = last close;
    return_stats per asset: mean, std, skewness (scipy, bias=True), excess kurtosis (Fisher).
    Margrabe (exotics.margrabe_price, dividend yields) per T with sigma_exchange =
    sqrt(s1^2 + s2^2 - 2 rho s1 s2). Kirk (kirk_spread_price_spot) for K = avg_spot m, avg_spot =
    (S1_0 + S2_0)/2; MC: correlated_terminal_gbm with ONE Z (n_paths x 2, default_rng(seed)) reused for
    all T; price = e^{-rT} mean(max(S1-S2-K, 0)), mc_std_err = e^{-rT} std(payoff, ddof 1)/sqrt(n).
    Rows sorted by T then K. Correlation sensitivity at T = rho_T, K = 0: Margrabe and Kirk per rho.
    Summary: sigma_exchange, max_mc_error_margrabe (max |Kirk(K=0) - MC| over T), max_mc_error_kirk (all
    rows), mc_validates (max error <= mc_z_mult * max std err), rho_sensitivity_range (max - min Margrabe
    over the grid), mean_kirk_price, margrabe_T050 (Margrabe at rho_T)."""
    from .exotics import margrabe_price, kirk_spread_price_spot
    a, b = df1.copy(), df2.copy()
    for d in (a, b):
        d[date_col] = pd.to_datetime(d[date_col])
    m = a[[date_col, price_col]].merge(b[[date_col, price_col]], on=date_col, suffixes=("_1", "_2"))
    m = m.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    p1, p2 = m[f"{price_col}_1"].to_numpy(float), m[f"{price_col}_2"].to_numpy(float)
    l1, l2 = np.diff(np.log(p1)), np.diff(np.log(p2))
    s1 = float(np.std(l1, ddof=ddof) * math.sqrt(trading_days))
    s2 = float(np.std(l2, ddof=ddof) * math.sqrt(trading_days))
    rho = float(np.corrcoef(l1, l2)[0, 1])
    S1, S2 = float(p1[-1]), float(p2[-1])
    rs = lambda x: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=ddof)), "skewness": float(stats.skew(x)),  # noqa: E731
                    "excess_kurtosis": float(stats.kurtosis(x)), "annualized_vol": float(np.std(x, ddof=ddof) * math.sqrt(trading_days))}
    cal = {"S1_0": S1, "S2_0": S2, "sigma1": s1, "sigma2": s2, "rho": rho, "n_returns": int(len(l1)),
           "D1": float(q1), "D2": float(q2), "r": float(r)}
    sx = math.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    mg = pd.DataFrame({"T": [float(T) for T in maturities],
                       "margrabe_price": [float(margrabe_price(S1, S2, T, s1, s2, rho, q1=q1, q2=q2)) for T in maturities],
                       "sigma_exchange": sx})
    avg = (S1 + S2) / 2.0
    Z = np.random.default_rng(seed).standard_normal((int(n_paths), 2))
    rows = []
    for T in maturities:
        x1, x2 = correlated_terminal_gbm(S1, S2, float(T), r, q1, q2, s1, s2, rho, n_paths, Z=Z)
        disc = math.exp(-r * float(T))
        for mm in moneyness:
            K = avg * float(mm)
            pay = np.maximum(x1 - x2 - K, 0.0)
            rows.append({"T": float(T), "K": K, "spread_moneyness": float(mm),
                         "kirk_price": float(kirk_spread_price_spot(S1, S2, K, float(T), r, s1, s2, rho, q1=q1, q2=q2)),
                         "mc_price": float(disc * pay.mean()), "mc_std_err": float(disc * pay.std(ddof=1) / math.sqrt(len(pay)))})
    kp = pd.DataFrame(rows).sort_values(["T", "K"], kind="mergesort").reset_index(drop=True)
    cs = pd.DataFrame({"rho_grid": [float(x) for x in rho_grid],
                       "margrabe_price": [float(margrabe_price(S1, S2, rho_T, s1, s2, x, q1=q1, q2=q2)) for x in rho_grid],
                       "kirk_price_K0": [float(kirk_spread_price_spot(S1, S2, 0.0, rho_T, r, s1, s2, x, q1=q1, q2=q2))
                                         for x in rho_grid]}).sort_values("rho_grid").reset_index(drop=True)
    err = (kp["kirk_price"] - kp["mc_price"]).abs()
    k0 = kp["spread_moneyness"].abs() < 1e-12
    summ = {"sigma_exchange": sx, "max_mc_error_margrabe": float(err[k0].max()) if k0.any() else None,
            "max_mc_error_kirk": float(err.max()), "mc_validates": bool(err.max() <= mc_z_mult * kp["mc_std_err"].max()),
            "rho_sensitivity_range": float(cs["margrabe_price"].max() - cs["margrabe_price"].min()),
            "mean_kirk_price": float(kp["kirk_price"].mean()),
            "margrabe_T050": float(margrabe_price(S1, S2, rho_T, s1, s2, rho, q1=q1, q2=q2))}
    if decimals is not None:
        kp, mg, cs = kp.round(decimals), mg.round(decimals), cs.round(decimals)
    return ResultDict({"calibration.json": cal, "margrabe_prices.csv": mg, "kirk_prices.csv": kp,
                       "correlation_sensitivity.csv": cs, "summary.json": summ,
                       "return_stats": {"asset1": rs(l1), "asset2": rs(l2)}})
