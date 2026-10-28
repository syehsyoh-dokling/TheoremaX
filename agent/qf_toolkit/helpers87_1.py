"""know87: two whole-task patterns whose programs failed on hand-written glue around correct helpers.

- "GARCH(1,1) conditional vol -> closed-form option price -> central-FD Greeks -> delta-gamma-vega VaR split":
  the helper outputs were re-descaled by hand (omega / (1e4 (1 - a - b)) on an omega that is already decimal) and
  the vol-of-vol was rebuilt from a hand-made recursion instead of the fitted conditional-volatility series.
  garch_dgv_var_report does the whole chain from the documented helper outputs, in decimal units.
- "back-adjust a close/volume series for splits and cash dividends": the per-action factors were computed in one
  cell and their product re-read from keys that were never stored (cumulative factor 1.0). corporate_action_adjust
  computes every per-action factor once and returns the table, the factors and the summary from the SAME numbers.

Every convention is a keyword with the textbook default or a switch read from the task's own words; no task
names, no data values. Conventions table: theory_library/conventions.jsonl 'dividend_adjustment_close'.
Self-tests: proven_work/fail72/selftest87_1.py.

API lines (one per function):
     garch_dgv_var_report(log_returns, price_fn, S0, n_contracts, fd_dS_pct, fd_dsigma, level, mean="constant", periods_per_year=252) -> ResultDict{results{portfolio_var, delta/gamma/vega_var_pct}, intermediates{garch_omega, garch_alpha, garch_beta, garch_persistence, long_run_variance, garch_vol, analytical_price, delta, gamma, vega}, vol_of_vol_daily, sigma_daily}; price_fn(S, sigma) closed form
     dividend_close_rule(text=None) -> 'ex_date' (close on the ex-date, nearest prior trading day if absent) | 'prior' (last close strictly before the ex-date)
     corporate_action_adjust(prices, actions, dividend_close=None, price_decimals=None) -> dict{table DataFrame[date, close_unadjusted, close_adjusted, volume_unadjusted, volume_adjusted], actions [per-action factors], results{n_actions_applied, cumulative_price_adjustment_factor, cumulative_volume_adjustment_factor, first_date_adjusted_close, last_date_unadjusted_close, total_dividends_per_original_share}}
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


# ============================================================================= GARCH -> option -> DGV VaR
def garch_dgv_var_report(log_returns, price_fn, S0, n_contracts, fd_dS_pct, fd_dsigma, level, *,
                         mean="constant", periods_per_year=252):
    """GARCH(1,1) conditional vol into a closed-form option price, central finite-difference Greeks and a
    parametric delta-gamma-vega VaR decomposition, all from ONE fit and in decimal units.

    g = garch11_forecast(log_returns, mean, scale=100) (arch fit on percent returns, results back in DECIMAL:
    omega, long_run_variance = omega/(1-alpha-beta), sigma2_next = one-step-ahead variance after the last return);
    garch_vol = sqrt(A sigma2_next) (the conditional vol at the final observation, annualised) prices the option and
    every Greek (the long-run variance is reported only); gk = fd_greeks(price_fn, S0, garch_vol, fd_dS_pct*S0,
    fd_dsigma) (relative spot bump, absolute vol bump); vol-of-vol = std(diff(sqrt(A) cond_vol_t), ddof=1)
    rescaled to one day (the sqrt(A) factors cancel) = g['vol_of_vol_daily']; sigma_daily = garch_vol/sqrt(A);
    delta_gamma_vega_var(n, delta, gamma, vega, S0, sigma_daily, vov_daily, level): portfolio_var = z sqrt(total
    variance) (a VaR amount, not the variance), *_var_pct = shares of VARIANCE on 0-100 summing to 100.
    price_fn(S, sigma) is the closed-form pricer, e.g. lambda s, v: barrier_price(s, K, B, T, r, v, kind='call',
    barrier='down-and-out'). Returns ResultDict{results, intermediates {name: {'value': x}}, flat values}:
        for name, obj in json_deliverables(rep).items(): emit(name, obj)"""
    from .helpers75_1 import garch11_forecast, fd_greeks, delta_gamma_vega_var
    from ._rdict import ResultDict
    r = np.asarray(log_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    A = float(periods_per_year)
    g = garch11_forecast(r, mean=mean, scale=100.0, dist="normal", periods_per_year=periods_per_year)
    garch_vol = float(math.sqrt(A * float(g["sigma2_next"])))
    gk = fd_greeks(price_fn, float(S0), garch_vol, float(fd_dS_pct) * float(S0), float(fd_dsigma))
    sig_d = garch_vol / math.sqrt(A)
    vov = float(g["vol_of_vol_daily"])
    dv = delta_gamma_vega_var(float(n_contracts), gk["delta"], gk["gamma"], gk["vega"], float(S0), sig_d, vov,
                              level=float(level))
    vals = {"garch_omega": float(g["omega"]), "garch_alpha": float(g["alpha"]), "garch_beta": float(g["beta"]),
            "garch_persistence": float(g["persistence"]), "long_run_variance": float(g["long_run_variance"]),
            "garch_vol": garch_vol, "analytical_price": float(gk["price"]), "delta": float(gk["delta"]),
            "gamma": float(gk["gamma"]), "vega": float(gk["vega"])}
    res = {"portfolio_var": float(dv["portfolio_var"]), "delta_var_pct": float(dv["delta_var_pct"]),
           "gamma_var_pct": float(dv["gamma_var_pct"]), "vega_var_pct": float(dv["vega_var_pct"])}
    out = {"results": res, "intermediates": {k: {"value": v} for k, v in vals.items()},
           "vol_of_vol_daily": vov, "sigma_daily": sig_d, "total_variance": float(dv["total_variance"])}
    out.update(vals)
    out.update(res)
    return ResultDict(out, aliases={"price": "analytical_price", "omega": "garch_omega", "alpha": "garch_alpha",
                                    "beta": "garch_beta", "persistence": "garch_persistence",
                                    "vov_daily": "vol_of_vol_daily", "var": "portfolio_var"})


# ============================================================================= corporate actions
_EX_DATE = (r"close[_ ]on[_ ](?:the[_ ])?ex[_ -]?(?:dividend[_ ])?date|close (?:price )?(?:on|at) the ex[- ](?:dividend )?"
            r"date|ex[- ]date close")
_PRIOR = (r"(?:previous|prior|preceding|last) (?:trading )?(?:day'?s? )?close (?:before|prior to) the ex|cum[- ]dividend "
          r"close|close (?:on|of) the (?:trading )?day before the ex|close_prev|prev(?:ious)?_close")


def dividend_close_rule(text=None):
    """Which close enters a cash dividend's price factor (P - D)/P. 'ex_date' when the task's words name the close
    ON the ex-dividend date ('close_on_ex_date'; an ex-date that is not a trading day falls back to the nearest
    prior trading day); 'prior' when they name the cum-dividend close (the last close strictly before the ex-date),
    and when the task is silent (CRSP / Yahoo adjustment factor 1 - D/P_{t-1})."""
    t = _text(text).lower()
    if re.search(_PRIOR, t):
        return "prior"
    if re.search(_EX_DATE, t):
        return "ex_date"
    return "prior"


def _split_ratio(a):
    raw = a.get("ratio", a.get("split_ratio", a.get("factor")))
    if isinstance(raw, str):
        m = re.match(r"\s*([\d.]+)\s*[:/-]\s*([\d.]+)\s*$", raw)
        if m:
            return float(m.group(1)) / float(m.group(2))
        return float(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]) / float(raw[1])
    return float(raw)


def corporate_action_adjust(prices, actions, *, dividend_close=None, price_decimals=None,
                            date_col="date", close_col="close", volume_col="volume"):
    """Backward adjustment of a close/volume series for stock splits and cash dividends (CRSP-style factors).

    Actions are processed in chronological order. For each action at date d the rows STRICTLY BEFORE d
    (date < d) are multiplied by its own factor (so every past row carries the product of the factors of all
    later actions; the most recent prices stay unadjusted):
      split 'a:b' (new_shares = a/b old_shares): s = a/b, price factor 1/s, volume factor s;
      cash dividend D: price factor (P - D)/P, volume factor 1, P = the close chosen by dividend_close
        ('ex_date': the close on d, or on the nearest prior trading day when d is not in the series;
         'prior': the last close strictly before d; None: dividend_close_rule(task text)).
      Dividend closes are the UNADJUSTED closes (a dividend amount is in the share terms of its own date).
    An action with no row before it adjusts nothing and contributes factor 1.
    cumulative_price_adjustment_factor = product of all price factors applied (the number that multiplies the
    oldest unadjusted price), cumulative_volume_adjustment_factor = product of all volume factors;
    total_dividends_per_original_share = sum of the dividend amounts as given. Volumes are rounded to integers;
    close_adjusted is rounded to price_decimals when given (first_date_adjusted_close follows the table).
    Returns dict{table, actions, results} - emit(<csv name>, out['table']); emit(<json name>, out['results'])."""
    df = prices.copy() if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    acts = actions.get("corporate_actions", actions.get("actions", actions)) if isinstance(actions, dict) else actions
    acts = sorted((dict(a) for a in acts), key=lambda a: pd.Timestamp(a["date"]))
    rule = dividend_close if dividend_close is not None else dividend_close_rule()
    if rule not in ("ex_date", "prior"):
        raise ValueError("dividend_close in {'ex_date', 'prior', None}")
    dates = df[date_col].to_numpy()
    close = df[close_col].to_numpy(dtype=float)
    vol = df[volume_col].to_numpy(dtype=float)
    pf_row = np.ones(len(df))
    vf_row = np.ones(len(df))
    rows, Fp, Fv, divs = [], 1.0, 1.0, 0.0
    for a in acts:
        d = np.datetime64(pd.Timestamp(a["date"]))
        n_before = int(np.searchsorted(dates, d, side="left"))       # rows with date < d
        kind = str(a.get("type", "")).lower()
        f_p, f_v, p_used = 1.0, 1.0, None
        if "split" in kind:
            s = _split_ratio(a)
            f_p, f_v = 1.0 / s, s
        elif "div" in kind:
            D = float(a.get("amount", a.get("dividend", 0.0)))
            divs += D
            k = int(np.searchsorted(dates, d, side="right")) - 1 if rule == "ex_date" else n_before - 1
            if k >= 0:
                p_used = float(close[k])
                f_p = (p_used - D) / p_used
        if n_before == 0:
            f_p, f_v = 1.0, 1.0
        pf_row[:n_before] *= f_p
        vf_row[:n_before] *= f_v
        Fp *= f_p
        Fv *= f_v
        rows.append({"date": pd.Timestamp(a["date"]).strftime("%Y-%m-%d"), "type": a.get("type"),
                     "price_factor": f_p, "volume_factor": f_v, "close_used": p_used, "rows_adjusted": n_before})
    adj = close * pf_row
    if price_decimals is not None:
        adj = np.round(adj, int(price_decimals))
    table = pd.DataFrame({"date": df[date_col].dt.strftime("%Y-%m-%d"), "close_unadjusted": close,
                          "close_adjusted": adj, "volume_unadjusted": df[volume_col].to_numpy(),
                          "volume_adjusted": np.rint(vol * vf_row).astype(np.int64)})
    res = {"n_actions_applied": int(len(acts)), "cumulative_price_adjustment_factor": float(Fp),
           "cumulative_volume_adjustment_factor": float(Fv),
           "first_date_adjusted_close": float(adj[0]) if len(adj) else float("nan"),
           "last_date_unadjusted_close": float(close[-1]) if len(close) else float("nan"),
           "total_dividends_per_original_share": float(divs)}
    return {"table": table, "actions": rows, "results": res, "dividend_close": rule}
