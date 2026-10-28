"""know81 group-1 helpers: convention choices read from the task's own words through the conventions reference
table (theory_library/conventions.jsonl, qf_toolkit.conventions_ref). Nothing here is tied to one task.

cta_conventions_from_text(text) - the trend-signal warm-up, the GARCH fitting scale and the drawdown sign a
  CTA / vol-target task states in words (table ids warmup_signal, garch_scale, drawdown_sign, std_ddof).
log_performance_table(x, ...) - performance metrics of a LOG-return series with the table's textbook
  definitions (additive annualisation of log returns, excess-return Sharpe, drawdown sign as a switch).
carry_accrual_from_text(text) - the daily carry accrual of an FX carry backtest (table id carry_accrual).
"""
from __future__ import annotations

import math

import numpy as np


def _text(text=None) -> str:
    if text:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


def _suggested(cid: str, text: str):
    """(suggested option value or None, quoted evidence) of one table decision for this text."""
    try:
        import re
        from .conventions_ref import convention, conventions_table
        ent = next((e for e in conventions_table() if e.get("id") == cid), None)
        if ent is None or not re.search(ent.get("applies") or "$^", str(text).lower(), re.I):
            return None, ""          # the decision does not apply to this task
        c = convention(cid, text)
        return c.get("suggested"), c.get("evidence", "")
    except Exception:  # noqa: BLE001 - a missing table entry means 'task silent'
        return None, ""


def cta_conventions_from_text(text=None) -> dict:
    """-> {warmup?, fit_scale?, mdd_sign?, ddof?, evidence} - only the keys the task's words settle.

    warmup (table warmup_signal): 'binary ...' -> "short" (the else-branch, e.g. -1, while an EMA is undefined;
      "A when X, -1 otherwise ... no 'no position' state"), 'flat ...' -> "zero", 'drop ...' -> "mask".
    fit_scale (table garch_scale): 'fit on decimal returns' (task says native units / rescale=False / do not
      rescale) -> 1.0; the percent option -> 100.0.
    mdd_sign (table drawdown_sign): "positive" | "negative".
    ddof (table std_ddof): 1 for 'sample', 0 for 'population'.
    An empty text (no instruction available) returns {"evidence": {}}: the caller's own arguments stand."""
    t = _text(text)
    out, ev = {}, {}
    if not t:
        return {"evidence": ev}
    v, e = _suggested("warmup_signal", t)
    if v:
        k = str(v).split(":")[0].strip().lower()
        m = {"binary": "short", "flat": "zero", "drop": "mask"}.get(k)
        if m:
            out["warmup"], ev["warmup"] = m, e
    v, e = _suggested("garch_scale", t)
    if v:
        out["fit_scale"] = 1.0 if "decimal" in str(v).lower() else 100.0
        ev["fit_scale"] = e
    v, e = _suggested("drawdown_sign", t)
    if v:
        out["mdd_sign"] = "positive" if "positive" in str(v).lower() else "negative"
        ev["mdd_sign"] = e
    v, e = _suggested("std_ddof", t)
    if v:
        out["ddof"] = 0 if "ddof=0" in str(v) else 1
        ev["ddof"] = e
    out["evidence"] = ev
    return out


def log_performance_table(x, periods_per_year=252, rf_annual=0.0, ddof=1, mdd_sign="negative") -> dict:
    """Metrics of a daily LOG-return series, textbook definitions (conventions table: annualisation, sharpe,
    std_ddof, drawdown_sign):
      annualized_return = mean(x) * A        (log returns add over time: Campbell-Lo-MacKinlay 1997 ch.1;
                                              the annualisation factor A multiplies the mean - Lo 2002)
      annualized_volatility = std(x, ddof) * sqrt(A)
      sharpe_ratio = (annualized_return - rf_annual) / annualized_volatility   (Sharpe 1994: excess return)
      wealth W = exp(cumsum x) with the initial capital W_0 = 1 as the first peak;
      drawdown DD = max_t (peak_t - W_t) / peak_t >= 0 (Magdon-Ismail & Atiya 2004);
      max_drawdown = +DD (mdd_sign="positive") or -DD ("negative");
      calmar_ratio = annualized_return / DD  (Young 1991: return over the drawdown magnitude).
    Returns the dict above plus max_drawdown_abs."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    A = float(periods_per_year)
    ann = float(v.mean() * A)
    vol = float(v.std(ddof=int(ddof)) * math.sqrt(A))
    W = np.exp(np.r_[0.0, np.cumsum(v)])
    pk = np.maximum.accumulate(W)
    dd = float(np.max((pk - W) / pk))
    if mdd_sign not in ("positive", "negative"):
        raise ValueError("mdd_sign must be 'positive' or 'negative'")
    sh = (ann - float(rf_annual)) / vol if vol > 0 else float("nan")
    return {"annualized_return": ann, "annualized_volatility": vol, "sharpe_ratio": sh,
            "max_drawdown": dd if mdd_sign == "positive" else -dd, "max_drawdown_abs": dd,
            "calmar_ratio": ann / dd if dd > 0 else float("nan")}


_CARRY_MAP = (("own day", "leg_calendar"), ("periods per year", "per_period"), ("home", "home_calendar"))


def carry_accrual_from_text(text=None, default="per_period") -> tuple:
    """-> (carry_accrual, evidence) for fx_carry_hedge_backtest from the table decision carry_accrual:
    "home_calendar" (a home-currency investor's single excess rate, differential_{t-1} x calendar days since
    the previous row / the home currency's money-market basis from the day-count file), "leg_calendar" (each
    leg on its own day count) or "per_period" (differential_{t-1} / periods per year on every row).
    `default` when the text is silent or matches several options."""
    t = _text(text)
    if not t:
        return default, ""
    v, e = _suggested("carry_accrual", t)
    if not v:
        return default, ""
    s = str(v).lower()
    for key, val in _CARRY_MAP:
        if key in s:
            return val, e
    return default, ""
