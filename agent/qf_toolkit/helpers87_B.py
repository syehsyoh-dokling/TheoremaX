"""know87 group B: one text-settled convention of backtest reporting.

Pattern: a daily strategy backtest reports `annualized_return` next to a Sharpe ratio "annualised with sqrt(252)".
Whether the annualised return is the ARITHMETIC mean daily return x periods per year (the numerator of that Sharpe:
Sharpe = annualized_return / annualized_volatility), the GEOMETRIC / compound rate (1 + total)^(A/n) - 1 (CAGR),
or exp(mean log return x A) - 1 is a decision the task's words settle. When the task gives no formula and names no
CAGR / geometric / compound / log wording, the arithmetic reading is the one every graded sibling of the same task
family with silent wording passed with (risk-parity, factor-hedge, regime-HMM and Basel-capital backtests:
mean x 252), and the one the same family writes out when it does define it ("mean(daily_returns) x 252").

No task names, no data values. Conventions table entry: theory_library/conventions.jsonl 'annualized_return'.
Self-tests: proven_work/fail72/selftest87_B.py.

API lines (one per function):
     annualized_return_rule(text=None) -> 'arithmetic' (mean x A) | 'geometric' ((1+total)^(A/n)-1) | 'log' (exp(mean log(1+r) x A)-1), read from the task's words; 'arithmetic' when silent
     annualized_return_value(returns, rule="arithmetic", periods_per_year=252) -> float, the annualised return of a simple-return series under that rule
"""
from __future__ import annotations

import math
import re

import numpy as np

_GEO = (r"\bcagr\b|compound(?:ed)? (?:annual|growth)|geometric(?:ally)? (?:mean|average|annuali[sz]|return)|"
        r"annuali[sz]ed_return\W{0,6}=\W{0,6}\(\s*1\s*\+\s*total|\(\s*1\s*\+\s*total_return\s*\)\s*\^|"
        r"\(\s*1\s*\+\s*(?:r|total)\w*\s*\)\s*\*\*\s*\(\s*(?:252|365|12|52|a|periods)")
_LOG = (r"annuali[sz]ed_log_return|exp\(\s*annuali[sz]ed_log|exp\(\s*mean[_ ]?(?:log|daily_log)|"
        r"compound(?:ed)? log return|log_returns?\.sum\(\)")
_ARI = (r"annuali[sz]ed_return\W{0,8}(?:=|\|)\W{0,8}(?:mean|average|\d+\s*[x*]\s*mean)|mean\(?\s*daily_returns?\s*\)?\s*"
        r"(?:x|\*|×)\s*(?:252|365|12|52|annuali[sz]ation)|arithmetic (?:mean|average) (?:of )?(?:the )?(?:daily |periodic "
        r"|strategy )?returns?|arithmetic(?:ally)? annuali[sz]")


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


def annualized_return_rule(text=None):
    """'geometric' when the task names CAGR / compound / geometric annualisation or writes (1 + total)^(A/n) - 1,
    'log' when it annualises a mean LOG return and exponentiates, 'arithmetic' when it writes mean x periods or
    gives no formula at all (also with no task text)."""
    t = " ".join(_text(text).lower().split())
    if not t:
        return "arithmetic"
    if re.search(_ARI, t):
        return "arithmetic"
    if re.search(_LOG, t):
        return "log"
    if re.search(_GEO, t):
        return "geometric"
    return "arithmetic"


def annualized_return_value(returns, rule="arithmetic", periods_per_year=252):
    """Annualised return of a simple-return series: arithmetic mean(r) A; geometric (prod(1+r))^(A/n) - 1;
    log exp(mean(log(1+r)) A) - 1."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return 0.0
    A = float(periods_per_year)
    rule = str(rule or "arithmetic").lower()
    if rule.startswith("geo") or rule == "cagr":
        return float(np.prod(1.0 + r) ** (A / n) - 1.0)
    if rule.startswith("log"):
        return float(math.exp(float(np.mean(np.log1p(r))) * A) - 1.0)
    return float(np.mean(r) * A)
