"""know80 group-1 helpers: conventions read from the task's own words.

note_conventions_from_text(text) - the structured-note decomposition choices an instruction may state in words
(markup base, the participation divisor, the VaR horizon). Defaults are the textbook readings; a keyword is only
switched when the instruction says so. structured_note_report(..., markup_base=None, n_options_base=None,
risk_horizon_days=None) calls it on qf_toolkit.instruction_text().

delta_hedge_trade(delta, notional, convention) - the spot hedge of an option position: the trade that
neutralises the option's delta is -delta x notional (sell base against a long call); 'exposure' gives +delta x N.
"""
import re

_MARKUP_FAIR = re.compile(
    r"mark[- ]?up[^.\n]{0,120}?(?:relative to|over|divided by|as a (?:percent(?:age)?|fraction|share) of|per unit of)"
    r"\s+(?:the\s+)?(?:note'?s?\s+)?(?:fair|theoretical|model|intrinsic)\s+value"
    r"|\(\s*(?:price|notional|issue[ _]price)\s*-\s*fair[ _]value\s*\)\s*/\s*fair[ _]value", re.I)
_MARKUP_NOTIONAL = re.compile(
    r"mark[- ]?up[^.\n]{0,120}?(?:relative to|over|divided by|as a (?:percent(?:age)?|fraction|share) of|per unit of)"
    r"\s+(?:the\s+)?(?:notional|issue price|face value|par)", re.I)
_NOPT_STRIKE = re.compile(r"(?:participation|n_options|number of options)[^.\n]{0,120}?"
                          r"(?:divided by|per unit of|/)\s*(?:the\s+)?(?:strike|K\b)", re.I)
_HORIZON = re.compile(r"\b(\d{1,3})[- ](?:business[- ]|trading[- ])?day\s+(?:var|value[- ]at[- ]risk|es\b|"
                      r"expected shortfall|horizon|holding period)", re.I)


def note_conventions_from_text(text=None) -> dict:
    """-> {markup_base, n_options_base, risk_horizon_days, evidence} for a zero-bond + option note.

    markup_base: "fair" only when the text puts the markup over the fair / theoretical value (markup over
      cost, 100 (N - fair)/fair); otherwise "notional" (issuer margin on the issue price, 100 (N - fair)/N).
    n_options_base: "K" only when the text divides the participation by the strike; otherwise "S0"
      (participation p on the underlying's return: p N/S0 options).
    risk_horizon_days: the first "<h>-day VaR / horizon / holding period" in the text, else 1 (1-day VaR/ES).
    text=None reads qf_toolkit.instruction_text() ('' when unavailable -> all defaults)."""
    if text is None:
        try:
            from .helpers78_4 import instruction_text
            text = instruction_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
    t = str(text)
    ev = {}
    mb = "notional"
    m = _MARKUP_FAIR.search(t)
    if m and not _MARKUP_NOTIONAL.search(t):
        mb, ev["markup_base"] = "fair", m.group(0)[:120]
    nb = "S0"
    m = _NOPT_STRIKE.search(t)
    if m:
        nb, ev["n_options_base"] = "K", m.group(0)[:120]
    h = 1
    m = _HORIZON.search(t)
    if m:
        h, ev["risk_horizon_days"] = int(m.group(1)), m.group(0)[:120]
    return {"markup_base": mb, "n_options_base": nb, "risk_horizon_days": h, "evidence": ev}


def delta_hedge_trade(delta, notional, convention="offset"):
    """Spot hedge amount of an option position in base-currency (underlying) units.
    convention="offset": the hedge TRADE = -delta x notional (a long call with delta +0.25 on 10m base is hedged
    by selling 2.5m base; a long put with delta -0.25 by buying 2.5m); "exposure": +delta x notional."""
    if convention not in ("offset", "exposure"):
        raise ValueError("convention must be 'offset' or 'exposure'")
    return (-1.0 if convention == "offset" else 1.0) * float(delta) * float(notional)
