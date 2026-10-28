"""Convention checks that only fire when the instruction itself gives the evidence.

The playbook already TELLS the model that VaR/ES are positive loss magnitudes and that
stated levels are written as given; graded runs still shipped `var_95 = -0.051` and a
confidence column holding 1.0 where the task named 0.999. Text did not bind the model, so
these rules are executed: a violation becomes an invariant failure the repair loop sees.

Every check here is evidence-bound, because a self-check that is wrong destroys correct
answers (v6's examiner rewrites did exactly that):

* risk_sign: only when the task talks about VaR / expected shortfall at all, only on
  fields named as those measures, and never when the task says they are reported as
  negative numbers.
* rounded_level: only a literal the task itself writes with >= 3 decimals (0.999, 0.995),
  only in a low-cardinality column of probabilities, and only when that column holds the
  literal's 2-decimal rounding instead of the literal.

A rule that cannot be tied to the task's own text (for example "the cleaned table must
end at the last input timestamp") is deliberately NOT here.
"""

from __future__ import annotations

import math
import re

try:
    import pandas as pd
except ImportError:  # pragma: no cover - the image carries pandas
    pd = None

from .universal import Violation

#: The task mentions a tail-risk measure at all.
_MENTIONS_RISK = re.compile(
    r"\bVaR\b|\bCVaR\b|\bES\b|value[- ]at[- ]risk|expected[- ]shortfall|conditional[- ]value",
    re.IGNORECASE)
#: The task states that these measures are reported as negative numbers.
_NEGATIVE_CONVENTION = re.compile(
    r"(?:\bvar\b|value[- ]at[- ]risk|shortfall|\bcvar\b|\bes\b)[^.\n]{0,60}"
    r"\b(?:as|is|are|be|reported\s+as)\s+(?:a\s+)?negative\b"
    r"|negative\s+(?:numbers?|values?|returns?)[^.\n]{0,40}(?:\bvar\b|shortfall|\bcvar\b)",
    re.IGNORECASE)
#: A field named as a VaR / ES value: var, var_95, var_99_normal, cvar_95, es_975,
#: expected_shortfall, value_at_risk_99 ... but not variance-like or test-statistic names.
_RISK_NAME = re.compile(
    r"(?:^|_)(?:c?var|es|etl|expected_shortfall|value_at_risk)(?:_?\d{1,3}(?:p\d+)?)?(?:_|$)",
    re.IGNORECASE)
_NOT_A_RISK_VALUE = re.compile(
    r"ratio|change|diff|delta|error|breach|exceed|violation|count|hit|p_?value|pval|stat|"
    r"lr_|_lr|test|kupiec|christoffersen|rank|weight|contrib|share|pct_of|residual|resid|"
    r"variance|covar|garch|return|exponent|index|level|alpha|conf",
    re.IGNORECASE)

#: Decimal literals such as 0.999 / 0.9975 written in the task.
_LEVEL_LITERAL = re.compile(r"(?<![\d.])0\.\d{3,4}(?!\d)")


def _risk_name(name: str) -> bool:
    return bool(_RISK_NAME.search(name)) and not _NOT_A_RISK_VALUE.search(name)


def risk_sign(frame=None, payload=None, instruction: str = "") -> list[Violation]:
    """VaR / ES values must be non-negative loss magnitudes unless the task says otherwise."""
    if not instruction or not _MENTIONS_RISK.search(instruction):
        return []
    if _NEGATIVE_CONVENTION.search(instruction):
        return []
    out: list[Violation] = []

    def flag(name, value, scale):
        if value < -1e-9 * max(scale, 1.0):
            out.append(Violation(
                "convention",
                f"{name!r} = {value:.6g} is negative, but VaR / expected shortfall are "
                "positive loss magnitudes and the task does not state a negative "
                "convention; report the loss as a positive number"))

    if frame is not None and pd is not None and len(frame):
        for column in frame.select_dtypes(include="number").columns:
            if not _risk_name(str(column)):
                continue
            values = frame[column].dropna()
            if len(values):
                worst = float(values.min())
                flag(column, worst, float(values.abs().max()))
    if payload is not None:
        for key, value in _numeric_items(payload):
            if _risk_name(key):
                flag(key, value, abs(value))
    return out[:4]


def _numeric_items(payload, prefix: str = "", depth: int = 0):
    """(key, number) pairs from a JSON payload, three levels deep."""
    if depth > 3:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                yield name, float(value)
            elif isinstance(value, (dict, list)):
                yield from _numeric_items(value, name, depth + 1)
    elif isinstance(payload, list):
        for item in payload[:200]:
            yield from _numeric_items(item, prefix, depth + 1)


def rounded_level(frame, instruction: str = "") -> list[Violation]:
    """A level the task writes as 0.999 must not appear in the output as 1.0 (or 0.99x -> 0.99)."""
    if frame is None or pd is None or not instruction or not len(frame):
        return []
    # Levels only: confidence / probability levels live in [0.5, 1). A replay
    # over 303 graded-pass outputs flagged 19 correct answers when small
    # literals (0.013, 0.001) "rounded" to a legitimate 0.0 in an error column.
    literals = {float(x) for x in _LEVEL_LITERAL.findall(instruction)}
    literals = {x for x in literals if 0.5 <= x < 1.0}
    if not literals:
        return []
    out: list[Violation] = []
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        distinct = sorted({float(v) for v in values.unique()})
        if not distinct or len(distinct) > 12 or min(distinct) < 0.5 or max(distinct) > 1:
            continue
        present = lambda x: any(abs(x - d) <= 1e-9 for d in distinct)  # noqa: E731
        for literal in sorted(literals):
            if present(literal):
                continue
            for digits in (2, 1):
                rounded = round(literal, digits)
                if abs(rounded - literal) > 1e-12 and present(rounded):
                    out.append(Violation(
                        "convention",
                        f"column {column!r} holds {rounded} where the task states "
                        f"{literal}: the value was rounded on write; write stated "
                        "levels exactly (no float_format / round on key columns)"))
                    break
    return out[:4]


def check(path, contract, instruction: str) -> list[Violation]:
    """Run every evidence-bound convention check on one written deliverable."""
    import json
    import pathlib

    p = pathlib.Path(path)
    if not instruction or not p.is_file():
        return []
    try:
        if contract.fmt == "json":
            payload = json.loads(p.read_text(encoding="utf-8"))
            return risk_sign(payload=payload, instruction=instruction)
        if pd is None:
            return []
        if contract.fmt == "csv":
            frame = pd.read_csv(p)
        elif contract.fmt == "parquet":
            frame = pd.read_parquet(p)
        else:
            return []
    except Exception:  # noqa: BLE001 - unreadable files are universal.check_contract's job
        return []
    return risk_sign(frame=frame, instruction=instruction) + rounded_level(frame, instruction)
