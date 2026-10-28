"""Reference table of academic / market conventions and its relation to a task's own words (know81).

`theory_library/conventions.jsonl` lists, per DECISION a quant task leaves to the solver (money-market day count,
annualisation, VaR quantile, GARCH mean, EMA seed, ...), every standard OPTION with the textbook / regulatory
source and a `when` pattern - the words a task uses when it means that option. Nothing in the table comes from a
task's data, checks or a stored solution: it is a reference book.

The relation: a decision is RELEVANT when its `applies` pattern occurs in the task text; an option is SUGGESTED
when its `when` pattern occurs there (the matched words are quoted back); with no match the table's default
stands. The model sees the decisions and decides; a program may look one up at run time:

    from qf_toolkit import convention
    c = convention("hist_var")          # reads the task text itself (instruction_text())
    c["suggested"], c["evidence"], c["options"], c["default"], c["source"]
"""
from __future__ import annotations

import json
import os
import pathlib
import re

_TABLE = pathlib.Path(__file__).resolve().parent.parent / "theory_library" / "conventions.jsonl"
_CACHE: list | None = None


def conventions_table() -> list[dict]:
    """Every decision in the reference table (a list of dicts; [] when the table is missing)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = []
        try:
            for line in _TABLE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        _CACHE.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
    return _CACHE


def _task_text(text=None) -> str:
    if text:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return os.environ.get("T1_INSTRUCTION", "")


def _quote(m: re.Match, text: str, width: int = 50) -> str:
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return " ".join(text[a:b].split())


def resolve(entry: dict, text: str) -> dict:
    """One decision against one task text: which options its words point to (with the quoted words)."""
    low = text.lower()
    hits = []
    for opt in entry.get("options", []):
        pat = opt.get("when")
        if not pat:
            continue
        m = re.search(pat, low, re.I)
        if m:
            hits.append({"value": opt.get("value"), "evidence": _quote(m, text)})
    return {"id": entry.get("id"), "decision": entry.get("decision"),
            "suggested": hits[0]["value"] if len(hits) == 1 else None,
            "candidates": hits, "evidence": hits[0]["evidence"] if len(hits) == 1 else "",
            "options": [o.get("value") for o in entry.get("options", [])],
            "default": entry.get("default"), "source": entry.get("source")}


def relevant_conventions(text=None) -> list[dict]:
    """Every decision of the table this task touches, resolved against its words."""
    t = _task_text(text)
    low = t.lower()
    out = []
    for e in conventions_table():
        pat = e.get("applies")
        try:
            if pat and re.search(pat, low, re.I):
                out.append(resolve(e, t))
        except re.error:
            continue
    return out


def convention(decision_id: str, text=None) -> dict:
    """One decision by id ('mm_daycount', 'hist_var', 'garch_mean', ...), resolved against the task text.
    KeyError for an unknown id (the ids are listed by conventions_table())."""
    for e in conventions_table():
        if e.get("id") == decision_id:
            return resolve(e, _task_text(text))
    raise KeyError(f"unknown convention {decision_id!r}; known: {[e.get('id') for e in conventions_table()]}")


def conventions_block(text=None, max_items: int = 8, max_chars: int = 1800) -> str:
    """The prompt block: the relevant decisions, the option the task's own words point to (quoted), the
    alternatives, the default when the task is silent, and the source. '' when nothing applies."""
    rows = relevant_conventions(text)
    if not rows:
        return ""
    # decisions whose option the task's words settle come first - they are the ones to get right
    rows.sort(key=lambda r: (r["suggested"] is None, len(r["candidates"]) == 0))
    lines = []
    for r in rows[:max_items]:
        if r["suggested"]:
            pick = f"TASK WORDS POINT TO: {r['suggested']} (\"{r['evidence']}\")"
        elif r["candidates"]:
            pick = "TASK WORDS MATCH SEVERAL: " + " | ".join(c["value"] for c in r["candidates"])
        else:
            pick = f"TASK SILENT -> default: {r['default']}"
        lines.append(f"- {r['decision']}: {pick}. Options: {' / '.join(map(str, r['options']))}. [{r['source']}]")
    body = "\n".join(lines)[:max_chars]
    return ("CONVENTION TABLE (academic / market references; decide each with the task's own words - "
            "quote them - and use the default only when the task is silent; "
            "a program can call convention('<id>')):\n" + body)
