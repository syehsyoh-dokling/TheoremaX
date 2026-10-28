"""Line-level repair for a crash that repeats on the same line.

30 Sep batch 1: compound-geske crashed 11 times on `float(mc_price[idx])` with the runtime state
showing mc_price was a scalar; each repair re-sent the whole 200-line program with the bug intact.
The same loop shape cost barrier-garch-var 8, var-es 9 and yield-curve-pca 6 attempts. Asking for
the whole program again invites the same program; asking for an EDIT of the numbered lines
focuses the model on the defect and keeps the reply far below the 4,000-token cut.

The model answers with blocks
    @@ REPLACE <start>-<end>
    <replacement lines>
    @@ END
(1-based, inclusive; the replacement may be longer or shorter). Blocks are applied bottom-up; the
result must parse, else None and the caller falls back to a normal rewrite."""
from __future__ import annotations

import ast
import re

from . import code_writer

SYSTEM = """You fix a bug in a Python program by EDITING specific lines. You are shown the program with
line numbers, the error, and the runtime values of the variables on the failing line.
Reply ONLY with one or more edit blocks and nothing else:
@@ REPLACE <start>-<end>
<the new lines that replace lines start..end, with correct indentation>
@@ END
Line numbers are 1-based and inclusive. Fix the LOGIC the runtime values expose (a scalar used as an
array, a missing key, a wrong shape) - wherever it is, not only the line that raised. Keep every
other line unchanged. Do not re-send the whole program."""

_BLOCK = re.compile(r"@@\s*REPLACE\s+(\d+)\s*-\s*(\d+)\s*\n(.*?)\n?@@\s*END", re.S)
MAX_LINES_SHOWN = 400


def _numbered(source: str) -> str:
    lines = source.splitlines()[:MAX_LINES_SHOWN]
    return "\n".join(f"{i + 1:4d}| {line}" for i, line in enumerate(lines))


def apply(source: str, reply: str) -> str | None:
    """Apply the REPLACE blocks of `reply` to `source`; None if nothing valid."""
    lines = source.splitlines()
    blocks = []
    for m in _BLOCK.finditer(reply or ""):
        start, end = int(m.group(1)), int(m.group(2))
        if not (1 <= start <= end <= len(lines)):
            return None
        body = m.group(3)
        body = re.sub(r"^```(?:python)?\s*\n|\n```\s*$", "", body)
        blocks.append((start, end, body.splitlines()))
    if not blocks:
        return None
    blocks.sort(key=lambda b: b[0])
    for (s1, e1, _), (s2, _, _) in zip(blocks, blocks[1:]):
        if s2 <= e1:
            return None          # overlapping edits: ambiguous, do not guess
    for start, end, new in reversed(blocks):
        lines[start - 1:end] = new
    patched = "\n".join(lines) + "\n"
    try:
        ast.parse(patched)
    except SyntaxError:
        return None
    return patched


def patch(source: str, detail: str, *, telemetry: list | None = None) -> str | None:
    user = ("PROGRAM (line numbers are NOT part of the code):\n" + _numbered(source)
            + "\n\nIT FAILED TWICE IN A ROW ON THE SAME LINE WITH:\n" + (detail or "")[-2500:]
            + "\n\nReturn the edit blocks now.")
    reply = code_writer.complete(SYSTEM, user, telemetry=telemetry, variant=60, want_code=False)
    return apply(source, reply or "")
