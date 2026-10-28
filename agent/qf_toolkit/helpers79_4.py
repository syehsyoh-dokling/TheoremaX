"""know79 group 4: generic helpers for the filings / data-engineering units that still failed in know78.

Task patterns covered (each helper is written for the PATTERN; conventions are keywords with textbook
defaults; no task names, no data values):
- insider-filing (SEC Form 4) reconstruction: which footnotes describe an OWNERSHIP bucket (the post-
  transaction holding and the ownership-nature cells, not the price or the transaction code), and one
  label for the direct bucket used consistently across every filing of the task;
- library API migration (Polars 0.x -> 1.x) that must PRESERVE the old outputs: the behaviour changes
  that are not renames - joins on null keys, join row order, LazyFrame.fetch(n).
Wired in place into helpers75_1.parse_form4, helpers76_2.form4_sale_pressure_bundle and
helpers75_4.polars_0x_to_1x_source. Self-tests: proven_work/fail72/selftest79_4.py.
"""
from __future__ import annotations

import re
from collections import Counter

# ============================================================================= Form 4 ownership footnotes
#: XML sections of a Form 4 row whose footnotes describe the OWNERSHIP bucket (Table I columns 5-7).
FORM4_OWNERSHIP_SECTIONS = ("postTransactionAmounts", "ownershipNature")


def form4_ownership_footnote_ids(el, sections=FORM4_OWNERSHIP_SECTIONS):
    """Footnote ids attached to the ownership part of a Form 4 nonDerivativeTransaction / Holding element
    -> list in document order, de-duplicated.

    SEC Form 4 Table I: column 5 'Amount of Securities Beneficially Owned Following Reported
    Transaction(s)' (postTransactionAmounts/sharesOwnedFollowingTransaction), column 6 'Ownership Form:
    Direct (D) or Indirect (I)' and column 7 'Nature of Indirect Beneficial Ownership' (ownershipNature).
    A footnote on ANY of these cells qualifies the ownership bucket (e.g. 'includes RSUs', 'held by the X
    Trust'); footnotes on the price (weighted-average ranges) or on the transaction code (10b5-1 plans) do
    not. Pass sections=('ownershipNature',) for the nature-only convention."""
    out = []
    for sec in sections:
        x = el.find(sec)
        if x is None:
            continue
        for f in x.iter("footnoteId"):
            i = f.get("id")
            if i and i not in out:
                out.append(i)
    return out


_POINTER = re.compile(r"(see\s+footnotes?.*|\(?\d+\)?|footnote.*)", re.I)


def form4_direct_label(natures, default="Direct"):
    """One label for the DIRECT ownership bucket across all filings of a task.

    natures: iterable of (ownership_form, nature_of_ownership_text) pairs from every row (transactions
    and holdings) of every filing. The direct bucket is ONE economic bucket whatever the wording; when
    some filing names it (a D-form row whose nature text is a real label that starts with 'direct',
    e.g. 'Direct holdings' / 'Direct ownership'), that wording is used for every filing (the most common
    one; ties -> first seen), else `default`. Pointer texts ('See footnote', '(1)') and empty cells never
    name it. Returns the label string."""
    seen = Counter()
    order = []
    for form, nat in natures:
        t = re.sub(r"\s+", " ", str(nat or "")).strip()
        if not str(form or "").upper().startswith("D") or not t or _POINTER.fullmatch(t):
            continue
        if not t.lower().startswith("direct"):
            continue
        if t not in seen:
            order.append(t)
        seen[t] += 1
    if not seen:
        return default
    best = max(seen.values())
    return next(t for t in order if seen[t] == best)


# ============================================================================= Polars 0.x -> 1.x semantics
def _call_span(src, open_idx):
    """Index just past the ')' matching the '(' at open_idx (strings respected), or -1."""
    depth, i, q = 0, open_idx, None
    while i < len(src):
        c = src[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


_JOIN = re.compile(r"\.join\(")


def polars_join_semantics_0x(src, *, null_keys_match=True, maintain_order=True):
    """Keep Polars 0.x JOIN behaviour in migrated 1.x source text -> new source.

    Two behaviour changes that are not renames (Polars 0.20 / 1.x upgrade guides):
    * null keys: 0.19 joins matched null to null; 0.20+ do not (join_nulls=False, renamed nulls_equal in
      1.x). null_keys_match=True adds nulls_equal=True to every DataFrame/LazyFrame .join( (and renames
      an explicit join_nulls= to nulls_equal=), so a lookup row keyed None still matches.
    * row order: 1.x joins do not promise an order (maintain_order='none'); 0.x returned the left rows
      in order, then (full joins) the unmatched right rows. maintain_order=True adds
      maintain_order='left' (left / inner / semi / anti) or 'left_right' (full / outer).
    String/list .join( (str.join, list.join, ', '.join) is left alone. Idempotent: an argument already
    present is not added twice."""
    src = re.sub(r"\bjoin_nulls\s*=", "nulls_equal=", src)
    out, pos = [], 0
    for m in _JOIN.finditer(src):
        if m.start() < pos:
            continue
        before = src[max(0, m.start() - 8):m.start()]
        if re.search(r"(\b(str|list|arr)|[\"'])$", before):
            continue
        end = _call_span(src, m.end() - 1)
        if end < 0:
            continue
        args = src[m.end():end - 1]
        add = []
        if null_keys_match and "nulls_equal" not in args:
            add.append("nulls_equal=True")
        if maintain_order and "maintain_order" not in args:
            how = re.search(r"how\s*=\s*[\"'](\w+)[\"']", args)
            h = how.group(1) if how else "inner"
            if h != "cross":
                add.append("maintain_order=" + ("'left_right'" if h in ("full", "outer", "outer_coalesce") else "'left'"))
        if not add:
            continue
        body = args.rstrip()
        trail = args[len(body):]
        sep = " " if body.endswith(",") else ", "
        out.append(src[pos:m.end()] + body + sep + ", ".join(add) + trail + ")")
        pos = end
    out.append(src[pos:])
    return "".join(out)


_FETCH = re.compile(r"\.fetch\(\s*([^()]*?)\s*\)")


def polars_fetch_to_head(src, style="head"):
    """LazyFrame.fetch(n) (removed in 1.x) -> a 1.x query with the same RESULT, in migrated source text.

    style='head' (default): `.fetch(n)` -> `.head(n).collect()` - the first n rows of the query result
      (the upgrade-guide replacement: collect + head; what a migration that must preserve the
      deliverable's rows keeps).
    style='n_rows': the debugging semantics (n rows read from the scan, then the rest of the query):
      handled by helpers75_4._pl_fetch_to_nrows - use it only when the task says the sample is taken
      at the source.
    A bare `.fetch()` becomes `.collect()`."""
    if style != "head":
        return src

    def rep(m):
        n = m.group(1).strip()
        n = re.sub(r"^n_rows\s*=\s*", "", n)
        return f".head({n}).collect()" if n else ".collect()"
    return _FETCH.sub(rep, src)


# ============================================================================= whole-task fast path (K1)
_WHOLE_TASK = re.compile(r"\b(\w+_(?:bundle|auto|and_run|report))\s*\(")


def frame_fastpath_program(frame):
    """The runnable whole-task program of a curated frame -> source text, or None.

    For the engine's fast path (fix79_4 K1): a curated frame whose text starts with 'DECISIVE' and whose
    STEPS begin with plain Python lines (import, one call of a whole-task helper *_bundle / *_auto /
    *_and_run / *_report, emit) is run BEFORE any reading / planning request, so a unit whose model replies
    get cut still writes its deliverables. Rules: steps are taken in order while each parses as Python on
    its own; a line with '<', '...' or a parse error ends the program; the program must call a whole-task
    helper and emit(); otherwise None (the normal path runs)."""
    import ast
    if not isinstance(frame, dict) or not frame.get("curated") or frame.get("proven"):
        return None
    if not str(frame.get("frame", "")).lstrip().upper().startswith("DECISIVE"):
        return None
    lines = []
    for s in frame.get("steps", []):
        s = str(s).strip()
        if not s or "<" in s or "..." in s:
            break
        try:
            ast.parse(s)
        except SyntaxError:
            break
        lines.append(s)
    prog = "\n".join(lines)
    if not lines or "emit(" not in prog or not _WHOLE_TASK.search(prog):
        return None
    try:
        ast.parse(prog)
    except SyntaxError:
        return None
    return prog + "\n"
