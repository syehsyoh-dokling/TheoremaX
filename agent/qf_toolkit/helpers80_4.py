"""know80 group 4: generic ENGINE aids for the units whose model never reached (or never passed) the first
code cell because the reading / plan phase used the budget, plus a role-aware missing-column report.

Nothing here runs a stored program or replaces the model's plan: the model still writes the plan and every
cell. These functions make the plan reply SHORT enough to arrive whole, recognise a cut plan reply for what
it is (know79: a cut reply was "parsed" as its first inner quantity dict, so the cut was never seen), keep
the plan phase inside a time box, and turn a KeyError on a column into a per-table report the model can act
on (know79 multimodal: three identical `KeyError: 'asof_date'` cells in a row).

Task patterns covered (no task names, no data values):
- "fill this JSON skeleton" deliverables with many keys built from a few stems and a suffix family
  (<stem>__inA / __inB / __BoverA, <stem>_2024 / _2025, <metric>_mean / _std ...);
- plan replies cut at the completion-token limit (a JSON object that never closes);
- multi-table panels whose tables name the same role differently (asof_date / rebalance_date /
  bulletin_date; ticker / symbol) - the usual cause of a KeyError in the loading cell.
Self-tests: proven_work/fail72/selftest80_4.py.
"""
from __future__ import annotations

import csv
import difflib
import json
import os
import re

#: Engine-side aids: importable as qf_toolkit.helpers80_4.<name>, NOT exported into the model's toolkit API
#: (qf_toolkit/__init__ skips these names) - the model never needs them and they would only lengthen prompts.
_ENGINE_ONLY = ("json_skeleton_keys", "skeleton_key_groups", "compact_skeleton_listing",
                "compact_instruction_for_plan", "salvage_json_object", "parse_plan_reply", "plan_cut_problem",
                "plan_round_affordable", "replan_improves", "column_role", "role_column_candidates",
                "csv_headers", "state_frames", "missing_column_report", "keyerror_column_report",
                "plan_column_audit")

# ============================================================================= JSON skeleton compaction
_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S | re.I)
_SKEL_LINE = re.compile(r'^\s*"([^"]+)"\s*:\s*(<[^>]*>|"[^"]*"|[-\w.]+|\{|\[)', re.M)


def json_skeleton_keys(text):
    """Top-level keys of the fenced JSON skeleton(s) in a task text -> [(key, type_text)] in order.

    The skeleton is usually NOT valid JSON (`"x": <int>`), so keys are read line by line: a line
    `"key": <type>` (or a literal / '{' / '[') counts. Only the largest fenced block is used (the output
    skeleton, not an input example). Returns [] when there is no fenced JSON block."""
    blocks = [b for b in _FENCE.findall(text or "") if '":' in b]
    if not blocks:
        return []
    block = max(blocks, key=lambda b: len(_SKEL_LINE.findall(b)))
    return [(k, t.strip()) for k, t in _SKEL_LINE.findall(block)]


def skeleton_key_groups(keys, min_family=3, sep="__"):
    """Group template keys by STEM and SUFFIX FAMILY -> [{'stem', 'suffixes', 'type'}] in first-seen order.

    A suffix is the part after the LAST `sep` ('__inA'); it belongs to the family when at least
    `min_family` keys carry it. Keys without a family suffix are their own group (suffixes []).
    `keys` may be plain names or (name, type) pairs; a group's type is its first key's type (or
    'mixed' when the suffixes differ in type). Lossless: every key is stem + sep + suffix of one group."""
    pairs = [(k, "") if isinstance(k, str) else (str(k[0]), str(k[1])) for k in keys]
    counts = {}
    for k, _t in pairs:
        if sep in k:
            counts[k.rsplit(sep, 1)[1]] = counts.get(k.rsplit(sep, 1)[1], 0) + 1
    family = {s for s, c in counts.items() if c >= min_family}
    groups, index = [], {}
    for k, t in pairs:
        stem, suf = (k.rsplit(sep, 1) if sep in k and k.rsplit(sep, 1)[1] in family else (k, None))
        g = index.get(stem)
        if g is None:
            g = {"stem": stem, "suffixes": [], "type": t}
            index[stem] = g
            groups.append(g)
        elif t and g["type"] and t != g["type"]:
            g["type"] = "mixed"
        if suf is not None:
            g["suffixes"].append(suf)
    return groups


def compact_skeleton_listing(text_or_keys, min_keys=24, sep="__", max_chars=6000):
    """A short, LOSSLESS listing of a large JSON skeleton for a planning prompt ('' when the skeleton has
    fewer than `min_keys` keys). One line per stem: `<stem>{__a,__b,__c}: <type>`; header line states the
    key count and the suffix family, and that the keys are already known (copied from the task)."""
    keys = json_skeleton_keys(text_or_keys) if isinstance(text_or_keys, str) else list(text_or_keys)
    if len(keys) < min_keys:
        return ""
    groups = skeleton_key_groups(keys, sep=sep)
    fam = sorted({s for g in groups for s in g["suffixes"]})
    lines = [f"OUTPUT SKELETON: {len(keys)} keys = {len(groups)} stems"
             + (f" x suffix family {{{', '.join(sep + s for s in fam)}}}" if fam else "")
             + " (the full key list is in the TASK and is added to the plan automatically - do NOT list the "
               "keys one by one as quantities)"]
    for g in groups:
        suf = ("{" + ",".join(sep + s for s in g["suffixes"]) + "}") if g["suffixes"] else ""
        lines.append(f"- {g['stem']}{suf}: {g['type'] or '?'}")
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[:max_chars] + "\n- ..."


def compact_instruction_for_plan(text, min_keys=24, sep="__"):
    """The task text with its large fenced JSON skeleton replaced by compact_skeleton_listing (every stem,
    suffix and type kept) -> (text, replaced: bool). For the PLAN prompt only: the cells keep the full
    text, so every exact key is still in front of the model when it writes the program."""
    keys = json_skeleton_keys(text)
    if len(keys) < min_keys:
        return text, False
    listing = compact_skeleton_listing(keys, min_keys=min_keys, sep=sep)
    blocks = [m for m in _FENCE.finditer(text or "") if '":' in m.group(1)]
    m = max(blocks, key=lambda m: len(_SKEL_LINE.findall(m.group(1))))
    return text[:m.start()] + listing + "\n" + text[m.end():], True


#: Plan-sized advice for a skeleton task (the plan_gate SYSTEM asks for <= ~12 quantities).
SKELETON_PLAN_RULE = ("This task's output is a large key skeleton: plan ONE quantity per SOURCE (a cover-page "
                      "fact group, one per table or section, one per derived difference such as B - A), naming "
                      "the stems it fills - at most ~12 quantities in all. The exact keys are added from the task.")


# ============================================================================= cut-reply-aware JSON parsing
PLAN_KEYS = ("quantities", "method", "deliverables", "identities", "variants_chosen", "pitfalls_checked",
             "unused_inputs", "open_questions")


def salvage_json_object(text):
    """Parse the FIRST top-level JSON object of a reply -> (obj | None, cut: bool).

    A complete object is returned as is (cut False). An object that never closes (the reply was cut at the
    token limit) is closed at its last complete element: the open strings / arrays / objects are closed in
    order, a dangling key or partial value is dropped (cut True). Never returns an INNER object in place of
    the outer one (know79: the plan parser did that, so a cut plan looked like a parsed one with 'no
    quantities')."""
    if not text:
        return None, False
    start = text.find("{")
    if start < 0:
        return None, False
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return (obj if isinstance(obj, dict) else None), False
    except ValueError:
        pass
    s = text[start:]
    stack, in_str, esc = [], False, False
    cuts = []                      # (position, stack snapshot) just after a complete element
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            cuts.append((i + 1, tuple(stack)))
            if not stack:
                break
        elif ch == ",":
            cuts.append((i, tuple(stack)))
    # first pass: cut only between LIST elements or after a closed container (a partial record is dropped
    # whole); second pass: any element boundary (a dangling key / partial value inside an object dropped)
    for strict in (True, False):
        for pos, st in reversed(cuts[-400:]):
            if not st or (strict and st[-1] != "]" and s[pos - 1] not in "}]"):
                continue
            cand = s[:pos].rstrip().rstrip(",") + "".join(reversed(st))
            try:
                obj = json.loads(cand)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj, True
    return None, True


def plan_cut_problem(plan, max_quantities=12):
    """The send-back line for a plan reply that was CUT at the token limit (salvaged or not): how far it
    got and how to make the next one arrive whole. The model still writes the plan; it is told the size."""
    nq = len([q for q in (plan or {}).get("quantities") or [] if isinstance(q, dict)]) if isinstance(plan, dict) else 0
    return (f"YOUR PLAN REPLY WAS CUT at the length limit" + (f" after {nq} quantities" if nq else "")
            + f" - nothing after the cut arrived (method, identities, deliverables). Re-plan SHORTER: at most "
              f"{max_quantities} quantities (group keys/columns that come from the same source into ONE quantity "
              "naming them), one short line per string, method <= 10 steps.")


def parse_plan_reply(text, plan_keys=PLAN_KEYS):
    """A plan reply -> (plan dict | None, cut: bool). The plan is the outer object and must carry at least
    one plan key; an object of another shape (a lone quantity, a variant) is NOT a plan (None)."""
    obj, cut = salvage_json_object(text)
    if isinstance(obj, dict) and any(k in obj for k in plan_keys):
        return obj, cut
    return None, cut


# ============================================================================= plan time box
PLAN_BUDGET_SHARE = 0.35


def plan_round_affordable(elapsed_sec, budget_sec, round_seconds, share=PLAN_BUDGET_SHARE, rounds_done=1):
    """May ANOTHER plan round start? True when the elapsed time plus the costliest plan round so far stays
    within `share` of the whole budget. The first round always runs (rounds_done=0 -> True): the model
    always plans; the box only stops RE-plans that would leave the cells too little time (know79: explore +
    three plan rounds with continuations left 29-44 s of a 900 s budget for the code)."""
    if rounds_done <= 0:
        return True
    worst = max([float(x) for x in (round_seconds or [])] or [0.0])
    return float(elapsed_sec) + worst <= share * float(budget_sec)


def replan_improves(gap_counts):
    """Soft re-plans only while they help: False when the latest round did not reduce the number of gaps
    of the best earlier round (a second identical send-back costs a full request and changes nothing)."""
    g = [int(x) for x in gap_counts or []]
    if len(g) < 2:
        return True
    return g[-1] < min(g[:-1])


# ============================================================================= role-aware missing column
_ROLE_TOKENS = {
    "date": ("date", "dt", "time", "timestamp", "asof", "day", "period", "month", "datetime", "ts"),
    "id": ("ticker", "symbol", "id", "cik", "permno", "isin", "cusip", "sedol", "issuer", "asset", "secid",
           "gvkey", "code"),
    "price": ("price", "close", "px", "last", "mid"),
    "return": ("ret", "return", "returns", "rtn"),
    "weight": ("weight", "wgt", "w"),
}


def column_role(name):
    """The role of a column name from its tokens: 'date', 'id', 'price', 'return', 'weight' or ''."""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]
    for role, words in _ROLE_TOKENS.items():
        if any(t in words for t in toks):
            return role
    return ""


def role_column_candidates(missing, columns, n=3):
    """Columns of a table that can stand in for a missing one: same role ('asof_date' -> 'rebalance_date',
    'bulletin_date'), then by name similarity; plus close spellings of any role. Ordered, at most n."""
    cols = [str(c) for c in columns]
    role = column_role(missing)
    same = [c for c in cols if role and column_role(c) == role and c != missing]
    same.sort(key=lambda c: -difflib.SequenceMatcher(None, str(missing).lower(), c.lower()).ratio())
    close = difflib.get_close_matches(str(missing), cols, n=n, cutoff=0.75)
    out = []
    for c in same + close:
        if c not in out and c != missing:
            out.append(c)
    return out[:n]


def csv_headers(paths, max_files=40):
    """{base name: [columns]} from the header line of every .csv/.tsv path (data rows are not read)."""
    out = {}
    for p in list(paths or [])[:max_files]:
        p = str(p)
        if not p.lower().endswith((".csv", ".tsv")):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace", newline="") as fh:
                out[os.path.basename(p)] = next(csv.reader(fh, delimiter="\t" if p.lower().endswith(".tsv") else ","), [])
        except (OSError, StopIteration):
            continue
    return out


_STATE_FRAME = re.compile(r"^\s*([A-Za-z_]\w*): DataFrame shape=\([^)]*\) columns=\[([^\]]*)\]", re.M)


def state_frames(state_text):
    """{variable: [columns]} for every DataFrame line of a RUNTIME STATE text."""
    out = {}
    for m in _STATE_FRAME.finditer(state_text or ""):
        out[m.group(1)] = [c.strip().strip("'\"") for c in m.group(2).split(",") if c.strip()]
    return out


def missing_column_report(missing, tables, max_lines=10):
    """Per-table answer to "column X does not exist": which tables HAVE it, which name the same role
    differently (with the rename that aligns them), and which have no such role at all (a static table:
    merge it on its ids, do not sort / filter it by that column). tables: {name: [columns]}. '' if empty."""
    if not tables:
        return ""
    has, alias, none = [], [], []
    for name, cols in tables.items():
        cols = [str(c) for c in cols]
        if missing in cols:
            has.append(name)
            continue
        cand = role_column_candidates(missing, cols, n=1)
        if cand:
            alias.append(f"{name}: no '{missing}', its {column_role(missing) or 'closest'} column is '{cand[0]}' "
                         f"-> use '{cand[0]}' there, or .rename(columns={{'{cand[0]}': '{missing}'}}) when it "
                         "plays the same part, before sort/merge")
        else:
            none.append(name)
    if not has and not alias:
        return ""          # not a column any table speaks to (a dict key, a typo of a computed name)
    lines = []
    if has:
        lines.append(f"tables WITH '{missing}': {', '.join(has[:8])}")
    lines += alias[:max_lines]
    if none:
        lines.append(f"tables with no {column_role(missing) or 'such'} column (do not sort/filter them by "
                     f"'{missing}'; merge them on their ids): {', '.join(none[:8])}")
    return "\n".join(lines[:max_lines + 2])


def keyerror_column_report(error_text, state_text="", input_paths=None):
    """For an error text with `KeyError: '<col>'`: missing_column_report over the DataFrames in the runtime
    state and the CSV inputs (input_paths, default the T1_INPUT_FILES env) -> '' when not a column error."""
    m = (re.search(r"None of \[Index\(\[['\"]([^'\"]+)['\"]", error_text or "")
         or re.search(r"KeyError: ['\"]([^'\"\[\]]+)['\"]", error_text or ""))
    if not m:
        return ""
    if input_paths is None:
        try:
            input_paths = json.loads(os.environ.get("T1_INPUT_FILES") or "[]")
        except ValueError:
            input_paths = []
    tables = dict(state_frames(state_text))
    for k, v in csv_headers(input_paths).items():
        tables.setdefault(k, v)
    return missing_column_report(m.group(1), tables)


def plan_column_audit(plan, headers):
    """Notes for plan quantities whose source cites `<file>:<column>` (or <file>[<column>]) that the file
    does not have, with the same-role column it does have. headers: {file base name: [columns]}."""
    notes = []
    if not isinstance(plan, dict):
        return notes
    pat = re.compile(r"([\w.\-]+\.(?:csv|tsv|parquet))\s*(?::|\[\s*['\"]?)\s*([A-Za-z_][\w ]*)")
    for q in plan.get("quantities") or []:
        if not isinstance(q, dict):
            continue
        for f, col in pat.findall(str(q.get("source", ""))):
            cols = headers.get(os.path.basename(f))
            col = col.strip()
            if cols is None or col in cols or col.lower() == "column":
                continue
            cand = role_column_candidates(col, cols, n=1)
            notes.append(f"{f} has no column '{col}'" + (f" - it is '{cand[0]}' there" if cand else
                                                           f" (columns: {', '.join(cols[:10])})"))
    return list(dict.fromkeys(notes))
