"""know78 group 4: generic helpers for the filings / alt-data / data-engineering units that still failed.

Task patterns covered (each helper is written for the PATTERN; conventions are keywords with textbook
defaults; no task names, no data values):
- reading the task's own instruction at run time (typed JSON templates, signed component lists);
- equity curve -> periodic simple returns (n points give n - 1 returns);
- multi-agent committee alpha from signed component lists, with every input discovered by its role;
- report-pair extraction: fuzzy table-row lookup (label + section + column header + unit multiplier)
  and the inline-XBRL canonical value of a transformed fact (ixt date / boolean formats).
Self-tests: proven_work/fail72/selftest78_4.py.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re

import numpy as np
import pandas as pd


# ============================================================================= the task's own instruction
_INSTRUCTION_ENVS = ("T1_INSTRUCTION_FILE", "T1_INSTRUCTION_PATH", "T1_INSTRUCTION", "T1_INSTRUCTION_TEXT")
_INSTRUCTION_PLACES = ("/input/instruction.md", "/app/instruction.md", "/app/data/instruction.md", "instruction.md")


def instruction_text(path=None):
    """The task instruction as text ('' when it cannot be found). Looks at `path`, then the env vars
    T1_INSTRUCTION_FILE / T1_INSTRUCTION_PATH (a file) or T1_INSTRUCTION / T1_INSTRUCTION_TEXT (the text
    itself), then /input/instruction.md, /app/instruction.md. Use it to feed parse_json_template /
    signed_component_lists without retyping the instruction's lists."""
    cands = [path] if path else []
    for env in _INSTRUCTION_ENVS:
        v = os.environ.get(env)
        if v:
            cands.append(v)
    cands += list(_INSTRUCTION_PLACES)
    for c in cands:
        s = str(c)
        if "\n" in s or len(s) > 400:
            return s
        try:
            p = pathlib.Path(s)
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return ""


# ============================================================================= equity curve -> returns
def equity_curve_returns(equity, *, first="drop", base=None):
    """Simple periodic returns of an equity curve: r_t = E_t / E_{t-1} - 1.

    n equity points give n - 1 returns (first='drop', the textbook definition: the first point has no
    predecessor). first='zero' keeps a leading 0.0 (n values); first='base' uses base (e.g. the starting
    capital) as E_{-1} so the first return is E_0/base - 1. Annualise afterwards: mean * A,
    std(ddof=1) * sqrt(A). Returns a float ndarray."""
    e = np.asarray(equity, float)
    if e.size == 0:
        return e
    r = e[1:] / e[:-1] - 1.0
    if first == "drop":
        return r
    if first == "zero":
        return np.concatenate([[0.0], r])
    if first == "base":
        if base is None:
            raise ValueError("first='base' needs base= (the equity before the first point)")
        return np.concatenate([[e[0] / float(base) - 1.0], r])
    raise ValueError("first must be 'drop', 'zero' or 'base'")


# ============================================================================= signed component lists
_SIGNED_ITEM = re.compile(r"^\s*[-*]\s*`?\s*([+\-−])\s*(.+?)`?\s*$")
_NAMED_SCORE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


def signed_component_lists(text=None, *, name_pattern=r"_(score|signal|factor|alpha)$"):
    """Instruction lists of signed components -> {name: [(sign, expression), ...]} in order.

    Reads blocks such as
        ### A. `filing_agent_score`
        - `+ revenue_growth_ttm`
        - `- log1p(theme_risk_count_30d)`
        - `+ article_count_7d / max(article_count_30d, 1)`
    The block name is the last backticked identifier matching name_pattern in the heading / sentence
    before the items; signs '+' -> +1, '-' (or U+2212) -> -1; the expression is kept verbatim (it is
    evaluated later by eval_feature). text=None reads instruction_text()."""
    t = instruction_text() if text is None else str(text)
    out, cur = {}, None
    rx = re.compile(name_pattern)
    for line in t.splitlines():
        m = _SIGNED_ITEM.match(line)
        if m and cur is not None:
            sgn = -1 if m.group(1) in "-−" else 1
            expr = m.group(2).strip().strip("`").strip()
            if expr:
                out.setdefault(cur, []).append((sgn, expr))
            continue
        names = [n for n in _NAMED_SCORE.findall(line) if rx.search(n)]
        if names and not line.lstrip().startswith(("-", "*")):
            cur = names[-1]
            if cur in out:          # a later mention (e.g. 'X is the mean of ...') keeps the same list
                continue
    return {k: v for k, v in out.items() if v}


# ============================================================================= committee alpha, inputs by role
def _role_files(data_dir):
    roots = [pathlib.Path(str(data_dir))] if data_dir else []
    for env in ("T1_DATA_DIR", "DATA_DIR"):
        if os.environ.get(env):
            roots.append(pathlib.Path(os.environ[env]))
    roots += [pathlib.Path(p) for p in ("/app/data", "/input/data", "/input/environment/data", "data")]
    for r in roots:
        if r.is_dir() and any(r.glob("*.csv")):
            return r, sorted(r.glob("*.csv")), sorted(r.glob("*.json"))
    raise FileNotFoundError(f"no CSV inputs found in {[str(r) for r in roots]}")


def committee_alpha_auto(data_dir=None, agents=None, params=None, **overrides):
    """ONE call for a committee-alpha task whose inputs are CSV files named by ROLE -> DICT{file: obj}
    (the 7 outputs of committee_process_alpha_bundle); emit each value as is.

    Roles by file-name words: '*universe*' (ids + attributes), '*rebalance*' (dates), '*return*'
    (forward returns), '*bulletin*' / '*event*' (dated events), '*_v1*' / '*_v2*' (reference vintages),
    every other CSV = a feature table named by its stem without '_features' (the table whose name holds
    'news' / 'gdelt' is the news table; tables joined on a bucket column count as shared context, not
    missing-cell tables). params = params.json (any *.json) by default. agents: {score column: [(sign,
    expression)]}; None -> signed_component_lists(instruction_text()). overrides go to
    committee_process_alpha_bundle (e.g. missing_side='count', annualization='arithmetic')."""
    from .helpers77_1 import committee_process_alpha_bundle
    root, csvs, jsons = _role_files(data_dir)
    rd = lambda p: pd.read_csv(p)  # noqa: E731
    role = {}
    feats = {}
    for p in csvs:
        st = p.stem.lower()
        if "universe" in st:
            role["universe"] = p
        elif "rebalance" in st:
            role["rebalance"] = p
        elif "return" in st:
            role["returns"] = p
        elif "bulletin" in st or re.search(r"(^|_)events?($|_)", st):
            role["events"] = p
        elif re.search(r"(^|_)v1($|_)", st):
            role["v1"] = p
        elif re.search(r"(^|_)v2($|_)", st):
            role["v2"] = p
        else:
            feats[re.sub(r"_features?$", "", st)] = p
    missing = [k for k in ("universe", "rebalance", "returns") if k not in role]
    if missing:
        raise FileNotFoundError(f"committee_alpha_auto: no file for role(s) {missing} in {root}")
    if params is None:
        params = next((j for j in jsons if "param" in j.stem.lower()), jsons[0] if jsons else None)
    P = params if isinstance(params, dict) else (json.loads(pathlib.Path(str(params)).read_text()) if params else {})
    if agents is None:
        agents = signed_component_lists(instruction_text())
        if not agents:
            raise ValueError("committee_alpha_auto: pass agents={score column: [(sign, expression), ...]} "
                             "(the instruction text is not available at run time)")
    uni = rd(role["universe"])
    # table names drive '<name>_missing_cells' columns: use the names the instruction asks for when one
    # is contained in the file stem (gdelt_news_features -> news), else the stem without '_features'
    _itxt = instruction_text()
    wanted = sorted(set(re.findall(r"`?([a-z][a-z0-9_]*?)_missing_cells`?", _itxt)), key=len, reverse=True)
    ren = {}
    for k in list(feats):
        w = next((w for w in wanted if w in k), None)
        ren[k] = w if w and w not in ren.values() else k
    feats = {ren[k]: p for k, p in feats.items()}
    tables = {k: rd(p) for k, p in feats.items()}
    id_like = {"ticker", "asof_date", "date"}
    bucket_tables = [k for k, t in tables.items()
                     if not ({"ticker"} & set(t.columns)) and (set(t.columns) & set(uni.columns)) - id_like]
    news = next((k for k in tables if re.search(r"news|gdelt", k)), None)
    kw = dict(events=rd(role["events"]) if "events" in role else None,
              vintage_v1=rd(role["v1"]) if "v1" in role else None,
              vintage_v2=rd(role["v2"]) if "v2" in role else None, params=P,
              missing_cell_tables=sorted([k for k in tables if k not in bucket_tables],
                                         key=lambda k: (_itxt.find(f"{k}_missing_cells") % (len(_itxt) + 1)
                                                        if f"{k}_missing_cells" in _itxt else len(_itxt) + 1, k)),
              news_table=news)
    kw.update(overrides)
    out = committee_process_alpha_bundle(uni, rd(role["rebalance"]), tables, rd(role["returns"]), agents,
                                         None, **kw)
    return {k: v for k, v in out.items() if "." in str(k)}     # deliverable files only


# ============================================================================= inline XBRL canonical values
def ixt_canonical(text, fmt):
    """Inline-XBRL transformed text -> its canonical XBRL value (the value a taxonomy-aware reader sees).
    ixt date formats ('date-monthname-day-year-en', 'date-month-day-year', 'datedoteu', ...) -> ISO
    'YYYY-MM-DD' (a format without a year is returned unchanged); boolean formats ('boolballotbox',
    'fixed-true' / 'fixed-false') -> True/False; everything else -> the text unchanged."""
    f = str(fmt or "").lower()
    s = str(text or "").strip()
    if "fixed-true" in f:
        return True
    if "fixed-false" in f:
        return False
    if "boolballotbox" in f or "ballotbox" in f:
        return s in ("☒", "☑", "x", "X")
    if "date" in f and re.search(r"\d{4}", s):
        try:
            dayfirst = "eu" in f or "day-month" in f or "daymonth" in f
            return pd.Timestamp(pd.to_datetime(s.replace(".", " ").strip(), dayfirst=dayfirst)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return s
    return s


# ============================================================================= fuzzy table lookup (report pairs)
_STOP = {"the", "of", "a", "an", "to", "in", "by", "for", "and", "as", "at", "on", "is", "from", "with", "or",
         "its", "per", "that", "this", "be", "are", "was", "were", "inc", "value", "amount", "amounts"}
_SELECTORS = {"total", "average", "minimum", "maximum", "median", "net", "gross"}
_DASH = {"-", "—", "–", "�", "‒", "―"}
_UNIT_RX = re.compile(r"\bin (thousands|millions|billions)\b", re.I)
_MULT = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
_YEAR_RX = re.compile(r"^(?:fiscal\s+)?(19|20)\d{2}$", re.I)


def _toks(s):
    s = re.sub(r"\bU\.S\.", "US", str(s))
    s = re.sub(r"\([0-9a-z]\)", " ", s)               # footnote markers (1), (a)
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", s):
        w = w.lower()
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _cell_value(c):
    """A table cell -> float, 0.0 for a dash cell, None for text, 'skip' for '$' / '%' / ')' cells."""
    t = str(c).strip()
    if t in ("", "$", "%", ")", "(", "$(", "%)"):
        return "skip"
    if t.replace("%", "").strip() in _DASH:
        return 0.0
    m = re.fullmatch(r"\(?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\)?\s*%?", t)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    return -v if t.startswith("(") and ")" in t else v


def _row_values(cells):
    lab_i = next((i for i, c in enumerate(cells) if str(c).strip()), None)
    if lab_i is None:
        return "", [], False, []
    vals, pct, raws = [], False, []
    for c in cells[lab_i + 1:]:
        v = _cell_value(c)
        if "%" in str(c):
            pct = True
        if v == "skip":
            continue
        vals.append(v)
        raws.append(str(c).strip())
    return str(cells[lab_i]).strip(), vals, pct, raws


def _is_header_like(row, vals):
    """No numeric cell, or every numeric cell is a bare year (a column-header row such as '| 2025 | 2024')."""
    nums = [v for v in vals if isinstance(v, float)]
    if not nums:
        return True
    if any(str(c).strip().startswith("$") for c in row):
        return False
    return all(v.is_integer() and 1990 <= v <= 2100 for v in nums)


def _column_labels(block, n):
    """Column labels for a data row with n values from the header rows above it (multi-level headers:
    the deepest row is right-aligned over the values; a shallower row's last cell spans that part and its
    other cells label the leading columns). None when the header cannot be aligned."""
    if not block:
        return None
    deep = max(block, key=len)
    if len(deep) == n + 1:
        labels = list(deep[1:])
        off = 0
    elif len(deep) <= n:
        off = n - len(deep)
        labels = [""] * off + list(deep)
    else:
        return None
    for u in block:
        if u is deep:
            continue
        if len(u) in (n, n + 1):
            uu = u[1:] if len(u) == n + 1 else u
            labels = [f"{a} {b}".strip() for a, b in zip(uu, labels)]
        elif len(u) >= 2:
            lead, last = u[:-1], u[-1]
            if off >= len(lead):
                for j, c in enumerate(lead):
                    labels[j] = f"{c} {labels[j]}".strip()
                for j in range(off, n):
                    labels[j] = f"{last} {labels[j]}".strip()
            else:
                for j in range(n):
                    labels[j] = f"{u[-1]} {labels[j]}".strip()
        elif len(u) == 1:
            labels = [f"{u[0]} {x}".strip() for x in labels]
    return labels


def _footnote_unit(text, marker, anchor, value_text=None):
    """'(k) ... in thousands/millions/billions' footnote for a header marker (k), searched in the 6000
    characters after the table (located by its row label followed by the row's own value text, else
    the label alone) -> multiplier or None."""
    if not marker:
        return None
    pos = -1
    if anchor and value_text:
        vals = [value_text] if isinstance(value_text, str) else list(value_text)
        rx = re.escape(anchor) + "".join(r"[\s$|%()]{0,12}" + re.escape(v) for v in vals)
        m0 = re.search(rx, text)
        pos = m0.start() if m0 else -1
    if pos < 0 and anchor:
        pos = text.find(anchor)
    seg = text[pos: pos + 6000] if pos >= 0 else text
    m = re.search(r"\(" + re.escape(marker) + r"\)\s*[^\n]{0,100}?\bin (thousands|millions|billions)\b", seg, re.I)
    return _MULT[m.group(1).lower()] if m else None


_PER_UNIT = re.compile(r"\bper share\b|\bprice paid\b|\bprice\b|\bage\b|\bpercent(age)?\b|\bratio\b|\(years\)|"
                       r"\binterest rate\b|\bweighted[- ]average\b.*\brate\b|\byield\b", re.I)


def sec_table_lookup(src, stem, typ="float", *, detail=False, min_label_hits=1):
    """Best table cell for a field NAME (no hand spec): fuzzy match of the field's words against every
    table row of an SEC report (zip / html), with the table's unit multiplier applied.

    score(row) = 3 x |field words in the row label| + 2 x |in the nearest section line above| + 1 x
    |in earlier section lines / header rows| + 3 x |in the chosen column label| - 0.5 x |label words
    not in the field| - 3 per field word found nowhere (label, sections, headers, column) - 4 per
    selector word (total, average, minimum, maximum, median, net, gross) found nowhere; a row needs >=
    min_label_hits label words. Column labels come from the header rows (multi-level headers
    composed); the column sharing most field words wins, else the latest year column, else the first
    value (most recent period in SEC layout). Dash cells are 0.0 so columns stay aligned. Unit: '(in
    thousands | millions | billions)' in the table, or the footnote '(k) ... in thousands' of a marked
    column header, multiplies the number unless the row is a percentage or a per-unit quantity
    (per share, price, rate, age, years). typ 'str': the label cell of the best row matched on ANY cell
    (a person table: the name whose description holds the field words); a numeric field whose words
    include 'age' takes that row's 'Age' column. Returns the value (None if nothing scores) or
    (value, info) with detail=True."""
    from .helpers76_2 import _sec_doc
    facts, tables, text = _sec_doc(src)
    Sset = set(_toks(stem))
    want_str = typ == "str"
    person = want_str or "age" in Sset
    best = None
    for ti, tb in enumerate(tables):
        ttxt = " ".join(" ".join(r) for r in tb)
        unit_m = _UNIT_RX.search(ttxt)
        mult = _MULT[unit_m.group(1).lower()] if unit_m else 1.0
        sections, block, in_data = [], [], False
        for ri, row in enumerate(tb):
            label, vals, pct, raws = _row_values(row)
            nonempty = [str(c).strip() for c in row if str(c).strip()]
            if not label:
                continue
            Ltoks = set(_toks(label))
            if person:
                if _is_header_like(row, vals) and len(nonempty) >= 2 and not (Sset - {"officier", "officer", "name", "age"}) & set(_toks(" ".join(nonempty))):
                    block = [nonempty]
                    continue
                other = set(_toks(" ".join(nonempty[1:])))
                core = Sset - {"officier", "officer", "name", "age"}
                h = len(core & (Ltoks | other))
                if h < max(1, min(2, len(core))):
                    continue
                ans = label
                if not want_str:
                    hc = block[0] if block else []
                    j = next((k for k, c in enumerate(hc) if _toks(c) == ["age"]), None)
                    if j is None or j >= len(nonempty) or _cell_value(nonempty[j]) in (None, "skip"):
                        continue
                    ans = _cell_value(nonempty[j])
                sc = 3.0 * h - 0.05 * len(other - Sset)
                cand = (sc, -ti, -ri, ans, {"table": ti, "row": ri, "label": label, "score": sc})
                if best is None or cand[:3] > best[:3]:
                    best = cand
                continue
            if _is_header_like(row, vals):
                if len(nonempty) >= 2:
                    if in_data:
                        block, in_data = [], False
                    block.append(nonempty)
                else:
                    sections.append(label)
                continue
            in_data = True
            lab_hits = Sset & Ltoks
            if len(lab_hits) < min_label_hits:
                continue
            near = set(_toks(sections[-1])) if sections else set()
            far = set(_toks(" ".join(sections[:-1]))) | set(_toks(" ".join(" ".join(b) for b in block)))
            labels = _column_labels(block, len(vals))
            rest = Sset - lab_hits
            col, col_hits = None, set()
            if labels:
                for j, hc in enumerate(labels):
                    hh = rest & set(_toks(hc))
                    if len(hh) > len(col_hits) and isinstance(vals[j], float):
                        col, col_hits = j, hh
                if col is None:
                    yrs = [(j, int(re.findall(r"(?:19|20)\d{2}", hc)[-1])) for j, hc in enumerate(labels)
                           if re.search(r"(?:19|20)\d{2}", hc) and isinstance(vals[j], float)]
                    if yrs:
                        col = max(yrs, key=lambda x: x[1])[0]
            if col is None:
                col = next((j for j, v in enumerate(vals) if isinstance(v, float)), None)
            if col is None:
                continue
            found = lab_hits | (near & Sset) | (far & Sset) | col_hits
            sc = (3.0 * len(lab_hits) + 2.0 * len((near & Sset) - lab_hits) + 1.0 * len((far & Sset) - lab_hits - near)
                  + 3.0 * len(col_hits) - 0.5 * len(Ltoks - Sset) - 3.0 * len(Sset - found)
                  - 6.0 * len((Sset & _SELECTORS) - found))
            if raws[col].replace("%", "").strip() in _DASH:
                sc -= 4.0          # a dash placeholder is weak evidence for an asked field
            per_unit = bool(_PER_UNIT.search(label)) or pct
            m = mult
            if not per_unit and labels:
                mk = re.search(r"\((\d)\)", labels[col] or "")
                vt = [r for r in raws if isinstance(_cell_value(r), float)] or None
                fm = _footnote_unit(text, mk.group(1) if mk else None, label, vt)
                if fm:
                    m = fm
            val = vals[col] if per_unit else vals[col] * m
            cand = (sc, -ti, -ri, val, {"table": ti, "row": ri, "label": label, "col": col,
                                        "col_label": (labels[col] if labels else None),
                                        "mult": 1.0 if per_unit else m, "score": sc})
            if best is None or cand[:3] > best[:3]:
                best = cand
    if best is None:
        return (None, {}) if detail else None
    val = best[3]
    if typ == "int" and isinstance(val, float):
        val = int(round(val))
    elif typ == "float" and isinstance(val, (int, float)):
        val = float(val)
    return (val, best[4]) if detail else val


def report_pair_sources(data_dir=None, pattern="*.zip", exclude=r"taxonomy|glossary|schema"):
    """(A, B) report files of a report-pair task: the bundles matching `pattern` (default every zip that
    is not a glossary / taxonomy), sorted by name - SEC accession numbers sort chronologically, so A is the
    older report. Pass the paths yourself when the instruction names A and B differently."""
    roots = [pathlib.Path(str(data_dir))] if data_dir else []
    # know79: the same data-dir env the other input finders honour (T1_DATA_DIR) before the fixed places
    roots += [pathlib.Path(p) for p in (os.environ.get("T1_DATA_DIR", ""), "/app/data", "/input/environment/data", "data") if p]
    for r in roots:
        if r.is_dir():
            hits = sorted(p for p in r.glob(pattern) if not re.search(exclude, p.name, re.I))
            if len(hits) >= 2:
                return str(hits[0]), str(hits[1])
    raise FileNotFoundError(f"report_pair_sources: fewer than two {pattern} reports in {[str(r) for r in roots]}")
