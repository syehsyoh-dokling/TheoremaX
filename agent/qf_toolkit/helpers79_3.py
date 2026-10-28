"""know79 group 3 (regime portfolios / factor models): the task's own JSON contract and two text-settled conventions.

Task patterns behind the know78 losses of this group (single PERFECT/WRONG verifiers):
- "report helper -> results.json + solution.json written from an instruction template": the model re-typed the
  deliverables by hand and (a) negated a max_drawdown the helper already gave as the positive magnitude the task asks
  for, (b) dropped an intermediate the instruction names only in prose ("this subset count is reported as the
  `num_scored_nonzero` intermediate"), (c) hit a KeyError on a factor-named beta (Mkt_RF_beta_residual vs
  market_beta_residual). json_deliverables builds EVERY JSON file from the instruction's own templates (keys, types,
  {"value", "note"} wrappers, prose-only intermediates, positive-magnitude rules) out of any helper result.
- "IPCA / any ALS whose sign is fixed once after convergence": per_sweep_sign_rule reads whether the task applies the
  eigenvector sign rule inside every sweep (ipca package) or only after the final rotation (then the sweep-to-sweep
  sign flips keep max|dGamma| O(1): the convergence flag and the iteration count follow that rule).
- "month-end rebalancing on a sample that stops mid-month": month_end_sample_rule reads whether the task defines the
  rebalance as the "last trading day of each calendar month" (strict: a sample-end row before the month's last
  business day is not one).

No task names, no data values. Self-tests: proven_work/fail72/selftest79_3.py.

API lines (one per function):
     json_deliverables(values, text=None, *, files=None, strict=False) -> {file name: object} for EVERY JSON output the instruction templates; for name, obj in json_deliverables(rep).items(): emit(name, obj)
     output_json_templates(text=None) -> {file: {"keys": {dotted key: type}, "wrapped": {key: bool}, "notes": {key: note}}} parsed from the instruction's ```json blocks (+ prose-only intermediates)
     positive_magnitude_patterns(text=None) -> list of key regexes the task says to report as a positive magnitude (drawdown, CVaR, VaR, named keys)
     canonical_factor_name(name) -> 'market' for Mkt_RF / Mkt-RF / MKT / mkt_rf / excess market, else lower snake case ('SMB' -> 'smb')
     per_sweep_sign_rule(text=None) -> bool: False when the task flips factor signs only after the final rotation / after convergence
     month_end_sample_rule(text=None) -> bool: False when the task defines rebalances as the last trading day of each calendar month (a mid-month sample end is not one)
"""
from __future__ import annotations

import math
import re

import numpy as np

_TYPE_WORDS = {"string": "str", "str": "str", "text": "str", "date": "str", "int": "int", "integer": "int",
               "float": "float", "number": "float", "double": "float", "bool": "bool", "boolean": "bool",
               "list": "list", "array": "list", "object": "dict", "dict": "dict"}
_SECTIONS = ("results", "intermediates", "solution", "metrics", "summary", "checkpoints")


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def canonical_factor_name(name):
    """Conventional short name of a factor column: the excess market factor (Mkt_RF, Mkt-RF, MKT, MKTRF, mkt_rf,
    market_excess, rm_rf) -> 'market'; any other name -> lower case with non-alphanumerics as '_' ('SMB' -> 'smb',
    'HML' -> 'hml', 'Mom' -> 'mom'). Used to alias '<factor>_beta_residual' keys to the names tasks write."""
    s = str(name).strip()
    n = _norm(s)
    if n in ("mktrf", "mkt", "market", "marketexcess", "excessmarket", "rmrf", "mktexcess", "marketrf"):
        return "market"
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ============================================================================= instruction JSON templates
def _blocks(text):
    """[(file name, block text)] - each ```json fence attached to the last OUTPUT *.json file named before it (a
    path under /output/, or any *.json named in a heading / bullet inside an 'Output' / 'Deliverables' section)."""
    out, cur, buf, infence, in_out = [], None, None, False, False
    for line in str(text).splitlines():
        s = line.strip()
        if not infence:
            if s.startswith("#"):
                in_out = bool(re.search(r"output|deliverable|write|produce", s.lower())) or (
                    in_out and s.startswith("###"))
            m = re.findall(r"([\w.\-/]+\.json)\b", s)
            if m and (s.startswith("#") or s.startswith("`") or s.startswith("-") or s.startswith("*")):
                if "/output/" in m[-1] or in_out:
                    cur = m[-1].split("/")[-1]
                elif "/data/" in m[-1] or "/input" in m[-1]:
                    cur = None
            if s.startswith("```") and cur is not None:
                infence, buf = True, []
            continue
        if s.startswith("```"):
            out.append((cur, "\n".join(buf)))
            infence, cur = False, None
            continue
        buf.append(line)
    return out


def _parse_block(block):
    keys, wrapped, notes, stack = {}, {}, {}, []
    for line in block.splitlines():
        s = re.sub(r"\s//.*$", "", line).strip()
        m = re.match(r'^"([^"]+)"\s*:\s*(.*)$', s)
        if m:
            key, rest = m.group(1), m.group(2).strip()
            full = ".".join(stack + [key])
            if rest.startswith("{") and not rest.rstrip(",").rstrip().endswith("}"):
                stack.append(key)
                continue
            t = re.search(r"<\s*([A-Za-z_]+)", rest)
            keys[full] = _TYPE_WORDS.get(t.group(1).lower(), "float") if t else (
                "bool" if rest.startswith(("true", "false")) else "float" if re.match(r"-?\d", rest) else "str")
            wrapped[full] = bool(re.match(r'^\{\s*"value"\s*:', rest))
            nm = re.search(r'"note"\s*:\s*"([^"]*)"', rest)
            if nm:
                notes[full] = nm.group(1)
            continue
        if s.startswith("}") and stack:
            stack.pop()
    return keys, wrapped, notes


_PROSE_INTER = re.compile(r"(?:reported as|report(?:ed)?|recorded as|given as) (?:the |an? )?[`'\"]?([a-z_][a-z0-9_]*)[`'\"]?"
                          r"\s+intermediate", re.I)


def output_json_templates(text=None):
    """{file name: {"keys": {dotted key: type}, "wrapped": {key: True when the template writes {"value": ...}},
    "notes": {key: the template's "note" text}}} for every ```json block that follows a heading / bullet naming a
    *.json file. Intermediates the instruction names only in prose ("... is reported as the `x` intermediate") are
    appended to the template that has an 'intermediates' group (type int for num_/n_/count names, else float,
    wrapped like their siblings)."""
    t = _text(text)
    out = {}
    for fname, block in _blocks(t):
        if fname in out:
            continue
        keys, wrapped, notes = _parse_block(block)
        if keys:
            out[fname] = {"keys": keys, "wrapped": wrapped, "notes": notes}
    extra = []
    for m in _PROSE_INTER.finditer(t):
        nm = m.group(1)
        if nm.lower() not in ("the", "an", "a", "value") and nm not in extra:
            extra.append(nm)
    for fname, spec in out.items():
        groups = {k.split(".")[0] for k in spec["keys"] if "." in k}
        if "intermediates" not in groups:
            continue
        sib = [k for k in spec["keys"] if k.startswith("intermediates.")]
        wrap = any(spec["wrapped"].get(k) for k in sib)
        for nm in extra:
            full = "intermediates." + nm
            if full not in spec["keys"]:
                spec["keys"][full] = "int" if re.match(r"(num|n|count)_|.*_count$", nm) else "float"
                spec["wrapped"][full] = wrap
        break
    return out


def positive_magnitude_patterns(text=None):
    """Key regexes for every quantity the task says to report as a POSITIVE magnitude (sentences / bullets that
    contain 'positive magnitude', 'positive number' or 'as a positive'): drawdown words -> 'drawdown', CVaR /
    expected shortfall -> 'cvar|expected_shortfall', VaR -> '(^|_)var(_|\\d|$)', plus every `backticked` key."""
    t = _text(text)
    pats = []
    for sent in re.split(r"(?<=[.!?])\s+|\n\s*[-*]\s+|\n\n", t):
        low = sent.lower()
        if not re.search(r"positive (magnitude|number|value)|as a positive|reported positive", low):
            continue
        if "drawdown" in low:
            pats.append(r"drawdown")
        if re.search(r"\bcvar\b|expected shortfall|conditional var", low):
            pats.append(r"cvar|expected_shortfall")
        if re.search(r"\bvar\b|value[- ]at[- ]risk", low):
            pats.append(r"(^|_)var(_|\d|$)")
        for b in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", sent):
            pats.append(r"^" + re.escape(b.lower()) + r"$")
    return list(dict.fromkeys(pats))


# ============================================================================= values -> deliverables
def _flatten(values, flat=None):
    flat = {} if flat is None else flat
    if not isinstance(values, dict):
        try:
            values = dict(values)
        except (TypeError, ValueError):
            return flat
    for k, v in values.items():
        if isinstance(v, dict) and (str(k).lower() in _SECTIONS or str(k).lower().endswith(".json")):
            _flatten(v, flat)
            continue
        if isinstance(v, dict) and "value" in v and len(v) <= 3:
            v = v["value"]
        if isinstance(v, dict):
            continue
        flat.setdefault(str(k), v)
        m = re.match(r"^(.*)_(beta|beta_residual|loading|exposure|tstat)$", str(k))
        if m:
            flat.setdefault(canonical_factor_name(m.group(1)) + "_" + m.group(2), v)
    return flat


def _cast(v, typ):
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    try:
        if typ == "bool":
            return bool(v)
        if typ == "int":
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return int(round(float(v)))
        if typ == "float":
            f = float(v)
            return f if math.isfinite(f) else None
        if typ == "str":
            return str(v)
        return v
    except (TypeError, ValueError):
        return v


def json_deliverables(values, text=None, *, files=None, strict=False):
    """EVERY JSON deliverable the instruction templates, filled from a helper result -> {file name: object}.

    values: one result dict or a list of them (a *_report output with 'results' / 'intermediates', a flat
    ResultDict, {name: {'value': x}} - all accepted). Lookup per template key (its last dotted part): exact name,
    then case/underscore-insensitive, then the factor alias (Mkt_RF_beta_residual -> market_beta_residual).
    Values are cast to the template type (int / float / bool / str; NaN -> None). Keys written {"value": ...} in the
    template are wrapped the same way, with the template's "note" text when it has one. Quantities the task says to
    report as a positive magnitude (positive_magnitude_patterns) are written as abs(value). Prose-only intermediates
    ('reported as the `x` intermediate') are included. files: restrict to these file names. Missing values -> None and a
    printed NOTE (strict=True raises KeyError instead). Raises ValueError when the instruction has no JSON template.
        for name, obj in json_deliverables(rep).items(): emit(name, obj)"""
    t = _text(text)
    tmpl = output_json_templates(t)
    if files is not None:
        tmpl = {f: s for f, s in tmpl.items() if f in set(files)}
    if not tmpl:
        raise ValueError("json_deliverables: no ```json output template found in the instruction - emit the report's "
                         "own 'results' / {'intermediates': ...} dicts instead")
    flat = {}
    for v in (values if isinstance(values, (list, tuple)) else [values]):
        _flatten(v, flat)
    by_norm = {}
    for k, v in flat.items():
        by_norm.setdefault(_norm(k), v)
    pos = [re.compile(p) for p in positive_magnitude_patterns(t)]
    out, missing = {}, []
    for fname, spec in tmpl.items():
        obj = {}
        for full, typ in spec["keys"].items():
            parts = full.split(".")
            leaf = parts[-1]
            if leaf in flat:
                v = flat[leaf]
            elif _norm(leaf) in by_norm:
                v = by_norm[_norm(leaf)]
            else:
                v = None
                missing.append(f"{fname}:{full}")
            v = _cast(v, typ)
            if isinstance(v, float) and any(p.search(leaf.lower()) for p in pos):
                v = abs(v)
            if spec["wrapped"].get(full):
                cell = {"value": v}
                if full in spec["notes"]:
                    cell["note"] = spec["notes"][full]
                v = cell
            d = obj
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[leaf] = v
        out[fname] = obj
    if missing:
        msg = f"json_deliverables: no value for {missing} - pass the dict that holds them (or compute them)"
        if strict:
            raise KeyError(msg)
        print("NOTE " + msg, flush=True)
    try:   # know80 group 3: remember the template-complete files (emit restores keys a re-typed dict drops)
        from .helpers80_3 import register_deliverables
        register_deliverables(out)
        out_dir = pathlib.Path(os.environ.get("OUTPUT_DIR", "/app/output"))
        if out_dir.is_dir():
            for _fn, _obj in out.items():
                (out_dir / _fn).write_text(json.dumps(_obj, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return out


# ============================================================================= text-settled conventions
def per_sweep_sign_rule(text=None):
    """Whether an iterative factor estimator (IPCA / ALS) fixes eigenvector signs INSIDE every sweep.

    True (the ipca-package default, and the answer when no task text is found) unless the task applies the sign
    rule once after the final rotation: 'after the rotation flip the sign', 'after convergence, normalize ...
    sign', 'After this step, every factor ... mean >= 0' with no 'each iteration / every sweep' sign wording.
    Without a per-sweep sign the eigen step flips signs from sweep to sweep, max|dGamma| stays O(1) and the
    reported convergence flag / iteration count are those of that literal procedure (the final rotation and sign
    flip make every reported factor quantity identical either way)."""
    t = _text(text).lower()
    if not t:
        return True
    if re.search(r"(each|every) (iteration|sweep|step)[^.]{0,80}sign|sign[^.]{0,80}(each|every) (iteration|sweep)", t):
        return True
    if re.search(r"after (the )?(final )?rotation[^.]{0,40}flip|after convergence[^#]{0,1500}flip the sign", t, re.S):
        return False
    return True


def month_end_sample_rule(text=None):
    """Whether the LAST row of a sample counts as a month-end rebalance even when the sample stops before its
    calendar month ends. False when the task defines rebalances as the 'last trading day of each calendar month' /
    'last business day of the month' (a row followed, in the calendar, by more trading days of the same month is not
    that day; weights set there would never earn a return); True when no task text is found (old behaviour).

    know81 (conventions table entry 'month_end_rebalance'): the know79 reading above is withdrawn. With no exchange
    calendar supplied, the trading days ARE the sample's rows, so 'last trading day of each calendar month' is the
    last observed row of each calendar month of the sample - the final month included (the month-end formation
    dates of Jegadeesh & Titman 1993 are taken from the data's own calendar; pandas groupby(month).last()).
    False only when the task asks for complete months ('complete / full calendar months only', 'drop / exclude the
    partial (final) month')."""
    t = _text(text).lower()
    if not t:
        return True
    if re.search(r"(?:complete|full) (?:calendar )?months? only|only (?:complete|full) (?:calendar )?months|"
                 r"(?:drop|exclude|discard|ignore)s? (?:the |any )?(?:incomplete|partial) (?:final |last )?month", t):
        return False
    return True
