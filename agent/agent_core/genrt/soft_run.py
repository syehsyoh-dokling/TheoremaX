"""Run a generated program with its `assert` statements turned into recorded checks.

2 Oct autopsy of 252 agent-only runs: 185 attempts died on an AssertionError the program raised
itself, 56% of all attempts were crashes, and 36% of runs ended with a deliverable never written.
An assert that fires BEFORE the files are written loses a possibly correct answer; one that fires
after them still costs a whole repair round, whether the check or the number was wrong.

Here every `assert test, msg` becomes `if not (test): record(msg)`: the program runs to its end
and writes what it computed, and the failed checks are printed as `SOFT_ASSERT_FAILED line N: msg`
for the repair loop (test_runner) to act on. The AST is compiled directly, so line numbers in
tracebacks and in the repair prompt are those of the original source. Any other exception
behaves exactly as before.

Usage: python soft_run.py SCRIPT [args...]
"""
import ast
import json
import os
import sys

# know93: run as a SCRIPT (python soft_run.py candidate.py - the whole-program path) this module is `__main__`;
# the toolkit's guards find the runtime by the name `soft_run` (the raw inputs INPUTS loaded, guard D), and found
# nothing on that path - a program that filled the NaNs before a nan_count helper got nan_count 0 there, while the
# same program in the cells kernel was corrected. The module is registered under its own name as well.
sys.modules.setdefault("soft_run", sys.modules[__name__])

_FAILED: list[str] = []
_EMIT_FLAGS: dict = {}      # know91: deliverable base name -> guard messages of its latest emit
_EMITTED_CELL: list = []    # know94: deliverables emitted by the cell now running (the kernel clears it)
_EMIT_FROM_BUNDLE: dict = {}  # know95: deliverable base name -> its latest emit was a bundle's own object


def _record(message, lineno):
    text = "" if message is None else str(message)
    _FAILED.append(f"SOFT_ASSERT_FAILED line {lineno}: {text[:300]}")


def _rows(x):
    """know89 (guard B): `for item in frame: item['col']` - iterating a DataFrame yields its COLUMN NAMES, so a
    loop body that reads each item by key crashes ('string indices must be integers'). Such a loop gets the
    frame's records (to_dict('records')); anything that is not a DataFrame is returned unchanged."""
    try:
        import pandas as _pd
        if isinstance(x, _pd.DataFrame):
            print("[guard89 records] a DataFrame iterated as records: for-loop runs over df.to_dict('records')",
                  file=sys.stderr, flush=True)
            return x.to_dict("records")
    except Exception:  # noqa: BLE001
        pass
    return x


def _reads_by_key(body, name: str) -> bool:
    """The loop body subscripts the loop variable with a string constant or calls its .get(...)."""
    for stmt in body:
        for n in ast.walk(stmt):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == name
                    and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
                return True
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == name):
                return True
    return False


class _Soften(ast.NodeTransformer):
    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        # know89 (guard B): only a loop whose variable is read BY KEY gets the records wrapper; on a DataFrame
        # that loop crashes as written, on anything else __t1_rows__ is the identity
        if isinstance(node.target, ast.Name) and _reads_by_key(node.body, node.target.id):
            node.iter = ast.copy_location(ast.Call(func=ast.Name(id="__t1_rows__", ctx=ast.Load()),
                                                   args=[node.iter], keywords=[]), node.iter)
        return node

    def visit_Assert(self, node: ast.Assert):
        call = ast.Expr(ast.Call(
            func=ast.Name(id="__soft_assert_record__", ctx=ast.Load()),
            args=[node.msg if node.msg is not None else ast.Constant(None), ast.Constant(node.lineno)],
            keywords=[]))
        new = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=[call], orelse=[])
        return ast.fix_missing_locations(ast.copy_location(new, node))


_LOADED: list = []     # know89: (input name, DataFrame) as INPUTS loaded them - the raw data a helper may need

_BINARY_SUFFIX = (".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".xlsx", ".xls", ".xlsm", ".pdf", ".png", ".jpg",
                  ".jpeg", ".gif", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".db", ".sqlite", ".docx")


def _is_binary(path) -> bool:
    try:
        if str(path).lower().endswith(_BINARY_SUFFIX):
            return True
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(4096)
    except OSError:
        return False


class _Inputs(dict):
    """INPUTS[name]: the input file `name` (its basename, or a relative path) loaded on first
    access - a DataFrame for csv / tsv / parquet / feather, a dict or list for json, text
    otherwise. Keys are the files the task ships (os.environ T1_INPUT_FILES, a JSON list)."""

    def __init__(self, paths):
        super().__init__()
        self._paths = {}
        for p in paths:
            self._paths.setdefault(os.path.basename(p), p)
            self._paths.setdefault(p, p)

    # know82: full mapping protocol - INPUTS.items() / keys / values / iteration / len / in / get
    # all see EVERY shipped input (know81: the dict stayed empty until a key was read, so
    # {k: v for k, v in INPUTS.items()} was {} and a pooled-file study received no files).
    def _names(self):
        return sorted(set(os.path.basename(p) for p in self._paths.values()))

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def __contains__(self, key):
        return key in self._paths or os.path.basename(str(key)) in self._paths

    def items(self):
        return [(k, self[k]) for k in self._names()]

    def values(self):
        return [self[k] for k in self._names()]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __missing__(self, key):
        path = self._paths.get(key) or self._paths.get(os.path.basename(str(key)))
        if path is None:
            raise KeyError(f"no input named {key!r}; the inputs are {sorted(set(os.path.basename(p) for p in self._paths.values()))}")
        from qf_toolkit import io as qio
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".txt":
            # a .txt is a table only when its first lines share a delimiter count
            with open(path, encoding="utf-8", errors="replace") as handle:
                head = [handle.readline() for _ in range(5)]
            counts = {sep: [ln.count(sep) for ln in head if ln.strip()] for sep in (",", "\t", "|", ";")}
            tabular = any(c and min(c) >= 1 and len(set(c)) == 1 for c in counts.values())
            suffix = ".csv" if tabular else ".text"
        if suffix in (".csv", ".tsv", ".parquet", ".pqt", ".feather"):
            value = qio.read_table(path, **({"sep": "\t"} if suffix == ".tsv" else {}))
            try:
                import prep
                value = prep.normalise_frame(value, path)
            except Exception:  # noqa: BLE001
                pass
        elif suffix == ".json":
            value = qio.read_json(path)
        elif suffix in (".jsonl", ".ndjson"):
            # know73: one JSON object per line -> a DataFrame of records (sentiment posts.jsonl)
            import json as _json
            import pandas as _pd
            rows = []
            with open(path, encoding="utf-8", errors="replace") as handle:
                for ln in handle:
                    ln = ln.strip()
                    if ln:
                        try:
                            rows.append(_json.loads(ln))
                        except ValueError:
                            pass
            value = _pd.DataFrame(rows) if rows and all(isinstance(r, dict) for r in rows) else rows
        elif _is_binary(path):
            # know89 (guard D): a binary input (zip archive, workbook, pdf, image ...) has no text content - its
            # decoded bytes handed to a helper made every field None. INPUTS gives its PATH, which every reader
            # (zipfile, pd.read_excel, the toolkit's *_auto helpers) accepts
            print(f"[guard89 input] INPUTS[{key!r}] is a binary file: its path is returned", file=sys.stderr, flush=True)
            value = path
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                value = handle.read()
        self[key] = value
        try:     # know89 guard D: a private copy of a small input's numeric columns that carry NaNs
            if hasattr(value, "columns") and value.size <= 2_000_000:
                num = value.select_dtypes("number")
                if num.shape[1] and bool(num.isna().to_numpy().any()):
                    _LOADED.append((os.path.basename(str(key)), num.copy()))
        except Exception:  # noqa: BLE001
            pass
        return value

    def keys(self):
        return sorted(set(os.path.basename(p) for p in self._paths.values()))


def _contracts() -> dict:
    """{filename: [required columns]} from T1_CONTRACTS (agent.py); {} outside the agent."""
    try:
        return {str(k): list(v) for k, v in json.loads(os.environ.get("T1_CONTRACTS") or "{}").items()}
    except (ValueError, AttributeError):
        return {}


def _norm_name(s) -> str:
    cleaned = "".join(ch for ch in str(s).lower() if ch.isalnum())
    for pfx in ("avg", "mean"):
        if cleaned.startswith(pfx) and len(cleaned) > len(pfx) + 3:
            return cleaned[len(pfx):]
    return cleaned


def _align_contract(filename: str, obj):
    """8 Oct (contract emitter): a DataFrame or dict headed for a deliverable with known required
    columns gets its columns renamed to the contract's exact spelling (case / space / underscore
    tolerant) and ordered contract-first; missing ones are reported, never invented."""
    required = _contracts().get(filename)
    if not required:
        return obj
    try:
        if hasattr(obj, "columns") and hasattr(obj, "rename"):
            have = {_norm_name(c): c for c in obj.columns}
            renames = {}
            for want in required:
                got = have.get(_norm_name(want))
                if got is not None and got != want and want not in obj.columns:
                    renames[got] = want
            if renames:
                obj = obj.rename(columns=renames)
            missing = [c for c in required if c not in obj.columns]
            ordered = [c for c in required if c in obj.columns] + [c for c in obj.columns if c not in required]
            obj = obj[ordered]
            if missing:
                print(f"CONTRACT_NOTE {filename}: missing columns {missing}; present {list(obj.columns)[:20]}", flush=True)
            elif renames:
                print(f"CONTRACT_NOTE {filename}: renamed {renames}", flush=True)
        elif isinstance(obj, dict):
            have = {_norm_name(k): k for k in obj}
            for want in required:
                got = have.get(_norm_name(want))
                if got is not None and got != want and want not in obj:
                    obj[want] = obj[got]
            def _has_path(d, key):   # know87: a dotted contract key ("intermediates.x") is a nested path
                if key in d:
                    return True
                cur = d
                for part in str(key).split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        return False
                    cur = cur[part]
                return True
            missing = [k for k in required if not _has_path(obj, k)]
            if missing:
                print(f"CONTRACT_NOTE {filename}: missing keys {missing}", flush=True)
    except Exception:  # noqa: BLE001 - alignment is an aid, never a failure cause
        pass
    return obj


_FILE_KEY = (".json", ".csv", ".parquet", ".pqt", ".tsv", ".txt", ".md", ".py", ".html", ".png", ".xlsx")


def _bundle_entry(base: str, obj):
    """know87 (round 2): a bundle {file name: object} emitted under ONE of its own file names
    (emit('solution.json', json_deliverables(...)) or emit('x.json', rep) with rep keyed by file names) is
    that file's entry - a deliverable cannot contain itself. Only when EVERY key is a file name and the
    emitted name is one of them; anything else is written as given."""
    try:
        if (isinstance(obj, dict) and obj and dict.__contains__(obj, base)
                and all(isinstance(k, str) and k.lower().endswith(_FILE_KEY) for k in obj.keys())):
            print(f"[emit {base}: the object was a bundle of {sorted(obj.keys())}; wrote its {base!r} entry]",
                  file=sys.stderr, flush=True)
            return dict.__getitem__(obj, base)
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        pass
    return obj


# ----------------------------------------------------------------- know89 emit guards (flip census)
import re as _re

_POS_CACHE: dict = {}


def _positive_drawdown_keys():
    """Guard C: keys the TASK TEXT states as a positive magnitude / non-negative number ("`max_drawdown` is a
    positive magnitude", "Maximum drawdown: report as positive magnitude"). Returns (names, generic) where
    generic is True when the text states it for 'maximum drawdown' without naming a key."""
    text = os.environ.get("T1_INSTRUCTION", "")
    if text in _POS_CACHE:
        return _POS_CACHE[text]
    names, generic = set(), False
    for m in _re.finditer(r"`?\b(\w*draw_?down\w*)`?[^.\n]{0,40}?\b(?:positive magnitude|non-negative number|"
                          r"positive number|as a positive|reported positive)", text, _re.I):
        nm = m.group(1).lower()
        if "max" in nm:
            names.add(nm)
        else:                      # a bare 'drawdown' (a series column) is never flipped; the maximum is meant
            generic = True
    if _re.search(r"maximum drawdown[^.\n]{0,40}\bpositive magnitude", text, _re.I):
        generic = True
    _POS_CACHE[text] = (names, generic)
    return names, generic


def _is_stated_key(key, names, generic) -> bool:
    k = str(key).lower()
    if k in names:
        return True
    return generic and bool(_re.fullmatch(r"(?:max(?:imum)?_?)draw_?down", k))


def _positive_magnitude(obj, names, generic, path=""):
    """Negative values under a stated key -> their magnitude (dict, {'value': x} nesting, list of records)."""
    fixed = []
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if _is_stated_key(k, names, generic):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                    obj[k] = abs(v)
                    fixed.append(f"{path}{k}")
                elif isinstance(v, dict) and isinstance(v.get("value"), (int, float)) \
                        and not isinstance(v.get("value"), bool) and v["value"] < 0:
                    v["value"] = abs(v["value"])
                    fixed.append(f"{path}{k}.value")
                continue
            if isinstance(v, (dict, list)):
                fixed += _positive_magnitude(v, names, generic, f"{path}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                fixed += _positive_magnitude(v, names, generic, f"{path}{i}.")
    return fixed


def _guard_convention(base, obj):
    try:
        names, generic = _positive_drawdown_keys()
        if not names and not generic:
            return obj
        if isinstance(obj, (dict, list)):
            import copy
            cand = copy.deepcopy(obj)          # the program's own object is never changed in place
            fixed = _positive_magnitude(cand, names, generic)
            if fixed:
                obj = cand
        elif hasattr(obj, "columns") and hasattr(obj, "copy"):
            cols = [c for c in obj.columns if _is_stated_key(c, names, generic)]
            fixed = []
            for c in cols:
                try:
                    if (obj[c] < 0).any():
                        obj = obj.copy() if not fixed else obj
                        obj[c] = obj[c].abs()
                        fixed.append(str(c))
                except TypeError:
                    pass
        else:
            return obj
        if fixed:
            print(f"[guard89 convention] {base}: the task states {sorted(names) or ['maximum drawdown']} as a positive "
                  f"magnitude - wrote |x| for {fixed[:6]}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        pass
    return obj


def _helper_pointer(base, cols) -> list:
    """Toolkit names whose API line names this deliverable or two of its contract columns."""
    try:
        import qf_toolkit
        stem = os.path.splitext(base)[0].lower()
        out = []
        for ln in qf_toolkit.API.splitlines():
            m = _re.match(r"\s+([a-z_][a-z0-9_]+)\(", ln)
            if not m:
                continue
            low = ln.lower()
            hits = sum(1 for c in cols if str(c).lower() in low)
            if (len(stem) > 3 and stem in low) or hits >= 2:
                out.append(m.group(1))
        return out[:3]
    except Exception:  # noqa: BLE001
        return []


_REPR_CELL = _re.compile(r"^(?:(?:Range|Datetime|Multi|Period|Int64|Float64|Timedelta)?Index\(|<(?:bound method|function|"
                         r"built-in|pandas\.|numpy\.)|(?:Series|DataFrame)\(|Name: \w+, (?:dtype|Length))")


def _pd_num(s):
    try:
        import pandas as _pd
        return _pd.to_numeric(s, errors="coerce")
    except Exception:  # noqa: BLE001
        return None


def _guard_sanity(base, obj):
    """Guard E: an emitted TABLE with no rows, or a contract column that is entirely NaN, is recorded as a FAILED
    check (the repair loop runs) with a pointer to the toolkit helper - the file is still written as given."""
    try:
        if not (hasattr(obj, "columns") and hasattr(obj, "shape")):
            return
        required = _contracts().get(base) or []
        ptr = _helper_pointer(base, required or list(obj.columns))
        tail = (f" - the toolkit computes it: {', '.join(ptr)}" if ptr else "")
        if obj.shape[0] == 0:
            _record(f"emit({base!r}): the table has NO rows (an empty result: a filter / bucket / merge removed "
                    f"everything){tail}", 0)
            print(f"[guard89 sanity] {base}: empty table", file=sys.stderr, flush=True)
            return
        cols = [c for c in (required or []) if c in obj.columns]
        dead = [c for c in cols if obj[c].isna().all()]
        if dead:
            _record(f"emit({base!r}): column(s) {dead[:6]} are entirely NaN{tail}", 0)
            print(f"[guard89 sanity] {base}: all-NaN {dead[:6]}", file=sys.stderr, flush=True)
        # know90: a STANDARD ERROR is positive by definition; a column of zeros / negatives is a sign slip
        # (the inverse of the negated Hessian of a NEGATIVE log-likelihood gives a negative variance, clipped
        # to 0) and every confidence interval built from it has zero width
        se_cols = [c for c in obj.columns if _re.search(r"(^|_)(se|std_?err(or)?|stderr|standard_error)(_|$)",
                                                         str(c), _re.I)]
        bad_se = []
        for c in se_cols:
            try:
                v = _pd_num(obj[c])
                if v is not None and v.notna().any() and bool((v.dropna() <= 0).all()):
                    bad_se.append(str(c))
            except Exception:  # noqa: BLE001
                pass
        # know90: a cell holding the PRINTED FORM of a Python object (str(df.index) -> 'RangeIndex(start=0, ...)',
        # a bound method, a whole Series) instead of a value - the numbers can all be right and the column is still
        # unreadable
        repr_cols = []
        for c in obj.columns:
            try:
                s = obj[c]
                if getattr(s, "ndim", 1) != 1 or not (s.dtype == object or str(s.dtype).startswith("str")):
                    continue
                txt = s.dropna().astype(str)
                if len(txt) and txt.str.match(_REPR_CELL).mean() >= 0.5:
                    repr_cols.append(str(c))
            except Exception:  # noqa: BLE001
                pass
        if repr_cols:
            _record(f"emit({base!r}): column(s) {repr_cols[:6]} hold the PRINTED FORM of a Python object (e.g. "
                    f"{str(obj[repr_cols[0]].dropna().astype(str).iloc[0])[:60]!r}), not values - write the values "
                    "themselves (a date column from the index: df.index.strftime('%Y-%m-%d') or reset_index(); "
                    "never str(df.index) / str(series) in one cell)", 0)
            print(f"[guard90 sanity] {base}: object reprs in {repr_cols[:6]}", file=sys.stderr, flush=True)
        if bad_se:
            _record(f"emit({base!r}): standard-error column(s) {bad_se[:6]} are all <= 0 - a standard error is "
                    "positive; the covariance of an MLE is the INVERSE of the Hessian of the NEGATIVE "
                    "log-likelihood (do not negate it again); for a GPD fit qf_toolkit.gpd_mle_se(excesses) "
                    "gives se_xi / se_sigma", 0)
            print(f"[guard90 sanity] {base}: non-positive standard errors {bad_se[:6]}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        pass


def _same_value(a, b) -> bool:
    try:
        if a is b:
            return True
        if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
            return True
        if hasattr(a, "equals") and type(a) is type(b):
            return bool(a.equals(b))
        r = a == b
        if isinstance(r, bool):
            return r
        import numpy as _np
        return bool(_np.array_equal(_np.asarray(a), _np.asarray(b)))
    except Exception:  # noqa: BLE001
        return False


def _candidates():
    try:
        from qf_toolkit._rdict import recent_results
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], set()
    for res in recent_results():
        stack = [(res, 0)]
        while stack:
            o, d = stack.pop(0)
            if id(o) in seen:
                continue
            seen.add(id(o))
            out.append(o)
            if isinstance(o, dict) and d < 2:
                stack += [(v, d + 1) for v in dict.values(o) if isinstance(v, dict) or hasattr(v, "columns")]
    return out


def _guard_restore(base, obj):
    """Guard F (flip census): a deliverable the program RESHAPED from a toolkit result (a hand-typed key list for
    conform_to_keys, a column subset, a re-typed summary) and that thereby LOST keys / columns the task's contract
    requires gets them back from that result - only when every key / column the program kept is identical to the
    result's own (the same object, reshaped), and only the missing contract entries are added."""
    try:
        required = [str(k) for k in (_contracts().get(base) or []) if "." not in str(k)]
        if not required:
            return obj
        if isinstance(obj, dict) and not getattr(obj, "_t1_bundle", False):
            missing = [k for k in required if k not in obj]
            if not missing or len(obj) < 2:
                return obj
            for cand in _candidates():
                if not isinstance(cand, dict) or cand is obj:
                    continue
                if not all(dict.__contains__(cand, k) for k in missing):
                    continue
                if all(dict.__contains__(cand, k) and _same_value(v, dict.__getitem__(cand, k)) for k, v in obj.items()):
                    new = dict(obj)
                    for k in missing:
                        new[k] = dict.__getitem__(cand, k)
                    order = [k for k in required if k in new] + [k for k in new if k not in required]
                    if [k for k in obj if k in required] == [k for k in required if k in obj]:
                        new = {k: new[k] for k in order}          # the program kept the contract order
                    print(f"[guard89 contract] {base}: restored {missing[:6]} from the toolkit result the object was "
                          "reshaped from", file=sys.stderr, flush=True)
                    return new
        elif hasattr(obj, "columns") and hasattr(obj, "shape"):
            missing = [c for c in required if c not in obj.columns]
            if not missing or obj.shape[1] < 1:
                return obj
            for cand in _candidates():
                if not hasattr(cand, "columns") or cand is obj or len(cand) != len(obj):
                    continue
                if not all(c in cand.columns for c in missing) or not all(c in cand.columns for c in obj.columns):
                    continue
                if all(_same_value(obj[c].reset_index(drop=True), cand[c].reset_index(drop=True)) for c in obj.columns):
                    new = obj.copy()
                    for c in missing:
                        new[c] = cand[c].to_numpy()
                    print(f"[guard89 contract] {base}: restored columns {missing[:6]} from the toolkit table the "
                          "frame was cut from", file=sys.stderr, flush=True)
                    return new
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        pass
    return obj


# ----------------------------------------------------------------- know91 guard G (bundle drift)
def _bundle_parts():
    """(path, object) for every DataFrame / dict inside the latest WHOLE-TASK bundle results (helpers wrapped by
    helpers79_1.memoize_bundle carry `_t1_bundle`), down to two levels."""
    try:
        from qf_toolkit._rdict import recent_results
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], set()
    for res in recent_results():
        if not (isinstance(res, dict) and getattr(res, "_t1_bundle", False)):
            continue
        stack = [(res, "", 0)]
        while stack:
            o, path, d = stack.pop(0)
            if id(o) in seen:
                continue
            seen.add(id(o))
            if path:
                out.append((path, o))
            if isinstance(o, dict) and d < 2:
                for k, v in dict.items(o):
                    if isinstance(v, dict) or hasattr(v, "columns"):
                        stack.append((v, f"{path}[{k!r}]", d + 1))
    return out


def _num(v):
    if isinstance(v, dict) and "value" in v and len(v) <= 3:
        v = v["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            import numpy as _np
            if isinstance(v, _np.generic) and not isinstance(v, _np.bool_):
                return float(v)
        except Exception:  # noqa: BLE001
            pass
        return None
    return float(v)


def _far(a, b) -> bool:
    """Different beyond rounding: relative > 1e-3 AND absolute > 1e-4 (a task's 4-6 decimal rounding never trips)."""
    if a != a and b != b:
        return False
    if a != a or b != b:
        return True
    d = abs(a - b)
    return d > 1e-4 and d > 1e-3 * max(abs(a), abs(b))


def _guard_bundle_drift(base, obj):
    """Guard G (know91): an emitted table / dict with the SAME SHAPE as a part of a whole-task bundle result the
    program called (same columns and row count, or most of the same keys) but DIFFERENT numbers means a part of
    the proven bundle was recomputed by hand - the census of every earlier run: the bundle used as is passed,
    a bundle mixed with hand-rebuilt parts failed. Recorded as a FAILED check (the repair loop runs); the file is
    still written as given."""
    try:
        parts = _bundle_parts()
        if not parts or obj is None:
            return
        if hasattr(obj, "columns") and hasattr(obj, "shape"):
            import numpy as _np
            for path, part in parts:
                if part is obj or not hasattr(part, "columns") or len(part) != len(obj) \
                        or set(map(str, part.columns)) != set(map(str, obj.columns)):
                    continue
                diff = []
                for c in obj.columns:
                    try:
                        a = _np.sort(_np.asarray(obj[c], dtype=float))
                        b = _np.sort(_np.asarray(part[c], dtype=float))
                    except (TypeError, ValueError):
                        continue                      # text columns: formatting is the program's business
                    if any(_far(float(x), float(y)) for x, y in zip(a, b)):
                        diff.append(str(c))
                if diff:
                    _record(f"emit({base!r}): column(s) {diff[:6]} differ from the toolkit bundle's own table "
                            f"{path} (same columns, same rows) - the bundle is the proven whole-task computation; "
                            f"emit that object as is instead of recomputing its parts", 0)
                    print(f"[guard91 bundle] {base}: differs from bundle part {path} in {diff[:6]}",
                          file=sys.stderr, flush=True)
                    return
        elif isinstance(obj, dict) and not getattr(obj, "_t1_bundle", False):
            mine = {str(k): _num(v) for k, v in obj.items()}
            mine = {k: v for k, v in mine.items() if v is not None}
            if len(mine) < 3:
                return
            for path, part in parts:
                if part is obj or not isinstance(part, dict):
                    continue
                theirs = {str(k): _num(v) for k, v in dict.items(part)}
                shared = [k for k in mine if theirs.get(k) is not None]
                if len(shared) < 3 or len(shared) < 0.8 * len(mine):
                    continue
                diff = [k for k in shared if _far(mine[k], theirs[k])]
                if diff:
                    _record(f"emit({base!r}): key(s) {diff[:6]} differ from the toolkit bundle's own result {path} "
                            f"(the same keys) - the bundle is the proven whole-task computation; emit that object "
                            f"as is instead of recomputing its parts", 0)
                    print(f"[guard91 bundle] {base}: differs from bundle part {path} in {diff[:6]}",
                          file=sys.stderr, flush=True)
                    return
    except Exception:  # noqa: BLE001 - an aid, never a failure cause
        pass


def _emit(name, obj):
    """emit(name, obj): write deliverable `name` NOW, in the format its extension asks for, and
    read it back. DataFrame -> csv / parquet / tsv (index=False); dict or list -> json (numpy and
    pandas values converted); str -> text (md, txt, html, ...); a matplotlib figure -> png."""
    from qf_toolkit import io as qio
    out_dir = os.environ.get("T1_OUTPUT_DIR") or "/app/output"
    path = name if os.path.isabs(str(name)) else os.path.join(out_dir, str(name))
    obj = _bundle_entry(os.path.basename(str(name)), obj)
    # know95: is this deliverable a whole-task bundle's own object, as the bundle returned it? Then a failed
    # check the PROGRAM wrote about it is far more likely a wrong check than a wrong bundle (a put-call-parity
    # run's correct bundle answer was "fixed" into a wrong one after its own assertion failed), so the kernel
    # does not hold `done` over it for that reason (the emit guards still apply)
    try:
        _EMIT_FROM_BUNDLE[os.path.basename(str(name))] = any(part is obj for _p, part in _bundle_parts())
    except Exception:  # noqa: BLE001
        _EMIT_FROM_BUNDLE[os.path.basename(str(name))] = False
    if isinstance(obj, dict) and str(name).lower().endswith(".json"):
        # know80 group 3: a JSON file json_deliverables already built from the task template keeps every template
        # key when a later cell re-types it (the re-typed dict's own values win; only missing keys come back)
        try:
            from qf_toolkit.helpers80_3 import fill_missing_from_deliverables
            obj = fill_missing_from_deliverables(os.path.basename(str(name)), obj)
        except Exception:  # noqa: BLE001 - an aid, never a failure cause
            pass
        # know81 group 3: a re-typed template wrapped an already-shaped {'value': x} once more -> lift it back
        try:
            from qf_toolkit.helpers81_3 import collapse_value_nesting
            obj = collapse_value_nesting(obj)
        except Exception:  # noqa: BLE001 - an aid, never a failure cause
            pass
    obj = _guard_restore(os.path.basename(str(name)), obj)        # know89 guard F
    obj = _align_contract(os.path.basename(str(name)), obj)
    obj = _guard_convention(os.path.basename(str(name)), obj)     # know89 guard C
    _n0 = len(_FAILED)
    _guard_sanity(os.path.basename(str(name)), obj)                # know89 guard E
    _guard_bundle_drift(os.path.basename(str(name)), obj)          # know91 guard G
    # know91: the guard verdict of the LATEST emit of each deliverable (an emit that fixes it clears it); the
    # cells loop does not accept `done` while a deliverable still carries one
    _EMIT_FLAGS[os.path.basename(str(name))] = list(_FAILED[_n0:])
    _EMITTED_CELL.append(os.path.basename(str(name)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".py" and isinstance(obj, tuple) and obj and isinstance(obj[0], str):
        obj = obj[0]          # know76: (source, todo) from a migration helper -> the source text
    if suffix == ".py" and not isinstance(obj, str):
        # know75: a code deliverable must be SOURCE TEXT; a function/module object is turned into its source
        import inspect
        try:
            obj = inspect.getsource(obj)
        except (TypeError, OSError):
            raise TypeError(f"emit({name!r}): a .py deliverable needs the SOURCE TEXT (a str); build it as a "
                            "string (e.g. polars_0x_to_1x_source(old_src)) and emit that")
    if hasattr(obj, "savefig"):
        obj.savefig(path, dpi=100)
        return path
    if hasattr(obj, "to_csv") and suffix in (".csv", ".tsv", ".txt"):
        qio.write_csv(obj, path, sep="\t" if suffix == ".tsv" else ",")
    elif hasattr(obj, "to_parquet") and suffix in (".parquet", ".pqt"):
        qio.write_parquet(obj, path)
    elif suffix == ".json" or isinstance(obj, (dict, list)):
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict(orient="records")
        qio.write_json(obj, path)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(obj if isinstance(obj, str) else str(obj))
    return path


def main() -> int:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    with open(script, encoding="utf-8") as handle:
        source = handle.read()
    try:
        # know87: a whole program gets the helper-unit identities of autofix._PRERUN_FN (the cell-oriented
        # drops/settings stay cell-only); notes go to stderr, never into the program's stdout
        import autofix
        source, _pre = autofix.prerun(source, fn_only=True)
        if _pre:
            print("[pre-run: " + "; ".join(_pre) + "]", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a rewrite aid, never a failure cause
        pass
    tree = _Soften().visit(ast.parse(source, filename=script))
    ast.fix_missing_locations(tree)
    try:
        paths = json.loads(os.environ.get("T1_INPUT_FILES") or "[]")
    except ValueError:
        paths = []
    scope = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__,
             "__soft_assert_record__": _record, "__t1_rows__": _rows,
             # 6 Oct: the model's programs were too long for the 4,000-token reply (median
             # 2,434 tokens, a quarter above 3,700). Reading inputs and writing deliverables
             # with read-back checks is boilerplate; these two names take it off the program.
             "INPUTS": _Inputs(paths), "emit": _emit}
    try:
        import prep
        prep.install(scope)
    except Exception:  # noqa: BLE001
        pass
    try:
        exec(compile(tree, script, "exec"), scope)
    finally:
        if _FAILED:
            sys.stdout.flush()
            print("\n".join(_FAILED[:12]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
