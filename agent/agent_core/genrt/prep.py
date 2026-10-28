"""Pre-resolution of the failures the model kept making - done in Python before the model sees
the data, and in the runtime while its code runs (6 Oct, owner's instruction: every collected
failure goes into Python, not back to the model).

Census of 3 x 12-unit batches (s12_40..42, 209 crashes) and the 2 Oct autopsy of 252 runs:
  - inputs read as ONE column (`Length mismatch: Expected axis has 1 elements`): wrong delimiter
  - `Cannot convert input [...]` / tz-aware or epoch-ms timestamps fed to to_datetime
  - numeric columns that arrived as strings (`ufunc add float64 <U3`, `rsub str float64`,
    `invalid literal for int()`): thousands separators, percent signs, blanks
  - `EmptyDataError: No columns to parse` / FileNotFoundError: the program re-read a deliverable
    it had not written yet, or a bare file name from the wrong directory
  - `TypeError: the JSON object must be str ... not dict`: json.loads on an already loaded input
  - `fit_garch11() got an unexpected keyword argument 'mean'` and friends: toolkit called with
    keywords it does not take (the call is right, the extra keyword is the model's habit)
  - `NameError: name 'skewness' is not defined`, `'chi2_gen' object has no attribute
    'chi2_contingency'`: a function the model remembers but did not import
  - `unknown money-market basis 'USD'`: a currency code where a day-count basis was expected

`normalise_frame` fixes the first three on load (and records what it did in
frame.attrs["prep_notes"], which the runtime state and the evidence show); `install` patches the
program's namespace for the rest: resolving readers, lenient json.loads, lenient toolkit calls,
the usual scientific names pre-bound, and `resolve_name` for a NameError the kernel can retry.
"""
from __future__ import annotations

import functools
import inspect
import os
import re
import types
import json as _stdjson
_JSON_LOADS = _stdjson.loads      # the unpatched loader, for our own use

import numpy as np
import pandas as pd

DATE_NAME = re.compile(r"(date|time|timestamp|_ts$|^ts$|asof|as_of|expir|maturity|settle|period|month|day)", re.I)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
NUMERIC_JUNK = re.compile(r"[,\s%$€£]")


def _is_text(s: pd.Series) -> bool:
    """object or pandas-3 StringDtype columns (pandas 3 reads text as `str`, not object)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _notes(frame: pd.DataFrame) -> list[str]:
    frame.attrs.setdefault("prep_notes", [])
    return frame.attrs["prep_notes"]


def resniff(frame: pd.DataFrame, path) -> pd.DataFrame:
    """A one-column frame whose header carries another delimiter was read with the wrong `sep`."""
    if frame.shape[1] != 1 or path is None or not os.path.isfile(str(path)):
        return frame
    header = str(frame.columns[0])
    for sep in (";", "\t", "|"):
        if header.count(sep) >= 1:
            try:
                new = pd.read_csv(path, sep=sep)
            except Exception:  # noqa: BLE001
                continue
            if new.shape[1] > 1:
                _notes(new).append(f"delimiter {sep!r} (the file is not comma-separated)")
                return new
    return frame


def coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Object columns that are numbers in disguise ('1,234.5', '12.5%', ' 3 ') become floats.
    A percent column keeps its printed magnitude (12.5, not 0.125) and the note says so."""
    for c in frame.columns:
        s = frame[c]
        if not _is_text(s) or len(s) == 0:
            continue
        text = s.dropna().astype(str).str.strip()
        if text.empty or text.str.match(ISO_DATE).mean() > 0.5:
            continue
        cleaned = text.str.replace(NUMERIC_JUNK, "", regex=True).replace({"": np.nan, "-": np.nan, "nan": np.nan})
        as_num = pd.to_numeric(cleaned, errors="coerce")
        share = as_num.notna().sum() / max(1, cleaned.notna().sum())
        if share >= 0.95 and as_num.notna().sum() >= 2 and not text.str.fullmatch(r"-?\d+").all():
            pct = text.str.contains("%").mean() > 0.5
            full = pd.to_numeric(s.astype(str).str.strip().str.replace(NUMERIC_JUNK, "", regex=True)
                                 .replace({"": np.nan, "-": np.nan, "nan": np.nan}), errors="coerce")
            frame[c] = full
            _notes(frame).append(f"column {c!r}: numeric strings -> float" + (" (percent sign stripped; values still in percent units)" if pct else ""))
        elif share >= 0.95 and text.str.fullmatch(r"-?\d+").all() and text.str.len().max() <= 15:
            # digits stored as text (ids stay text when they are long or zero-padded)
            if not text.str.startswith("0").any():
                frame[c] = pd.to_numeric(s, errors="coerce")
                _notes(frame).append(f"column {c!r}: integer strings -> number")
    return frame


def parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Date-like TEXT columns become tz-naive datetime64 (UTC instant): ISO strings with or without
    zone. Epoch seconds / milliseconds / nanoseconds stay numbers (know90) and get a note."""
    for c in frame.columns:
        s = frame[c]
        name_hint = bool(DATE_NAME.search(str(c)))
        if pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                frame[c] = s.dt.tz_convert("UTC").dt.tz_localize(None)
                _notes(frame).append(f"column {c!r}: tz-aware -> UTC naive")
            continue
        if _is_text(s):
            text = s.dropna().astype(str).str.strip()
            if text.empty or text.str.match(ISO_DATE).mean() < 0.9:
                continue
            try:
                parsed = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                try:
                    parsed = pd.to_datetime(s, utc=True, errors="coerce")
                except (TypeError, ValueError):
                    continue
            if parsed.notna().sum() >= 0.9 * s.notna().sum():
                frame[c] = parsed.dt.tz_localize(None)
                _notes(frame).append(f"column {c!r}: ISO text -> datetime64 (UTC naive)")
        elif (name_hint and pd.api.types.is_integer_dtype(s) and s.notna().any()
              and not re.search(r"(^|_)id$", str(c), re.I)):   # know75: update_id is an id, not a date
            mag = float(np.nanmedian(np.abs(s.to_numpy(dtype=float))))
            unit = "s" if 1e8 <= mag < 1e11 else "ms" if 1e11 <= mag < 1e14 else "us" if 1e14 <= mag < 1e17 \
                else "ns" if mag >= 1e17 else None
            if unit:
                # know90: epoch INTEGERS stay integers, as in the file. The task states its times in the same
                # unit (start_time_ms in a JSON spec, a bucket length in ms), so a program compares a column with
                # those numbers; a column silently turned into datetime64 crashed that comparison
                # ('Invalid comparison between dtype=datetime64[ms] and int') on every repair round.
                _notes(frame).append(f"column {c!r}: epoch {unit} integers (kept as numbers; "
                                     f"pd.to_datetime(col, unit='{unit}') gives datetimes)")
    return frame


def normalise_frame(frame: pd.DataFrame, path=None) -> pd.DataFrame:
    """Everything above, plus stripped column names. Never reorders rows or drops anything."""
    try:
        frame = resniff(frame, path)
        stripped = [str(c).strip() for c in frame.columns]
        if stripped != [str(c) for c in frame.columns]:
            frame.columns = stripped
            _notes(frame).append("column names stripped of blanks")
        frame = coerce_numeric(frame)
        frame = parse_dates(frame)
    except Exception:  # noqa: BLE001 - normalisation is an aid, never a failure cause
        pass
    return frame


# --------------------------------------------------------------------------- runtime patches
_ORIGINAL: dict[str, object] = {}
RUNTIME_NOTES: list[str] = []


def _roots() -> list[str]:
    roots = []
    for key in ("T1_OUTPUT_DIR",):
        if os.environ.get(key):
            roots.append(os.environ[key])
    try:
        import json as _json
        for p in _JSON_LOADS(os.environ.get("T1_INPUT_FILES") or "[]"):
            d = os.path.dirname(p)
            if d and d not in roots:
                roots.append(d)
    except ValueError:
        pass
    for extra in ("/input", "/app/output", "."):
        if extra not in roots:
            roots.append(extra)
    return roots


def resolve_path(name):
    """A bare or wrong-directory file name -> the file that exists (inputs first, then outputs)."""
    s = str(name)
    if os.path.isfile(s):
        return s
    base = os.path.basename(s)
    for root in _roots():
        cand = os.path.join(root, base)
        if os.path.isfile(cand):
            RUNTIME_NOTES.append(f"read {s!r} resolved to {cand!r}")
            return cand
    return s


def _reader(original, kind):
    @functools.wraps(original)
    def wrapper(path, *args, **kwargs):
        # know88: a table reader handed an already-loaded INPUTS entry (DataFrame) or one of its columns (Series)
        # returns it as a frame; pandas died on it with "argument of type 'method' is not iterable" (Series.mode)
        if kind in ("csv", "parquet") and isinstance(path, pd.DataFrame):
            RUNTIME_NOTES.append(f"read_{kind} received an already-loaded DataFrame; returned a copy")
            return path.copy()
        if kind in ("csv", "parquet") and isinstance(path, pd.Series):
            RUNTIME_NOTES.append(f"read_{kind} received a Series (a column, not a file); returned it as a one-column frame")
            return path.to_frame()
        if isinstance(path, (str, os.PathLike)):
            resolved = resolve_path(path)
            if not os.path.isfile(resolved) and os.path.basename(str(path)) in _emitted_names():
                raise FileNotFoundError(f"{path!r} is a DELIVERABLE this program has not emitted yet - use "
                                        "the variable that holds it, not the file")
            if os.path.isfile(resolved) and os.path.getsize(resolved) == 0:
                raise pd.errors.EmptyDataError(f"{resolved!r} is EMPTY (0 bytes) - it was never written; "
                                               "recompute from the inputs instead of reading it")
            path = resolved
            # know82: a csv reader pointed at a non-table file (params.json, a .py / .md source) parsed it
            # as a table and raised ParserError three times in a row; dispatch by the file's own type
            if kind == "csv" and os.path.isfile(str(path)):
                suf = os.path.splitext(str(path))[1].lower()
                if suf == ".json":
                    RUNTIME_NOTES.append(f"read_csv on {os.path.basename(str(path))}: a JSON file - returned the parsed JSON")
                    with open(path, encoding="utf-8") as handle:
                        return _JSON_LOADS(handle.read())
                if suf in (".parquet", ".pqt"):
                    return _ORIGINAL.get("pd.read_parquet", pd.read_parquet)(path)
                if suf in (".py", ".md", ".sql", ".yaml", ".yml", ".toml", ".xml", ".html"):
                    RUNTIME_NOTES.append(f"read_csv on {os.path.basename(str(path))}: a text file - returned its text")
                    with open(path, encoding="utf-8", errors="replace") as handle:
                        return handle.read()
        frame = original(path, *args, **kwargs)
        if kind == "csv" and isinstance(frame, pd.DataFrame) and not any(k in kwargs for k in ("sep", "delimiter")):
            frame = normalise_frame(frame, path)
        return frame
    return wrapper


def _emitted_names() -> set[str]:
    out = os.environ.get("T1_OUTPUT_DIR") or ""
    try:
        import json as _json
        return set(_JSON_LOADS(os.environ.get("T1_DELIVERABLES") or "[]")) if out else set()
    except ValueError:
        return set()


def lenient_loads(original):
    @functools.wraps(original)
    def loads(s, *args, **kwargs):
        if isinstance(s, (dict, list)):
            RUNTIME_NOTES.append("json.loads received an already-parsed object; returned it as is")
            return s
        if isinstance(s, str) and "\n" not in s and os.path.isfile(resolve_path(s)) and not s.lstrip().startswith(("{", "[")):
            with open(resolve_path(s), encoding="utf-8") as handle:
                return original(handle.read(), *args, **kwargs)
        return original(s, *args, **kwargs)
    return loads


_KW_RE = re.compile(r"got an unexpected keyword argument '([^']+)'")


def lenient_call(fn):
    """A toolkit function that drops a keyword it does not take (recorded) instead of dying, and
    whose positional-count error names its real signature."""
    if not callable(fn) or isinstance(fn, type):
        return fn
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        dropped = []
        for _ in range(8):
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                m = _KW_RE.search(str(exc))
                if m and m.group(1) in kwargs:
                    dropped.append(m.group(1))
                    kwargs = {k: v for k, v in kwargs.items() if k != m.group(1)}
                    RUNTIME_NOTES.append(f"{fn.__name__}: keyword {m.group(1)!r} is not a parameter; ignored. "
                                         f"Signature: {fn.__name__}{sig}")
                    continue
                if "positional argument" in str(exc):
                    raise TypeError(f"{exc}. The signature is {fn.__name__}{sig} - pass the extra values "
                                    "by keyword, or drop them") from None
                raise
        return fn(*args, **kwargs)
    wrapper.__signature__ = sig
    return wrapper


def lenient_toolkit() -> None:
    """Wrap every public function of qf_toolkit and its submodules (idempotent)."""
    try:
        import qf_toolkit
    except Exception:  # noqa: BLE001
        return
    if _ORIGINAL.get("toolkit"):
        return
    _ORIGINAL["toolkit"] = True
    modules = [qf_toolkit] + [m for m in vars(qf_toolkit).values()
                              if isinstance(m, types.ModuleType) and m.__name__.startswith("qf_toolkit")]
    for mod in modules:
        for name, obj in list(vars(mod).items()):
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            if getattr(obj, "__module__", "").startswith("qf_toolkit"):
                setattr(mod, name, lenient_call(obj))


COMMON_IMPORTS = (
    "import numpy as np\nimport pandas as pd\nimport math, json, os, re, itertools, collections, datetime\n"
    "import scipy\nfrom scipy import stats, optimize, interpolate, linalg, special, integrate\n"
    "from scipy.stats import norm, t as student_t, chi2, skew, kurtosis\n"
    "from datetime import date, datetime as dt_datetime, timedelta\n"
)
_MODULES_FOR_NAMES = ("scipy.stats", "scipy.optimize", "scipy.interpolate", "scipy.linalg", "scipy.special",
                      "numpy", "math", "pandas", "qf_toolkit", "qf_toolkit.riskx", "qf_toolkit.stats",
                      "qf_toolkit.exotics", "qf_toolkit.fx", "qf_toolkit.credit_rates", "qf_toolkit.copulas",
                      "qf_toolkit.aggregate_loss", "qf_toolkit.backtest", "qf_toolkit.attribution",
                      "qf_toolkit.compound", "qf_toolkit.event_study", "statsmodels.api", "statsmodels.tsa.stattools")
_ALIASES = {"skewness": ("scipy.stats", "skew"), "kurt": ("scipy.stats", "kurtosis"),
            "normcdf": ("scipy.stats", "norm"), "N": ("scipy.stats", "norm")}


def resolve_name(name: str):
    """(value, where) for a name the program used without importing it; (None, '') if unknown."""
    import importlib
    if name in _ALIASES:
        modname, attr = _ALIASES[name]
        try:
            mod = importlib.import_module(modname)
            value = getattr(mod, attr)
            return (value.cdf if name in ("normcdf", "N") else value), f"{modname}.{attr}"
        except Exception:  # noqa: BLE001
            return None, ""
    for modname in _MODULES_FOR_NAMES:
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(mod, name) and not name.startswith("_"):
            return getattr(mod, name), f"{modname}.{name}"
        if modname in ("scipy", "numpy", "pandas", "statsmodels.api") and name in ("sm", "sp"):
            return mod, modname
    return None, ""


def where_is(attr: str) -> str:
    """Modules that carry attribute `attr` (for "'chi2_gen' object has no attribute 'chi2_contingency'")."""
    import importlib
    hits = []
    for modname in _MODULES_FOR_NAMES:
        try:
            if hasattr(importlib.import_module(modname), attr):
                hits.append(f"{modname}.{attr}")
        except Exception:  # noqa: BLE001
            continue
    return ", ".join(hits[:4])


def install(ns: dict) -> None:
    """Patch the program's namespace and the shared modules it will use."""
    import builtins
    import json as _json
    if "pd.read_csv" not in _ORIGINAL:
        _ORIGINAL["pd.read_csv"] = pd.read_csv
        pd.read_csv = _reader(pd.read_csv, "csv")
        _ORIGINAL["pd.read_parquet"] = pd.read_parquet
        pd.read_parquet = _reader(pd.read_parquet, "parquet")
        _ORIGINAL["pd.read_json"] = pd.read_json
        pd.read_json = _reader(pd.read_json, "json")
        _ORIGINAL["json.loads"] = _json.loads
        _json.loads = lenient_loads(_json.loads)
        _ORIGINAL["open"] = builtins.open

        @functools.wraps(_ORIGINAL["open"])
        def _open(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, os.PathLike)) and "r" in str(mode) and not os.path.isfile(str(file)):
                file = resolve_path(file)
            return _ORIGINAL["open"](file, mode, *args, **kwargs)
        builtins.open = _open
    lenient_toolkit()
    read_guard()
    try:
        exec(COMMON_IMPORTS, ns)
        exec("from qf_toolkit import *\n", ns)
    except Exception:  # noqa: BLE001
        pass
    # 8 Oct (lapis tengah): every JSON input flattened into CFG - no config key is guessed
    try:
        ns["CFG"] = load_cfg()
    except Exception:  # noqa: BLE001
        ns["CFG"] = {}


def load_cfg() -> dict:
    """{dotted.path: value} over every JSON input (T1_INPUT_FILES), plus each file under its stem."""
    paths = _JSON_LOADS(os.environ.get("T1_INPUT_FILES") or "[]")
    cfg: dict = {}
    for p in paths:
        if not str(p).lower().endswith(".json") or os.path.basename(p) == "manifest.json":
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as handle:
                obj = _JSON_LOADS(handle.read())
        except (OSError, ValueError):
            continue
        for k, v in _flatten(obj).items():
            cfg.setdefault(k, v)
        cfg.setdefault(os.path.splitext(os.path.basename(p))[0], obj)
    return cfg


def _flatten(obj, prefix="", out=None, depth=0) -> dict:
    out = {} if out is None else out
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out[key] = v
            if isinstance(v, (dict, list)):
                _flatten(v, key, out, depth + 1)
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        for i, v in enumerate(obj[:50]):
            _flatten(v, f"{prefix}[{i}]", out, depth + 1)
    return out


def read_guard() -> None:
    """qf_toolkit.io.read_json / read_table / find_input given an object that is ALREADY loaded
    (INPUTS[...]) return it instead of dying on FileNotFoundError (Colab census: 26 crashes)."""
    try:
        from qf_toolkit import io as qio
    except Exception:  # noqa: BLE001
        return
    if _ORIGINAL.get("read_guard"):
        return
    _ORIGINAL["read_guard"] = True
    import qf_toolkit

    def guard(fn, kind):
        @functools.wraps(fn)
        def wrapper(name, *args, **kwargs):
            if isinstance(name, (dict, list)) and kind in ("json", "find"):
                RUNTIME_NOTES.append(f"{fn.__name__} received an already-loaded object; returned it")
                return name
            if hasattr(name, "columns") and kind in ("table", "find"):
                RUNTIME_NOTES.append(f"{fn.__name__} received a DataFrame; returned it")
                return name
            if isinstance(name, dict) and kind == "table":
                return pd.DataFrame(name)
            return fn(name, *args, **kwargs)
        return wrapper
    for attr, kind in (("read_json", "json"), ("read_table", "table"), ("read_csv", "table"), ("find_input", "find")):
        fn = getattr(qio, attr, None)
        if inspect.isfunction(fn):
            g = guard(fn, kind)
            setattr(qio, attr, g)
            if hasattr(qf_toolkit, attr):
                setattr(qf_toolkit, attr, g)


def take_notes() -> list[str]:
    notes, RUNTIME_NOTES[:] = list(RUNTIME_NOTES), []
    return notes
