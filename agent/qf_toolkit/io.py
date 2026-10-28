"""Reading inputs and writing deliverables without the crashes graded runs kept hitting.

Crash census over 5,607 attempts (16 Sep): KeyError on a guessed column or key in 54 units,
FileNotFoundError on a guessed input path in 19, `Object of type bool is not JSON serializable`
in 5, and deliverables rounded on write. Each helper here turns one of those into either the
right behaviour or an error message that names what IS available.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd

#: Where a unit's inputs can live inside the container, most specific first.
INPUT_ROOTS = ("/input/environment/data", "/input", "/app/data", "/app/output", "/tmp/agent-work", ".")


def find_input(name: str, roots=INPUT_ROOTS) -> pathlib.Path:
    """Locate an input file by name (or relative path) under the usual roots.

    Raises FileNotFoundError listing the data files that do exist.
    """
    candidate = pathlib.Path(name)
    if candidate.is_file():
        return candidate
    for root in roots:
        base = pathlib.Path(root)
        if (base / name).is_file():
            return base / name
    for root in roots:
        base = pathlib.Path(root)
        if base.is_dir():
            hits = sorted(p for p in base.rglob(candidate.name) if p.is_file())
            if hits:
                return hits[0]
    seen = []
    for root in roots:
        base = pathlib.Path(root)
        if base.is_dir():
            seen += [str(p) for p in base.rglob("*") if p.is_file()
                     and p.suffix.lower() in (".csv", ".parquet", ".pqt", ".json", ".txt", ".xlsx")][:40]
    raise FileNotFoundError(f"input {name!r} not found; data files present: {seen[:40]}")


def read_table(name, *, parse_dates=None, **kwargs) -> pd.DataFrame:
    """Read csv / parquet / pqt / feather / json (records or columns) into a DataFrame.

    `name` may be a bare filename; it is resolved with find_input. `parse_dates` is a list
    of column names converted with pandas.to_datetime (errors raise, never silently NaT).
    """
    path = find_input(str(name))
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pqt"):
        frame = pd.read_parquet(path, **kwargs)
    elif suffix == ".feather":
        frame = pd.read_feather(path, **kwargs)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload) if isinstance(payload, (list, dict)) else pd.DataFrame([payload])
    else:
        frame = _robust_csv(path, kwargs)
    for column in parse_dates or ():
        frame[col(frame, column)] = pd.to_datetime(frame[col(frame, column)])
    return frame


def _header_row(lines: list[str], sep: str) -> int:
    """Index of the first line that has the delimiter count most of the following lines have -
    a file that opens with a title / notes block starts its table there."""
    counts = [ln.count(sep) for ln in lines[:200]]
    body = [c for c in counts if c > 0]
    if not body:
        return 0
    modal = max(set(body), key=body.count)
    for i, c in enumerate(counts):
        if c == modal and sum(1 for d in counts[i:i + 6] if d == modal) >= min(4, len(counts) - i):
            return i
    return 0


def _robust_csv(path: pathlib.Path, kwargs: dict) -> pd.DataFrame:
    """pandas.read_csv, then the three recoveries the 2 Oct autopsy of 252 runs asked for (text
    inputs that crashed the model's own read_csv: UnicodeDecodeError, `Expected 1 fields in line
    8, saw 2`, a title block above the header). What was done is recorded in
    frame.attrs["read_note"]. An explicit encoding / skiprows / sep / header from the caller is
    respected and nothing is guessed around it."""
    try:
        return pd.read_csv(path, **kwargs)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as first:
        if any(k in kwargs for k in ("encoding", "skiprows", "sep", "delimiter", "header")):
            raise
        if path.stat().st_size == 0:
            raise pd.errors.EmptyDataError(f"{path} is EMPTY (0 bytes) - it was never written; "
                                           "do not read it, recompute from the inputs") from first
        raw = path.read_bytes()
        encoding = "utf-8"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding, text = "latin-1", raw.decode("latin-1")
        lines = text.splitlines()
        sep = max((",", ";", "\t", "|"), key=lambda s: sum(ln.count(s) for ln in lines[:200]))
        skip = _header_row(lines, sep)
        frame = pd.read_csv(path, encoding=encoding, sep=sep, skiprows=skip, **kwargs)
        frame.attrs["read_note"] = f"encoding={encoding!r}, sep={sep!r}, skiprows={skip}"
        return frame


def read_csv(name, **kwargs) -> pd.DataFrame:
    """Alias of `read_table` - the name a pandas user reaches for first."""
    return read_table(name, **kwargs)


def read_json(name):
    """Load a JSON input by name (resolved with find_input)."""
    return json.loads(find_input(str(name)).read_text(encoding="utf-8"))


def _norm(text) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def col(frame: pd.DataFrame, name: str) -> str:
    """The actual column name matching `name` ignoring case, spaces, '_' and '-'.

    Raises KeyError naming the available columns and the closest matches, which is what a
    repair round needs instead of a bare KeyError: 'close'.
    """
    if name in frame.columns:
        return name
    wanted = _norm(name)
    for column in frame.columns:
        if _norm(column) == wanted:
            return column
    close = difflib.get_close_matches(str(name), [str(c) for c in frame.columns], n=3, cutoff=0.5)
    raise KeyError(f"column {name!r} not found; available: {list(frame.columns)[:60]}; closest: {close}")


def key(mapping: dict, name: str):
    """mapping[name] with the same tolerant matching and error message as col()."""
    if name in mapping:
        return mapping[name]
    wanted = _norm(name)
    for k in mapping:
        if _norm(k) == wanted:
            return mapping[k]
    close = difflib.get_close_matches(str(name), [str(k) for k in mapping], n=3, cutoff=0.5)
    raise KeyError(f"key {name!r} not found; available: {list(mapping)[:60]}; closest: {close}")


def to_native(obj):
    """Recursively convert numpy / pandas values into JSON-native Python values.

    numpy bool/int/float -> bool/int/float, arrays -> lists, Timestamp -> ISO string,
    NaN/inf -> None (JSON has no NaN; strict graders reject it).
    """
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        # know72: a date with no clock reads as YYYY-MM-DD (graders compare strings; the 14 Oct census
        # had three units fail on '2020-03-16 00:00:00' against '2020-03-16'). A real time keeps the
        # ISO form; a tz-aware stamp is written in UTC without the offset.
        ts = pd.Timestamp(obj)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def _out_dir() -> pathlib.Path:
    """The run's output directory: T1_OUTPUT_DIR (set by the agent), else T1_OUT_DIR, else /app/output."""
    return pathlib.Path(os.environ.get("T1_OUTPUT_DIR") or os.environ.get("T1_OUT_DIR") or "/app/output")


def _deliverable_path(path) -> pathlib.Path:
    """know94: a BARE file name ('results.csv', no directory) is a deliverable and goes to the output
    directory; before, it was written to the working directory, the deliverable stayed missing and was
    replaced by an empty header-only stub (a credit-portfolio run lost all nine files this way). A path
    with a directory part is used as given."""
    p = pathlib.Path(path)
    if not p.is_absolute() and len(p.parts) == 1:
        return _out_dir() / p
    return p


def write_json(obj, path, *, indent: int = 2) -> pathlib.Path:
    """Write JSON safely (numpy types, bools, NaN handled) and read it back to prove it parses.
    A bare file name is written to the output directory."""
    path = _deliverable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(obj), indent=indent) + "\n", encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))
    return path


def write_csv(frame: pd.DataFrame, path, *, index: bool = False, **kwargs) -> pathlib.Path:
    """Write CSV with full precision (no float_format unless the task states decimals).
    A bare file name is written to the output directory."""
    path = _deliverable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, **kwargs)
    pd.read_csv(path)
    return path


def write_parquet(frame: pd.DataFrame, path, *, index: bool = False) -> pathlib.Path:
    """Write parquet (pyarrow) and read it back. A bare file name is written to the output directory."""
    path = _deliverable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index)
    pd.read_parquet(path)
    return path


def output_path(name: str, out_dir: str | None = None) -> pathlib.Path:
    """Deliverable path under the output directory (/app/output unless given)."""
    base = pathlib.Path(out_dir) if out_dir else _out_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base / name
