"""Phase 0 of the classroom loop: look before writing.

Sixty percent of v5's failures were programs that never ran to the end -
KeyError on a column that does not exist, tz-naive against tz-aware,
NameError on a config key the model assumed. Every one is a guess about
the inputs. This phase asks the model for a READING program only: open
every input at its exact path and print what is actually there. The run's
stdout becomes the EVIDENCE block of every later prompt; the model then
writes against facts it printed itself, which the trails show it trusts
over any summary we hand it.

Cheap by construction: one completion, one short run, one repair at most,
capped at a small share of the budget. Anything that goes wrong here is
itself the first lesson (a wrong path, an unreadable format) and is worth
more than the seconds it cost.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap
import time

from . import code_writer
from .contracts import TaskSpec
from .deadline import Deadline

#: Share of the budget phase 0 may use, and the hard cap on one run.
BUDGET_SHARE = 0.08
RUN_CAP_SEC = 90.0
#: The explorer is not worth starting on tiny budgets: the reading program
#: plus its run would eat what the solution needs.
MIN_BUDGET_SEC = 300.0
STDOUT_CAP = 7000

SYSTEM = textwrap.dedent("""\
    You write a short Python program that ONLY READS inputs and PRINTS what
    is in them. It computes nothing and writes no files. Rules:
      * Open every input at the exact absolute path given.
      * For each table (csv/parquet): print columns with dtypes, shape, the
        index / date column's first and last values and inferred frequency,
        the first 3 rows, describe() of numeric columns (rounded), unique
        counts of any id-like column, and any timezone information.
      * For each JSON: print every key with its value type; print scalar
        values in full; for lists print length and the first element.
      * For text files: print the first 5 lines.
      * Wrap each file in try/except and print the exception if it fails -
        never let one file stop the others.
      * Keep total output under about 150 lines: truncate long prints.
      * Import only numpy, pandas, pyarrow, json, pathlib. No fences, no
        commentary: Python source only.
    """)


def _user(spec: TaskSpec) -> str:
    files = "\n".join(f"  - {spec.input_root}/{f}" for f in spec.data_files)
    return (f"INPUT FILES (read with EXACTLY these paths):\n{files}\n\n"
            f"TASK EXCERPT (for orientation only; do not solve it):\n"
            f"{(spec.instruction or '')[:1500]}\n\n"
            "Write the reading program now.")


#: A fixed reading program: the same facts the model-written one is asked for (columns + dtypes,
#: shape, first/last dates, head, describe, JSON keys and scalars, text heads), with no model
#: request and no guessed names. The 25 Sep autopsy counted 53 KeyErrors on column / key names
#: across six sweeps; this puts the real names in front of the model before it writes any code.
PROFILE_PROGRAM = r'''
import json, sys, pathlib
import pandas as pd
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
LIMIT = 160
def table(df, name):
    print(f"== {name}: shape {df.shape}")
    print("columns + dtypes:", {c: str(t) for c, t in df.dtypes.items()})
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower() or str(df[c].dtype).startswith("datetime"):
            s = df[c].dropna()
            if len(s):
                print(f"   {c}: first {s.iloc[0]!r} last {s.iloc[-1]!r}")
                # 1 Oct: an unsorted file silently shifted every return; say it before any code.
                try:
                    t = pd.to_datetime(s, errors="coerce")
                    keys = [k for k in df.columns if k != c and df[k].dtype == object
                            and 1 < df[k].nunique(dropna=True) <= 25]
                    if keys:
                        bad = [g for g, part in df.assign(_t=pd.to_datetime(df[c], errors="coerce"))
                               .groupby(keys[0])["_t"] if not part.dropna().is_monotonic_increasing]
                        if bad:
                            print(f"   WARNING {c}: NOT sorted ascending within {keys[0]} {bad[:5]}"
                                  " - sort by date before diff/returns/rolling")
                        dup = int(df.duplicated([keys[0], c]).sum())
                    else:
                        if not t.dropna().is_monotonic_increasing:
                            print(f"   WARNING {c}: NOT sorted ascending - sort by date before"
                                  " diff/returns/rolling")
                        dup = int(t.duplicated().sum())
                    if dup:
                        print(f"   WARNING {c}: {dup} duplicated date rows")
                except Exception:
                    pass
    print(df.head(3).to_string()[:1500])
    num = df.select_dtypes("number")
    if num.shape[1]:
        print(num.describe().round(6).to_string()[:1500])
    for c in df.columns[:30]:
        if df[c].dtype == object and df[c].nunique(dropna=True) <= 25:
            print(f"   {c} values: {sorted(map(str, df[c].dropna().unique()))[:25]}")
def walk_json(obj, prefix="", depth=0):
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:60]:
            if isinstance(v, (dict, list)):
                print(f"   {prefix}{k}: {type(v).__name__}({len(v)})")
                walk_json(v, prefix + str(k) + ".", depth + 1)
            else:
                print(f"   {prefix}{k}: {type(v).__name__} = {str(v)[:120]}")
    elif isinstance(obj, list) and obj:
        print(f"   {prefix}[0] of {len(obj)}: {str(obj[0])[:200]}")
        if isinstance(obj[0], (dict, list)):
            walk_json(obj[0], prefix + "[0].", depth + 1)
for p in map(pathlib.Path, sys.argv[1:]):
    try:
        suf = p.suffix.lower()
        if suf in (".csv", ".tsv", ".txt"):
            # the same tolerant reader the solution is told to use (encoding, title block, delimiter)
            try:
                from qf_toolkit.io import _robust_csv
                df = _robust_csv(p, {"sep": "\t"} if suf == ".tsv" else {})
            except ImportError:
                df = pd.read_csv(p, sep="\t" if suf == ".tsv" else ",")
            table(df, p.name)
            if df.attrs.get("read_note"):
                print(f"   NOTE {p.name}: plain pd.read_csv FAILS on this file; qf_toolkit.read_table"
                      f"('{p.name}') reads it ({df.attrs['read_note']})")
        elif suf == ".parquet":
            table(pd.read_parquet(p), p.name)
        elif suf in (".xlsx", ".xls"):
            for sheet, df in pd.read_excel(p, sheet_name=None).items():
                table(df, f"{p.name}[{sheet}]")
        elif suf == ".json":
            obj = json.loads(p.read_text())
            print(f"== {p.name}: {type(obj).__name__}")
            walk_json(obj)
        else:
            lines = p.read_text(errors="replace").splitlines()
            print(f"== {p.name}: {len(lines)} lines")
            for l in lines[:5]:
                print("   " + l[:200])
    except Exception as e:
        print(f"== {p.name}: could not read ({type(e).__name__}: {e})")
'''


def _profile(spec: TaskSpec, work_dir: pathlib.Path, cap: float) -> tuple[bool, str]:
    """Run PROFILE_PROGRAM over every input file (no model request)."""
    paths = [f"{spec.input_root}/{f}" for f in spec.data_files]
    script = work_dir / "profile_inputs.py"
    script.write_text(PROFILE_PROGRAM, encoding="utf-8")
    from .test_runner import generated_env
    try:
        proc = subprocess.run([sys.executable, str(script), *paths], cwd=str(work_dir),
                              capture_output=True, text=True, timeout=cap,
                              env=generated_env())
    except subprocess.TimeoutExpired:
        return False, ""
    out = (proc.stdout or "")[:STDOUT_CAP]
    return proc.returncode == 0 and len(out) > 40, out


def _run(source: str, work_dir: pathlib.Path, cap: float) -> tuple[bool, str]:
    script = work_dir / "explore.py"
    script.write_text(source, encoding="utf-8")
    from .test_runner import generated_env
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=str(work_dir),
                              capture_output=True, text=True, timeout=cap,
                              env=generated_env())
    except subprocess.TimeoutExpired:
        return False, f"reading program exceeded {cap:.0f}s"
    out = (proc.stdout or "")[:STDOUT_CAP]
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1200:]
        return False, out + "\n--- reading program crashed ---\n" + tail
    return True, out


def run(spec: TaskSpec, work_dir: pathlib.Path, deadline: Deadline,
        telemetry: list | None = None) -> tuple[str, str]:
    """Returns (evidence, note). Evidence may be partial; never raises."""
    if not spec.data_files:
        return "", "explore skipped: no input files"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    allowance = deadline.budget * BUDGET_SHARE
    cap = min(RUN_CAP_SEC, max(20.0, allowance / 2))
    # The fixed profiler first: no request spent, nothing guessed. The model-written reader
    # below is only the fallback for inputs the profiler cannot read.
    try:
        ok, out = _profile(spec, work_dir, cap)
    except Exception:
        ok, out = False, ""
    if ok:
        spent = time.monotonic() - started
        deadline.record_attempt(spent)
        return out.strip(), f"explore: fixed profiler in {spent:.0f}s, {len(out)} chars of evidence"
    if not code_writer.model_available():
        return "", "explore skipped: no model"
    if deadline.budget < MIN_BUDGET_SEC:
        return "", "explore skipped: budget too small"

    source = code_writer.complete(SYSTEM, _user(spec), telemetry=telemetry,
                                  variant=7, want_code=True)
    if not source:
        return "", "explore skipped: no reading program"
    ok, out = _run(source, work_dir, cap)
    if not ok and time.monotonic() - started < allowance:
        # One repair: the crash text is the most specific evidence there is
        # about the inputs (wrong path, encoding, format).
        fix = code_writer.complete(
            SYSTEM, _user(spec) + "\n\nYOUR PREVIOUS READING PROGRAM FAILED "
            "WITH:\n" + out[-1500:] + "\n\nReturn the complete corrected "
            "reading program.", telemetry=telemetry, variant=8, want_code=True)
        if fix:
            ok2, out2 = _run(fix, work_dir, cap)
            if ok2 or len(out2) > len(out):
                ok, out = ok2, out2
    spent = time.monotonic() - started
    deadline.record_attempt(spent)
    note = (f"explore: {'ok' if ok else 'partial'} in {spent:.0f}s, "
            f"{len(out)} chars of evidence")
    return out.strip(), note
