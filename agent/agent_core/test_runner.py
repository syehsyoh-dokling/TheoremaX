"""Run a generated solution and classify what happened.

Classification is the product here, not the stdout. The repair loop can only act
on a FailureKind, and the local matrix is only useful if the kinds are distinct.
Reading a traceback into the right bucket is what makes the difference between
"try again" and "stop, this cannot be fixed from here".
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

from .contracts import FailureKind

#: A module the image does not carry cannot be installed at runtime - the eval
#: network reaches the model endpoint and nothing else. Retrying is pure waste.
_IMPORT = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
_TRACEBACK_LAST = re.compile(r"^([A-Za-z_.]*Error|[A-Za-z_.]*Exception): (.*)$", re.M)
_STATE_BLOCK = re.compile(r"=== RUNTIME STATE AT FAILURE.*?=== END STATE ===", re.S)


#: Compatibility layer loaded as `sitecustomize` into every generated program (genrt/).
GENRT = pathlib.Path(__file__).resolve().parent / "genrt"
_DID_YOU_MEAN = re.compile(
    r"NameError: name '([A-Za-z_]\w*)' is not defined\. Did you mean: '([A-Za-z_]\w*)'\?")
#: Deterministic typo repairs per run: each is a rerun, never a model request.
MAX_NAME_FIXES = 3
#: Asserts as recorded checks (genrt/soft_run.py); T1_SOFT_ASSERTS=off restores the plain run.
SOFT_ASSERTS = os.environ.get("T1_SOFT_ASSERTS", "on").strip().lower() != "off"
_SOFT_LINE = re.compile(r"^SOFT_ASSERT_FAILED line \d+: .*$", re.M)
_SOFT_SEEN: set[str] = set()


def _soft_key(line: str) -> str:
    """The check's identity across rewrites: its message with line number and digits removed."""
    return re.sub(r"[\d.]+", "#", line.split(":", 1)[-1]).strip()[:120]


def generated_env(extra: list[str] | None = None) -> dict:
    """Environment for running model-written code: the compat layer first, then the agent root
    (generated programs `import qf_toolkit`; the image sets no PYTHONPATH)."""
    env = os.environ.copy()
    parts = [str(GENRT), str(pathlib.Path(__file__).resolve().parents[1]), *(extra or [])]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # 27 Sep: under the read-only root matplotlib warns about /nonexistent/.config on every run, and
    # those lines filled the traceback tail the repair prompt shows. A writable cache dir silences it.
    env.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
    return env


def fix_name_typo(script: pathlib.Path, output: str) -> str | None:
    """Apply Python's own "Did you mean" suggestion for an undefined name, in place.

    The autopsy found crashes like `curran_vs_mc_mc_error` for `curran_vs_mc_max_error`; the
    interpreter already names the fix, so it costs a rerun instead of a repair request. Returns
    a note, or None when there is nothing safe to do."""
    m = _DID_YOU_MEAN.search(output)
    if not m:
        return None
    wrong, right = m.group(1), m.group(2)
    src = script.read_text(encoding="utf-8")
    new = re.sub(rf"(?<![\w.]){re.escape(wrong)}(?!\w)", right, src)
    if new == src:
        return None
    script.write_text(new, encoding="utf-8")
    return f"auto-fixed NameError {wrong!r} -> {right!r}"


def run_solution(script: pathlib.Path, cwd: pathlib.Path,
                 timeout_sec: float,
                 input_dir: pathlib.Path | str | None = None) -> tuple[FailureKind, str, float]:
    started = time.monotonic()
    extra = [str(cwd)] + ([str(input_dir)] if input_dir else []) + ["/input"]
    env = generated_env(extra)
    notes: list[str] = []
    for _ in range(MAX_NAME_FIXES + 1):
        kind, detail, _sec = _run_once(script, cwd, env,
                                       max(1.0, timeout_sec - (time.monotonic() - started)))
        if kind is not FailureKind.RUNTIME_ERROR:
            break
        note = fix_name_typo(script, detail)
        if note is None:
            break
        notes.append(note)
    if notes:     # appended, never prefixed: callers read the head of `detail`
        detail = f"{detail}\n[{'; '.join(notes)}]"
    return kind, detail, time.monotonic() - started


def _run_once(script: pathlib.Path, cwd: pathlib.Path, env: dict,
              timeout_sec: float) -> tuple[FailureKind, str, float]:
    started = time.monotonic()
    # 2 Oct: the program runs under genrt/soft_run.py, which turns its asserts into recorded
    # checks (it finishes and writes its files; the failed checks come back on stdout).
    command = [sys.executable, str(script)]
    if SOFT_ASSERTS:
        command = [sys.executable, str(GENRT / "soft_run.py"), str(script)]
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return FailureKind.TIMEOUT, f"exceeded {timeout_sec:.0f}s", time.monotonic() - started

    seconds = time.monotonic() - started
    if proc.returncode == 0:
        failed = _SOFT_LINE.findall(proc.stdout or "")
        # Each distinct failed check is reported ONCE: the program gets one repair round for it.
        # If the same check still fails afterwards the files stand as written - a check the model
        # cannot satisfy is as likely to be wrong as the number, and the output contract and the
        # invariants still judge the files.
        fresh = [f for f in failed if _soft_key(f) not in _SOFT_SEEN]
        _SOFT_SEEN.update(_soft_key(f) for f in failed)
        if fresh:
            detail = ("the program finished and wrote its files, but these checks of its own "
                      "FAILED:\n" + "\n".join(fresh[:8]) + "\nFix the computation each one "
                      "tests (or the check, if the task does not state that property). "
                      "--- output tail ---\n" + (proc.stdout or "")[-600:])
            return FailureKind.INVARIANT_VIOLATION, detail, seconds
        return FailureKind.PASS, proc.stdout[-2000:], seconds

    combined = (proc.stderr or "") + (proc.stdout or "")
    missing = _IMPORT.search(combined)
    if missing:
        return (FailureKind.IMPORT_ERROR,
                f"missing module {missing.group(1)!r} - not installable at runtime",
                seconds)

    # The one-line summary routes the classification; the tail is what the
    # repair prompt actually needs - a bare "KeyError: 'x'" without the
    # frames around it sent the model hunting in the wrong function.
    # the runtime-state block (genrt/sitecustomize.py) is carried whole, outside the 1000-char tail,
    # so program output printed after it can never push it out of the repair prompt
    state = _STATE_BLOCK.search(combined)
    if state:
        combined = combined.replace(state.group(0), "")
    last = _TRACEBACK_LAST.findall(combined)
    summary = f"{last[-1][0]}: {last[-1][1]}" if last else "non-zero exit"
    detail = f"{summary}\n--- traceback tail ---\n{combined[-1000:]}"
    if state:
        detail += "\n" + state.group(0)[:1400]
    return FailureKind.RUNTIME_ERROR, detail, seconds


# `run_unit_checks` - the dev-only runner for the grader's own checks - used
# to live here and therefore shipped inside the submission image: dead code
# that executes `checks/test.sh`, which is the first thing an auditor of the
# "never depend on checks/ at runtime" rule goes looking for. It now lives in
# `dev_tools/checks_dev.py`, which is not COPY'd into the image.
