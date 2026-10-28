"""Cell-by-cell solving with a cached runtime state (6 Oct).

Why: the 2 Oct autopsy of 252 runs found the model's programs too long for the 4,000-token reply
(median 2,434 tokens, a quarter above 3,700; 19-35% of replies cut), and crashes of mismatched
array shapes the model could not see. Here the program is written as short cells. Each complete
cell runs at once in a persistent kernel (genrt/kernel.py); its variables stay cached there, and
the kernel's description of them (type, shape, columns, head) goes back to the model with the next
request. A cut reply costs nothing: the cells before the cut have run, and the model continues
from the next cell. A cell that crashes is rewritten alone, with the traceback and the state.

Every request is still one admitted House request; the loop stops at the ledger and the deadline.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import time

from . import code_writer, test_runner
from .contracts import Attempt, FailureKind, TaskSpec
from .deadline import Deadline

MARK = re.compile(r"^# ?%%.*$", re.M)
MAX_CELL_LINES = 60
MAX_ROUNDS = 22
MIN_REQUESTS = 2
#: know72: 100 s, not 240. Four units of the 14 Oct census burnt 500+ s in two timed-out simulation
#: cells each and reached the deadline with nothing written; a cell that needs more than 100 s is a
#: cell that must be re-sized (paths / steps) or vectorised, and the model is told so.
CELL_TIMEOUT_CAP = 100.0
#: know90: the one longer run a cell gets after its first timeout (see run_cells)
CELL_TIMEOUT_EXTENDED = 260.0
SAME_CELL_FAILS = 3

CELL_RULES = """\
THE PROGRAM IS WRITTEN ONE CELL PER REPLY. A cell starts with a marker line `# %% <short title>`
and is at most ~40 lines. Each cell runs at once in ONE Python session that keeps every variable
(numpy as np, pandas as pd, math, json and every qf_toolkit helper are already imported); you then
see the variables it created (type, shape, columns, head) and write the next cell against them.
Order: 1) inputs from INPUTS and the alignment of their dates / ids; then one cell per computed
quantity, in the plan's order; emit() each deliverable in the cell where its values exist. When
every deliverable has been emitted, reply with exactly `# %% done`. Reply with ONE cell only: no
prose, no code fence, nothing after the cell."""
SCAFFOLD_RULES = """\
THE PROGRAM IS WRITTEN ONE CELL PER REPLY, AND THE CELLS ARE ALREADY DECIDED FOR YOU. Each request
names the cell to write (`# %% <title>` - use that exact title), what it must compute and emit(),
the required columns / keys, the task sentences and plan lines about it. Write ONLY that cell, at
most ~40 lines, against the RUNTIME STATE (variables that exist now - real names, shapes, columns;
numpy as np, pandas as pd, math, json and every qf_toolkit helper are already imported). emit() the
deliverable inside the cell. Reply with ONE cell only: no prose, no code fence, nothing after it."""
DONE = re.compile(r"^# ?%% *done\s*$", re.M | re.I)


def split_cells(text: str) -> list[tuple[str, str]]:
    """[(title, code)] from a reply; without markers the whole reply is one cell."""
    text = code_writer._extract_code(text) or text
    marks = list(MARK.finditer(text))
    if not marks:
        return [("program", text.strip())] if text.strip() else []
    cells = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        title = m.group().lstrip("# %").strip() or f"cell {i + 1}"
        code = text[m.end():end].strip("\n")
        if code.strip():
            cells.append((title[:60], code))
    return cells


def _parses(code: str) -> bool:
    try:
        compile(code, "<cell>", "exec")
        return True
    except SyntaxError:
        return False


def _parsing_prefix(code: str) -> str | None:
    lines = code.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln[:1].strip() and not ln.startswith((")", "]", "}"))]
    for i in reversed(starts):
        head = "".join(lines[:i])
        if head.strip() and _parses(head):
            return head
    return None


class Kernel:
    def __init__(self, work_dir: pathlib.Path, input_dir):
        self.work_dir = work_dir
        self.env = test_runner.generated_env([str(work_dir)] + ([str(input_dir)] if input_dir else []) + ["/input"])
        self.proc = None
        self.start()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(test_runner.GENRT / "kernel.py")], cwd=str(self.work_dir), env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def run(self, code: str, timeout: float) -> dict:
        """{"ok", "stdout", "error", "vars", "soft", "seconds"}; a timeout kills the kernel."""
        started = time.monotonic()
        try:
            self.proc.stdin.write(json.dumps({"cmd": "run", "code": code}) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self.start()
            return {"ok": False, "error": "kernel restarted; cell not run", "vars": [], "soft": [],
                    "stdout": "", "seconds": 0.0}
        import threading
        box: list = []
        reader = threading.Thread(target=lambda: box.append(self.proc.stdout.readline()), daemon=True)
        reader.start()
        reader.join(timeout)
        if reader.is_alive() or not box or not box[0]:
            self.close()
            return {"ok": False, "error": f"TimeoutError: the cell ran longer than {timeout:.0f}s. Rewrite it to finish in under {timeout/3:.0f}s: "
                                          "cut paths/steps/grid by 10x (the tolerance, not the maximum, sizes a simulation), vectorise over paths with numpy "
                                          "(no Python loop per path or per day), reuse variables already in the kernel; a smaller correct answer beats none "
                    "(vectorise; size the simulation to the tolerance)", "vars": [], "soft": [],
                    "stdout": "", "seconds": time.monotonic() - started, "timeout": True}
        try:
            out = json.loads(box[0])
        except ValueError:
            out = {"ok": False, "error": "kernel reply unreadable", "vars": [], "soft": [], "stdout": ""}
        out["seconds"] = time.monotonic() - started
        return out

    def close(self) -> None:
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except OSError:
            pass


def _task_lines(instruction: str, names: list[str], limit: int = 2200) -> str:
    """The task's own sentences that mention any of `names` (a deliverable, its columns)."""
    text = instruction or ""
    pieces = re.split(r"(?<=[.\n])\s+", text)
    keep, used = [], 0
    for p in pieces:
        low = p.lower()
        if any(n and n.lower() in low for n in names) and p.strip():
            s = p.strip()[:400]
            if used + len(s) > limit:
                break
            keep.append(s)
            used += len(s)
    return "\n".join(keep)


def _plan_lines(plan: str, names: list[str], limit: int = 1600) -> str:
    """Plan JSON lines (quantities, method steps) that mention any of `names`."""
    keep, used = [], 0
    for line in (plan or "").splitlines():
        low = line.lower()
        if any(n and n.lower() in low for n in names):
            s = line.strip()[:300]
            if used + len(s) > limit:
                break
            keep.append(s)
            used += len(s)
    return "\n".join(keep)


def _short_base(spec: TaskSpec, output_dir) -> str:
    """8 Oct (lapis tengah): the compact context for every cell request - the task text, the inputs
    with their profiles and CFG keys, the contract, the task-specific toolkit lines and the closest
    frame. No theory cards, no playbook, no worked examples: Python already used those to write each
    cell's brief (scaffold_steps), so the model reads ~4-6k tokens instead of ~25k."""
    from . import intake, toolkit_hints, frame_bank
    parts = [f"TASK\n{(spec.instruction or '').strip()[:9000]}",
             "\nINPUTS - already loaded as INPUTS[<file name>] (DataFrame for tables, dict/list for json, "
             "text otherwise); dates parsed, numeric strings converted, delimiter detected:\n"
             + "\n".join(f"  - INPUTS[{pathlib.Path(f).name!r}]" for f in spec.data_files)]
    try:
        cfg, _notes = intake.load_configs(spec.input_root, spec.data_files)
        if cfg:
            parts.append("\nCFG - every JSON config flattened (use these keys, never guess):\n" + intake.cfg_listing(cfg))
    except Exception:  # noqa: BLE001
        pass
    parts.append(f"\nDELIVERABLES - emit(<file name>, <object>) writes under {output_dir} and reads it back; "
                 "columns are aligned to the contract's exact spelling:\n"
                 + "\n".join(code_writer._contract_block(c) for c in spec.contracts))
    profile = code_writer._data_profile(spec)
    if profile:
        parts.append("\nINPUT DATA PROFILES:\n" + profile[:3500])
    if spec.evidence:
        parts.append("\nEVIDENCE (printed from the real inputs):\n" + spec.evidence[:2500])
    hint = toolkit_hints.block(spec.instruction or "", code_writer.qf_toolkit_api()) or ""
    if hint:
        parts.append("\n" + hint[:3000])
    frames = frame_bank.block(spec.instruction or "", "", k=1)
    if frames:
        parts.append("\n" + frames[:1800])
    try:   # know81: the conventions reference table, related to this task by its own words
        from qf_toolkit.conventions_ref import conventions_block
        conv = conventions_block(spec.instruction or "", max_items=6, max_chars=1200)
        if conv:
            parts.append("\n" + conv)
    except Exception:  # noqa: BLE001
        pass
    if getattr(spec, "plan", ""):
        parts.append("\nAPPROVED PLAN (quantities, sources, method order):\n" + spec.plan[:3500])
    return "\n".join(parts)


def _theory_lines(spec: TaskSpec, names: list[str], limit: int = 1400) -> str:
    """9 Oct: the formula / convention lines of the family theory cards that mention this
    deliverable's name, stem or columns - the part of the theory the short base dropped."""
    try:
        from . import theory
        text = theory.select_library(spec.normalised_category, spec.instruction or "")
    except Exception:  # noqa: BLE001
        return ""
    keys = [n.lower() for n in names if n and len(n) >= 4]
    # the quantity words in the column names too (var_dollar -> var, dollar)
    for n in list(names):
        for part in re.split(r"[_\-\s]+", str(n)):
            if len(part) >= 4:
                keys.append(part.lower())
    keys = list(dict.fromkeys(keys))[:20]
    out, used = [], 0
    for line in text.splitlines():
        low = line.lower()
        if line.startswith("[family") or not line.strip():
            continue
        if any(k in low for k in keys) and ("=" in line or any(w in low for w in ("formula", "default", "use ", "convention"))):
            s = line.strip()[:260]
            if used + len(s) > limit:
                break
            out.append(s)
            used += len(s)
    return "\n".join(out)


def scaffold_steps(spec: TaskSpec) -> list[dict]:
    """The cells Python decides (6 Oct, owner's design): cell 0 loads and aligns the inputs; then
    ONE cell per deliverable, in the task's order, each carrying the file name, its format and
    required columns / keys, the task sentences and plan lines that speak about it."""
    from . import frame_bank
    best = frame_bank.select(spec.instruction or "", k=1)
    frame_line = ""
    if best:
        f = best[0]
        frame_line = ("\nMATCHED FRAME: " + str(f.get("title", "")) + " - "
                      + " | ".join(str(x) for x in f.get("formulas", [])[:6]))[:700]
        if f.get("proven"):
            # 13 Oct: a proven method brings its ordered steps and the conventions that made it pass
            steps = [str(x) for x in f.get("steps", []) if str(x).strip()]
            conv = [str(x) for x in f.get("conventions", []) if str(x).strip()]
            frame_line += ("\nPROVEN STEPS: " + " ".join(f"{j}) {s}" for j, s in enumerate(steps[:12], 1))
                           + ("\nCONVENTIONS: " + " | ".join(conv[:6]) if conv else ""))[:1500]
        elif f.get("curated"):
            # know78: a curated frame's FIRST sentence carries its decisive instruction ("the whole task is one
            # call to X; emit its files") - per-file cells saw only title + formulas and rebuilt files by hand
            head = str(f.get("frame", "")).strip()
            cut = head.find(". ")
            frame_line += "\nFRAME LEAD: " + (head[:cut + 1] if 0 < cut < 300 else head[:300])
    from . import intake
    graph = intake.task_graph(spec.instruction or "")
    # know71: MC settings the task states, enforced in every cell (MC-noise misses were 3 of 8 regressions)
    try:
        _cfg, _ = intake.load_configs(spec.input_root, spec.data_files)
    except Exception:  # noqa: BLE001
        _cfg = {}
    mc_line = intake.mc_block(spec.instruction or "", _cfg)
    mc_line = ("\n" + mc_line) if mc_line else ""
    # know71: toolkit calls the matched proven method relies on - call them, do not re-derive them
    must_call = ""
    if best and best[0].get("proven"):
        import re as _re
        from . import code_writer as _cw
        api = _cw.qf_toolkit_api()
        names = sorted({n for n in _re.findall(r"\b([a-z_][a-z0-9_]{3,})\(", " ".join(
            str(x) for x in best[0].get("steps", []) + best[0].get("formulas", []))) if f"{n}(" in api})
        if names:
            must_call = ("\nTOOLKIT CALLS THAT SOLVED THIS KIND OF TASK (call these from qf_toolkit instead of "
                         "re-deriving them): " + ", ".join(names[:10]))
    steps = [{"title": "load_inputs", "file": None,
              "brief": "Load every input from INPUTS (tables are already parsed: dates as datetime64, numbers "
                       "numeric), read parameters from CFG (flattened JSON configs), sort each table by ITS OWN "
                       "date column (tables may name it differently - use the INPUT EVIDENCE / RUNTIME STATE "
                       "columns), merge a table without dates on its ids, and keep them in clearly named "
                       "variables. If the FRAME LEAD's call reads the inputs itself, call it here instead of "
                       "loading by hand. Emit nothing yet."}]
    for c in spec.contracts:
        names = [c.filename, pathlib.Path(c.filename).stem] + [k for k in c.required_columns if len(k) >= 3][:12]
        cols = ", ".join(c.required_columns) if c.required_columns else "(as the task describes)"
        rules = intake.rules_for(graph, names)
        theory_lines = _theory_lines(spec, names)
        steps.append({"title": pathlib.Path(c.filename).stem[:40], "file": c.filename,
                      "brief": f"Compute and emit('{c.filename}', ...) - format {c.fmt}; required "
                               f"columns/keys: {cols}. Keep the cell under 40 lines: if the computation is "
                               "longer, emit this file from the variables you already have and leave the rest "
                               "to the next cell.\nTASK SENTENCES ABOUT IT:\n"
                               + (_task_lines(spec.instruction, names) or "(none found - follow the task)")
                               + ("\nRULES (must / only / exactly) THAT APPLY:\n- " + "\n- ".join(rules) if rules else "")
                               + "\nPLAN LINES ABOUT IT:\n" + (_plan_lines(spec.plan, names) or "(none)")
                               + ("\nTHEORY LINES ABOUT IT:\n" + theory_lines if theory_lines else "")
                               + frame_line + must_call + mc_line})
    return steps


def _state_text(state: dict[str, str]) -> str:
    if not state:
        return "(no variables yet)"
    return "\n".join(f"  {v}"[:260] for v in list(state.values())[-28:])


def _settings(endpoint: str) -> dict:
    s = {"model": os.environ.get("MODEL_NAME"), "temperature": 0.0, "seed": 0,
         "max_tokens": code_writer.MAX_COMPLETION_TOKENS}
    s.update(code_writer._thinking_settings(endpoint))
    return s


def _scaffold_on() -> bool:
    return os.environ.get("T1_SCAFFOLD", "on").strip().lower() != "off"


def _step_demand(step: dict, n: int, total: int) -> str:
    return (f"WRITE CELL {n}/{total} NOW - marker line exactly `# %% {step['title']}`.\n"
            + step["brief"] + "\nOne cell only.")


def solve(spec: TaskSpec, output_dir: pathlib.Path, work_dir: pathlib.Path, deadline: Deadline,
          telemetry: list | None, self_check) -> tuple[list[Attempt], str | None]:
    """Returns (attempts, the accepted program as one source)."""
    endpoint = os.environ.get("MODEL_ENDPOINT", "")
    if not endpoint or not os.environ.get("MODEL_NAME"):
        return [], None
    base = (_short_base(spec, output_dir) if os.environ.get("T1_SHORT_BASE", "on").strip().lower() != "off"
            else code_writer._prompt(spec, spec.contracts, str(output_dir), None))
    scaffold = _scaffold_on()
    steps = scaffold_steps(spec) if scaffold else []
    if scaffold:
        # know81: a whole-task-call frame is a two-step program (call, emit); when the time left cannot pay
        # for one model request per file, the scaffold takes that shape - the model still writes both cells
        try:
            from qf_toolkit import helpers81_4 as _h81
            steps, _collapsed = _h81.maybe_collapse(steps, deadline.remaining, deadline.elapsed,
                                                    int(code_writer._ledger.get("requests", 0)))
            if _collapsed and telemetry is not None:
                telemetry.append({"event": "scaffold_collapsed", "cells": len(steps),
                                  "remaining_sec": round(deadline.remaining, 1)})
        except Exception:  # noqa: BLE001 - an aid; the full scaffold stays when it fails
            pass
    si = 0                      # index of the scaffold step being written
    auditing = False            # every scaffold step done: now fixing what the audit reports
    system = code_writer.SYSTEM + "\n" + (SCAFFOLD_RULES if scaffold else CELL_RULES)
    kernel = Kernel(work_dir, spec.input_root)
    accepted: list[tuple[str, str]] = []
    state: dict[str, str] = {}
    attempts: list[Attempt] = []
    demand = (_step_demand(steps[0], 1, len(steps)) if scaffold
              else "Write CELL 1 now (inputs and their alignment). One cell only.")
    fails: dict[str, int] = {}
    timeouts: dict[str, int] = {}
    emit_flags: dict = {}
    audits_seen: set[str] = set()
    source = None
    # know76: keep the BEST audited answer. A late fix cell that makes the deliverables worse (the 14 Oct
    # full runs had units fall from all-tests-pass to near zero in the same version) must not replace a
    # better set already written: every audit snapshots the tree when it has fewer problems than any
    # earlier audit, and the end restores that snapshot if the final tree audits worse.
    best = {"n": None, "tree": None}

    def _problems(kind, detail) -> int:
        return 0 if kind is FailureKind.PASS else max(1, len([p for p in str(detail).split(";") if p.strip()]))

    def _remember(kind, detail) -> None:
        try:
            from .repair_loop import _snapshot_tree
            n = _problems(kind, detail)
            if best["n"] is None or n < best["n"]:
                best["n"], best["tree"] = n, _snapshot_tree(output_dir)
        except Exception:  # noqa: BLE001 - bookkeeping must never break a run
            pass
    try:
        for round_ in range(MAX_ROUNDS):
            if code_writer.requests_remaining() < MIN_REQUESTS or not deadline.can_afford():
                attempts.append(Attempt(len(attempts), FailureKind.TIMEOUT if deadline.remaining < 60
                                        else FailureKind.MODEL_UNAVAILABLE,
                                        f"[cells] stopped: {code_writer.requests_remaining()} requests, "
                                        f"{deadline.remaining:.0f}s left", 0.0))
                break
            kept = "\n\n".join(f"# %% {t}\n{c}" for t, c in accepted)
            user = (base + "\n\nCELLS ALREADY RUN - kept in the session, do NOT repeat them:\n"
                    + (kept[-9000:] if kept else "(none yet)")
                    + "\n\nRUNTIME STATE (variables now defined):\n" + _state_text(state)
                    + "\n\n" + demand)
            started = time.monotonic()
            content, finish = code_writer._call(
                endpoint, _settings(endpoint),
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                float(os.environ.get("MODEL_TIMEOUT_SEC", 180.0)), telemetry)
            write_seconds = time.monotonic() - started
            if content is None:
                if round_ == 0:
                    return [], None        # the endpoint answered nothing: the other paths stay open
                attempts.append(Attempt(len(attempts), FailureKind.MODEL_UNAVAILABLE, "[cells] no reply", write_seconds))
                break
            if DONE.search(content) and not split_cells(DONE.sub("", content)):
                # know91: a deliverable whose LATEST emit still carries a guard verdict (object text in a cell,
                # non-positive standard errors, a hand-recomputed part of a toolkit bundle ...) is not done; the
                # verdict was only shown in the state before, and `done` wrote it out as it was
                flagged = {k: v for k, v in emit_flags.items() if v}
                fkey = "emit-guards:" + ";".join(sorted(flagged))
                if flagged and fkey not in audits_seen:
                    audits_seen.add(fkey)
                    lines = "\n".join(f"- {m.split(': ', 1)[-1][:400]}" for v in flagged.values() for m in v[:1])
                    attempts.append(Attempt(len(attempts), FailureKind.RUNTIME_ERROR,
                                            f"[cells] done refused: guard verdicts on {sorted(flagged)}"[:1500],
                                            write_seconds))
                    demand = ("You said done, but these written deliverables failed a check when they were "
                              "emitted:\n" + lines + "\nWrite ONE cell that fixes this and emit() the deliverable "
                              "again (a later emit overwrites), or `# %% done` if the check is wrong for this task.")
                    continue
                kind, detail = self_check(output_dir, spec.contracts, spec)
                _remember(kind, detail)
                attempts.append(Attempt(len(attempts), kind, f"[cells] done; audit: {detail}"[:1500], write_seconds))
                key = detail[:160]
                if kind is FailureKind.PASS or key in audits_seen:
                    break
                audits_seen.add(key)
                demand = ("You said done, but the audit of the written deliverables says:\n" + detail[:1500]
                          + "\nWrite ONE cell that fixes this (recompute what is wrong and emit() the "
                          "deliverable again; a later emit overwrites), or `# %% done` if nothing can be done.")
                continue
            cells = split_cells(DONE.sub("", content))[:1]      # one cell per reply
            cut = finish == "length"
            if cut and cells and not _parses(cells[-1][1]):
                prefix = _parsing_prefix(cells[-1][1])
                cells[-1] = (cells[-1][0], prefix) if prefix else (cells[-1][0], "")
                cells = [c for c in cells if c[1].strip()]
            if telemetry is not None:
                telemetry.append({"event": "cells", "round": round_, "cells": len(cells), "cut": cut})
            error_cell = None
            run_seconds = 0.0
            for title, code in cells:
                per = max(20.0, min(CELL_TIMEOUT_CAP, deadline.remaining - 30.0))
                # know90: a cell that already timed out ONCE and still has a large budget behind it gets one
                # longer run (a rolling re-fit over ~1000 windows is honest work of 1-3 minutes on one CPU,
                # and three 100-s timeouts in a row spent the whole budget with nothing written)
                if timeouts.get(title, 0) == 1 and deadline.remaining > 420.0:
                    per = max(per, min(CELL_TIMEOUT_EXTENDED, deadline.remaining - 240.0))
                res = kernel.run(code, per)
                if isinstance(res.get("emit_flags"), dict):
                    emit_flags = res["emit_flags"]        # know91: verdicts of the latest emit per deliverable
                run_seconds += res.get("seconds", 0.0)
                for v in res.get("vars", []):
                    state[v.split(":", 1)[0]] = v
                if res.get("notes"):
                    state["__runtime__"] = "RUNTIME NOTES: " + "; ".join(res["notes"])[:400]
                if res.get("ok"):
                    accepted.append((title, code))
                    if res.get("soft"):
                        state["__checks__"] = "FAILED CHECKS: " + "; ".join(res["soft"])[:400]
                    continue
                error_cell = (title, code, res)
                if res.get("timeout"):
                    kernel.start()
                    for _t, c in accepted:            # restore the cached state
                        kernel.run(c, CELL_TIMEOUT_CAP)
                break
            deadline.record_attempt(write_seconds + run_seconds)
            if error_cell is not None:
                title, code, res = error_cell
                fails[title] = fails.get(title, 0) + 1
                if res.get("timeout"):
                    timeouts[title] = timeouts.get(title, 0) + 1
                kind = FailureKind.TIMEOUT if res.get("timeout") else FailureKind.RUNTIME_ERROR
                attempts.append(Attempt(len(attempts), kind, f"[cells] cell {title!r}: {res.get('error', '')}"[:1500],
                                        write_seconds + run_seconds))
                if fails[title] >= SAME_CELL_FAILS:
                    break
                demand = (f"CELL {title!r} FAILED and did not run:\n{res.get('error', '')[:1200]}\n"
                          f"The cell was:\n{code[:2500]}\nRewrite THAT cell (same marker title) and every "
                          "cell after it, using the RUNTIME STATE above (real shapes and columns). "
                          "Do not repeat the cells already run.")
                from . import known_errors
                fix = known_errors.hint(res.get("error", ""), _state_text(state))
                if fix:
                    demand += "\n" + fix
                if scaffold:
                    goal = steps[si]["brief"] if si < len(steps) else "fix what the audit reported"
                    demand = (f"CELL {title!r} FAILED and did not run:\n{res.get('error', '')[:1200]}\n"
                              f"The cell was:\n{code[:2500]}\nRewrite THAT cell (same marker `# %% {title}`) "
                              "using the RUNTIME STATE above (real names, shapes and columns). "
                              + ("The previous rewrite failed the same way: take a DIFFERENT route (another "
                                 "toolkit function, a simpler formula, explicit column names from the state). "
                                 if fails[title] >= 2 else "")
                              + "Its goal is unchanged:\n" + goal[:1800]
                              + ("\n" + fix if fix else ""))
                continue
            if not cells:
                attempts.append(Attempt(len(attempts), FailureKind.MISSING_OUTPUT, "[cells] reply held no cell", write_seconds))
                demand = "Your reply held no cell. Write the NEXT cell now (one `# %% title` line, then code), or `# %% done`."
                continue
            last_title = accepted[-1][0] if accepted else "?"
            attempts.append(Attempt(len(attempts), FailureKind.PASS,
                                    f"[cells] cell {last_title!r} ran" + (" (reply cut, kept its parsing part)" if cut else ""),
                                    write_seconds + run_seconds))
            demand = (f"Cell {last_title!r} ran. Write the NEXT cell (one cell only), or reply `# %% done` "
                      "when every deliverable has been emitted.")
            if not scaffold:
                continue
            if not auditing:
                step = steps[si]
                missing = step["file"] and not (output_dir / step["file"]).exists()
                if missing and cut and fails.get("cont:" + step["title"], 0) < 2:
                    # 9 Oct: a cell cut at the token limit ran up to its last complete statement; the
                    # model continues from there instead of rewriting (a rewrite is cut again)
                    fails["cont:" + step["title"]] = fails.get("cont:" + step["title"], 0) + 1
                    tail = "\n".join(accepted[-1][1].splitlines()[-4:])[:600] if accepted else ""
                    demand = (f"Your reply was CUT at the 4,000-token limit. Cell {last_title!r} ran up to its last "
                              f"complete statement; its last lines were:\n{tail}\nCONTINUE the same step in ONE short "
                              f"cell (marker `# %% {step['title']}_cont`): write ONLY what remains, reuse the variables "
                              f"already defined, and end with emit('{step['file']}', ...). Do not repeat what ran.")
                    continue
                if missing and fails.get("emit:" + step["title"], 0) < 1:
                    fails["emit:" + step["title"]] = 1
                    demand = (f"Cell {last_title!r} ran but '{step['file']}' was NOT emitted. Write ONE cell "
                              f"(marker `# %% {step['title']}`) that builds it from the variables in the RUNTIME "
                              f"STATE and calls emit('{step['file']}', ...).\n" + step["brief"][:1500])
                    continue
                si += 1
                if si < len(steps):
                    demand = _step_demand(steps[si], si + 1, len(steps))
                    continue
                auditing = True
            # know91: every planned cell ran, but a deliverable's latest emit still carries a guard verdict
            flagged = {k: v for k, v in emit_flags.items() if v}
            fkey = "emit-guards:" + ";".join(sorted(flagged))
            if flagged and fkey not in audits_seen:
                audits_seen.add(fkey)
                lines = "\n".join(f"- {m.split(': ', 1)[-1][:400]}" for v in flagged.values() for m in v[:1])
                attempts.append(Attempt(len(attempts), FailureKind.RUNTIME_ERROR,
                                        f"[cells] scaffold complete; guard verdicts on {sorted(flagged)}"[:1500], 0.0))
                demand = ("Every planned cell ran, but these deliverables failed a check when they were emitted:\n"
                          + lines + "\nWrite ONE cell (marker `# %% fix`) that fixes this from the RUNTIME STATE and "
                          "emit() the deliverable again (a later emit overwrites).")
                continue
            kind, detail = self_check(output_dir, spec.contracts, spec)
            _remember(kind, detail)
            attempts.append(Attempt(len(attempts), kind, f"[cells] scaffold complete; audit: {detail}"[:1500], 0.0))
            key = detail[:160]
            if kind is FailureKind.PASS or key in audits_seen:
                break
            audits_seen.add(key)
            demand = ("Every planned cell ran, but the audit of the written deliverables says:\n" + detail[:1500]
                      + "\nWrite ONE cell (marker `# %% fix`) that fixes this: recompute what is wrong from the "
                      "RUNTIME STATE and emit() the deliverable again (a later emit overwrites).")
    finally:
        kernel.close()
        try:
            if best["tree"] is not None:
                kind, detail = self_check(output_dir, spec.contracts, spec)
                if _problems(kind, detail) > best["n"]:
                    from .repair_loop import _restore_tree
                    _restore_tree(output_dir, best["tree"])
                    attempts.append(Attempt(len(attempts), FailureKind.PASS if best["n"] == 0 else kind,
                                            f"[cells] restored the best audited answer ({best['n']} problems "
                                            f"vs {_problems(kind, detail)} at the end)", 0.0))
        except Exception:  # noqa: BLE001
            pass
    if accepted:
        source = "\n\n".join(f"# %% {t}\n{c}" for t, c in accepted)
        (work_dir / "cells_program.py").write_text(source, encoding="utf-8")
    return attempts, source
