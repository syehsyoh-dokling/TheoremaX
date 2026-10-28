"""Write, run, self-check, repair - under a clock.

FinanceZero makes one model call and stops. It does not run what it wrote, so a
program that raises, or one that runs and emits a negative gamma, scores exactly
the same as one that never compiled. Running the code is the cheapest gain
available here.

The second gain is the self-check. The reward is binary and the grader asserts
invariants, so a solution that executes cleanly and produces wrong-shaped output
still scores zero. Checking the contract before declaring success turns a silent
zero into another repair iteration.
"""

from __future__ import annotations

import re
import os
import pathlib
import time

from . import (code_writer, consensus, examiner, explore, fc_gate, patcher, plan_gate, solution_memory,
               staging, test_runner)
from .contracts import Attempt, FailureKind, TaskSpec
from .deadline import Deadline
from invariant_library import conventions, universal

#: Independent candidates in the consensus vote. Two is the minimum that
#: yields a comparison; the vote runs only while the clock clearly affords a
#: full extra write-run-check, so a slow unit never pays for it. Dev knob
#: only (the eval room sets nothing, so the default IS the eval behaviour).
CANDIDATES = 2
#: Fraction of the budget that must remain for a second candidate to start:
#: the first solution already proved what a full attempt costs on this unit,
#: and the reconciliation round may need a third.
CONSENSUS_MIN_REMAINING = 0.35


def _candidates_wanted() -> int:
    try:
        return max(1, int(os.environ.get("T1_CANDIDATES", CANDIDATES)))
    except ValueError:
        return CANDIDATES


#: House admits 25 requests per unit. A vote costs up to ~6 of them (second
#: candidate with continuations, one repair, a reconciliation), so it only
#: runs while that many remain beyond a reserve for crash repairs.
VOTE_MIN_REQUESTS = 8
#: A new write costs 1 request plus up to MAX_CONTINUATIONS when the reply is cut.
MIN_REQUESTS_FOR_ATTEMPT = 3
#: The examiner costs one request, plus one repair when it confirms a defect.
EXAMINER_MIN_REQUESTS = 4


def _consensus_affordable(deadline: Deadline) -> bool:
    return (_candidates_wanted() >= 2 and deadline.can_afford()
            and deadline.remaining >= deadline.budget * CONSENSUS_MIN_REMAINING
            and code_writer.requests_remaining() >= VOTE_MIN_REQUESTS)

#: Base iteration allowance, extended up to the hard cap while the budget
#: clearly affords more rounds. Five was a constant from the no-model era;
#: with typical model runs using 100-300s of an 1800s card, extra rounds are
#: free realisation-of-mistake opportunities.
MAX_ITERATIONS = 5
MAX_ITERATIONS_HARD = 10
#: A timed-out attempt earns exactly one repair round with a "make it cheaper"
#: hint. Zero rounds threw away >1000s of budget on any task whose honest run
#: costs more than the old fixed cap; unlimited rounds would let a hopeless
#: task eat the clock.
MAX_TIMEOUT_REPAIRS = 1


def _source_frame(spec: TaskSpec):
    """The input table a row-key check compares against, when unambiguous.

    Only when the unit carries exactly one tabular data file: with several,
    guessing which one the identifier must survive from risks a self-check
    that is wrong in either direction. Lazy and forgiving - a failure to load
    input is never allowed to fail the *output*.
    """
    tabular = [f for f in spec.data_files
               if f.lower().endswith((".parquet", ".csv"))]
    if len(tabular) != 1:
        return None
    try:
        import pandas as pd
        path = pathlib.Path(spec.input_root) / tabular[0]
        if tabular[0].lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None


def _self_check(output_dir: pathlib.Path, contracts,
                spec: TaskSpec | None = None) -> tuple[FailureKind, str]:
    problems: list[str] = []
    source = _source_frame(spec) if (
        spec is not None and any(c.row_key for c in contracts)) else None
    instruction = spec.instruction if spec is not None else ""
    for contract in contracts:
        violations = universal.check_contract(
            output_dir / contract.filename, contract,
            source_frame=source if contract.row_key else None)
        # Executed conventions, bound to the task's own text. The playbook
        # states the same rules; graded runs broke them anyway.
        if os.environ.get("T1_CONVENTION_CHECKS", "on") != "off":
            violations += conventions.check(
                output_dir / contract.filename, contract, instruction)
        problems += [str(v) for v in violations]
    if not problems:
        return FailureKind.PASS, ""
    if any(p.startswith("file_exists") for p in problems):
        return FailureKind.MISSING_OUTPUT, "; ".join(problems[:8])
    if any(p.startswith(("columns_present", "readable")) for p in problems):
        return FailureKind.SCHEMA_MISMATCH, "; ".join(problems[:8])
    return FailureKind.INVARIANT_VIOLATION, "; ".join(problems[:8])


#: A stage costs one write plus at most one repair. Below this many requests
#: left, starting another stage buys a program that cannot be fixed.
STAGE_MIN_REQUESTS = 4
STAGE_MIN_SECONDS = 90.0
#: One repair per stage. A stage that fails twice is a stage whose deliverables
#: the later stages must live without; spending more starves the stages after it.
STAGE_REPAIRS = 1


def _reply_was_cut(telemetry: list | None) -> bool:
    """Did the endpoint truncate a reply at the 4,000-token cap on this unit?"""
    for event in telemetry or ():
        if event.get("event") == "output_cut" or event.get("finish_reason") == "length":
            return True
    return False


def _failing_contracts(spec: TaskSpec, output_dir: pathlib.Path, detail: str) -> tuple:
    """The deliverables the full audit complained about, or that are simply absent."""
    named = tuple(c for c in spec.contracts
                  if c.filename in detail or not (output_dir / c.filename).is_file())
    return named or spec.contracts


def _delivered(spec: TaskSpec, output_dir: pathlib.Path) -> int:
    return sum(1 for c in spec.contracts if (output_dir / c.filename).is_file())


def _run_stages(spec: TaskSpec, output_dir: pathlib.Path,
                work_dir: pathlib.Path, deadline: Deadline,
                telemetry: list | None) -> tuple[list[Attempt], str | None]:
    """Write the deliverables in groups of two, each group its own program.

    Returns (attempts, last source). The caller falls back to the ordinary
    monolithic loop when the full contract still is not satisfied, so a bad
    split can only cost requests, never the answer.
    """
    stages = staging.plan(spec.contracts)
    out: list[Attempt] = []
    last_source: str | None = None
    seen: set[str] = set()      # failure signatures already answered once
    # 2 Oct autopsy: 0 of 32 staged runs passed. Stage 1 is told to save its prepared tables to
    # staging.SCRATCH and "create the directory"; programs that did not died on `OSError: Cannot
    # save file into a non-existent directory`, and every later stage then read nothing.
    try:
        pathlib.Path(staging.SCRATCH).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    for si, group in enumerate(stages):
        label = f"[stage {si + 1}/{len(stages)}]"
        # 27 Sep compound-geske: the last stage (summary.json, a few seconds of work) was never
        # started with 202 s and 11 requests left, because can_afford() predicts from the WORST
        # attempt (213 s, the monolithic try). run_solution already caps each program at the time
        # that is left, so a stage only needs room for one model call plus a short run.
        if deadline.remaining < STAGE_MIN_SECONDS or code_writer.requests_remaining() < STAGE_MIN_REQUESTS:
            out.append(Attempt(len(out), FailureKind.MODEL_UNAVAILABLE,
                               f"{label} not started: "
                               f"{code_writer.requests_remaining()} House requests "
                               f"and {deadline.remaining:.0f}s left", 0.0))
            break
        note = staging.note(si, stages, str(output_dir))
        error: str | None = None
        previous: str | None = None
        repeats = 0
        for round_ in range(STAGE_REPAIRS + 1):
            # 1 Oct cme-hdd: rounds inside one stage never looked at the clock and the unit ran
            # 1376 s on a 900 s budget (free-lane model calls take 50-100 s each).
            if round_ and (deadline.remaining < STAGE_MIN_SECONDS
                           or code_writer.requests_remaining() < STAGE_MIN_REQUESTS):
                out.append(Attempt(len(out), FailureKind.MODEL_UNAVAILABLE,
                                   f"{label} repair round {round_} not started: "
                                   f"{deadline.remaining:.0f}s left", 0.0))
                break
            started = time.monotonic()
            source, provenance = code_writer.write(
                spec, group, str(output_dir), error, True, telemetry=telemetry,
                previous_source=previous, attempt_index=si * 10 + round_,
                stage_note=note)
            write_seconds = time.monotonic() - started
            if source is None:
                out.append(Attempt(len(out), FailureKind.MODEL_UNAVAILABLE,
                                   f"{label} no program returned", write_seconds))
                break
            previous = last_source = source
            script = work_dir / f"stage_{si}_{round_}.py"
            script.write_text(source, encoding="utf-8")
            per_attempt = max(60.0, min(deadline.remaining - 30.0,
                                        max(5.0, deadline.remaining)))
            kind, detail, seconds = test_runner.run_solution(
                script, work_dir, per_attempt, input_dir=spec.input_root)
            deadline.record_attempt(write_seconds + seconds)
            if kind is FailureKind.PASS:
                # Only THIS stage's deliverables are audited here; the full
                # contract is checked once every stage has run.
                kind, detail = _self_check(output_dir, group, spec)
            out.append(Attempt(len(out), kind,
                               f"{label}[{provenance}] {detail}"[:1500],
                               write_seconds + seconds))
            if kind is FailureKind.PASS or not kind.repairable:
                break
            signature = f"{kind.value}:{detail[:120]}"
            error = (f"{kind.value}: {detail}" + _toolkit_import_hint(detail)
                     + _was_cut(detail) + identity_hint(detail))
            if signature in seen:
                # Same complaint as a previous stage or round: the plain repair
                # prompt has already been answered with the same program once.
                error += repeat_hint(detail)
                repeats += 1
                if repeats > 1:
                    break       # three shots at one signature is the allowance
            seen.add(signature)

    # The full contract, once every stage has had its turn.
    kind, detail = _self_check(output_dir, spec.contracts, spec)
    out.append(Attempt(len(out), kind,
                       f"[stages] full contract audit: {detail}"[:1500], 0.0))
    if kind is FailureKind.PASS:
        return out, last_source
    # Batch A: bollinger spent 6 stage programs AND 2 monolithic rewrites on one
    # unit (24 of 24 requests) because a failing audit threw the whole staged
    # answer away. Repair the deliverables that actually failed instead - one
    # short program, scoped to them, with the audit as the complaint.
    attempt, source = _complete_missing(spec, output_dir, work_dir, deadline, telemetry, detail,
                                        len(out), "stage repair")
    if attempt is not None:
        out.append(attempt)
        last_source = source or last_source
    return out, last_source


def _complete_missing(spec: TaskSpec, output_dir: pathlib.Path, work_dir: pathlib.Path,
                      deadline: Deadline, telemetry: list | None, detail: str, index: int,
                      label: str) -> tuple[Attempt | None, str | None]:
    """One short program scoped to the deliverables that failed or are absent, the others left
    alone. Used after the stages, and (27 Sep full run: 8 units ended with some files written and
    the rest stubbed) after a single program that wrote only part of the answer."""
    failing = _failing_contracts(spec, output_dir, detail)
    if not (len(failing) < len(spec.contracts) and deadline.remaining >= STAGE_MIN_SECONDS
            and code_writer.requests_remaining() >= STAGE_MIN_REQUESTS):
        return None, None
    note = ("\nREPAIR STAGE. The other deliverables are written and correct; "
            "leave them alone (you may READ them from the write directory). Write ONLY: "
            + ", ".join(c.filename for c in failing)
            + ". The audit of your earlier attempt said:\n" + detail[:800])
    started = time.monotonic()
    source, provenance = code_writer.write(
        spec, failing, str(output_dir), None, True, telemetry=telemetry,
        attempt_index=90 + index, stage_note=note)
    write_seconds = time.monotonic() - started
    if not source:
        return None, None
    script = work_dir / f"complete_{index}.py"
    script.write_text(source, encoding="utf-8")
    per_attempt = max(60.0, min(deadline.remaining - 30.0, max(5.0, deadline.remaining)))
    rkind, rdetail, seconds = test_runner.run_solution(
        script, work_dir, per_attempt, input_dir=spec.input_root)
    deadline.record_attempt(write_seconds + seconds)
    if rkind is FailureKind.PASS:
        rkind, rdetail = _self_check(output_dir, spec.contracts, spec)
    return (Attempt(index, rkind, f"[{label}][{provenance}] {rdetail}"[:1500],
                    write_seconds + seconds), source)


#: Clean rewrites allowed when the repair thread repeats itself, and what must still be left.
MAX_FRESH_RESTARTS = 2
FRESH_MIN_REQUESTS = 8
FRESH_MIN_BUDGET_SHARE = 0.4


def _fresh_restart_affordable(deadline: Deadline, used: int) -> bool:
    return (used < MAX_FRESH_RESTARTS
            and code_writer.requests_remaining() >= FRESH_MIN_REQUESTS
            and deadline.remaining >= deadline.budget * FRESH_MIN_BUDGET_SHARE
            and deadline.can_afford())


def solve(spec: TaskSpec, output_dir: pathlib.Path, work_dir: pathlib.Path,
          deadline: Deadline, *, prefer_model: bool = True,
          telemetry: list | None = None) -> list[Attempt]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[Attempt] = []
    error: str | None = None
    previous_source: str | None = None
    timeout_repairs = 0
    model_failures = 0
    template_tried = False
    previous_failure: str | None = None
    risk_round_used = False
    risk_snapshot: dict[str, bytes] = {}
    fresh_restarts = 0
    completion_tried = False
    patch_pending: str | None = None
    patched_signatures: set[str] = set()

    # A checker-verified program for THIS task id (README rule 8: our own solutions to public
    # units travel in the image) is replayed first, with no model phases in front of it: the
    # reading and planning requests exist to write a program, and this one is already written.
    # An exam id matches nothing, so the phases below run as usual there.
    replaying = (os.environ.get("T1_SOLUTION_MEMORY", "off").strip().lower() == "on"
                 and solution_memory.has_solution(spec.task_id))
    if replaying:
        spec.notes.append("verified solution on file for this task id: replaying it first")

    # Phase 0 - look before writing. The reading program's output becomes
    # the EVIDENCE block of every prompt below.
    if prefer_model and not replaying and os.environ.get("T1_EXPLORE", "1") != "0":
        evidence, note = explore.run(spec, work_dir, deadline, telemetry)
        spec.evidence = evidence
        spec.notes.append(note)

    # Phase P - the answer framework, validated by code before any program is written; an
    # incomplete plan is sent back with the full notes of similar solved tasks (plan_gate.py).
    if prefer_model and not replaying and os.environ.get("T1_PLAN", "on").strip().lower() != "off":
        try:
            spec.notes.append(plan_gate.run(spec, telemetry, deadline))
        except Exception as error:          # a planning failure never blocks the unit
            spec.notes.append(f"plan error: {type(error).__name__}: {error}")

    # Cells with a cached runtime state (cells.py, 6 Oct): short cells run one by one in a
    # persistent kernel; a cut reply loses nothing and every cell sees the real shapes. When the
    # cells leave at least half the deliverables written the monolithic loop is skipped; the
    # accepted cells become `previous_source` so the later phases can still repair from them.
    if (prefer_model and not replaying and os.environ.get("T1_CELLS", "on").strip().lower() != "off"
            and code_writer.requests_remaining() >= 2 * STAGE_MIN_REQUESTS):
        try:
            from . import cells
            cell_attempts, cell_source = cells.solve(spec, output_dir, work_dir, deadline, telemetry,
                                                     _self_check)
        except Exception as error:  # noqa: BLE001 - the kernel path never blocks the unit
            cell_attempts, cell_source = [], None
            spec.notes.append(f"cells error: {type(error).__name__}: {error}")
        attempts += cell_attempts
        if cell_source:
            previous_source = cell_source
            spec.notes.append(f"cells: {len(cell_attempts)} round(s), {len(cell_source)} chars kept")
        kind, detail = _self_check(output_dir, spec.contracts, spec)
        if kind is FailureKind.PASS or _delivered(spec, output_dir) * 2 >= len(spec.contracts):
            spec.staged = True
        else:
            error = f"{kind.value}: {detail}"

    # Stage-first for units too big for one 4,000-token reply (staging.first): the same staged
    # path the loop takes after a cut, taken BEFORE the first long reply spends the requests. If
    # the split collapses (under half the deliverables written) the monolithic loop still runs.
    # (not after a cells attempt: know39 showed the stage path only spending what the cells left)
    cells_tried = any(a.detail.startswith("[cells]") for a in attempts)
    if (prefer_model and not replaying and not spec.staged and not cells_tried
            and staging.first(spec.contracts, spec.instruction or "")
            and code_writer.requests_remaining() >= 2 * STAGE_MIN_REQUESTS):
        spec.staged = True
        spec.notes.append(f"stage-first: {len(spec.contracts)} deliverables, "
                          f"{len(spec.instruction or '')} instruction chars")
        staged, staged_source = _run_stages(spec, output_dir, work_dir, deadline, telemetry)
        attempts += staged
        previous_source = staged_source or previous_source
        kind, detail = _self_check(output_dir, spec.contracts, spec)
        if not (kind is FailureKind.PASS or _delivered(spec, output_dir) * 2 >= len(spec.contracts)):
            spec.staged = False
            error = f"{kind.value}: {detail}"

    # Phase 1 - answer everything: the floor answer, never destroyed later.
    for index in range(MAX_ITERATIONS_HARD):
        if spec.staged:
            break
        if index >= MAX_ITERATIONS and deadline.remaining < deadline.budget * 0.5:
            # Extra rounds beyond the base five only while MOST of the
            # budget is still unspent - late extensions risk the reserve.
            break
        if not deadline.can_afford():
            attempts.append(Attempt(index, FailureKind.TIMEOUT,
                                    f"stopped early with {deadline.remaining:.0f}s "
                                    "left rather than risk being killed mid-write"))
            break
        if index and code_writer.requests_remaining() <= MIN_REQUESTS_FOR_ATTEMPT:
            # A fresh attempt can cost 1 + MAX_CONTINUATIONS requests. Starting one
            # that cannot finish spends the last of the House allowance on a program
            # that will be cut; the answer already on disk is worth more.
            attempts.append(Attempt(index, FailureKind.MODEL_UNAVAILABLE,
                                    "[budget] stopped: "
                                    f"{code_writer.requests_remaining()} House requests left"))
            break

        # The model call costs wall clock too - up to 180s per iteration -
        # and an unmeasured cost made `predicted_attempt_cost` optimistic by
        # exactly that much, which is how a run gets killed at the cap.
        # After two failed model attempts, a template that matches the
        # contract gets one shot: a proven template must never lose forever
        # to a completion that keeps crashing.
        force_template = (
            not template_tried and model_failures >= 2
            and code_writer.match_template(spec, spec.contracts) is not None)

        write_started = time.monotonic()
        patched = None
        if patch_pending and previous_source:
            # the same line crashed twice: ask for an EDIT of that program (patcher.py)
            patched = patcher.patch(previous_source, patch_pending, telemetry=telemetry)
            patch_pending = None
        if patched:
            source, provenance = patched, "model-patch"
        else:
            source, provenance = code_writer.write(
                spec, spec.contracts, str(output_dir), error, prefer_model,
                telemetry=telemetry, previous_source=previous_source,
                attempt_index=index, force_template=force_template,
                # a fresh rewrite must not replay attempt 0 (temperature 0): its own variant
                # carries the independent-route hint and a different seed
                variant=(20 + fresh_restarts) if fresh_restarts else 0)
        write_seconds = time.monotonic() - write_started
        if source is None:
            attempts.append(Attempt(index, FailureKind.MODEL_UNAVAILABLE,
                                    "no model and no template matched"))
            break
        if provenance == "template":
            template_tried = True

        # Byte-identical source will fail byte-identically. If a template is
        # still in reserve, spend the next iteration on it; otherwise stop.
        if (source == previous_source and error and failing_line(error)
                and f"identical:{error[:200]}" not in patched_signatures
                and code_writer.requests_remaining() > MIN_REQUESTS_FOR_ATTEMPT):
            # the rewrite came back byte-identical: edit the failing line instead of stopping
            patched_signatures.add(f"identical:{error[:200]}")
            patch_pending = error
            attempts.append(Attempt(index, FailureKind.RUNTIME_ERROR,
                                    f"[{provenance}] regenerated source identical; line-level "
                                    "patch next", write_seconds))
            continue
        if source == previous_source:
            model_failures = max(model_failures, 2)
            if (not template_tried and code_writer.match_template(
                    spec, spec.contracts) is not None):
                attempts.append(Attempt(index, FailureKind.RUNTIME_ERROR,
                                        f"[{provenance}] regenerated source "
                                        "identical; switching to template",
                                        write_seconds))
                continue
            attempts.append(Attempt(index, FailureKind.RUNTIME_ERROR,
                                    f"[{provenance}] regenerated source is "
                                    "identical to the failed attempt; stopping",
                                    write_seconds))
            break
        previous_source = source

        script = work_dir / f"solution_{index}.py"
        script.write_text(source, encoding="utf-8")

        # The cap is the remaining budget, not an arbitrary constant: a fixed
        # 600s cap killed any honest 700s solution with >1000s still on the
        # clock. The floor may never punch through what is actually left.
        per_attempt = max(60.0, deadline.remaining - 30.0)
        per_attempt = min(per_attempt, max(5.0, deadline.remaining))
        try:
            kind, detail, seconds = test_runner.run_solution(
                script, work_dir, per_attempt, input_dir=spec.input_root)
        except TypeError:
            kind, detail, seconds = test_runner.run_solution(
                script, work_dir, per_attempt)
        deadline.record_attempt(write_seconds + seconds)

        if provenance == "solution_memory":
            if kind is FailureKind.PASS:
                # Verified by the unit's own checker (reward 1.0): the output is final. No shape
                # gate, no vote, no examiner - each of those can only replace a graded answer
                # with an unverified one.
                attempts.append(Attempt(index, FailureKind.PASS,
                                        "[solution_memory][evidence] checker-verified "
                                        "program replayed", write_seconds + seconds))
                return attempts
            # The stored program failed here (different data layout or limits): fall back to
            # the normal path, reading the inputs first so the model gets real evidence.
            attempts.append(Attempt(index, kind, f"[solution_memory] {detail}"[:1500],
                                    write_seconds + seconds))
            if prefer_model and os.environ.get("T1_EXPLORE", "1") != "0":
                evidence, note = explore.run(spec, work_dir, deadline, telemetry)
                spec.evidence = evidence
                spec.notes.append(note)
            error = f"{kind.value}: {detail}"
            previous_source = None
            continue

        if kind is FailureKind.PASS:
            kind, detail = _self_check(output_dir, spec.contracts, spec)
        elif kind in (FailureKind.RUNTIME_ERROR, FailureKind.TIMEOUT):
            # A generated program can write every real deliverable and then
            # die in its own assert or in post-processing.  The grader scores
            # the files, not the program's exit code, so re-read the output
            # before throwing that candidate away.  This is deliberately
            # limited to a complete contract audit: a partial or unreadable
            # tree remains a normal failure and gets a repair prompt.
            recovered_kind, _ = _self_check(
                output_dir, spec.contracts, spec)
            if recovered_kind is FailureKind.PASS:
                detail = (f"post-failure contract audit passed after "
                          f"{kind.value}: candidate kept as uncertain")
                kind = FailureKind.PASS

        attempts.append(Attempt(index, kind,
                                f"[{provenance}] {detail}"[:1500],
                                write_seconds + seconds))

        if kind is FailureKind.PASS:
            # Output-level evidence outranks code-shape evidence: when a
            # second independent candidate is affordable, the consensus vote
            # (below the loop) is the honesty driver. The calibrated shape
            # gate is the fallback for units too slow to afford a vote - a
            # nudge that buys ONE cross-check round, never a hard block.
            if (provenance == "model" and not risk_round_used
                    and not _consensus_affordable(deadline)
                    and deadline.can_afford()
                    and fc_gate.risk_of(source) >= fc_gate.RISK_THRESHOLD):
                risk_round_used = True
                # Snapshot the passing deliverables: if the cross-checked
                # rewrite lands worse, the audited-but-passing answer is
                # restored - the gate may spend budget, never destroy value.
                for contract in spec.contracts:
                    target = output_dir / contract.filename
                    if target.is_file():
                        risk_snapshot[contract.filename] = target.read_bytes()
                attempts[-1] = Attempt(index, FailureKind.INVARIANT_VIOLATION,
                                       "[fc-gate] self-check passed but the "
                                       "risk profile is high; demanding a "
                                       "cross-checked rewrite",
                                       attempts[-1].seconds)
                error = fc_gate.CROSS_CHECK_DEMAND
                continue
            break
        if kind is FailureKind.TIMEOUT:
            if timeout_repairs >= MAX_TIMEOUT_REPAIRS or not deadline.can_afford():
                break
            timeout_repairs += 1
            error = (f"timeout: the previous attempt exceeded {per_attempt:.0f}s. "
                     "Produce a substantially cheaper implementation - "
                     "vectorise, reduce grid/path counts to what the stated "
                     "tolerance needs, avoid per-row Python loops.")
            continue
        if not kind.repairable:
            break
        if (kind is FailureKind.MISSING_OUTPUT and not completion_tried
                and 0 < _delivered(spec, output_dir) < len(spec.contracts)):
            # part of the answer is on disk: complete THAT, scoped to the missing files, rather
            # than a full rewrite that can lose what is already right (27 Sep full run)
            completion_tried = True
            attempt, source = _complete_missing(spec, output_dir, work_dir, deadline, telemetry,
                                                detail, index, "complete missing")
            if attempt is not None:
                attempts.append(attempt)
                previous_source = source or previous_source
                if attempt.kind is FailureKind.PASS:
                    kind, detail = attempt.kind, attempt.detail
                    break
        if (staging.wanted(spec.contracts) and _reply_was_cut(telemetry)
                and code_writer.requests_remaining() >= 2 * STAGE_MIN_REQUESTS
                and deadline.can_afford()):
            # The reply really was truncated at the House cap, and this unit has
            # enough deliverables to split. Batch A-C: 89 stage programs, median
            # ~8,000 characters, against monoliths of 10,900-21,100 - the same
            # answer in a shape that arrives whole. Units that were never cut
            # keep the single program they have always passed with.
            spec.staged = True
            staged, staged_source = _run_stages(spec, output_dir, work_dir,
                                                deadline, telemetry)
            attempts += staged
            previous_source = staged_source or previous_source
            kind, detail = _self_check(output_dir, spec.contracts, spec)
            if kind is FailureKind.PASS or _delivered(spec, output_dir) * 2 >= len(spec.contracts):
                break
            spec.staged = False          # the split collapsed too; carry on
            error = f"{kind.value}: {detail}"
            continue
        if provenance == "model":
            model_failures += 1
        # Two consecutive attempts failing on the byte-same complaint means
        # the feedback channel is exhausted: a third answer to the same
        # message will not differ. Measured on the 30x1 post-mortem - extra
        # iterations against a relational error were spent, not used.
        signature = f"{kind.value}:{detail[:200]}"
        if (signature == previous_failure and kind is FailureKind.RUNTIME_ERROR
                and failing_line(detail) and previous_source
                and signature not in patched_signatures
                and code_writer.requests_remaining() > MIN_REQUESTS_FOR_ATTEMPT):
            # the same line crashed twice: an edit of the failing program, not a third rewrite
            patched_signatures.add(signature)
            patch_pending = detail
            attempts.append(Attempt(index, kind, "same line failed twice; line-level patch next",
                                    0.0))
            continue
        if signature == previous_failure:
            # A template still in reserve outranks giving up (the exemplar
            # fails the model path identically twice before its template
            # rescue - stopping here would re-open that regression).
            if (not template_tried and code_writer.match_template(
                    spec, spec.contracts) is not None):
                model_failures = max(model_failures, 2)
            elif _fresh_restart_affordable(deadline, fresh_restarts):
                # The repair channel is exhausted, the budget is not: the 25 Sep autopsy found
                # most failed units stopped here or at the crash cap with time left and a median
                # of 6 of 25 requests used. A clean rewrite - no failed source, no error thread -
                # is an independent draw, not a third answer to the same complaint.
                fresh_restarts += 1
                attempts.append(Attempt(index, kind,
                                        f"same failure twice; fresh rewrite {fresh_restarts} "
                                        f"({code_writer.requests_remaining()} requests, "
                                        f"{deadline.remaining:.0f}s left)", 0.0))
                previous_source = None
                previous_failure = None
                error = None
                continue
            else:
                attempts.append(Attempt(index, kind,
                                        "same failure twice in a row; stopping "
                                        "rather than spending budget on a third",
                                        0.0))
                break
        previous_failure = signature
        error = f"{kind.value}: {detail}" + _toolkit_import_hint(detail) + _was_cut(detail)
        if signature == previous_failure or failing_line(detail):
            error += repeat_hint(detail)
        error += identity_hint(detail)
        if _unstated_assert(detail, spec.instruction):
            error += "\n" + SELF_ASSERT_HINT

    if (risk_snapshot and attempts
            and attempts[-1].kind is not FailureKind.PASS):
        for filename, payload in risk_snapshot.items():
            (output_dir / filename).write_bytes(payload)
        attempts.append(Attempt(len(attempts), FailureKind.PASS,
                                "[fc-gate] cross-checked rewrite did not "
                                "verify; restored the earlier passing answer",
                                0.0))

    # Phases 2-3 - self-triage, then spend what is left on the diagnosed
    # cause in leverage order. Each round re-triages; the floor answer is
    # replaced only by an answer that self-checks (and only earns a `pass`
    # claim with vote evidence).
    if prefer_model:
        attempts += _fix_list(spec, output_dir, work_dir, deadline,
                              attempts, previous_source, error, telemetry)
    return attempts


# ---------------------------------------------------------------------------
# Phase 2: triage.  Phase 3: the fix list.
# ---------------------------------------------------------------------------

#: Buckets, in the order the v5 leverage table ranks them.
CRASH, SELFCHECK, VOTE_PENDING, SECOND_CRASHED, DISAGREE, INCONCLUSIVE, \
    AGREE, EXAMINED, DONE = ("crash", "selfcheck", "vote_pending",
                             "second_crashed", "disagree", "inconclusive",
                             "agree", "examined", "done")
MAX_FIX_ROUNDS = 6
#: The crash bucket gets at most this many extra repairs beyond phase 1.
MAX_CRASH_ROUNDS = 2

def _toolkit_import_hint(detail: str) -> str:
    """A name imported from qf_toolkit that does not exist gets the list of names that do
    (v6.9: `from qf_toolkit import OPT` read a section title as a function)."""
    if "from 'qf_toolkit'" not in (detail or ""):
        return ""
    try:
        import qf_toolkit
        public = [n for n in dir(qf_toolkit) if not n.startswith("_") and n != "API"]
        names = sorted(n for n in public
                       if callable(getattr(qf_toolkit, n)) or n.isupper())
    except Exception:  # noqa: BLE001
        return ""
    return ("\nqf_toolkit has no such name. The ONLY importable names are: "
            + ", ".join(names) + ". Import one of these or write the helper yourself.")


#: A program cut at the 4,000-token House reply cap comes back as a SyntaxError on the seam.
#: Continuing it costs one request each time (9 of 26 ever-pass failures ended this way, and
#: 7 more ran out of the 25-request allowance), so the repair asks for a SHORTER program.
TOO_LONG_HINT = (
    "\nYour previous program was cut off by the reply limit, so it never parsed. Write a "
    "SHORTER program (aim for under 150 lines): import qf_toolkit helpers instead of "
    "re-deriving them, drop comments and prints, compute and WRITE each deliverable as soon "
    "as its values exist, and keep verification to the identities the task states.")


def _was_cut(detail: str) -> str:
    cut = ("unterminated string literal", "was never closed", "unexpected EOF",
           "invalid syntax", "unmatched", "does not match opening parenthesis")
    return TOO_LONG_HINT if any(c in (detail or "") for c in cut) else ""


CRASH_HINT = (
    "The program keeps crashing before it writes anything. Use ONLY the "
    "column names, keys, dtypes and paths shown in the EVIDENCE block; "
    "before every access that raised, print the object's keys/columns and "
    "guard the access. Return the complete program.")

#: A self-assert whose wording the instruction never uses is the model's own
#: hypothesis. Measured: 59% of assert failures, and they kill programs that
#: had already written correct deliverables (ohlc-realized-vol died on
#: "yz_efficiency should exceed 1" in 35 iterations across runs).
#: The word match is a SIGNAL, not a verdict: a stated identity can be
#: paraphrased. So the hint asks the model to decide, defaulting to the
#: warning form (a hypothesis assert can destroy a correct answer, while a
#: downgraded stated-identity assert only loses a check), and explicitly
#: protects a stated identity that happens to be worded differently.
SELF_ASSERT_HINT = (
    "The wording of that failing check does not appear in the task text. "
    "Decide which it is. If it is YOUR expectation - an estimator you think "
    "should dominate, an ordering or sign you consider natural - it is a "
    "hypothesis: keep the computation, replace the assert with a printed "
    "warning, and carry on. If the task really does require it in different "
    "words, keep the assert and fix the computation instead. Either way, "
    "write the deliverables BEFORE any verification block runs, so a check "
    "can never cost you a finished answer.")
_ASSERT_RE = re.compile(r"AssertionError:?\s*([^\n|]{0,180})")
_ASSERT_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


#: Traceback frames of the generated program itself (not the library).
_FRAME = re.compile(r'File "[^"]*(?:agent-work|/tmp)[^"]*", line (\d+)[^\n]*\n\s*([^\n]+)')


def failing_line(detail: str) -> str:
    """The last line of the generated program the traceback blames.

    Batch B: fama-french repeated `e**2[:, None]` - an operator-precedence slip -
    in 6 successive programs, and option-put-call-parity repeated its own 8
    times. The complaint reached the model; the offending expression did not.
    """
    frames = _FRAME.findall(detail)
    return frames[-1][1].strip() if frames else ""


def repeat_hint(detail: str) -> str:
    line = failing_line(detail)
    if not line:
        return ""
    return ("\nTHIS EXACT LINE FAILED AGAIN:\n    " + line[:300]
            + "\nDo not send it back with cosmetic changes. Rewrite THAT "
              "expression a different way - split it into named steps, check "
              "operator precedence and bracket placement, and print the shape "
              "and dtype of each part before combining them.")


#: An assert whose wording is a conservation, range or sign law is never the
#: model's private hypothesis, whatever words the task used. Batch C ended 17 of
#: its attempts on exactly these: "variance decomposition does not sum to 100,
#: 101.69", "spread=1.51, sum=-2.93", "all cf_price must be positive",
#: "u1 should be in [0,1]", "sum of scheduled_quantity is 0, expected 50000".
_IDENTITY = re.compile(
    r"sum|total|add up|adds to|equals?\b|mismatch|match(es)?\b|consistent|"
    r"identity|parity|balance|non-?negative|positive|negative|probab|"
    r"\[0\s*,\s*1\]|between 0 and 1|100\b|weights?\b|conserv", re.I)


def is_identity_violation(detail: str) -> bool:
    match = _ASSERT_RE.search(detail or "")
    return bool(match and _IDENTITY.search(match.group(1)))


def identity_hint(detail: str) -> str:
    """The complaint to send back when the program's own law failed.

    Batch C: programs now run far enough to check themselves and then break
    their own rule - a sign flipped, a unit unconverted, a term left out of a
    sum. The model knows the law (it wrote the check); what it needs is to be
    stopped from deleting the check and made to look at the two sides.
    """
    match = _ASSERT_RE.search(detail or "")
    if not match:
        return ""
    return ("\nYOUR OWN CHECK FAILED: " + match.group(1).strip()[:200]
            + "\nThat check is a law of the problem, not an opinion: keep it "
              "exactly as it is. The arithmetic under it is what is wrong. "
              "Before combining terms, print EACH component of both sides with "
              "its name, sign and unit, and compare them - a decomposition that "
              "misses 101.69 against 100 is one term counted twice, scaled "
              "wrongly (percent against fraction, annual against daily), or "
              "carrying the wrong sign. Fix the computation so the check passes "
              "on its own, and never relax, delete or widen the tolerance of a "
              "check to get past it.")


def _unstated_assert(detail: str, instruction: str) -> bool:
    """Does the failing assert's wording appear in the task text at all?"""
    match = _ASSERT_RE.search(detail or "")
    if not match:
        return False
    if _IDENTITY.search(match.group(1)):
        # A conservation or range law is never a private hypothesis, so it must
        # never be downgraded to a printed warning - that hides the defect the
        # grader is about to find.
        return False
    text = (instruction or "").lower()
    words = {w.lower() for w in _ASSERT_WORD.findall(match.group(1))}
    words -= {"must", "should", "value", "values", "expected", "error",
              "assert", "check", "result", "results", "than", "with"}
    if not words:
        return False
    return len({w for w in words if w in text}) < 2


def triage(attempts: list[Attempt]) -> str:
    """Which bucket of the leverage table the unit is in right now."""
    if not attempts:
        return CRASH
    last = attempts[-1]
    trail = " ".join(a.detail for a in attempts)
    if not last.passed:
        if last.kind in (FailureKind.RUNTIME_ERROR, FailureKind.IMPORT_ERROR,
                         FailureKind.TIMEOUT, FailureKind.MODEL_UNAVAILABLE):
            return CRASH
        return SELFCHECK
    if "[examiner]" in last.detail:
        return EXAMINED
    if last.detail.startswith("[template]"):
        return DONE          # a proven template is its own evidence
    if "[consensus]" not in trail:
        return VOTE_PENDING
    if "[evidence]" in last.detail:
        return AGREE
    if "second candidate unusable" in trail and "DISAGREE" not in trail:
        return SECOND_CRASHED
    if "DISAGREE" in trail:
        return DISAGREE
    return INCONCLUSIVE


def _fix_list(spec: TaskSpec, output_dir: pathlib.Path, work_dir: pathlib.Path,
              deadline: Deadline, attempts: list[Attempt],
              last_source: str | None, last_error: str | None,
              telemetry) -> list[Attempt]:
    out: list[Attempt] = []
    crash_rounds = 0
    examined = False
    voted = False
    source = last_source
    error = last_error
    for _ in range(MAX_FIX_ROUNDS):
        if not deadline.can_afford():
            break
        bucket = triage(attempts + out)
        index = len(attempts) + len(out)
        if bucket in (CRASH, SELFCHECK):
            if bucket == CRASH and crash_rounds >= MAX_CRASH_ROUNDS:
                break
            if bucket == CRASH:
                crash_rounds += 1
            hint = ((CRASH_HINT if bucket == CRASH else "")
                    + _toolkit_import_hint(error or "") + _was_cut(error or ""))
            fixed, kind, detail, seconds, tree = _candidate(
                spec, output_dir, work_dir, deadline, variant=0,
                error=f"{error or ''}\n{hint}".strip(), previous_source=source,
                prefer_model=True, telemetry=telemetry, index=index)
            if fixed is None:
                break
            source = fixed
            if kind is FailureKind.PASS:
                _restore_tree(output_dir, tree)   # the floor becomes this answer
                out.append(Attempt(index, FailureKind.PASS,
                                   f"[model][fix:{bucket}] repaired", seconds))
            else:
                error = f"{kind.value}: {detail}"
                out.append(Attempt(index, kind,
                                   f"[model][fix:{bucket}] {detail}"[:1500], seconds))
            continue
        if bucket == VOTE_PENDING and not voted:
            voted = True
            if source and not spec.staged and _consensus_affordable(deadline):
                out += _cross_validate(spec, output_dir, work_dir, deadline,
                                       source, True, telemetry, start_index=index)
                continue
            # No budget for a vote: the answer stays; the verdict will say so.
            # A staged unit lands here by design - a second candidate would be
            # one more monolithic program, the shape staging exists to avoid.
            out.append(Attempt(index, FailureKind.PASS,
                               ("[consensus] staged answer: no vote (a second "
                                "monolith is what staging avoids); first answer "
                                "stands (flagged)" if spec.staged else
                                "[consensus] no vote (disabled or unaffordable); "
                                "first answer stands (flagged)"), 0.0))
            continue
        if bucket in (AGREE, SECOND_CRASHED, DISAGREE, INCONCLUSIVE,
                      VOTE_PENDING) and not examined and source:
            # The examiner: one adversarial audit, then at most one repair
            # driven by its findings. Runs on evidenced answers too - two
            # writers can share a misreading (v5b: 6 majority installs, 0
            # graded pass).
            examined = True
            if code_writer.requests_remaining() < EXAMINER_MIN_REQUESTS:
                # An agreement nobody audited is not evidence: over 812 claims
                # that path graded 69%, against 92% when the examiner also
                # found nothing. Unaudited answers ship unclaimed.
                _unclaim(out, "the examiner could not run: "
                              f"{code_writer.requests_remaining()} House requests left")
                break
            verdict, findings = examiner.examine(spec, output_dir, source,
                                                 telemetry)
            if verdict == "unavailable":
                _unclaim(out, "no examiner verdict available")
                break
            # Only an EXECUTED failing check is a finding. v6.1 gate-1: the
            # examiner's prose called two correct answers wrong (invented
            # arithmetic) and the claim rate fell to 1 of 26.
            confirmed = (examiner.verify(findings, output_dir, work_dir,
                                         spec.input_root)
                         if verdict == "wrong" else [])
            if not confirmed:
                # An examiner that DID see something wrong, whose check simply
                # could not be executed, is not a clean bill of health. Over
                # 812 pass claims: "ok, no finding" graded 92%, "findings not
                # confirmed" only 66% - the same precision as no examiner at
                # all. Only the clean verdict earns a claim.
                note = ("no rule violation found" if verdict == "ok" else
                        f"{len(findings)} finding(s) not confirmed by execution")
                out.append(Attempt(index, FailureKind.PASS,
                                   f"[examiner] ok: {note}"
                                   + ("[evidence]" if bucket == AGREE
                                      and verdict == "ok" else " (flagged)"), 0.0))
                break
            findings = confirmed
            # ADVISORY ONLY. Gate-1 run 1 of v6: the examiner's installed
            # rewrites graded 3 of 8, and three of the failures were stable
            # units that were RIGHT before the "fix" - the same shape as
            # v4's reconciliation. A model rewrite is a fresh guess; the
            # examiner's value is its verdict (5 of 5 "ok" graded pass),
            # so its findings downgrade the claim and are recorded, and the
            # floor answer stays untouched.
            out.append(Attempt(index, FailureKind.PASS,
                               "[examiner] wrong (executed checks failed): "
                               + "; ".join(f"{f.get('where', '?')}: {f.get('executed', '')}"
                                           for f in findings)[:600]
                               + " - answer kept (flagged)", 0.0))
            break
        break
    return out


def _unclaim(attempts: list[Attempt], why: str) -> None:
    """Withdraw a pass claim in place, keeping the answer and the reason.

    The deliverables are untouched - only the claim goes. Rewriting the last
    attempt rather than appending another PASS row keeps `triage` and the
    verdict reading the same single row they always read.
    """
    if not attempts or "[evidence]" not in attempts[-1].detail:
        return
    last = attempts[-1]
    attempts[-1] = Attempt(last.index, last.kind,
                           last.detail.replace("[evidence]", "")
                           + f" - not claimed ({why}) (flagged)",
                           last.seconds, last.artifacts)


def _snapshot_tree(root: pathlib.Path) -> dict[str, bytes]:
    """Every file under the output root - not just the contract files.

    The v3 sweep destroyed graded-1.0 answers: a second candidate crashed
    after overwriting deliverables the contract extractor had not listed,
    and a contract-only restore left those overwritten. The vote may only
    ever touch the real output through a whole-tree restore.
    """
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def _restore_tree(root: pathlib.Path, files: dict[str, bytes]) -> None:
    for p in list(root.rglob("*")):
        if p.is_file() and str(p.relative_to(root)) not in files:
            p.unlink()
    for name, payload in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _candidate(spec: TaskSpec, output_dir: pathlib.Path,
               work_dir: pathlib.Path, deadline: Deadline, *, variant: int,
               error: str | None, previous_source: str | None,
               prefer_model: bool, telemetry, index: int
               ) -> tuple[str | None, FailureKind, str, float, dict[str, bytes]]:
    """One independent write-run-check, with the real output dir restored
    afterwards no matter what. Returns (source, kind, detail, secs, tree)
    where `tree` is what the candidate wrote (whole output tree)."""
    before = _snapshot_tree(output_dir)
    started = time.monotonic()
    source, provenance = code_writer.write(
        spec, spec.contracts, str(output_dir), error, prefer_model,
        telemetry=telemetry, previous_source=previous_source,
        attempt_index=index, variant=variant)
    write_seconds = time.monotonic() - started
    if source is None or provenance != "model":
        return (None, FailureKind.MODEL_UNAVAILABLE, "no independent candidate",
                write_seconds, {})
    script = work_dir / f"candidate_{variant}.py"
    script.write_text(source, encoding="utf-8")
    per_attempt = max(30.0, deadline.remaining - 30.0)
    try:
        try:
            kind, detail, seconds = test_runner.run_solution(
                script, work_dir, per_attempt, input_dir=spec.input_root)
        except TypeError:
            kind, detail, seconds = test_runner.run_solution(
                script, work_dir, per_attempt)
        if kind is FailureKind.PASS:
            kind, detail = _self_check(output_dir, spec.contracts, spec)
        produced = _snapshot_tree(output_dir)
    finally:
        _restore_tree(output_dir, before)
    deadline.record_attempt(write_seconds + seconds)
    return source, kind, detail, write_seconds + seconds, produced


def _cross_validate(spec: TaskSpec, output_dir: pathlib.Path,
                    work_dir: pathlib.Path, deadline: Deadline,
                    first_source: str, prefer_model: bool, telemetry,
                    start_index: int) -> list[Attempt]:
    """Vote between the passing answer and an independent second candidate.

    Agree: the first answer stands, now with output-level evidence.
    Disagree: one reconciliation round with the discrepancy spelled out;
    its answer stands if it verifies, else the first answer is restored.
    The first answer is never destroyed - the vote spends budget, not value.
    """
    # Every candidate runs against the real output dir (the instruction may
    # name it absolutely) but `_candidate` restores the whole tree after
    # each one: the first answer is on disk, untouched, at every return
    # below except the one that explicitly installs a reconciled winner.
    first_print = consensus.fingerprint(output_dir, spec.contracts)
    out: list[Attempt] = []
    if not first_print:
        out.append(Attempt(start_index, FailureKind.PASS,
                           "[consensus] no numeric deliverable to compare; "
                           "first answer stands (flagged)", 0.0))
        return out

    source_b, kind, detail, seconds, tree_b = _candidate(
        spec, output_dir, work_dir, deadline, variant=1, error=None,
        previous_source=None, prefer_model=prefer_model, telemetry=telemetry,
        index=start_index)
    if (kind is not FailureKind.PASS and kind.repairable and source_b
            and source_b != first_source and deadline.can_afford()):
        # v3 sweep: 37 of 62 votes died because the second candidate
        # crashed on its first run - no evidence either way, and that path
        # graded at 41% precision. One repair round, exactly as the main
        # loop would grant, turns most of those into a real comparison.
        out.append(Attempt(start_index, kind,
                           f"[consensus] second candidate failed once "
                           f"({detail[:200]}); one repair round", seconds))
        source_b, kind, detail, more, tree_b = _candidate(
            spec, output_dir, work_dir, deadline, variant=1,
            error=f"{kind.value}: {detail}", previous_source=source_b,
            prefer_model=prefer_model, telemetry=telemetry,
            index=start_index + 1)
        seconds += more
    if kind is not FailureKind.PASS or source_b == first_source:
        why = ("identical to the first" if source_b == first_source
               else f"{kind.value}: {detail[:300]}")
        out.append(Attempt(start_index, FailureKind.PASS,
                           f"[consensus] second candidate unusable ({why}); "
                           "first answer stands (flagged)", seconds))
        return out

    second_print = _fingerprint_tree(tree_b, work_dir / "vote_b", spec)
    agree, mismatches = consensus.compare(first_print, second_print)
    if agree is None:
        out.append(Attempt(start_index, FailureKind.PASS,
                           "[consensus] nothing comparable; first answer "
                           "stands (flagged)", seconds))
        return out
    if agree:
        # The one path that earns a `pass` claim: two independently written
        # programs landing on the same numbers (v3/v4 graded precision 70%
        # vs 9-25% on every other path).
        out.append(Attempt(start_index, FailureKind.PASS,
                           f"[consensus][evidence] independent candidates agree "
                           f"on {len(first_print)} quantities; first answer "
                           "stands", seconds))
        return out

    out.append(Attempt(start_index, FailureKind.INVARIANT_VIOLATION,
                       "[consensus] independent candidates DISAGREE: "
                       + "; ".join(mismatches[:6]), seconds))
    if not deadline.can_afford():
        if _prefer_self_consistent(output_dir, tree_b, first_print, second_print, out,
                                   start_index + 1, 0.0, "no budget to reconcile"):
            return out
        out.append(Attempt(start_index + 1, FailureKind.PASS,
                           "[consensus] no budget to reconcile; first answer "
                           "stands (flagged)", 0.0))
        return out

    demand = consensus.RECONCILE_DEMAND.format(
        mismatches="\n".join(mismatches[:20]), other=source_b[:3500])
    source_c, kind, detail, seconds, tree_c = _candidate(
        spec, output_dir, work_dir, deadline, variant=2, error=demand,
        previous_source=first_source, prefer_model=prefer_model,
        telemetry=telemetry, index=start_index + 1)
    if kind is FailureKind.PASS:
        third_print = _fingerprint_tree(tree_c, work_dir / "vote_c", spec)
        with_first, _ = consensus.compare(first_print, third_print)
        with_second, _ = consensus.compare(second_print, third_print)
        if with_first or with_second:
            # A majority: the third computation confirms one of the two.
            # Installing the confirmed one (the third's tree) is the
            # adjudicated answer with evidence behind it.
            _restore_tree(output_dir, tree_c)
            out.append(Attempt(start_index + 1, FailureKind.PASS,
                               "[consensus][evidence] reconciled answer "
                               "verified; sides with the "
                               f"{'first' if with_first else 'second'} "
                               "candidate", seconds))
            return out
        # v4 sweep: 12 of 14 "reconciled" installs agreed with NOBODY and
        # graded 0 - a third different answer is not adjudication, it is a
        # third guess. The first answer stays; the claim is uncertain.
        if _prefer_self_consistent(output_dir, tree_b, first_print, second_print, out,
                                   start_index + 1, seconds, "reconciled answer agrees with neither"):
            return out
        out.append(Attempt(start_index + 1, FailureKind.PASS,
                           "[consensus] reconciled answer agrees with neither "
                           "candidate; first answer stands (flagged)", seconds))
        return out
    if _prefer_self_consistent(output_dir, tree_b, first_print, second_print, out,
                               start_index + 1, seconds, f"reconciliation failed ({kind.value})"):
        return out
    out.append(Attempt(start_index + 1, FailureKind.PASS,
                       f"[consensus] reconciliation failed ({kind.value}); "
                       "first answer stands (flagged)", seconds))
    return out


#: 30 Sep compound-geske: two independent candidates disagreed only on the program's own cross-method
#: error (max_mc_error 0.110 vs 0.011); "first answer stands" kept the worse one and the unit failed
#: on mc_validates. When the vote cannot be settled, the candidate whose OWN error-type fields
#: (error / diff / gap / discrepancy, not standard errors) are clearly smaller is the more
#: self-consistent computation and is installed instead.
_SELF_ERR_TOKENS = {"error", "err", "relerr", "abserr", "diff", "gap", "discrepancy", "deviation",
                    "rmse", "mae"}
_NOT_ERR_TOKENS = {"std", "se", "stderr", "standard"}


def _self_consistency_edge(first_print: dict, second_print: dict) -> int:
    """>0 when the SECOND candidate's own error fields are clearly (2x) smaller more often."""
    import re as _re
    edge = 0
    for key in set(first_print) & set(second_print):
        last = _re.split(r"[.\[\]]", key)[-1] or key
        toks = set(_re.findall(r"[a-z]+", last.lower()))
        if not toks & _SELF_ERR_TOKENS or toks & _NOT_ERR_TOKENS:
            continue
        a, b = abs(first_print[key]), abs(second_print[key])
        if b < 0.5 * a:
            edge += 1
        elif a < 0.5 * b:
            edge -= 1
    return edge


def _prefer_self_consistent(output_dir, tree_b, first_print, second_print, out, index, seconds, why):
    if _self_consistency_edge(first_print, second_print) > 0:
        _restore_tree(output_dir, tree_b)
        out.append(Attempt(index, FailureKind.PASS,
                           f"[consensus] {why}; the second candidate's own cross-method errors are "
                           "clearly smaller - installed it (flagged)", seconds))
        return True
    return False


def _fingerprint_tree(tree: dict[str, bytes], scratch: pathlib.Path,
                      spec: TaskSpec) -> dict[str, float]:
    """Fingerprint a candidate's output tree without touching the real one."""
    scratch.mkdir(parents=True, exist_ok=True)
    _restore_tree(scratch, tree)
    return consensus.fingerprint(scratch, spec.contracts)
