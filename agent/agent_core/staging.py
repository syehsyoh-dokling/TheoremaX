"""Split a heavy unit into short stages, each one a program of its own.

House caps every reply at 4,000 output tokens (~16,000 characters of Python).
The 17 Sep ever-pass run measured what that costs: units asking for three or
more deliverables wrote a median 10,900-16,000 character program, 11 of 27 of
them overran the cap, and 7 of 27 passed. Units asking for one or two files
never overran it and passed 10 of 16. The length, not the finance, is what
fails - and a program cut mid-statement writes NOTHING, so the whole unit
scores zero even where the first two deliverables were already correct.

Staging answers that: deliverables are handed out in groups of at most two,
each group written as a standalone program that reads the inputs itself (and
may read the files earlier stages already wrote). Every stage is short enough
to arrive whole, and a stage that fails costs only its own deliverables.

Cost: one request per stage plus at most one repair each, against the 24 the
House allows - roughly 6 of them for a three-stage unit, where a single long
program routinely spent 5 on continuations alone and still ended broken.
"""

from __future__ import annotations

import os
import pathlib

#: Below this many deliverables the monolith is not the problem: 2-file units
#: never once overran the cap, so staging would add handoffs and buy nothing.
MIN_CONTRACTS_FOR_STAGES = 3
#: Deliverables per stage. Two keeps each program near the length band that
#: actually passes (median 6,200-6,500 characters) - but a unit asking for five
#: or more files carries enough shared preparation that two at a time still
#: overran the cap (batch B: fomc-tone 3 stage programs over it, etf and
#: intraday 2 each), so those go one file at a time.
CONTRACTS_PER_STAGE = 2
MANY_CONTRACTS = 5
#: Where a stage leaves work the next stage can reuse. /tmp is the only writable
#: place in the exam room and it is wiped between units, never graded.
SCRATCH = "/tmp/stage_prep"


def _per_stage(contracts) -> int:
    return 1 if len(contracts) >= MANY_CONTRACTS else CONTRACTS_PER_STAGE
#: More stages than this means more handoffs than the 24-request allowance can
#: repair. Six single-file stages still cost only six writes.
MAX_STAGES = 6


def enabled() -> bool:
    return os.environ.get("T1_STAGING", "on").strip().lower() != "off"


def wanted(contracts) -> bool:
    """Is this unit heavy enough to be worth splitting?

    Deliverable count alone was the wrong test. Batch D: momentum-backtest asks
    for 4 files, passed for months as one program, and FAILED once it was split;
    with staging off it passed again. The count says how much is asked for, not
    whether the reply can carry it - so the count only gates the split, and the
    trigger is the cut itself (see repair_loop: a reply actually truncated at
    the 4,000-token cap).
    """
    return enabled() and len(contracts) >= MIN_CONTRACTS_FOR_STAGES


#: Stage-FIRST (27 Sep full know13 run, 76 units): waiting for a cut before splitting came too late -
#: 24 units were truncated >= 5 times and none passed, the first long reply having spent the requests.
#: Calibrated on that run: >= 7 deliverables, or >= 3 with an instruction of >= 8,000 characters,
#: catches 17 of the 24 heavily cut units and touches NONE of the 22 passes (momentum-backtest, 4
#: files / 5.2k chars, the unit that once broke when split, stays a single program).
FIRST_MIN_CONTRACTS = 7
FIRST_MIN_CONTRACTS_LONG = 3
FIRST_LONG_INSTRUCTION = 8000


def first(contracts, instruction: str) -> bool:
    """Split from the first attempt: the unit is too big to arrive as one reply."""
    if not enabled() or os.environ.get("T1_STAGE_FIRST", "on").strip().lower() == "off":
        return False
    n = len(contracts)
    return n >= FIRST_MIN_CONTRACTS or (n >= FIRST_MIN_CONTRACTS_LONG
                                        and len(instruction or "") >= FIRST_LONG_INSTRUCTION)


def plan(contracts) -> list[tuple]:
    """Deliverables in the order the task named them, grouped into stages.

    Order matters: the task names inputs-then-results-then-summary far more
    often than not, so a later stage can read what an earlier one wrote.
    """
    items = list(contracts)
    per = _per_stage(items)
    count = min(MAX_STAGES, -(-len(items) // per))
    # Even sizes: a last stage carrying the whole tail is the monolith again.
    base, extra = divmod(len(items), count)
    groups, at = [], 0
    for i in range(count):
        take = base + (1 if i < extra else 0)
        groups.append(tuple(items[at:at + take]))
        at += take
    return groups


def note(index: int, stages: list[tuple], output_dir: str) -> str:
    """The stage block for the prompt: scope, and what is REALLY on disk.

    Batch A: 3 attempts died reading a handoff file an earlier stage never
    wrote, because the note promised it unconditionally. A stage that failed
    leaves nothing behind, and the next stage has to be told so.
    """
    root = pathlib.Path(output_dir)
    mine = ", ".join(c.filename for c in stages[index])
    earlier = [c.filename for group in stages[:index] for c in group]
    done = [f for f in earlier if (root / f).is_file()]
    absent = [f for f in earlier if f not in done]
    later = [c.filename for group in stages[index + 1:] for c in group]
    lines = [f"\nSTAGE {index + 1} OF {len(stages)}.",
             "This unit is being answered in stages, each a SHORT standalone "
             "program. Your reply is cut off at 4,000 tokens, so a long program "
             "never arrives; a short one does.",
             f"WRITE IN THIS STAGE, and nothing else: {mine}"]
    if done:
        lines.append("ALREADY WRITTEN by the earlier stages, in the write "
                     f"directory: {', '.join(done)}. Read them from "
                     f"{output_dir} instead of recomputing what they hold, "
                     "and do not modify or delete them.")
    if absent:
        lines.append("An earlier stage FAILED, so these are NOT on disk and you "
                     f"cannot read them: {', '.join(absent)}. Compute whatever you "
                     "need from the inputs yourself.")
    if later:
        lines.append(f"LEFT TO A LATER STAGE (do NOT write these): {', '.join(later)}.")
    lines.append("Your program must be complete on its own: read the inputs it "
                 "needs, compute, write your stage's files, and stop. No "
                 "placeholders, no TODOs, no work deferred to a later stage.")
    # Batch B: 8 stage programs still overran the cap, all on units whose data
    # preparation is heavy - every stage re-read, re-cleaned and re-aligned the
    # same inputs. Preparation done once and saved is the length that comes off.
    if index == 0 and len(stages) > 1:
        lines.append(f"SAVE YOUR PREPARED DATA: after cleaning and aligning the "
                     f"inputs, write the prepared table(s) to {SCRATCH}/ (create "
                     "the directory) as parquet or csv. The later stages read "
                     "them instead of repeating this work.")
    else:
        saved = sorted(p.name for p in pathlib.Path(SCRATCH).glob("*")) \
            if pathlib.Path(SCRATCH).is_dir() else []
        if saved:
            lines.append(f"REUSE THE PREPARED DATA: stage 1 saved these under "
                         f"{SCRATCH}/: {', '.join(saved)}. Read them rather than "
                         "repeating the preparation.")
        else:
            # Batch D: ou-jump's stage 2 died on FileNotFoundError for a scratch
            # file stage 1 never got far enough to write.
            lines.append(f"NOTHING WAS PREPARED: {SCRATCH}/ is empty, so read "
                         "and prepare the raw inputs yourself. Do not open any "
                         "file under that directory.")
    # Last word before the program: the task text above asks for more files than
    # this stage owns, and the task text is the louder instruction.
    lines.append(f"REMINDER, this overrides the file list in the task text: "
                 f"write ONLY {mine}. Any other file is another stage's job.")
    return "\n".join(lines)
