"""know81 group 4: STRUCTURE aid distilled from how a short whole-task program is organised.

A program that solves a 'one call returns every deliverable' task has two steps: make the call (it reads the
inputs itself; the compute takes seconds) and write the files. The know79 lanes spent the budget elsewhere:
the cell scaffold asks the model for one cell per deliverable (1 + N requests, about half a minute each on
the slow lanes), plus a load cell that loaded / sorted inputs by hand. When the time left cannot pay for the
1 + N requests, the scaffold collapses to the program's own two steps - the MODEL still writes both cells
(the call and the emit), nothing is run for it.

Engine-only (not in the model's toolkit API). Self-tests: proven_work/fail72/selftest81_4.py.
"""
from __future__ import annotations

import re

_ENGINE_ONLY = ("frame_lead_of", "bundle_lead", "cells_affordable", "collapse_bundle_steps", "maybe_collapse")

_CALL = re.compile(r"\b[a-z_][a-z0-9_]{3,}\(")
_WHOLE = re.compile(r"\b[a-z][a-z0-9_]*_(?:report|bundle|deliverables|study|suite|auto|oos)\(")


def frame_lead_of(steps):
    """The FRAME LEAD sentence a scaffold put into its per-file briefs ('' when none)."""
    for st in steps or []:
        m = re.search(r"\nFRAME LEAD: (.*)", str(st.get("brief", "")))
        if m:
            return m.group(1).strip()
    return ""


def bundle_lead(lead):
    """The lead when it names ONE decisive toolkit call that writes the deliverables (a 'DECISIVE: ... X(...)
    ... emit(' sentence), else None."""
    s = str(lead or "")
    if not re.match(r"\s*DECISIVE\b", s):
        return None
    if "emit(" in s and _CALL.search(s.split("emit(")[0]):
        return s
    # know82: a DECISIVE lead that names a whole-task toolkit call (a *_report / *_bundle / *_deliverables /
    # *_study / *_suite / *_auto / *_oos function returns every deliverable) is a bundle lead even when the
    # sentence does not spell out the emit loop (know81: fomc / american-fd / mc-greeks / Fama-MacBeth leads
    # were not recognised, the per-file cells re-ran or re-typed the bundle and the units failed).
    if _WHOLE.search(s):
        return s
    return None


def cells_affordable(remaining_sec, elapsed_sec, requests_done, n_cells, safety=1.5):
    """True when the time left pays for n_cells more model requests at the mean seconds per request so far
    (elapsed / requests; the first estimate is the elapsed time itself), with a safety factor."""
    per = float(elapsed_sec) / max(1, int(requests_done))
    return float(remaining_sec) >= n_cells * per * float(safety)


def _required(brief):
    m = re.search(r"required columns/keys: (.*?)\. Keep the cell", str(brief), re.S)
    return (m.group(1).strip() if m else "(as the task describes)")[:400]


def collapse_bundle_steps(steps, lead, max_chars=2600):
    """[load, file_1 .. file_N] -> [call, emit_all]: the call cell makes the lead's call once and keeps its
    result; the emit cell writes every file and checks each against its required columns / keys."""
    files = [st for st in steps if st.get("file")]
    listing = "; ".join(f"{st['file']}: {_required(st.get('brief', ''))}" for st in files)[:max_chars]
    call = {"title": "load_inputs", "file": None,
            "brief": ("STRUCTURE (the time left does not pay for one cell per file): the FRAME LEAD names ONE toolkit "
                      "call that reads the inputs itself and returns every deliverable, so this program has two "
                      "cells. THIS cell makes that call ONCE and keeps its result in a variable (outs / out / r as "
                      "the lead writes it), printing only its keys. Do not load, sort or parse the inputs by hand; "
                      "a glossary / taxonomy / README input is context, not data. Emit nothing yet.\nFRAME LEAD: "
                      + str(lead))}
    emit = {"title": "emit_all", "file": None,
            "brief": ("Emit EVERY deliverable from the previous cell's result (for name, obj in outs.items(): "
                      "emit(name, obj) - or emit('<file>', out) for a single file). Then compare each file with "
                      "its required columns / keys and fix ONLY a missing or misnamed one from the variables you "
                      "have. Required: " + listing)}
    return [call, emit]


def maybe_collapse(steps, remaining_sec, elapsed_sec, requests_done):
    """(steps, collapsed: bool). Collapse only when the matched frame's lead is a whole-task call AND the
    time left cannot pay for the full scaffold; otherwise the scaffold is returned unchanged."""
    lead = bundle_lead(frame_lead_of(steps))
    if not lead or len(steps) < 2:
        return steps, False
    # know82: a whole-task call is collapsed to (call, emit_all) ALWAYS, not only when time runs short.
    # know81 kept the per-file scaffold whenever it was affordable, and the later per-file cells then
    # re-ran the bundle (500k-path MC re-computed in every cell -> out of requests), rebuilt its tables
    # by hand (strings in float columns) or re-typed a file over the bundle's correct one (timestamps
    # lost their date format). One call + one emit of what it returned is the program's own structure.
    return collapse_bundle_steps(steps, lead), True
