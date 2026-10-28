"""Phase P - a mandatory answer framework BEFORE any code is written.

The 25 Sep A/B showed where the knowledge build still loses units: programs that run and pass
their own checks but miss one or two graded facts (a convention, an identity, a unit), with the
theory that settles it already in the prompt and unused. This phase makes the model commit, in a
checkable form, to where every quantity comes from before it writes a program:

  deliverables  - every output file and the fields/columns it carries;
  quantities    - every input the method needs, each GIVEN (file/column or instruction line),
                  DERIVED (the formula, citing the THEORY card or a NOTES entry) or CONVENTION
                  (the choice, citing the line that fixes it);
  method        - the ordered steps;
  identities    - the checks that must hold on the output (asserted in the program);
  open_questions- anything unresolved (must be empty to be accepted).

The plan is validated by CODE, not by the model. An incomplete plan is sent back with the list
of what is missing and the FULL theory notes of the most similar solved tasks, with the order to
read them and resolve every gap from them. At most two re-plans; a plan still incomplete after
that is recorded but NOT used - this phase may cost requests, it must never block a unit.
"""
from __future__ import annotations

import json
import time


from . import code_writer, pattern_retrieval, theory
from .contracts import TaskSpec

MAX_REPLANS = 2
_TABULAR = {"csv", "json", "parquet", "tsv", "xlsx"}
NOTES_CHARS = 5000
MIN_IDENTITIES = 2
STATUSES = ("given", "derived", "convention")

SYSTEM = """You plan the solution of a quantitative-finance coding task BEFORE any code is written, as a
worked-problem frame: DIKETAHUI (what is given, with units) - DITANYAKAN (what must be produced) -
RUMUS (the theory/toolkit formula linking them, and the pitfalls avoided) - JAWAB (the method steps
and the checks the answer must pass).
Reply with ONE JSON object and nothing else, keys IN THIS ORDER (a long reply is cut at the end, so
the parts only you can decide come first; the output files are already known to the checker):
{"quantities": [{"name": "<quantity>", "status": "given|derived|convention",
                 "source": "<file:column | instruction line | formula or toolkit call | convention>",
                 "unit": "<given: decimal/percent, per year/per period, days/years, currency, label>",
                 "shape": "<scalar | [N] over what | [T x N] rows x columns | table, one row per what>",
                 "reference": "<the THEORY or NOTES line / instruction sentence that justifies it>"}],
 "variants_chosen": [{"id": "V1", "choice": "a|b|c|n/a", "reference": "<TASK sentence, or the theory default>"}],
 "pitfalls_checked": [{"id": "P1", "relevant": true, "how": "<what this plan does about it>"}],
 "method": ["<step>", ...],
 "identities": ["<a check on THIS task's outputs, written with their column/key names>", ...],
 "unused_inputs": [{"name": "<input key/column>", "reason": "<why it changes no output>"}],
 "open_questions": [],
 "deliverables": [{"file": "<exact name>", "contents": ["<column or key>", ...]}]}
Rules: BUILD THE FRAME ON THE FAMILY'S "Worked frame" when the FAMILY THEORY carries one: it lists
what is typically given and asked in this family, the formula chain and the order of the steps with
the shape of each result. Adapt it to THIS task; do not copy quantities the task does not have.
DIKETAHUI - every given quantity carries its unit AND its shape, and every unit conversion is a
method step. Every derived quantity carries its shape; a step that combines two quantities says how
their shapes are aligned (same dates, same ids, broadcast over which axis).
RUMUS - a derived quantity's formula comes from a THEORY / FAMILY THEORY line (quote it) or a toolkit
call (name it); a formula the task writes out may be quoted from the task. PITFALLS - answer EVERY
numbered pitfall listed in the prompt (relevant or not, and how the plan handles it). VARIANTS -
decide EVERY numbered variant listed in the prompt: the theory default unless a TASK sentence asks for
another option (then quote that sentence); "n/a" only for a method this task does not use. A frame
with an undecided variant is incomplete and will be sent back.
List EVERY input the method needs; list output files whose contents you must decide (files
whose columns/keys the task lists verbatim may be omitted - they are added from the task). A DERIVED quantity
names its formula and cites where it comes from; a CONVENTION cites the sentence or theory line
that fixes it (day count, annualisation, units, sign, rounding, ordering). If something is not
stated, resolve it from the THEORY and NOTES provided - do not leave it open. List at least two
identities that the finished output of THIS task must satisfy, each naming the output
columns/keys it checks. Include the natural range, sign and monotonicity properties of the
quantities (prices >= 0, probabilities within [0, 1], ES beyond VaR, prices monotone in strike or
maturity where theory says so) and any parity or decomposition the method implies. Do not copy
identities from other tasks. Every parameter key and table column in the INPUT EVIDENCE must
appear in the source formula of the quantity it affects (multipliers, units, direction/sign,
day counts, filters, weights) - loading an input is not using it - or be listed in
unused_inputs with the reason it changes no output.
BE CONCISE - the reply is cut at 4,000 tokens: every string one short line, deliverable contents
as bare key/column names, at most ~12 quantities (the ones that carry a convention or a formula),
at most ~10 method steps."""


def _notes_block(spec: TaskSpec) -> str:
    """Full theory notes of the most similar solved tasks (the task's own entry excluded)."""
    entries = pattern_retrieval.load()
    menu = pattern_retrieval.candidates(spec.instruction or "", spec.normalised_category,
                                        task_id=spec.task_id, k=6, entries=entries)
    by_id = {e["task_id"]: e for e in entries}
    parts = []
    for ex in menu:
        notes = (by_id.get(ex.task_id) or {}).get("notes") or ""
        if notes.strip():
            parts.append(f"--- NOTES from a solved task ({ex.category}; tags "
                         f"{', '.join(list(ex.tags)[:6])}) ---\n{notes[:NOTES_CHARS]}")
        if len(parts) >= 2:
            break
    return "\n\n".join(parts)


def _prompt(spec: TaskSpec, notes: str, problems: list[str] | None, previous: str | None,
            pitfalls: list[str] | None = None, variants: list[dict] | None = None) -> str:
    parts = ["TASK:\n" + (spec.instruction or "")[:12000]]
    from . import toolkit_hints
    hint = toolkit_hints.block(spec.instruction or "", code_writer.qf_toolkit_api())
    if hint:
        parts.append(hint + "\n(name these in the quantities' source where they apply)")
    if spec.evidence:
        parts.append("INPUT EVIDENCE (real names, dtypes, keys):\n" + spec.evidence[:5000])
    if spec.contracts:
        parts.append("OUTPUT FILES DETECTED: " + ", ".join(c.filename for c in spec.contracts))
    theory_text = theory.select(spec.normalised_category, spec.instruction or "")
    parts.append("THEORY:\n" + theory_text)
    # 6 Oct: Python recognises the task and hands the model the closest textbook frames; the plan
    # is built on the one that fits (frame_bank.py), not on the model's own recall of a method.
    from . import frame_bank
    frames = frame_bank.block(spec.instruction or "", theory_text)
    if frames:
        parts.append(frames + "(build the plan's DIKETAHUI / DITANYAKAN / RUMUS / JAWAB on the fitting frame)")
    try:   # know81: the conventions reference table, related to this task by its own words
        from qf_toolkit.conventions_ref import conventions_block
        conv = conventions_block(spec.instruction or "")
        if conv:
            parts.append(conv + "\n(record each of these decisions in variants_chosen, quoting the task words)")
    except Exception:  # noqa: BLE001 - a reference aid, never a failure cause
        pass
    if variants:
        parts.append("VARIANTS TO DECIDE from the theory above - decide EVERY one in variants_chosen:\n"
                     + "\n".join(f"V{i + 1}. {v['decision']} Options: {v['options']} Default when silent: "
                                  f"({v['default']}) - {v['source']}" for i, v in enumerate(variants)))
    if pitfalls:
        parts.append("PITFALLS from the theory above - answer EVERY one in pitfalls_checked:\n"
                     + "\n".join(f"P{i + 1}. {p[:300]}" for i, p in enumerate(pitfalls)))
    api = code_writer.qf_toolkit_api()
    if api:
        parts.append("TOOLKIT (tested functions the program will import; when one computes a "
                     "quantity, name it in that quantity's source, e.g. \"toolkit "
                     "barrier_hit_prob(S0, H, T, mu, sigma)\", with the convention keywords):\n" + api)
    if notes:
        parts.append("NOTES (recipes and conventions from similar solved tasks):\n" + notes)
    if problems:
        parts.append("YOUR PREVIOUS PLAN WAS INCOMPLETE:\n" + (previous or "")[:4000]
                     + "\n\nMISSING OR INVALID:\n- " + "\n- ".join(problems)
                     + "\n\nRead the TASK, the THEORY and the NOTES above IN FULL, resolve every "
                       "item listed, and return the complete corrected plan.")
    parts.append("Return the plan JSON now.")
    return "\n\n".join(parts)


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return obj if isinstance(obj, dict) else None
        except ValueError:
            start = text.find("{", start + 1)
    return None


def _base(name) -> str:
    return str(name or "").strip().replace("\\", "/").rsplit("/", 1)[-1]


def validate(plan: dict | None, spec: TaskSpec) -> list[str]:
    """What is missing from a plan; [] means accepted."""
    if not isinstance(plan, dict):
        return ["the reply was not a JSON object"]
    problems: list[str] = []
    # compare by base name: '/app/output/results.json' and 'results.json' are the same deliverable
    files = {_base(d.get("file")) for d in plan.get("deliverables") or [] if isinstance(d, dict)}
    # 27 Sep full run: 9 plans ended "partial" because a many-file plan was cut at the 4,000-token cap
    # and listed only some deliverables, then was rejected 3 times for the same omission. The code
    # already knows the deliverables: a missing one is ADDED from the detected contract (with its
    # detected columns/keys), never a reason to reject the plan.
    if not isinstance(plan.get("deliverables"), list):
        plan["deliverables"] = []
    for c in spec.contracts:
        if c.source == "instruction" and _base(c.filename) not in files:
            plan["deliverables"].append({"file": c.filename,
                                         "contents": list(c.required_columns) or ["as the task states"],
                                         "added_from_contract": True})
    for d in plan.get("deliverables") or []:
        # an image, text or HTML deliverable has no columns (hull-white, fama-french, american-fd
        # plans were rejected over a .png "listing no columns")
        if (isinstance(d, dict) and not d.get("contents")
                and _base(d.get("file")).rsplit(".", 1)[-1].lower() in _TABULAR):
            problems.append(f"deliverable '{d.get('file')}' lists no columns/keys")
    quantities = [q for q in plan.get("quantities") or [] if isinstance(q, dict)]
    if not quantities:
        problems.append("no quantities listed")
    for q in quantities:
        name = str(q.get("name", "?"))
        status = str(q.get("status", "")).lower()
        if status not in STATUSES:
            problems.append(f"quantity '{name}': status must be one of {STATUSES}")
        if not str(q.get("source", "")).strip():
            problems.append(f"quantity '{name}': no source")
        if status in ("derived", "convention") and not str(q.get("reference", "")).strip():
            problems.append(f"quantity '{name}' ({status}): cite the THEORY/NOTES line or "
                            "instruction sentence that fixes it")
    if not plan.get("method"):
        problems.append("no method steps")
    open_q = [str(o) for o in plan.get("open_questions") or [] if str(o).strip()]
    for o in open_q:
        problems.append(f"open question must be resolved from THEORY/NOTES: {o[:160]}")
    return problems


#: 30 Sep batch 1: with the same image the model used the toolkit in one run (compound parity error
#: 1e-13, first-passage hit probabilities right) and re-derived the formulas in the next (parity 10.6,
#: probabilities wrong). A plan that names none of the toolkit calls recommended for the task is sent
#: back once (soft, like grounding); the plan then carries the choice into the code prompt.
TOOLKIT_GATE_MIN_NAMES = 2


def toolkit_problems(plan: dict | None, spec: TaskSpec) -> list[str]:
    import os
    if os.environ.get("T1_TOOLKIT_GATE", "on").strip().lower() == "off" or not isinstance(plan, dict):
        return []
    from . import toolkit_hints
    names = toolkit_hints.names_for(spec.instruction or "")
    if len(names) < TOOLKIT_GATE_MIN_NAMES:
        return []
    text = json.dumps(plan.get("quantities") or []) + " " + json.dumps(plan.get("method") or [])
    if any(n in text for n in names):
        return []
    return ["the plan names no toolkit function, but these tested calls cover this task: "
            + ", ".join(names[:10]) + ". Name the one that computes each quantity in that quantity's "
            "source (e.g. \"toolkit " + names[0] + "(...)\") and use it instead of re-deriving the formula; "
            "keep a hand-written formula only where no listed call applies"]


#: 30 Sep owner's framework - DIKETAHUI / DITANYAKAN / RUMUS / JAWAB. The batch-1 traces showed plans
#: citing the task text almost exclusively (theory cards ~0 of 20 references) while the card already
#: carried the fact the run got wrong (compound parity needs put-on-CALL). Three code checks make the
#: theory part of the frame real; all are soft (sent back, then accepted):
#:   DIKETAHUI - every given quantity states its unit ("0.05 decimal per year", "percent", "days")
#:   RUMUS     - a derived quantity's formula comes from a THEORY/NOTES line or a toolkit call, or is a
#:               formula the task itself writes (the reference then contains "=")
#:   PITFALLS  - the numbered pitfalls of the selected theory cards are each answered (relevant? how)
MAX_PITFALLS = 6    # 30 Sep know26: 10 pitfalls + 12 variants cut the plan at 4,000 tokens


def pitfalls_for(spec: TaskSpec) -> list[str]:
    """Pitfall lines of the theory cards selected for this task, most relevant to its text first."""
    import re
    text = theory.select_library(spec.normalised_category, spec.instruction or "")
    lines, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line.strip().lower().startswith("## pitfalls")
            continue
        if on and line.strip().startswith("- "):
            lines.append(line.strip()[2:].strip())
    words = set(re.findall(r"[a-z]{4,}", (spec.instruction or "").lower()))
    ranked = sorted(lines, key=lambda l: -len(set(re.findall(r"[a-z]{4,}", l.lower())) & words))
    return ranked[:MAX_PITFALLS]


#: 30 Sep owner's rule: a frame that leaves a method variant undecided is INCOMPLETE and must not be
#: solved yet (event-study: two Corrado statistics off by the same factor on every run - one silent
#: normalisation choice). Theory cards list "## Variants to decide" (options + textbook default); the
#: plan must decide each listed variant, and a choice other than the default needs a TASK sentence.
MAX_VARIANTS = 8   # know74: method cards add their own decisions
_VARIANT = __import__("re").compile(
    r"^- V\d+ ([a-z0-9_]+): (.+?) Options: (.+?) Default when silent: \(([a-z])\) - (.+)$")


def variants_for(spec: TaskSpec) -> list[dict]:
    """Decision points of the theory cards selected for this task, most relevant first."""
    import re
    text = theory.select_library(spec.normalised_category, spec.instruction or "")
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            on = line.strip().lower().startswith("## variants to decide")
            continue
        m = _VARIANT.match(line.strip()) if on else None
        if m:
            out.append({"key": m.group(1), "decision": m.group(2), "options": m.group(3),
                        "default": m.group(4), "source": m.group(5), "line": line.strip()[2:]})
    words = set(re.findall(r"[a-z]{4,}", (spec.instruction or "").lower()))
    # variants the TASK itself speaks to first (they need a task quote), then by word overlap
    out.sort(key=lambda v: (0 if _topical_sentence(v, spec.instruction or "") else 1,
                            -len(set(re.findall(r"[a-z]{4,}", v["line"].lower())) & words)))
    return out[:MAX_VARIANTS]


_GENERIC = {"default", "options", "silent", "option", "value", "values", "using", "which", "their",
            "where", "there", "these", "those", "with", "from", "each", "every", "other", "rather", "method",
            "standard", "returns", "return", "series", "price", "prices", "model", "rate", "rates", "daily",
            "annual", "period", "periods", "number", "should", "under", "after", "before", "first", "second"}


def _topical_sentence(variant: dict, task_text: str) -> str:
    """A TASK sentence about this variant's decision: >= 2 shared distinctive words (5+ letters), or
    one shared proper-name word (an author/test name such as kolari, corrado, patell)."""
    import re
    vtoks = {w for w in re.findall(r"[a-z]{5,}", (variant.get("key", "") + " " + variant.get("decision", "")
                                                  + " " + variant.get("options", "")).lower().replace("_", " "))
             if w not in _GENERIC}
    names = {w for w in re.findall(r"[a-z]{5,}", variant.get("source", "").lower()) if w not in _GENERIC}
    best, best_score = "", 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n\s*[-*]\s+|\n{2,}", task_text or ""):
        stoks = set(re.findall(r"[a-z]{5,}", sentence.lower()))
        score = len(stoks & vtoks) + 2 * len(stoks & names & vtoks)
        if (len(stoks & vtoks) >= 2 or stoks & names & vtoks) and score > best_score:
            best, best_score = " ".join(sentence.split()), score
    return best


def variant_problems(plan: dict | None, variants: list[dict], task_text: str) -> list[str]:
    if not variants or not isinstance(plan, dict):
        return []
    chosen = {}
    for item in plan.get("variants_chosen") or []:
        if isinstance(item, dict):
            chosen[str(item.get("id", "")).strip().upper()] = item
    missing, unsupported, unread = [], [], []
    words = _norm_words(task_text)
    grams = {" ".join(words[i:i + 4]) for i in range(len(words) - 3)}
    for i, v in enumerate(variants):
        vid = f"V{i + 1}"
        item = chosen.get(vid)
        choice = str((item or {}).get("choice", "")).strip().lower().strip("()")
        if not item or not choice:
            missing.append(vid)
            continue
        ref = _norm_words(str(item.get("reference", "")))
        quotes_task = any(" ".join(ref[j:j + 4]) in grams for j in range(max(0, len(ref) - 3)))
        # 30 Sep know26: the model answered every variant "(a)" and pasted the theory line even where
        # the TASK has a sentence about that very decision. If it does, the frame must quote it.
        topical = _topical_sentence(v, task_text)
        if topical and not quotes_task and choice not in ("n/a", "na", "none", "not used"):
            unread.append((vid, topical))
            continue
        if choice in ("n/a", "na", "none", "not used") or choice == v["default"]:
            continue
        if not quotes_task:
            unsupported.append(vid)
    problems = []
    for vid, sentence in unread[:6]:
        problems.append(f"FRAME INCOMPLETE - the TASK speaks to {vid} (\"{sentence[:220]}\"): read that "
                        f"sentence, decide {vid} from it, and quote it word for word as the reference")
    if missing:
        problems.append("FRAME INCOMPLETE - re-read the THEORY 'Variants to decide' and decide "
                        f"{', '.join(missing)} in variants_chosen: "
                        "{\"id\": \"V#\", \"choice\": \"a|b|c|n/a\", \"reference\": \"<the TASK sentence that "
                        "fixes it, or the theory default line>\"}")
    if unsupported:
        problems.append(f"FRAME INCOMPLETE - {', '.join(unsupported)} choose a non-default option without "
                        "quoting a TASK sentence that asks for it: quote that sentence word for word, or use "
                        "the theory default")
    return problems


def pitfall_problems(plan: dict | None, pitfalls: list[str]) -> list[str]:
    if not pitfalls or not isinstance(plan, dict):
        return []
    answered = set()
    for item in plan.get("pitfalls_checked") or []:
        if isinstance(item, dict) and str(item.get("how", "")).strip():
            answered.add(str(item.get("id", "")).strip().upper())
    missing = [f"P{i + 1}" for i in range(len(pitfalls)) if f"P{i + 1}" not in answered]
    if not missing:
        return []
    return [f"pitfalls_checked must answer EVERY listed theory pitfall ({', '.join(missing)} missing): "
            "{\"id\": \"P#\", \"relevant\": true|false, \"how\": \"what this plan does about it\"}"]


def unit_problems(plan: dict | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    missing = [str(q.get("name", "?")) for q in plan.get("quantities") or []
               if isinstance(q, dict) and str(q.get("status", "")).lower() == "given"
               and not str(q.get("unit", "")).strip()]
    if not missing:
        return []
    return [f"DIKETAHUI: given quantities {missing[:8]} state no unit - add \"unit\" to each (decimal or "
            "percent, per year or per period, days or years, currency, 'count'/'label' for non-numbers) "
            "and a method step for every conversion"]


def shape_problems(plan: dict | None) -> list[str]:
    """2 Oct: in a 12-unit sample 38 of 89 crashes were arrays of mismatched shape ("operands
    could not be broadcast", "All arrays must be of the same length"). The frame now states the
    shape of every given and derived quantity (the family's Worked frame shows how), so the
    mismatch is visible in the plan instead of in a traceback."""
    if not isinstance(plan, dict):
        return []
    missing = [str(q.get("name", "?")) for q in plan.get("quantities") or []
               if isinstance(q, dict) and str(q.get("status", "")).lower() in ("given", "derived")
               and not str(q.get("shape", "")).strip()]
    if not missing:
        return []
    return [f"DIKETAHUI/RUMUS: quantities {missing[:8]} state no shape - add \"shape\" to each "
            "(scalar, [N] over what, [T x N] rows x columns, table with one row per what), as the "
            "family's Worked frame does; two quantities combined in one formula must have "
            "compatible shapes, and a method step must say how they are aligned"]


def formula_problems(plan: dict | None, g: dict) -> list[str]:
    """RUMUS: derived formulas sourced only from the task text, with no formula written there."""
    if not isinstance(plan, dict):
        return []
    try:
        import qf_toolkit
        tk_names = {n for n in dir(qf_toolkit) if not n.startswith("_") and callable(getattr(qf_toolkit, n))}
    except Exception:  # noqa: BLE001
        tk_names = set()
    sections = (g or {}).get("sections") or {}
    weak = []
    for q in plan.get("quantities") or []:
        if not isinstance(q, dict) or str(q.get("status", "")).lower() != "derived":
            continue
        name = str(q.get("name", "?"))[:60]
        blob = f"{q.get('source', '')} {q.get('reference', '')}"
        if any(n in blob for n in tk_names if len(n) > 5):
            continue                                   # a tested toolkit call carries the formula
        if sections.get(name) in ("theory", "family_theory", "notes"):
            continue                                   # quoted from a theory card / solved-task note
        if "=" in str(q.get("reference", "")) or "=" in str(q.get("source", "")):
            continue                                   # the task (or the plan) writes the formula out
        weak.append(name)
    if not weak:
        return []
    return [f"RUMUS: derived quantities {weak[:8]} have no formula source - for each, write the formula "
            "and QUOTE the THEORY / FAMILY THEORY line it comes from, or name the toolkit call that "
            "computes it"]


def identity_problems(plan: dict | None) -> list[str]:
    """Soft requirement (27 Sep: 4 plans that were otherwise complete - realized-vol, which then
    PASSED, hull-white, ipca, earnings-surprise - were rejected 3 times over identities alone): sent
    back like a grounding gap, accepted after the re-plans. An identity counts only if it names
    something this task outputs (a file, column or key); one copied from another task checks nothing."""
    if not isinstance(plan, dict):
        return []
    names = {n.lower() for d in plan.get("deliverables") or [] if isinstance(d, dict)
             for n in [_base(d.get("file")).rsplit(".", 1)[0], *map(str, d.get("contents") or [])]
             if len(n) >= 2}
    bound = [i for i in plan.get("identities") or []
             if any(n in str(i).lower() for n in names)]
    if len(bound) < MIN_IDENTITIES:
        return [f"list at least {MIN_IDENTITIES} identities that name this task's output "
                "columns/keys (range, sign, monotonicity, parity or decomposition checks)"]
    return []


def render(plan: dict) -> str:
    """The accepted plan, as the code prompt shows it."""
    return json.dumps(plan, indent=1)[:9000]


#: Per-unit trace of Phase P, written into agent_report.json by agent.py (key "plan_trace"):
#: every round's problems, which solved tasks' notes were put in front of the model, and the
#: final status. It is what separates "our instructions/theory were missing" from "the model
#: had a complete, validated plan and still computed wrong".
TRACE: dict = {}


def _notes_sources(spec: TaskSpec) -> list[str]:
    entries = pattern_retrieval.load()
    menu = pattern_retrieval.candidates(spec.instruction or "", spec.normalised_category,
                                        task_id=spec.task_id, k=6, entries=entries)
    by_id = {e["task_id"]: e for e in entries}
    return [ex.task_id for ex in menu if (by_id.get(ex.task_id) or {}).get("notes")][:2]


def _norm_words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", (text or "").lower())


#: Section headers of the plan prompt, to name WHERE a reference was found.
_SECTIONS = (("TASK:", "instruction"), ("INPUT EVIDENCE", "evidence"), ("OUTPUT FILES", "contracts"),
             ("THEORY:", "theory"), ("FAMILY THEORY", "family_theory"), ("NOTES", "notes"))


def grounding(plan: dict | None, shown: str) -> dict:
    """Is every DERIVED / CONVENTION quantity's `reference` text actually in what the model was
    shown (instruction, evidence, theory cards, family library, notes)? A reference counts as
    grounded when at least half of its word 4-grams (or, for 3-5 word references, the whole
    phrase) occur in the shown text. Diagnostic only - it never blocks a plan.
    Returns {n, grounded, ungrounded: [...], no_reference: [...], by_section: {...}}."""
    out = {"n": 0, "grounded": 0, "ungrounded": [], "no_reference": [], "by_section": {}}
    if not isinstance(plan, dict):
        return out
    words = _norm_words(shown)
    joined = " " + " ".join(words) + " "
    grams = {" ".join(words[i:i + 4]) for i in range(len(words) - 3)}
    # section boundaries on the normalised text
    bounds = []
    for header, name in _SECTIONS:
        pos = shown.find(header)
        if pos >= 0:
            bounds.append((len(_norm_words(shown[:pos])), name))
    bounds.sort()

    def section_of(word_index: int) -> str:
        name = "other"
        for start, sec in bounds:
            if word_index >= start:
                name = sec
        return name

    for q in plan.get("quantities") or []:
        if not isinstance(q, dict):
            continue
        status = str(q.get("status", "")).lower()
        if status not in ("derived", "convention"):
            continue
        out["n"] += 1
        name = str(q.get("name", "?"))[:60]
        ref = _norm_words(str(q.get("reference", "")))
        if len(ref) < 3:
            out["no_reference"].append(name)
            continue
        where = -1
        if len(ref) <= 5:
            phrase = " " + " ".join(ref) + " "
            hit = joined.find(phrase)
            ok = hit >= 0
            if ok:
                where = joined[:hit].count(" ")
        else:
            # word coverage by 4-grams found in the shown text: a label the model prefixes
            # ("THEORY fixed-income:") must not sink a genuine quotation
            covered = [False] * len(ref)
            first = None
            for i in range(len(ref) - 3):
                g = " ".join(ref[i:i + 4])
                if g in grams:
                    first = first or g
                    for j in range(i, i + 4):
                        covered[j] = True
            ok = first is not None and sum(covered) * 5 >= len(ref) * 2     # >= 40% of words
            if ok:
                where = joined.find(" " + first + " ")
                where = joined[:where].count(" ") if where >= 0 else -1
        if ok:
            out["grounded"] += 1
            sec = section_of(where) if where >= 0 else "other"
            out["by_section"][sec] = out["by_section"].get(sec, 0) + 1
            out.setdefault("sections", {})[name] = sec
        else:
            out["ungrounded"].append(name)
            if status == "convention":
                out["_conventions_ungrounded"] = True
    return out


#: Grounding gate (26 Sep): the know6 diagnosis showed plans citing the task text almost only and the
#: theory cards almost never (kelly: 0 of 14 references found anywhere). A structurally complete
#: plan is sent back once more when a CONVENTION is ungrounded or under half of the derived and
#: convention quantities are grounded. After the re-plans it is accepted anyway (a unit is never
#: blocked), flagged "accepted_ungrounded" in the trace. T1_GROUNDING_GATE=off disables it.
GROUNDING_MIN_SHARE = 0.5


def grounding_problems(g: dict) -> list[str]:
    import os
    if os.environ.get("T1_GROUNDING_GATE", "on").strip().lower() == "off" or not g.get("n"):
        return []
    bad = list(g.get("ungrounded", [])) + list(g.get("no_reference", []))
    if not bad:
        return []
    share = g.get("grounded", 0) / g["n"]
    if share >= GROUNDING_MIN_SHARE and not g.get("_conventions_ungrounded"):
        return []
    return [f"quantity '{name}': its reference must QUOTE, word for word (at least five consecutive "
            "words), the TASK sentence, the THEORY or FAMILY THEORY line, or the NOTES line that "
            "fixes it - not a paraphrase or your own explanation. If no line fixes it, choose the "
            "convention the THEORY/FAMILY THEORY states for this situation and quote that line."
            for name in bad[:12]]


#: Input-usage audit (26 Sep): the know6-know8 runs showed plans that LOAD a parameter or column and
#: then never use it in any formula (fx: the pip multipliers and the trade direction were read, the
#: forward points came out unscaled and the P&L sign inverted). Every parameter-file key and every
#: column of a small table must reach a quantity or a computing step, or be declared irrelevant
#: under "unused_inputs" with a reason. Like the grounding gate it sends the plan back and never
#: blocks it. T1_INPUT_AUDIT=off disables it.
_META_KEYS = {"description", "notes", "note", "comment", "comments", "version", "author", "title",
              "source", "sources", "citation", "readme", "doc", "docs", "metadata", "about"}
_ID_LIKE = ("id", "date", "time", "timestamp", "name", "ticker", "symbol", "description", "comment",
            "label", "index", "unnamed")
#: bar columns a close-only method legitimately ignores (a range estimator names them anyway)
_OHLCV = {"open", "high", "low", "volume", "adj_close", "adj close", "adjclose"}
USAGE_MAX_COLUMNS = 14
USAGE_MAX_REPORTED = 8
_LOAD_VERBS = ("load", "read", "parse", "import", "open")


def input_fields(spec: TaskSpec) -> list[tuple[str, str]]:
    """(field, file) for every parameter-file key (depth <= 2) and every non-identifier column of
    a table with at most USAGE_MAX_COLUMNS columns. Reads headers and JSON only - no data rows."""
    import csv
    import pathlib
    root = pathlib.Path(spec.input_root or "/input")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    import re

    def add(key, fname):
        k = str(key).strip()
        low = k.lower()
        # "_"-prefixed keys are bookkeeping (never inputs); all-caps short names are asset tickers
        if (len(k) < 2 or low in seen or low in _META_KEYS or k.isdigit() or k.startswith("_")
                or re.fullmatch(r"[A-Z0-9^.=/\-]{1,8}", k)):
            return
        seen.add(low)
        out.append((k, fname))

    def walk(obj, fname):
        # parameter names only: top-level keys, the keys of records in a top-level list, and one
        # level into a thin wrapper ({"params": {...}}); deeper keys are data (pairs, tenors)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            return
        items = list(obj.items())[:60]
        for k, v in items:
            add(k, fname)
        if len(items) <= 2:
            for k, v in items:
                if isinstance(v, dict) and str(k).lower() not in _META_KEYS:
                    for kk in list(v)[:60]:
                        add(kk, fname)

    for rel in spec.data_files or ():
        path = root / rel
        suf = path.suffix.lower()
        if "manifest" in path.name.lower():
            continue
        try:
            if suf == ".json" and path.stat().st_size <= 200_000:
                walk(json.loads(path.read_text(encoding="utf-8", errors="replace")), path.name)
            elif suf in (".csv", ".tsv"):
                with path.open(encoding="utf-8", errors="replace", newline="") as fh:
                    header = next(csv.reader(fh, delimiter="\t" if suf == ".tsv" else ","), [])
                if 0 < len(header) <= USAGE_MAX_COLUMNS:
                    for col in header:
                        low = col.strip().lower()
                        if low in _OHLCV or any(low == t or low.endswith("_" + t)
                                                or low.startswith(t + "_") for t in _ID_LIKE):
                            continue
                        add(col, path.name)
        except (OSError, ValueError, StopIteration):
            continue
    return out


def _mentions(text: str, field: str) -> bool:
    import re
    f = field.lower()
    variants = {f, f.replace("_", " "), f.replace(" ", "_")}
    return any(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text) for v in variants if v)


def usage_problems(plan: dict | None, spec: TaskSpec) -> list[str]:
    import os
    if os.environ.get("T1_INPUT_AUDIT", "on").strip().lower() == "off" or not isinstance(plan, dict):
        return []
    fields = input_fields(spec)
    if not fields:
        return []
    used_text = " ".join(json.dumps(q) for q in plan.get("quantities") or [] if isinstance(q, dict))
    used_text += " " + " ".join(str(m) for m in plan.get("method") or []
                                if not str(m).strip().lower().startswith(_LOAD_VERBS))
    declared = " ".join(json.dumps(u) for u in plan.get("unused_inputs") or [])
    used_text, declared = used_text.lower(), declared.lower()
    unused = [(k, f) for k, f in fields if not _mentions(used_text, k) and not _mentions(declared, k)]
    return [f"input '{k}' ({f}) is loaded but enters no quantity or computing step: show in a "
            "quantity's source formula WHERE it enters the calculation (a multiplier or unit, a sign "
            "or direction, a day count, a filter, a weight), or list it under \"unused_inputs\": "
            "[{\"name\": ..., \"reason\": ...}] with why it changes no output"
            for k, f in unused[:USAGE_MAX_REPORTED]]


def _override_plan(spec: TaskSpec) -> str | None:
    """Control arm (dev only): T1_PLAN_OVERRIDE=<json file {task_id: plan text}> hands the model a
    given plan instead of its own - separating 'the framework was filled wrongly' from 'the code
    was written wrongly'. Unset or no entry for this task: None."""
    import os
    path = os.environ.get("T1_PLAN_OVERRIDE", "").strip()
    if not path:
        return None
    try:
        return (json.loads(open(path, encoding="utf-8").read()).get(spec.task_id) or None)
    except (OSError, ValueError):
        return None


def run(spec: TaskSpec, telemetry: list | None = None, deadline=None) -> str:
    """Build and validate the plan; returns a note. Sets spec.plan only for an accepted plan."""
    TRACE.clear()
    TRACE.update(status="skipped", rounds=[], notes_from=[], plan=None, grounding=None)
    override = _override_plan(spec)
    if override:
        spec.plan = str(override)[:9000]
        TRACE.update(status="override", plan={"text": spec.plan})
        return "plan: control-arm override (T1_PLAN_OVERRIDE)"
    if not code_writer.model_available():
        return "plan skipped: no model"
    notes = _notes_block(spec)
    try:
        pitfalls = pitfalls_for(spec)
    except Exception:  # noqa: BLE001 - the frame never blocks a unit
        pitfalls = []
    TRACE["pitfalls"] = len(pitfalls)
    try:
        variants = variants_for(spec)
    except Exception:  # noqa: BLE001 - the frame never blocks a unit
        variants = []
    TRACE["variants"] = len(variants)
    TRACE["notes_from"] = _notes_sources(spec)
    plan, problems, previous = None, None, None
    best = None     # (gap count, plan, grounding, gaps) of the best structurally complete plan
    shown = ""
    unparsed = 0
    round_secs: list[float] = []
    try:
        from qf_toolkit import helpers80_4 as _h80
    except Exception:  # noqa: BLE001
        _h80 = None
    for attempt in range(1 + MAX_REPLANS):
        # know80: plan time box - the first round always runs (the model always plans); a RE-plan starts only
        # while the plan phase stays inside its share of the budget, so the cells keep their time
        if (attempt >= 1 and deadline is not None and _h80 is not None
                and not _h80.plan_round_affordable(deadline.elapsed, deadline.budget, round_secs, rounds_done=attempt)):
            TRACE["rounds"][-1]["stopped"] = "plan time box"
            break
        t0 = time.monotonic()
        prompt = _prompt(spec, notes, problems, previous, pitfalls, variants)
        shown = shown or prompt
        # know80: no continuation chase for a plan reply - a cut plan is salvaged and re-asked SHORTER
        reply = code_writer.complete(SYSTEM, prompt, telemetry=telemetry, variant=30 + attempt,
                                     want_code=False, continuations=0)
        round_secs.append(time.monotonic() - t0)
        cut = False
        if _h80 is not None:
            try:
                candidate, cut = _h80.parse_plan_reply(reply)
            except Exception:  # noqa: BLE001
                candidate = _parse(reply)
        else:
            candidate = _parse(reply)
        found = validate(candidate, spec)
        if cut and _h80 is not None:
            found = [_h80.plan_cut_problem(candidate)] + list(found)
        TRACE["rounds"].append({"round": attempt + 1, "parsed": candidate is not None,
                                "problems": found[:20], "cut": cut,
                                "forced_reading": bool(problems)})
        if candidate is not None and (plan is None or len(found) <= len(problems or [])):
            plan = candidate
        # know79: two replies in a row with no parsable plan (cut at the length limit) - more re-plans only
        # burn the request budget the cells need (10-K / multimodal runs never reached a code cell)
        unparsed = (unparsed + 1) if (candidate is None or cut) else 0
        if unparsed >= 2 and best is None:
            TRACE["rounds"][-1]["stopped"] = "two unparsed plan replies"
            break
        if not found:
            g = grounding(candidate, shown + "\n" + notes)
            unused = usage_problems(candidate, spec)
            gaps = (variant_problems(candidate, variants, spec.instruction or "")
                    + toolkit_problems(candidate, spec) + pitfall_problems(candidate, pitfalls)
                    + formula_problems(candidate, g) + identity_problems(candidate)
                    + grounding_problems(g) + unit_problems(candidate)
                    + shape_problems(candidate) + unused)
            TRACE["rounds"][-1]["grounding"] = {"grounded": g["grounded"], "n": g["n"]}
            TRACE["rounds"][-1]["unused_inputs"] = len(unused)
            # 26 Sep fx run: round 2 left 1 unused input, round 3 went back to 8 and was the one
            # accepted. Keep the complete plan with the fewest gaps, not the last one.
            if best is None or len(gaps) < best[0]:
                best = (len(gaps), candidate, g, gaps)
            if gaps and attempt < MAX_REPLANS:
                # structurally complete, but conventions/derivations cite nothing the model was
                # shown: send it back to QUOTE the task, theory or notes line it relies on
                problems, previous = gaps, (reply or "")[:4000]
                continue
            _, candidate, g, gaps = best
            spec.plan = render(candidate)
            TRACE.update(status="accepted" if not gaps else "accepted_ungrounded", plan=candidate,
                         grounding=g)
            return f"plan accepted after {attempt + 1} request(s)" + (" (ungrounded)" if gaps else "")
        problems, previous = found, (reply or "")[:4000]
    if best is not None:
        # a re-plan broke the structure after an earlier round had passed it: use that round
        _, candidate, g, gaps = best
        spec.plan = render(candidate)
        TRACE.update(status="accepted_earlier_round" if not gaps else "accepted_ungrounded",
                     plan=candidate, grounding=g)
        return f"plan accepted from an earlier round ({len(gaps)} gaps)"
    if plan is not None:
        # a plan that never passed validation is NOT handed to the code writer: an unvalidated
        # framework (wrong file names, borrowed identities) cost units in the 25 Sep diagnosis
        TRACE.update(status="partial", plan=plan, grounding=grounding(plan, shown + "\n" + notes))
        return f"plan partial after {1 + MAX_REPLANS} requests ({len(problems or [])} gaps left); not used"
    TRACE["status"] = "failed"
    return "plan failed: no usable reply"
