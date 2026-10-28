"""Worked examples for a NEW task, chosen by similarity - never by task id.

The exam room's tasks have ids no library entry can carry, so an entry is
useful exactly to the degree that its *content* resembles the task at hand:
same category, overlapping card tags, overlapping instruction vocabulary,
the same kind of deliverable. That is what is scored here, with a small
BM25-style weighting so rare shared terms ("cliquet", "bootstrapping") count
for more than common ones ("returns", "daily").

An example must also share a METHOD with the task: one of its method tags has
to be named in the instruction as a phrase. Vocabulary overlap alone handed
American options a cliquet example and Geske compounds a lookback one, and
the model copied 21-22 lines of the wrong method's helpers (archived arm-on
run, 16 Sep). No example is better than a wrong one.

The entry for the task itself is ALWAYS excluded, even when present. In the
exam that exclusion is a no-op; in local measurement it is what makes the
number honest (leave-one-out): a library answering its own question measures
memory, not transfer.

Offline by construction: no model call, no network, a few hundred lines of
JSON read from the image.
"""
from __future__ import annotations

import collections
import json
import math
import os
import pathlib
import re
from dataclasses import dataclass

_LIBRARY_DIR = pathlib.Path(__file__).resolve().parent.parent / "pattern_library"

STOP = set("""a an the and or of to in for on with by from as is are be this that
these those it its into over under per each all any not no than then when where
which while using use used via must should may can will would output outputs
input inputs file files write read return returns given data value values set
compute calculate""".split())

#: Harness drills (deliverable naming, dtype contracts, canary and echo bait)
#: carry no finance method. Measured on the 45 no-entry units they were handed
#: to NLP, reporting and migration tasks as "worked examples" of nothing.
_DRILL_PREFIXES = ("t1s-",)

#: Tags that name a tool, a file format or a whole field rather than a method.
_GENERIC_TAGS = set("""python numpy scipy pandas polars plotly matplotlib statsmodels csv json
parquet json-input json-config nested-json cross-file multi-deliverable prose-deliverable
options option-pricing backtesting risk-management fixed-income portfolio portfolio-management
calibration estimation mle monotonicity identity numerics tolerance format performance
technical-analysis pnl-analysis adjusted-prices data-cleaning fundamental-analysis
time-series-analysis stochastic risk method edge-cases budget scale""".split())
#: A tag counts as method evidence only if few entries carry it: "var" or
#: "black-scholes" sit on many entries; "cip", "garch" or "early-exercise" do not.
MAX_IDENTITY_DF = 3


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9_\-]{2,}", text.lower())
            if w not in STOP]


def _phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def same_task(entry: dict, instruction: str) -> bool:
    """Is this entry the solution to the task being solved, recognised by CONTENT (not id)?

    Track 1 rule 8 (clarified 25 Sep 2026): our solutions may be examples for OTHER units only,
    however the image recognises the unit - by id or by content. An entry counts as the same task
    when every deliverable file it produced is named in this instruction AND most of its
    instruction keywords occur here too. Excluding a near-twin costs one example; using the
    task's own solution breaks the rule."""
    deliv = {str(d).lower() for d in entry.get("deliverables", []) if d}
    text = (instruction or "").lower()
    if not deliv or not all(d in text for d in deliv):
        return False
    kws = [k for k in entry.get("keywords", []) if k]
    if not kws:
        return True
    words = set(_tokens(text))
    # know78: a keyword inside a hyphenated name ("brenner-subrahmanyam", "newton-raphson") or a short one
    # ("li") is still the same word - _tokens keeps hyphens and drops 2-letter words, which let a proven frame
    # through on its own unit (8 of 17 keywords seen instead of 17)
    words |= set(re.findall(r"[a-z][a-z0-9]*", text))
    return sum(1 for k in kws if k in words) >= 0.6 * len(kws)


def _identity(entry: dict) -> set[str]:
    """What an entry IS about: its method tags, as normalised phrases."""
    return {p for p in (_phrase(t) for t in entry.get("tags", []))
            if p and p not in _GENERIC_TAGS}


def _named_in(phrase: str, instruction_phrase: str) -> bool:
    """The tag appears in the task as a whole phrase, singular or plural:
    "mean-reversion" matches "mean reversion", not "mean" and "reversion"
    in different sentences."""
    return (f"-{phrase}-" in instruction_phrase
            or f"-{phrase}s-" in instruction_phrase)


@dataclass(frozen=True)
class Example:
    task_id: str
    category: str
    tags: tuple[str, ...]
    deliverables: tuple[str, ...]
    source: str
    score: float
    notes: str = ""          # theory card of the solved task (t1_solutions notes.md), if any


def enabled() -> bool:
    return os.environ.get("T1_PATTERNS", "on").strip().lower() != "off"


def load(library_dir: pathlib.Path | None = None) -> list[dict]:
    d = library_dir or _LIBRARY_DIR
    entries = []
    for path in sorted(d.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return entries


def select(instruction: str, category: str, *, task_id: str | None = None,
           k: int = 2, entries: list[dict] | None = None) -> list[Example]:
    """Top-k entries by content similarity, self excluded, method-gated."""
    if not enabled():
        return []
    from .contracts import normalise_category

    entries = load() if entries is None else entries
    entries = [e for e in entries if e.get("task_id") != task_id
               and not str(e.get("task_id", "")).startswith(_DRILL_PREFIXES)
               and not same_task(e, instruction)]
    if not entries:
        return []

    query = _tokens(instruction)
    if not query:
        return []
    q_counts = collections.Counter(query)
    q_phrase = f"-{_phrase(instruction)}-"
    category = normalise_category(category) if category else category

    # Document frequencies over entry vocabularies (keywords + tags + title).
    docs = []
    for e in entries:
        vocab = set(e.get("keywords", [])) | set(_tokens(" ".join(e.get("tags", []))))
        vocab |= set(_tokens(e.get("title", "")))
        docs.append(vocab)
    n = len(docs)
    df = collections.Counter(t for vocab in docs for t in vocab)
    identities = [_identity(e) for e in entries]
    identity_df = collections.Counter(t for ident in identities for t in ident)

    scored = []
    for e, vocab, ident in zip(entries, docs, identities):
        method = [t for t in ident
                  if identity_df[t] <= MAX_IDENTITY_DF and _named_in(t, q_phrase)]
        if not method:
            continue
        s = 0.0
        for term, tf in q_counts.items():
            if term in vocab:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * (tf / (tf + 1.2))          # saturating tf
        if category and normalise_category(e.get("category")) == category:
            s *= 1.5                                    # same category matters most
        scored.append((s, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for s, e in scored[:k]:
        if s <= 0:
            break
        out.append(Example(task_id=e["task_id"], category=e.get("category", ""),
                           tags=tuple(e.get("tags", [])),
                           deliverables=tuple(e.get("deliverables", [])),
                           source=e.get("source", ""), score=round(s, 3),
                           notes=e.get("notes", "")))
    return out


def candidates(instruction: str, category: str, *, task_id: str | None = None,
               k: int = 8, entries: list[dict] | None = None) -> list[Example]:
    """The menu for the model picker: top-k by vocabulary and category, NO method gate.

    Keyword scoring cannot judge whether a method transfers (measured: tightening it fixed
    some units and broke others), so it only shortlists; the model judges meaning.
    """
    from .contracts import normalise_category

    entries = load() if entries is None else entries
    entries = [e for e in entries if e.get("task_id") != task_id
               and not str(e.get("task_id", "")).startswith(_DRILL_PREFIXES)
               and not same_task(e, instruction)]
    query = _tokens(instruction)
    if not entries or not query:
        return []
    q_counts = collections.Counter(query)
    category = normalise_category(category) if category else category
    docs = []
    for e in entries:
        vocab = set(e.get("keywords", [])) | set(_tokens(" ".join(e.get("tags", []))))
        vocab |= set(_tokens(e.get("title", "")))
        docs.append(vocab)
    n = len(docs)
    df = collections.Counter(t for vocab in docs for t in vocab)
    scored = []
    for e, vocab in zip(entries, docs):
        s = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * (tf / (tf + 1.2))
                for t, tf in q_counts.items() if t in vocab)
        if category and normalise_category(e.get("category")) == category:
            s *= 1.5
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [Example(task_id=e["task_id"], category=e.get("category", ""),
                    tags=tuple(e.get("tags", [])), deliverables=tuple(e.get("deliverables", [])),
                    source=e.get("source", ""), score=round(s, 3),
                    notes=e.get("notes", "")) for s, e in scored[:k]]


PICKER_SYSTEM = ("You choose worked examples for a quantitative-finance coding task. "
                 "Reply with one JSON object and nothing else.")
PICKER_TASK_CHARS = 3000


def picker_prompt(instruction: str, menu: list[Example], titles: dict[str, str]) -> str:
    """Task text (head) plus a numbered menu of solved tasks: id, title, method tags."""
    lines = []
    for ex in menu:
        tags = [t for t in ex.tags if _phrase(t) not in _GENERIC_TAGS][:8]
        lines.append(f"- {ex.task_id} | {titles.get(ex.task_id, '')[:90]} | tags: {', '.join(tags)}")
    return ("TASK (beginning):\n" + instruction[:PICKER_TASK_CHARS]
            + "\n\nSOLVED TASKS AVAILABLE AS WORKED EXAMPLES:\n" + "\n".join(lines)
            + "\n\nPick at most 2 solved tasks whose METHOD (model, estimator, pricing or "
              "numerical technique, data procedure) would be directly reused to solve THIS "
              "task. Sharing a topic, asset class or a few words is NOT enough. If none "
              'applies, pick none. Reply exactly: {"pick": ["<id>", ...]}')


def parse_pick(text: str | None, menu: list[Example], max_pick: int = 2) -> list[Example] | None:
    """Examples named in the reply, in menu order of mention; None if the reply is unusable."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        picked = json.loads(match.group(0)).get("pick")
    except (ValueError, AttributeError):
        return None
    if not isinstance(picked, list):
        return None
    by_id = {ex.task_id: ex for ex in menu}
    out = []
    for pid in picked:
        if isinstance(pid, str) and pid.strip() in by_id and by_id[pid.strip()] not in out:
            out.append(by_id[pid.strip()])
    return out[:max_pick]


#: Per-example source budget in the prompt. Whole programs are 60-300 lines;
#: the method is in the middle, the boilerplate at both ends. Raised 25 Sep from 6,000: the
#: verified solutions for the hard units run 11-12.5k characters, and at 6,000 an example was cut
#: before its method. Input tokens carry no per-unit allowance (README rule 1).
MAX_SOURCE_CHARS = 12500
#: Per-example theory-notes budget (know4: notes travel with every verified solution).
NOTES_CHARS = 5000


def render(examples: list[Example]) -> str:
    """The prompt block. Says what an example is FOR, because the failure
    mode of few-shot code is copying the wrong contract."""
    if not examples:
        return ""
    parts = ["\nWORKED EXAMPLES from similar, previously solved tasks. They show "
             "a METHOD and the conventions that were accepted. They are NOT this "
             "task: the deliverable names, shapes and paths come from THIS "
             "instruction only. Adapt the method; never copy a filename or a "
             "column set from an example.\n"]
    for i, ex in enumerate(examples, 1):
        src = ex.source
        if len(src) > MAX_SOURCE_CHARS:
            src = src[:MAX_SOURCE_CHARS] + "\n# ... (truncated)\n"
        parts.append(f"--- example {i}: category={ex.category}; "
                     f"tags={', '.join(ex.tags[:8])}; "
                     f"its deliverables were {', '.join(ex.deliverables) or 'n/a'} ---\n"
                     f"```python\n{src}\n```\n")
        if ex.notes:
            parts.append(f"THEORY NOTES of example {i} (method, formulas, conventions, "
                         f"identities, pitfalls):\n{ex.notes[:NOTES_CHARS]}\n")
    return "".join(parts)
