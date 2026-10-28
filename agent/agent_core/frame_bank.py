"""Matched frames: the textbook problem frames closest to a NEW task, chosen offline by Python.

6 Oct (owner's design): the model is not asked to recognise which formulas a task needs - Python
does it behind the scenes. `frame_bank/<family>.jsonl` holds ~50 generic frames per theory family
(given / asked / formulas / steps, symbolic parameters only). For a task, the families the theory
selector picks are searched first; every frame is scored by how many of its trigger phrases occur
in the task text as whole words, weighted BM25-style so a rare phrase ("curran", "patell") counts
more than a common one ("price", "returns"), with a bonus for title words and for deliverable
words (table, json, per contract) the task also uses. The best 3 frames (within a character
budget) are rendered under MATCHED FRAMES for the plan gate, the code writer and the scaffold.

Offline by construction: no model call, no network; a missing or malformed bank leaves the prompt
as it was. Nothing in the bank comes from a unit's checks or pinned values (frame_bank/README.md).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re

BANK_DIR = pathlib.Path(__file__).resolve().parent.parent / "frame_bank"
TOP_K = 3
BUDGET_CHARS = 3600
MIN_HITS = 2            # a frame needs two trigger phrases in the task (one for a single-trigger title)

_BANK: list[dict] | None = None
_DF: dict[str, int] = {}


def enabled() -> bool:
    return os.environ.get("T1_FRAMES", "on").strip().lower() != "off"


def load() -> list[dict]:
    """Every frame in the bank, with `family` filled from the file stem when absent."""
    global _BANK
    if _BANK is None:
        _BANK = []
        _DF.clear()
        try:
            for path in sorted(BANK_DIR.glob("*.jsonl")):
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        f = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(f, dict) or not f.get("frame"):
                        continue
                    f.setdefault("family", path.stem)
                    f["triggers"] = [str(t).strip().lower() for t in f.get("triggers", []) if str(t).strip()]
                    _BANK.append(f)
            for f in _BANK:
                for t in set(f["triggers"]):
                    _DF[t] = _DF.get(t, 0) + 1
        except OSError:
            _BANK = []
    return _BANK


def _count(phrase: str, text: str) -> int:
    """Whole-word occurrences of `phrase` (singular or plural) in the lower-cased task text."""
    if not phrase:
        return 0
    pat = r"(?<![a-z0-9])" + re.escape(phrase) + r"s?(?![a-z0-9])"
    return len(re.findall(pat, text))


def _score(frame: dict, text: str, n_frames: int, families: set[str]) -> tuple[float, int]:
    hits, score = 0, 0.0
    for t in set(frame["triggers"]):
        n = _count(t, text)
        if not n:
            continue
        hits += 1
        idf = math.log(1.0 + n_frames / (1.0 + _DF.get(t, 1)))
        score += (1.0 + math.log(n)) * idf * (1.5 if " " in t or "-" in t else 1.0)
    title_words = [w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", str(frame.get("title", "")).lower())
                   if w not in ("with", "from", "versus", "option", "options", "price", "model")]
    score += 0.3 * sum(1 for w in title_words if _count(w, text))
    if frame.get("family") in families:
        score *= 1.5           # the theory selector already recognised this family
    if frame.get("proven") and (frame.get("family") in families or hits >= 4):
        score *= 1.4           # 13 Oct: a proven method of the recognised family outranks a textbook frame
    elif frame.get("curated") and (frame.get("family") in families or hits >= 4):
        score *= 1.25          # know72: a curated method (theory + the task family's conventions)
    return score, hits


def _own_task(frame: dict, instruction: str) -> bool:
    """Track 1 rule 8: a PROVEN frame (distilled from our solution to a public unit) may serve
    OTHER units only. Recognised by content, never by id - pattern_retrieval.same_task."""
    if frame.get("curated") and not frame.get("proven"):
        # know72: a curated frame carries theory and the conventions the public instruction states,
        # never results; it is excluded from its own task only when T1_CURATED_OWN=exclude (an
        # honesty switch for local generalisation measurements).
        import os
        if os.environ.get("T1_CURATED_OWN", "allow") != "exclude":
            return False
    elif not frame.get("proven"):
        return False
    try:
        from . import pattern_retrieval
        return pattern_retrieval.same_task(frame, instruction)
    except Exception:  # noqa: BLE001
        return True   # when in doubt, leave the frame out


def select(instruction: str, families: list[str] | tuple[str, ...] | None = None,
           k: int = TOP_K) -> list[dict]:
    """The k best frames for this task text, best first (empty when nothing fits)."""
    if not enabled():
        return []
    bank = load()
    if not bank:
        return []
    text = (instruction or "").lower()
    if families is None:
        # the theory selector's family choice is the first recogniser; the frames refine it
        try:
            from . import theory
            families = families_in(theory.select_library(None, instruction or ""))
        except Exception:  # noqa: BLE001
            families = ()
    fams = set(families or ())
    ranked = []
    for f in bank:
        if _own_task(f, instruction or ""):
            continue
        s, hits = _score(f, text, len(bank), fams)
        if hits >= MIN_HITS or (hits == 1 and len(f["triggers"]) <= 2):
            ranked.append((-s, hits, f))
    ranked.sort(key=lambda r: (r[0], -r[1]))
    out, seen_titles = [], set()
    for _s, _h, f in ranked:
        key = str(f.get("title", ""))[:50].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        out.append(f)
        if len(out) >= k:
            break
    return out


def families_in(theory_text: str) -> list[str]:
    """Family names the theory selector rendered ([family: ...] blocks)."""
    return re.findall(r"\[family: ([a-z0-9\-]+)\]", theory_text or "")


def render(frames: list[dict], budget: int = BUDGET_CHARS) -> str:
    """The prompt block; '' when there is nothing to show."""
    if not frames:
        return ""
    parts, used = [], 0
    if any(f.get("proven") or f.get("curated") for f in frames):
        budget += 1800          # a proven/curated method carries its steps and conventions as well
    for i, f in enumerate(frames, 1):
        tag = (" PROVEN METHOD (passed a graded task of this kind at run time)" if f.get("proven")
               else " CURATED METHOD (theory + the conventions tasks of this kind state)" if f.get("curated") else "")
        block = (f"F{i}. {f.get('title', '')} [{f.get('family', '')}]{tag}\n"
                 f"   {str(f.get('frame', '')).strip()}\n")
        forms = [str(x) for x in f.get("formulas", []) if str(x).strip()]
        if forms:
            block += "   FORMULAS: " + " | ".join(forms[:8]) + "\n"
        if f.get("proven") or f.get("curated"):
            steps = [str(x) for x in f.get("steps", []) if str(x).strip()]
            if steps:
                block += "   STEPS: " + " ".join(f"{j}) {s}" for j, s in enumerate(steps[:12], 1)) + "\n"
            conv = [str(x) for x in f.get("conventions", []) if str(x).strip()]
            if conv:
                block += "   CONVENTIONS THAT MADE IT PASS: " + " | ".join(conv[:6]) + "\n"
        pits = [str(x) for x in f.get("pitfalls", []) if str(x).strip()]
        if pits:
            block += "   PITFALLS: " + " | ".join(pits[:3]) + "\n"
        if used + len(block) > budget:
            if not parts:
                # know77: an over-budget FIRST frame used to make the whole block empty (the FX frames
                # were never shown in know76). Keep its head - title, frame, formulas, the first steps
                # and conventions - cut at a line boundary.
                cut = block[:max(0, budget - used)]
                nl = cut.rfind("\n")
                parts.append((cut[:nl] if nl > len(cut) // 2 else cut) + "\n   [frame truncated]\n")
            break
        parts.append(block)
        used += len(block)
    if not parts:
        return ""
    return ("MATCHED FRAMES (textbook frames closest to THIS task, chosen by its wording; use the one "
            "that fits as the skeleton of the plan - same givens, asked quantities, formulas in order; "
            "the task's own numbers and conventions override anything here):\n" + "".join(parts))


def block(instruction: str, theory_text: str = "", k: int = TOP_K) -> str:
    """One call for the prompt builders: frames matched to the task, rendered."""
    try:
        return render(select(instruction, families_in(theory_text), k))
    except Exception:  # noqa: BLE001 - the bank is an aid, never a failure cause
        return ""
