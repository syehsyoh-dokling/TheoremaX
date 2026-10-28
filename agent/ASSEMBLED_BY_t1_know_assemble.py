"""Assemble the T1 'knowledge only' build context (upload of 24 Sep).

  * code: the uploaded v1 image's agent (its House route is the one proven live)
  * REMOVED: _canonicalize_deliverables (per-unit literal results, and 'IMPERFECT'->'PERFECT'
    rewriting), every batch*_recipes module, data_assets/, any solution_library
  * ADDED: pattern_library/ (v2: worked examples chosen by similarity, own entry excluded) and
    agent_core/theory.py (formula cards + the missing-input procedure), injected into every prompt
Then scan the pattern library for entries that embed long literal number runs (a sign of a copied
answer rather than a method) and report them.
"""
import pathlib
import re
import shutil

SP = pathlib.Path(__file__).resolve().parent
V1 = SP / "t1man" / "v1"
V2 = SP / "t1man" / "v2"
CTX = SP / "t1_know" / "ctx"
if CTX.exists():
    shutil.rmtree(CTX)
CTX.mkdir(parents=True)

ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
for d in ("agent_core", "invariant_library", "qf_toolkit"):
    shutil.copytree(V1 / d, CTX / d, ignore=ignore)
shutil.copytree(V2 / "pattern_library", CTX / "pattern_library", ignore=ignore)
shutil.copy(SP / "t1_know" / "theory.py", CTX / "agent_core" / "theory.py")

# ---- agent.py without the answer-writing function and its call
src = (V1 / "agent.py").read_text(encoding="utf-8")
start = src.index("def _canonicalize_deliverables(")
end = src.index("def _audit_contracts(")
src = src[:start] + src[end:]
call = "        _canonicalize_deliverables(out_dir, diagnostics, task_id=task_id, inp_dir=resolved_inp)\n"
assert src.count(call) == 1, "call site not found exactly once"
src = src.replace(call, "")
for banned in ("_canonicalize_deliverables", "data_assets", "verifier_data", "_recipes", "task_id =="):
    assert banned not in src, f"agent.py still mentions {banned}"
(CTX / "agent.py").write_text(src, encoding="utf-8")

# ---- inject the theory cards next to the playbook
cw = (CTX / "agent_core" / "code_writer.py").read_text(encoding="utf-8")
old_imp = "from . import consensus, pattern_retrieval, playbook, solution_memory"
assert cw.count(old_imp) == 1
cw = cw.replace(old_imp, "from . import consensus, pattern_retrieval, playbook, solution_memory, theory")
anchor = "                 + playbook.select(spec.normalised_category, spec.instruction))\n"
assert cw.count(anchor) == 1
cw = cw.replace(anchor, anchor +
    "    # Theory cards: the category's formulas, conventions and identities, and the procedure\n"
    "    # for incomplete inputs. Knowledge to derive from - never a unit's values.\n"
    "    parts.append(\"\\nTHEORY (formulas and conventions for this kind of task - derive the \"\n"
    "                 \"result from these and the inputs; state any assumption you make):\\n\"\n"
    "                 + theory.select(spec.normalised_category, spec.instruction))\n")
(CTX / "agent_core" / "code_writer.py").write_text(cw, encoding="utf-8")

# ---- nothing that looks up by task id may remain
for p in CTX.rglob("*.py"):
    t = p.read_text(encoding="utf-8", errors="replace")
    for banned in ("data_assets", "verifier_data", "batch1_recipes", "batch2_recipes", "batch3_recipes"):
        assert banned not in t, f"{p} mentions {banned}"
assert not (CTX / "solution_library").exists()

# ---- pattern library scan: long runs of high-precision literals
flag = []
for p in sorted((CTX / "pattern_library").glob("*.json")):
    t = p.read_text(encoding="utf-8", errors="replace")
    lits = re.findall(r"(?<![\w.])-?\d+\.\d{5,}", t)
    if len(lits) >= 12:
        flag.append((len(lits), p.name))
print("files:", sum(1 for _ in CTX.rglob("*") if _.is_file()))
print("pattern entries:", len(list((CTX / "pattern_library").glob("*.json"))) - 1)
print("entries with >= 12 high-precision literals:", flag or "none")
