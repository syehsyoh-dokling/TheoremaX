"""Offline self-test of plan_gate.validate (no model): an incomplete plan must be rejected with
the right reasons, a complete one accepted. Run from the image root: python -m agent_core.selftest_plan_gate"""
from agent_core import plan_gate
from agent_core.contracts import OutputContract, TaskSpec

spec = TaskSpec(task_id="t-demo", instruction="demo", input_root="/input",
                contracts=(OutputContract("prices.csv", "csv", source="instruction"),
                           OutputContract("summary.json", "json", source="instruction")))
bad = {"deliverables": [{"file": "prices.csv", "contents": ["strike", "price"]}],
       "quantities": [{"name": "rate", "status": "convention", "source": "ACT/365"},
                      {"name": "vol", "status": "guess", "source": ""}],
       "method": ["price"], "identities": ["price >= 0"],
       "open_questions": ["which day count?"]}
problems = plan_gate.validate(bad, spec)
for must in ("cite", "status must be", "no source", "open question"):
    assert any(must in p for p in problems), (must, problems)
# a deliverable the plan left out is ADDED from the detected contract, not a rejection (27 Sep)
assert not any("summary.json" in p for p in problems), problems
assert any(d.get("file") == "summary.json" and d.get("added_from_contract") for d in bad["deliverables"])
# an image deliverable needs no columns
img_spec = TaskSpec(task_id="t-img", instruction="demo", input_root="/input",
                    contracts=(OutputContract("chart.png", "png", source="instruction"),))
img_plan = {"deliverables": [{"file": "chart.png", "contents": []}], "quantities": [
    {"name": "x", "status": "given", "source": "a.csv:x"}], "method": ["plot"],
    "identities": ["chart has 2 lines", "chart axis labelled"], "open_questions": []}
assert not any("lists no columns" in p for p in plan_gate.validate(img_plan, img_spec))
good = {"deliverables": [{"file": "prices.csv", "contents": ["strike", "price"]},
                         {"file": "summary.json", "contents": ["n", "mean_price"]}],
        "quantities": [{"name": "rate", "status": "convention", "source": "ACT/365",
                        "reference": "THEORY fixed-income: ACT/365 when unstated"},
                       {"name": "spot", "status": "given", "source": "market.csv:close", "unit": "price", "shape": "[T] by date"}],
        "method": ["read", "price", "write"],
        "identities": ["price >= 0", "call decreasing in strike"], "open_questions": []}
assert plan_gate.validate(good, spec) == [], plan_gate.validate(good, spec)
# 25 Sep diagnosis bugs: a full output path names the same deliverable ...
pathed = dict(good, deliverables=[{"file": "/app/output/prices.csv", "contents": ["strike", "price"]},
                                  {"file": "/app/output/summary.json", "contents": ["n", "mean_price"]}])
assert plan_gate.validate(pathed, spec) == [], plan_gate.validate(pathed, spec)
# ... and an identity borrowed from another task (names nothing this task outputs) does not count
borrowed = dict(good, identities=["geometric Asian <= arithmetic Asian", "price >= 0"])
assert any("at least 2" in p for p in plan_gate.identity_problems(borrowed))
assert plan_gate.identity_problems(bad) and not plan_gate.identity_problems(good)
# grounding: a reference quoted from what the model was shown counts; an invented one does not
shown = "TASK:\nprice the options\nTHEORY:\nACT/365 when unstated for fixed income discounting"
g = plan_gate.grounding(good, shown)
assert g["n"] == 1 and g["grounded"] == 1 and g["by_section"] == {"theory": 1}, g
bad_ref = dict(good, quantities=[{"name": "rate", "status": "convention", "source": "x",
                                  "reference": "the central bank always uses thirty day months here"}])
g2 = plan_gate.grounding(bad_ref, shown)
assert g2["grounded"] == 0 and g2["ungrounded"] == ["rate"], g2
# grounding gate: an ungrounded convention sends the plan back; a grounded one passes
assert plan_gate.grounding_problems(g) == [], plan_gate.grounding_problems(g)
assert plan_gate.grounding_problems(g2) and "QUOTE" in plan_gate.grounding_problems(g2)[0]
assert plan_gate._parse('noise {"a": 1} {"b": 2}') == {"a": 1}
# input-usage audit: a loaded parameter or column that enters no formula sends the plan back;
# bookkeeping keys ("_"-prefixed), identifiers and manifests never count
import json as _json
import pathlib as _pl
import tempfile as _tf
_d = _pl.Path(_tf.mkdtemp())
(_d / "params.json").write_text(_json.dumps({"pip_multiplier": 10000, "_canary": "x"}))
(_d / "trades.csv").write_text("trade_id,direction,notional\nT1,buy,100\n")
(_d / "data_manifest.json").write_text(_json.dumps({"sha256": "0", "files": []}))
uspec = TaskSpec(task_id="t-usage", instruction="demo", input_root=str(_d),
                 data_files=("params.json", "trades.csv", "data_manifest.json"),
                 contracts=spec.contracts)
assert sorted(k for k, _ in plan_gate.input_fields(uspec)) == ["direction", "notional", "pip_multiplier"]
loose = dict(good, quantities=good["quantities"] + [{"name": "pnl", "shape": "scalar", "status": "derived",
                                                     "source": "pnl = notional * (F - K)", "reference": "x"}])
names = sorted(p.split("'")[1] for p in plan_gate.usage_problems(loose, uspec))
assert names == ["direction", "pip_multiplier"], names
tight = dict(loose, quantities=good["quantities"] + [
    {"name": "pnl", "shape": "scalar", "status": "derived", "source": "pnl = sign(direction) * notional * (F - K)", "reference": "x"},
    {"name": "points", "shape": "scalar", "status": "derived", "source": "points = (F - S) * pip_multiplier", "reference": "x"}])
assert plan_gate.usage_problems(tight, uspec) == [], plan_gate.usage_problems(tight, uspec)
declared = dict(loose, unused_inputs=[{"name": "direction", "reason": "all trades are buys"},
                                      {"name": "pip_multiplier", "reason": "points not asked"}])
assert plan_gate.usage_problems(declared, uspec) == []
# the run keeps the complete plan with the FEWEST gaps: rounds reply loose, tight, loose
import os as _os
from agent_core import code_writer as _cw
_os.environ["T1_GROUNDING_GATE"] = "off"
_replies = iter([_json.dumps(loose), _json.dumps(tight), _json.dumps(loose)])
_orig = (_cw.model_available, _cw.complete)
_cw.model_available = lambda: True
_cw.complete = lambda *a, **k: next(_replies)
try:
    note = plan_gate.run(uspec)
finally:
    _cw.model_available, _cw.complete = _orig
    del _os.environ["T1_GROUNDING_GATE"]
assert "pip_multiplier" in uspec.plan and plan_gate.TRACE["status"] == "accepted", (note, plan_gate.TRACE["status"])
# ... and a re-plan that breaks the structure does not throw away an earlier complete plan
_replies = iter([_json.dumps(loose), "not json", "not json"])
_os.environ["T1_GROUNDING_GATE"] = "off"
_cw.model_available = lambda: True
_cw.complete = lambda *a, **k: next(_replies)
uspec.plan = ""
try:
    note = plan_gate.run(uspec)
finally:
    _cw.model_available, _cw.complete = _orig
    del _os.environ["T1_GROUNDING_GATE"]
assert uspec.plan and "earlier round" in note, note
# DIKETAHUI / RUMUS / PITFALLS frame (30 Sep)
_pits = ["Compound parity pairs CoC with PoC", "Daily-grid MC extrema are biased"]
_fp = {"pitfalls_checked": [{"id": "P1", "relevant": True, "how": "compute PoC for parity"}]}
assert any("P2" in p for p in plan_gate.pitfall_problems(_fp, _pits))
_fp["pitfalls_checked"].append({"id": "p2", "relevant": False, "how": "no MC here"})
assert plan_gate.pitfall_problems(_fp, _pits) == []
assert plan_gate.unit_problems({"quantities": [{"name": "r", "status": "given", "source": "x"}]})
assert not plan_gate.unit_problems({"quantities": [{"name": "r", "status": "given", "source": "x",
                                                    "unit": "decimal/yr"}]})
_fq = {"quantities": [{"name": "price", "status": "derived", "source": "Geske formula",
                       "reference": "the task says so"}]}
assert plan_gate.formula_problems(_fq, {"sections": {"price": "instruction"}})
assert not plan_gate.formula_problems(_fq, {"sections": {"price": "theory"}})
_fq2 = {"quantities": [{"name": "price", "status": "derived",
                        "source": "toolkit compound_option_price(...)", "reference": ""}]}
assert not plan_gate.formula_problems(_fq2, {})
# VARIANTS TO DECIDE (30 Sep): undecided -> incomplete; a non-default choice needs a task sentence
_line = ("- V1 corrado_rank_scale: rank scaling. Options: (a) raw ranks; (b) standardized ranks. "
         "Default when silent: (a) - Corrado 1989")
_m = plan_gate._VARIANT.match(_line)
assert _m and _m.group(4) == "a", _m
_vars = [{"key": "corrado_rank_scale", "decision": "rank scaling", "options": "(a) raw; (b) std",
          "default": "a", "source": "Corrado 1989", "line": _line[2:]}]
_task = "Use standardized ranks K divided by T plus one for every event in the sample."
assert any("V1" in p for p in plan_gate.variant_problems({}, _vars, _task))
assert plan_gate.variant_problems({"variants_chosen": [{"id": "V1", "choice": "a", "reference": "default"}]},
                                  _vars, _task) == []
assert plan_gate.variant_problems({"variants_chosen": [{"id": "V1", "choice": "b", "reference": "I prefer it"}]},
                                  _vars, _task)
assert plan_gate.variant_problems({"variants_chosen": [{"id": "V1", "choice": "(b)",
                                   "reference": "Use standardized ranks K divided by T plus one"}]},
                                  _vars, _task) == []
print("PLAN GATE SELFTEST OK:", len(problems), "problems found in the bad plan")
