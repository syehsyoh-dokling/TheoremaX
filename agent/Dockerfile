# Agenthon 2026 Track 1 - "knowledge only" image (24 Sep).
#
# The agent SOLVES every task: no answer lookups of any kind. Removed relative to the image scored
# 0.6744: data_assets/ (ready-made deliverables and checker values), the batch recipe modules, and
# the per-unit reconciliation function in agent.py. Carried instead: the tested toolkit, the
# invariant library, the pattern library (worked examples chosen by content similarity - a task's
# own entry is always excluded) and theory cards (formulas, conventions, identities, and the
# procedure for incomplete inputs), injected into every model prompt.
FROM finance-bench-sandbox:clean

WORKDIR /agent
COPY agent.py /agent/agent.py
COPY agent_core/ /agent/agent_core/
COPY invariant_library/ /agent/invariant_library/
COPY qf_toolkit/ /agent/qf_toolkit/
COPY pattern_library/ /agent/pattern_library/
# Family theory library (26 Sep): formulas, implicit conventions and how to detect them, how to
# complete missing inputs, invariants, pitfalls - general knowledge from standard references, no
# task answers. Read by agent_core/theory.py for the plan and code prompts.
COPY theory_library/ /agent/theory_library/
# 13 Oct (know70): frame_bank/_proven.jsonl - the method frames distilled from every public unit we
# passed (formulas, ordered steps, conventions; symbolic, no numbers, no code). Rule 8: a proven
# frame is withheld on the task it came from (content match), offered on other tasks of its kind.
COPY frame_bank/ /agent/frame_bank/
# Track 1 README rules 8 and 9 (clarified 25 Sep 2026 05:00 UTC, in force from 5 Oct for Development
# and Final): our solutions to public units may be used ONLY as in-context examples for OTHER units
# (pattern_library, own entry excluded by task id); no stored answer or solution written in advance
# may be used on the unit it solves, and every pass must come from the House model at run time. So
# NO solution replay (solution_library is not in this image) and NO prepared templates.

LABEL qfbench2.interface_version="2.0"
ENV PYTHONUNBUFFERED=1
ENV T1_SOLUTION_MEMORY=off \
    T1_TEMPLATES=off
ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
