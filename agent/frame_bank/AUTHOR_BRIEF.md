# Brief for frame authors

Write `frame_bank/<family>.jsonl` for each family you are given, following `frame_bank/README.md`
(schema, rules). Context directory: `C:/Users/Saifuddin/Documents/Agenthon-2026/submissions/2026-09-25/t1_know4_ctx`.

Before writing, read ONLY:
- `frame_bank/README.md` (schema and rules)
- `theory_library/<family>.md` for each family (its keywords line, Worked frame, Core formulas,
  Variants, Pitfalls) - frames must be consistent with the card and may cite its numbered formulas.
- `qf_toolkit/*.py` signatures (grep `^def `) to name tested functions in RUMUS where one applies.

Do NOT open anything else: no task units, no `checks/`, no `pattern_library/`, no `solution_library`,
nothing under `agenthon-t1`, `simulators`, `t1all` or `submissions/*/units`. Frames come from
textbook knowledge (Hull, Haug, Glasserman, Jorion, McNeil-Frey-Embrechts, Tuckman, Brigo-Mercurio,
Campbell-Lo-MacKinlay, Fabozzi, Basel texts) and the family card only.

Content requirements per family: at least 50 DISTINCT frames covering the family's whole span -
every product / estimator / method the keywords line names, each with the variants a task could
pose (conventions, discrete vs continuous, with/without dividends or costs, per-contract table vs
scalar summary, calibration vs pricing vs checking, missing-input completion). Each frame says what
is GIVEN (with shapes), what is ASKED (with units), the FORMULAS in order, and the STEPS. Use
symbolic parameters only; no numeric results, no "typical values".

Validation (mandatory before you finish): write a small script to a file and run it (never
`python -` or heredocs) that loads every line with json.loads, checks the required keys
(id, family, title, triggers, deliverables, frame, formulas, pitfalls), 8-15 triggers each, line
length < 1800, frame length <= 800, at least 50 lines, ids unique, `family` equal to the file stem.
Report the per-file counts in your final message. Keep the final message short.
