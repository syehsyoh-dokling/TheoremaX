# Brief: proven method frames (rumus baku) from units that PASSED

Goal: every unit that already passed becomes ONE canonical METHOD FRAME: the formulas and the
ordered computation steps that made it pass, written generically so a Python middle layer can show
it to a model solving a NEW task of the same kind. It is NOT an answer key: no numeric results, no
values read from data, no pinned constants, no reference to the unit's tests. Constants that the
INSTRUCTION itself states (e.g. "use r = 5%", "252 trading days") may appear as symbols with the
stated meaning ("r as given", "annualise with the trading-day count the task states").

Read ONLY, per unit: `drafts/<unit>.md` (instruction, old notes, passed programs). You may grep
`qf_toolkit/*.py` in the context dir for `^def ` to name tested toolkit functions. Do NOT open any
`checks/` directory, nothing under `agenthon-t1`, no `pattern_library`, no `golden_bank`.

Write ONE JSON line per unit to your output file (JSONL, UTF-8, one object per line):
{"id": "pv-<unit-stem>", "family": "<one of the 27 family stems>", "title": "<=70 chars, what the method computes>",
 "proven": true, "derived_from": "<unit id>",
 "triggers": [8-15 lowercase phrases a task of this kind literally contains: product/method/estimator/output words],
 "keywords": [10-20 lowercase single words that characterise THIS unit's instruction (for the same-task exclusion)],
 "deliverables": [the output file names this unit's instruction names],
 "frame": "DIKETAHUI: ... DITANYAKAN: ... RUMUS: ... JAWAB: 1) ... 2) ...   (<= 900 chars, symbolic)",
 "formulas": [3-8 short formula strings, exact conventions: ddof, log vs simple, annualisation factor symbol, discounting, day-count],
 "steps": [4-12 ordered concrete steps: load -> transform -> compute -> write; name qf_toolkit functions where a passed program used them; name the output column words the task uses],
 "conventions": [2-6 one-line decisions that made the pass: e.g. 'returns = log(close_t/close_{t-1}) dropped first row', 'std with ddof=1', 'sort by date ascending before diff', 'JSON keys exactly as listed, floats not numpy types', 'dates written as YYYY-MM-DD'],
 "pitfalls": [0-3]}
Family stems: american-early-exercise backtest-conventions bonds-duration-immunization bsm-greeks-pde
copulas-and-dependence curve-bootstrapping day-count-and-calendars event-studies-and-filings evt-and-tails
execution-and-microstructure exotic-options factor-models-and-sorts fx-forwards-carry-xccy
implied-vol-and-parity jump-and-mean-reversion local-and-stochastic-vol monte-carlo-greeks-and-hedging
portfolio-construction portfolio-credit-risk rating-migration regulatory-capital short-rate-and-rate-options
structural-credit-and-spreads var-backtesting-and-data-prep var-es-methods volatility-models weather-derivatives
(pick the family the draft's FAMILIES line gives first unless the method clearly belongs elsewhere).

Rules: symbolic parameters only; never a number computed from the data or a test (a GOLDEN SOLVE.PY
may contain such numbers: ignore them, keep only the method); no unit ids inside `frame`/`steps`;
one line < 3500 chars; valid JSON (no trailing commas, no NaN). When sources disagree on a
convention, the HOUSE-MODEL passed program wins, then the golden solve.py, then old notes.

Validate before finishing: run `python validate_proven.py <your file>` from this directory (it is a
script file; never pipe python through stdin) and fix every complaint. Final message: counts only.
