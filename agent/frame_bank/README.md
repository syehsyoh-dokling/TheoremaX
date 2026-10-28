# Frame bank

One JSON-lines file per theory family (`<family>.jsonl`, same stem as `theory_library/<family>.md`).
Each line is ONE textbook problem frame: what a task of this kind gives, what it asks, which
formulas solve it and in which order. Frames are GENERIC (symbolic parameters, no numeric answers,
no unit-specific values, no file names from any unit). They are retrieved per task by
`agent_core/frame_bank.py` (closest frames to the task text, by trigger phrases) and shown to the
model as "matched frames", never as answers.

Rules (Track 1 rule 8/9 spirit): nothing here may come from a unit's `checks/`, its expected
outputs or any pinned value; frames are derived from textbooks and the family's theory card only.

## Line schema
```json
{"id": "exo-017", "family": "exotic-options", "title": "Down-and-out call, closed form vs MC",
 "triggers": ["down-and-out", "barrier", "knock-out", "rebate", "reiner-rubinstein", "monte carlo", "standard error"],
 "deliverables": ["per-contract table: price, mc_price, standard_error", "summary json: identity residuals"],
 "frame": "DIKETAHUI: S, K, H<S, T, r, q, sigma per contract; n_paths, n_steps, seed. DITANYAKAN: closed-form price, MC price and SE per contract; DO+DI=vanilla residual. RUMUS: Reiner-Rubinstein A-C+F (X>H) ... JAWAB: 1) ... 2) ... 3) ...",
 "formulas": ["DOc(X>H) = A - C + F", "SE = std(discounted payoffs, ddof=1)/sqrt(N)"],
 "pitfalls": ["discrete monitoring overprices DO: BGK shift H e^{-0.5826 sigma sqrt dt}"]}
```
- `triggers`: 8-15 lowercase phrases a task text about this problem would literally contain
  (product names, method names, output words). Whole-word matched.
- `frame`: DIKETAHUI / DITANYAKAN / RUMUS / JAWAB in one string, at most ~700 characters.
  Shapes in brackets where useful ([M] contracts, [T] dates, [N x T] paths).
- `formulas`: 1-6 short strings; `pitfalls`: 0-2.
- A whole line must stay under 1800 characters; the file holds at least 50 frames and they must
  differ (products, variants, conventions, deliverable shapes), not paraphrase one another.
- Name tested `qf_toolkit` functions in RUMUS where one computes the quantity (see `qf_toolkit/`).

## Proven method frames (`_proven.jsonl`, 13 Oct)

One line per public unit our agent has PASSED (the House model at run time, or an earlier graded
solution), distilled into the same generic frame: formulas, ordered steps, and the conventions that
made the pass (ddof, log vs simple returns, date formats, column order). Extra keys: `proven: true`,
`derived_from` (unit id), `keywords` and `deliverables` (used ONLY for the rule-8 exclusion:
`frame_bank._own_task` drops a proven frame on the task it came from, recognised by content through
`pattern_retrieval.same_task`). No numbers computed from data or tests, no file contents, no code:
`validate_proven.py` in the work folder rejects unit ids and precise decimals inside the frame.
Rendered with STEPS and CONVENTIONS lines and a PROVEN METHOD tag; scored 1.4x a textbook frame.
