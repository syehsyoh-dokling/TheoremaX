# TheoremaX — Quant Code from First Principles

Autonomous coding agent for quantitative-finance tasks. Agenthon 2026 (NeurIPS) Track 1 — Coding, Team 210.

The agent reads an unseen quantitative-finance task (option pricing, risk, credit, FX, backtests, event studies) and
delivers working analysis code and output files, theory-first and without stored answers. Before writing code it must
build a worked-problem frame — **Given** (every input with its unit), **Asked** (every deliverable and field),
**Formula** (each derivation tied to a theory card or a tested library call) and **Answer** (method plus the identities
the result must satisfy). A code-level checker rejects incomplete frames.

The language model (Nemotron-3-Super 120B through the organisers' hosted route) plans and writes every program. The local
Python layer supplies the theory library, a tested quant toolkit, a cell-by-cell sandbox that reports variable values at
the failing line, line-level patch repair, an independent second candidate with a self-consistency tie-break, and finance
invariants checked before any output is accepted.

## Versions

| Version | Role | Where |
|---|---|---|
| **know95** | Final submission (26 Oct 2026) | `agent/` |
| know96 | latest development build (adds a reshaped-bundle guard) | `versions/know95_to_know96.patch` |

Final image: `ghcr.io/syehsyoh-dokling/qf-solver@sha256:e407066077eed6bc0c748f2cbfdc721ce2103c6c34183568439144aeece851d8`
(descriptor in `submission/`). The model is called over the API; no weights are shipped.

## Layout

```
agent/            Final agent source (Dockerfile, agent_core, qf_toolkit, theory_library, frame_bank, pattern_library)
versions/         patch from the Final to the latest development build
tools/            build, run, check and diagnosis scripts used on the build server (tNN_* / t1_* scripts)
submission/       Final descriptor and the packaging script (the team key is entered at run time, never stored)
docs/reports/     per-unit analyses and design notes
data/             the organisers' public Track 1 repository (their licence applies, see its DATA-LICENSE.md)
```

## Build

```bash
docker build -t qf-solver:know95 agent/
docker run --rm -v "$PWD/task:/input:ro" -v "$PWD/out:/app/output" \
  -e MODEL_ENDPOINT=... -e MODEL_TOKEN=... -e MODEL_NAME=nvidia/nemotron-3-super-120b-a12b \
  qf-solver:know95 solve --task-dir /input --out /app/output --budget-sec 300
```

## Licence

Code: Apache-2.0. `data/` is a copy of the organisers' public material and remains under their licence.
