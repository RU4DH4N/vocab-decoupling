# Vocab Decoupling Research

## Layout

| Directory | Contents |
|---|---|
| `claims/` | One entry point per claim |
| `pipeline/` | Reusable graph steps: prepare, train, measure, score |
| `execution/` | Numerical workers the pipeline steps call |
| `models/`, `data/` | Model and corpus code |
| `orchestrator/` | Config loading, dependency graph, caching |
| `framework/` | Checkpoint, runtime and statistics utilities |
| `postprocess/` | Analysis, forecasting and graph rendering |
| `config/` | Configurations |

## Setup

```sh
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock
```

## Claims

| Claim | Entry point |
|---|---|
| C1. A frozen trunk's continuous messages carry information receivers can use beyond their own history | `claims/continuous_communication.py` |
| C2. Useful communication needs a learned layerwise protocol, not just input-layer injection or per-layer cross attention | `claims/learned_protocol.py` |
| C3. Correspondence must be bootstrapped with explicit alignment; likelihood can then preserve it | `claims/correspondence_bootstrap.py` |
| C4. Sender and receiver can run on different clocks | `claims/independent_clocks.py` |
| C5. Causally rolled future messages improve realization of the current event | `claims/planning_ahead.py` |
| C6. Fresh receivers can attach to the same frozen trunk | `claims/receiver_replacement.py` |
| C7. Quality gap to BPE and to the released BabyLlama, and the cost of replacing a receiver | `claims/remote_hypothesis.py` |

`claims/active.py` runs all of them in one graph.

## Running a claim

```sh
python claims/planning_ahead.py --config config/smoke.json --split selection --plan
python claims/planning_ahead.py --config config/smoke.json --split selection --run-missing
```

`--plan` shows what would run and `--run-missing` runs it. `python -m postprocess.analysis --output artifacts/<config> --split <split>`
turns the results into tables with confidence intervals. Results are written to `artifacts/<config>/claims/`.
`config/smoke.json`, `config/pilot.json` and `config/paper.json` are the same design at three scales; everything
else is derived and recorded in `artifacts/<config>/design.json`. Evaluate on `--split reporting` once, at the end.
