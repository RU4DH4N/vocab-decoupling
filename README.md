# Vocab Decoupling Research

[![DOI](https://zenodo.org/badge/1400919324.svg)](https://doi.org/10.5281/zenodo.23106806)

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

## Data

The data is the official BabyLM 2025 text release. Download the three archives from the OSF links in
`data/released_babylm.py`, then:

```sh
python -m data.released_babylm --train-archive train_100M.zip --dev-archive dev.zip \
    --test-archive test.zip --output datasets/babylm/released-2025 --lines-per-group 64
```

This writes the parquet files and `manifest.json` (committed, with their checksums).

| Split | Official split |
|---|---|
| `train` | train |
| `selection` | dev |
| `reporting` | test |

Each source file is cut into consecutive groups of 64 lines; the last group of each source may be shorter.
Groups are computational contexts, not original documents; we make no claim that the official splits are
disjoint at the document level.

| Source | Origin |
|---|---|
| `bnc_spoken` | British National Corpus, spoken portion |
| `childes` | CHILDES, child-directed speech |
| `gutenberg` | Project Gutenberg (children's stories) |
| `open_subtitles` | OpenSubtitles |
| `simple_wiki` | Simple English Wikipedia |
| `switchboard` | Switchboard Dialog Act Corpus |

Each source keeps its original licence; see the BabyLM 2025 release for terms.

## Claims

| Assumption before running | Result | Entry point |
|---|---|---|
| C1. The trunk's messages carry information the receiver can't get on its own | Supported | `claims/continuous_communication.py` |
| C2. A learned layer-by-layer protocol beats simpler ways of passing messages | Supported | `claims/learned_protocol.py` |
| C3. Receivers need explicit alignment to learn which message to read | Partly supported | `claims/correspondence_bootstrap.py` |
| C4. Trunk and receiver can run on different clocks | Supported | `claims/independent_clocks.py` |
| C5. Predicted future messages improve spelling of the current word | Not supported | `claims/planning_ahead.py` |
| C6. New receivers can attach to a frozen trunk | Supported | `claims/receiver_replacement.py` |
| C7. How it compares to BPE and BabyLlama, and what a new receiver costs | Measured | `claims/remote_hypothesis.py` |

`claims/active.py` runs all of them in one graph.

## Additive Claims

| Assumption before running | Result | Entry point |
|---|---|---|
| C8. Receiver size doesn't need to scale with trunk size | Supported | `claims/receiver_scaling.py` |
| C9. The BPE receiver's poor result came from the 16k vocabulary it was given rather than from BPE itself | Largely supported | `claims/receiver_vocabulary.py` |

## Running a claim

```sh
python claims/planning_ahead.py --config config/smoke.json --split selection --plan
python claims/planning_ahead.py --config config/smoke.json --split selection --run-missing
```

`--plan` shows what would run and `--run-missing` runs it. `python -m postprocess.analysis --output artifacts/<config> --split <split>`
turns the results into tables with confidence intervals. Results are written to `artifacts/<config>/claims/`.
`config/smoke.json`, `config/pilot.json` and `config/paper.json` are the same design at three scales; everything
else is derived and recorded in `artifacts/<config>/design.json`. Evaluate on `--split reporting` once, at the end.
C8 runs once per trunk config (`config/trunk-10m.json`, `config/trunk-20m.json`, `config/paper.json`) and C9 on
`config/paper.json`; `python -m postprocess.extension_analysis --scaling <outputs> --vocabulary <output> --split <split> --report <folder>`
turns them into tables.

## Artifacts

Generated artifacts are too large for git and are archived on Zenodo: [10.5281/zenodo.23102846](https://doi.org/10.5281/zenodo.23102846)

The archive is split into 19 parts; download all the files into the repository root.

To check them:

```sh
[ "$(cat vocab-decoupling-artifacts-v1.1.0.tar.zst.part-* | shasum -a 256 | cut -d' ' -f1)" = \
  "$(cut -d' ' -f1 vocab-decoupling-artifacts-v1.1.0.tar.zst.sha256)" ] && echo OK || echo MISMATCH
```

To restore `artifacts/`:

```sh
cat vocab-decoupling-artifacts-v1.1.0.tar.zst.part-* | zstd -d -c | tar -x
```
