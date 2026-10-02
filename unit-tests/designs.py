import json
from pathlib import Path

from execution.design import replicate_config, resolve

ROOT = Path(__file__).resolve().parents[1]
FACTS = {
    "train_events": 20_000,
    "train_bytes": 110_000,
    "selection_events": 2_000,
    "mean_event_bytes": 5.5,
    "max_unit_bytes": 15,
    "continuation_bytes": 150,
}
SELECTED = {
    "native": {"learning_rate": 1e-3},
    "baseline": {"learning_rate": 1e-3},
    "adapter": {"learning_rate": 1e-3},
    "trunk": {"learning_rate": 1e-3, "alignment_weight": 0.01},
}


def smoke_design() -> dict:
    inputs = json.loads((ROOT / "config/smoke.json").read_text())
    return resolve(inputs, FACTS, 4)


def smoke_config() -> dict:
    return replicate_config(smoke_design(), 0, SELECTED)
