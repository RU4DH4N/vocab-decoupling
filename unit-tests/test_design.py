import json
import math
from pathlib import Path

import pytest

from execution.design import (
    ADAPTER_FRACTION,
    SELECTION_FRACTION,
    SWEEP_FRACTION,
    check_inputs,
    continuation_bytes,
    prompt_shape,
    replicate_config,
    resolve,
    seed_for,
    statistics,
    unswept,
)
from execution.validation import validate

ROOT = Path(__file__).resolve().parents[1]
PAPER_FACTS = {
    "train_events": 99_195_775,
    "train_bytes": 540_973_212,
    "selection_events": 10_437_235,
    "mean_event_bytes": 5.45,
    "max_unit_bytes": 17,
    "continuation_bytes": 120,
}


def inputs(name):
    return json.loads((ROOT / "config" / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def paper():
    return resolve(inputs("paper"), PAPER_FACTS, 5000)


def test_every_config_is_a_valid_set_of_inputs():
    for name in ("smoke", "pilot", "paper"):
        check_inputs(inputs(name))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"batch_windows": 1}, "two windows"),
        ({"baseline_seeds": 4}, "baseline_seeds"),
        ({"passes": 0}, "passes"),
        ({"seeds": True}, "wrong type"),
        ({"surprise": 1}, "unknown"),
    ],
)
def test_bad_inputs_are_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        check_inputs({**inputs("smoke"), **change})


def test_baseline_is_parameter_matched_to_the_system(paper):
    counts = paper["parameters"]
    assert abs(counts["baseline_body"] / counts["matched_system"] - 1) < 0.1
    assert abs(counts["sender"] / inputs("paper")["trunk_parameters"] - 1) < 0.1


def test_steps_follow_the_exposure_budget(paper):
    batch, events = inputs("paper")["batch_windows"], inputs("paper")["context_events"]
    full = paper["steps"]["full"]
    assert full == math.ceil(PAPER_FACTS["train_events"] / (batch * events))
    assert paper["steps"]["adapter"] == math.ceil(full * ADAPTER_FRACTION)
    assert paper["steps"]["sweep"] == math.ceil(full * SWEEP_FRACTION)


def test_windows_fit_every_context(paper):
    shared = paper["shared"]
    events, unit = shared["events"], shared["corpus"]["max_unit_bytes"]
    assert shared["receiver_steps"] == (events + 1) * (unit + 1)
    assert shared["native"]["max_seq_len"] >= shared["receiver_steps"]
    assert shared["sender"]["max_seq_len"] >= events + shared["lookahead"]["horizon"]
    prompt, continuation = prompt_shape(inputs("paper"), PAPER_FACTS)
    assert shared["generation"]["prompt_events"] == prompt
    assert shared["generation"]["bytes"] == continuation


def test_seeds_are_deterministic_distinct_and_per_replicate(paper):
    first, second = replicate_config(paper, 0, {}), replicate_config(paper, 1, {})
    assert replicate_config(paper, 0, {}) == first
    assert seed_for(1, "a") == seed_for(1, "a") != seed_for(1, "b")
    for key in ("data_seed",):
        assert first[key] != second[key]
    assert (
        first["receivers"]["primary"]["seed"] != second["receivers"]["primary"]["seed"]
    )
    assert first["evaluation_seed"] == second["evaluation_seed"]
    assert first["sender"]["d_model"] == second["sender"]["d_model"]


def test_selected_rates_reach_every_job_and_can_be_cleared(paper):
    selected = {
        "native": {"learning_rate": 1e-3},
        "baseline": {"learning_rate": 2e-3},
        "adapter": {"learning_rate": 3e-3},
        "trunk": {"learning_rate": 4e-3, "alignment_weight": 0.1},
    }
    config = replicate_config(paper, 0, selected)
    validate(config)
    assert config["native_training"]["learning_rate"] == 1e-3
    assert config["baseline"]["learning_rate"] == 2e-3
    trunk = next(s for s in config["stages"] if s["name"] == "trunk")
    assert (trunk["learning_rate"], trunk["alignment_weight"]) == (4e-3, 0.1)
    for section in ("planner", "lookahead", "control_training"):
        assert config[section]["learning_rate"] == 3e-3
    cleared = unswept(config)
    assert cleared["native_training"]["learning_rate"] is None
    assert all(s["learning_rate"] is None for s in cleared["stages"])
    assert unswept(replicate_config(paper, 0, {})) == cleared


def test_statistics_cover_the_declared_quantile():
    facts = statistics([1] * 998 + [20, 30], 10)
    assert facts["max_unit_bytes"] == 20
    assert facts["train_events"] == 1000 and facts["selection_events"] == 10
    assert statistics([1] * 9_990 + [30] * 10, 1)["max_unit_bytes"] == 4


def test_selection_is_a_fixed_fraction_of_a_full_evaluation_pass(paper):
    shared = paper["shared"]
    windows = PAPER_FACTS["selection_events"] / (
        shared["batch_size"] * shared["events"]
    )
    assert shared["evaluation_batches"] == math.ceil(windows)
    assert shared["selection_batches"] == math.ceil(windows * SELECTION_FRACTION)


def test_continuation_budget_is_a_low_quantile_of_spans_within_documents():
    docs = [[1] * 40, [3] * 40, [2] * 3]
    assert continuation_bytes(docs, 4) == 4
    assert continuation_bytes([[5, 5, 5, 5]], 4) == 20
    with pytest.raises(ValueError, match="no training document"):
        continuation_bytes(docs, 41)
