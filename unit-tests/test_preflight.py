import copy

import numpy as np
import pytest
import torch
from corpora import documents
from designs import smoke_config

from data.protocol_corpus import EventCorpus
from data.tokenizers import train_bpe
from execution.preflight import check, longest_window
from execution.validation import validate
from framework.memory import TensorMeter

TRAIN = [
    "a b c d e f g h i j k l",
    "one two three four five six seven eight",
    "tiny words go here",
    *documents(6, 4, (20, 40)),
]


@pytest.fixture
def config():
    return smoke_config()


def test_longest_window_stays_inside_documents():
    small = EventCorpus(TRAIN[:3], ["x y"], 15)
    small_lengths = np.array([len(unit.encode()) + 1 for unit in small.vocab])
    rows = longest_window(small, 3, small_lengths)
    assert small.decode(rows[None])[0] == [" five", " six", " seven", " eight"]
    with pytest.raises(ValueError, match="no document"):
        longest_window(small, 12, small_lengths)


def test_longest_windows_train_forward_backward(config):
    corpus = EventCorpus(TRAIN, ["x y"], 15)
    meter = TensorMeter()
    texts = [" ".join(TRAIN)] * 4
    with meter:
        result = check(
            config,
            corpus,
            train_bpe(300, corpus.vocab * 4, False),
            train_bpe(300, texts, True),
            torch.device("cpu"),
            meter,
        )
    assert set(result["checks"]) == {"bytes", "bpe", "planner", "baseline"}
    assert len(result["checks"]["bytes"]["stages"]) == len(config["stages"])
    lengths = np.array([len(unit.encode()) + 1 for unit in corpus.vocab])
    rows = longest_window(corpus, config["events"], lengths)
    assert result["checks"]["bytes"]["receiver_steps"] == lengths[rows].sum()
    peaks = result["phase_peak_bytes"]
    stages = [s["name"] for s in config["stages"]]
    for kind in ("bytes", "bpe"):
        assert {f"{kind} build", f"{kind} lookahead"} <= set(peaks)
        for stage in stages:
            assert peaks[f"{kind} {stage}"] > 0
    assert peaks["planner"] > 0 and peaks["baseline"] > 0


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("baseline", "learning_rate", float("nan"), "baseline.learning_rate"),
        ("baseline", "weight_decay", -1, "baseline.weight_decay"),
        ("baseline", "warmup_steps", None, "baseline training schedule"),
        ("generation", "prompt_events", 40, "generation prompt"),
    ],
)
def test_invalid_run_fails_before_work(config, section, key, value, match):
    invalid = copy.deepcopy(config)
    invalid[section][key] = invalid[section]["steps"] if value is None else value
    with pytest.raises(ValueError, match=match):
        validate(invalid)


def test_short_bpe_context_rejected(config):
    config["baseline"]["model"]["max_seq_len"] = 16
    with pytest.raises(ValueError, match="unmerged byte-level"):
        validate(config)
