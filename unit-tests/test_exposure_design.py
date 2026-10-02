import numpy as np
import pytest
from corpora import documents
from designs import smoke_config

from data.protocol_corpus import EventCorpus
from execution.exposure_design import plan_exposure, verify_exposure
from framework.checkpoints import write_json


def test_exposure_is_precomputed_reproducibly_and_baseline_matches(tmp_path):
    config = smoke_config()
    corpus = EventCorpus(documents(4, 6, (20, 40)), documents(5, 2, (20, 40)), 15)
    plan = plan_exposure(config, corpus)
    assert plan == plan_exposure(config, corpus)
    primary = [v for k, v in plan["jobs"].items() if k.startswith("primary/")]
    expected = plan["jobs"]["baseline/model"]
    assert expected["target_bytes"] == sum(p["target_bytes"] for p in primary)
    assert expected["steps"] == sum(p["steps"] for p in primary)
    write_json(tmp_path / "exposure.json", plan)
    verify_exposure(
        tmp_path / "exposure.json",
        "baseline/model",
        expected["steps"],
        expected["target_bytes"],
    )
    with pytest.raises(ValueError, match="precomputed design"):
        verify_exposure(
            tmp_path / "exposure.json",
            "baseline/model",
            expected["steps"],
            expected["target_bytes"] + 1,
        )
    with pytest.raises(ValueError, match="precomputed design"):
        verify_exposure(
            tmp_path / "exposure.json",
            "baseline/model",
            expected["steps"] + 1,
            expected["target_bytes"],
        )


def test_document_window_index_reused_without_changing_samples():

    corpus = EventCorpus(["one two three four five six"], ["selection only"], 15)
    first = corpus.windows("train", 2, 3, np.random.default_rng(12))
    index = corpus._window_indices[("train", 2)]
    second = corpus.windows("train", 2, 3, np.random.default_rng(12))
    assert corpus._window_indices[("train", 2)] is index
    assert first.equal(second)


def test_restricted_corpus_keeps_the_same_units():

    corpus = EventCorpus(["one two three four five six"], ["selection only"], 15)
    rows = [
        corpus.windows("train", 2, 1, np.random.default_rng(seed)) for seed in (1, 2)
    ]
    subset, remapped = corpus.restricted(rows)
    assert len(subset.vocab) <= 6
    for original, local in zip(rows, remapped, strict=True):
        assert corpus.decode(original) == subset.decode(local)
