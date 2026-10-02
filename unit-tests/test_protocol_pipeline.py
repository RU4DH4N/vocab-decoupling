import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import execution.mauve as scoring
from data.protocol_corpus import EventCorpus, documents


@pytest.mark.parametrize("device,expected", [("cpu", -1), ("cuda", 0), ("cuda:1", 1)])
def test_mauve_uses_the_requested_device(tmp_path, monkeypatch, device, expected):

    compute = Mock(return_value=SimpleNamespace(mauve=0.5))
    monkeypatch.setitem(sys.modules, "mauve", SimpleNamespace(compute_mauve=compute))
    monkeypatch.setitem(
        sys.modules, "faiss", SimpleNamespace(omp_set_num_threads=Mock())
    )
    monkeypatch.setattr(
        scoring, "snapshot_download", Mock(return_value="cached-encoder")
    )
    config = {
        "enabled": True,
        "model": "test",
        "revision": "pinned",
        "max_tokens": 64,
        "buckets": 2,
        "seed": 1,
    }
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps({"samples": [{"reference": "reference", "text": "sample"}]})
    )
    scoring.mauve_score(
        generation, config, tmp_path / "mauve.json", torch.device(device)
    )
    assert compute.call_args.kwargs["device_id"] == expected


def test_document_windows_do_not_cross_boundaries(tmp_path):
    corpus = EventCorpus(["a b c d", "e f g h"], ["i j k l"], 15)
    corpus.save(tmp_path / "encoded", {})
    loaded = EventCorpus.load(tmp_path / "encoded")
    rows = loaded.windows("train", 2, 100, np.random.default_rng(0))
    for units in loaded.decode(rows):
        joined = "".join(units)
        assert joined in "a b c d" or joined in "e f g h"
    with pytest.raises(ValueError, match="no document"):
        loaded.windows("train", 4, 1, np.random.default_rng(0))


def test_firewall_utterances_are_grouped_by_document(tmp_path):
    path = tmp_path / "grouped.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["first line", "second line", "another document"],
                "document_sha256": ["a", "a", "b"],
            }
        ),
        path,
    )
    assert documents(path, 1000) == ["first line\nsecond line", "another document"]
    assert documents(path, 12) == ["first line\ns"]


def test_noncontiguous_document_rows_are_rejected(tmp_path):
    path = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table({"text": ["a", "b", "c"], "document_sha256": ["a", "b", "a"]}), path
    )
    with pytest.raises(ValueError, match="non-contiguous"):
        documents(path, 1000)
