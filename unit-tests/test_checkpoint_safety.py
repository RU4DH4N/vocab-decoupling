import json
import math

import pytest
import torch

from data.protocol_corpus import EventCorpus
from execution.protocol_io import ARCHITECTURE, check_inventory
from framework.checkpoints import (
    CheckpointError,
    file_sha256,
    load_checkpoint,
    read_sidecar,
    repair_sidecar,
    save_checkpoint,
    write_json,
)
from orchestrator.locks import Lock


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(2, 2)


def test_sidecar_records_the_committed_checkpoint_hash(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"steps": 3})
    sidecar = json.loads((tmp_path / "model.json").read_text())
    assert sidecar["checkpoint_sha256"] == file_sha256(path)
    assert read_sidecar(path) == {"steps": 3}
    assert load_checkpoint(path)["metadata"] == {"steps": 3}


def test_torn_write_is_refused_until_explicitly_repaired(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"steps": 3})
    stale_sidecar = (tmp_path / "model.json").read_text()
    save_checkpoint(path, _model(), {"steps": 4})
    (tmp_path / "model.json").write_text(stale_sidecar)
    with pytest.raises(CheckpointError, match="does not match the hash"):
        load_checkpoint(path)
    repair_sidecar(path)
    assert load_checkpoint(path)["metadata"] == {"steps": 4}
    assert read_sidecar(path) == {"steps": 4}


def test_substituted_checkpoint_bytes_are_refused_even_with_equal_metadata(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"steps": 3})
    torch.manual_seed(1)
    other = torch.nn.Linear(2, 2)
    torch.save(
        {"format_version": 1, "model": other.state_dict(), "metadata": {"steps": 3}},
        path,
    )
    with pytest.raises(CheckpointError, match="does not match the hash"):
        load_checkpoint(path)


def test_tampered_sidecar_with_matching_hash_is_rejected(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"steps": 3})
    write_json(
        tmp_path / "model.json",
        {"checkpoint_sha256": file_sha256(path), "metadata": {"steps": 99}},
    )
    with pytest.raises(CheckpointError, match="disagrees"):
        load_checkpoint(path)


def test_sidecar_requires_a_hash_and_nested_metadata(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"steps": 3})
    write_json(tmp_path / "model.json", {"steps": 3})
    with pytest.raises(CheckpointError, match="invalid checkpoint sidecar"):
        load_checkpoint(path)


def test_nan_metadata_does_not_break_the_sidecar_check(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"final_loss": math.nan, "nested": [1.0, math.nan]})
    assert math.isnan(load_checkpoint(path)["metadata"]["final_loss"])


def test_checkpoints_load_with_weights_only(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {"vocab": ["a", "b"], "config": {"x": 1.5}})
    payload = load_checkpoint(path)
    assert payload["metadata"]["vocab"] == ["a", "b"]


def test_live_lock_is_respected(tmp_path):
    lock = tmp_path / "exclusive-device.lock"
    with Lock(lock):
        with pytest.raises(RuntimeError, match="already in use"):
            with Lock(lock):
                pass


@pytest.mark.parametrize("version", [None, 0, -1, True, 2])
def test_checkpoint_requires_the_declared_format(tmp_path, version):
    path = tmp_path / "model.pt"
    torch.save({"format_version": version, "model": {}, "metadata": {}}, path)
    with pytest.raises(CheckpointError, match="unsupported checkpoint format"):
        load_checkpoint(path, verify_sidecar=False)


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "x" * 64])
def test_checkpoint_sidecar_requires_a_valid_hash(tmp_path, digest):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _model(), {})
    write_json(tmp_path / "model.json", {"checkpoint_sha256": digest, "metadata": {}})
    with pytest.raises(CheckpointError, match="invalid checkpoint sidecar"):
        load_checkpoint(path)


def test_released_lock_file_is_reusable(tmp_path):
    lock = tmp_path / "exclusive-device.lock"
    with Lock(lock):
        pass
    assert lock.exists()
    with Lock(lock):
        pass


def test_metadata_is_read_without_weights_and_must_match_the_checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, torch.nn.Linear(2, 2), {"vocab": ["a", "b"]})
    assert read_sidecar(path) == {"vocab": ["a", "b"]}
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(CheckpointError, match="does not match"):
        read_sidecar(path)


def test_inventory_is_checked_from_the_sidecar_before_any_model_loads(tmp_path):
    path = tmp_path / "stage.pt"
    corpus = EventCorpus(["a a a"], ["a a"], 4)
    save_checkpoint(
        path,
        torch.nn.Linear(2, 2),
        {"architecture": ARCHITECTURE, "vocab": corpus.vocab},
    )
    check_inventory(path, corpus, "selection")
    changed = EventCorpus(["b b b"], ["b b"], 4)
    with pytest.raises(ValueError, match="inventory differs"):
        check_inventory(path, changed, "selection")
    check_inventory(path, changed, "reporting")
