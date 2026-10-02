import json
from pathlib import Path

from orchestrator.context import load_context
from pipeline._base import EXCLUSIVE
from pipeline.bootstrap import Bootstrap
from pipeline.bpe_reference import BPEReference
from pipeline.measure_receiver import MeasureReceiver
from pipeline.planning import Planning
from pipeline.preflight import Preflight
from pipeline.prepare import Prepare
from pipeline.train_stage import TrainStage

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path, device):
    config = tmp_path / "inputs.json"
    config.write_text((ROOT / "config/smoke.json").read_text())
    train, selection = tmp_path / "train.parquet", tmp_path / "selection.parquet"
    for path in (train, selection):
        path.write_bytes(path.name.encode())
    return load_context(
        config, tmp_path / "output", train, selection, device, "selection", None
    )


def test_accelerator_weights_follow_memory_class(tmp_path):
    gpu = context(tmp_path, "cuda")
    assert Preflight(gpu).weight() == EXCLUSIVE
    assert TrainStage(gpu, 0, "fresh-bpe", 0).weight() == EXCLUSIVE
    assert TrainStage(gpu, 0, "primary", 2).weight() == 1
    assert MeasureReceiver(gpu, 0, "fresh-byte").weight() == 1
    assert BPEReference(gpu, 0, "train").weight() == EXCLUSIVE
    assert BPEReference(gpu, 0, "mauve").weight() == 1
    assert Planning(gpu, 0, "measure").weight() == 1
    assert Bootstrap(gpu, 0, "train").weight() == EXCLUSIVE
    assert Bootstrap(gpu, 0, "measure").weight() == 1
    assert Prepare(gpu).weight() == 0


def test_cpu_steps_never_claim_accelerator_slots(tmp_path):
    cpu = context(tmp_path, "cpu")
    assert TrainStage(cpu, 0, "fresh-bpe", 0).weight() == 0


def test_remote_plans_share_the_accelerator():
    for path in (ROOT / "config/remote").glob("*.json"):
        for claim in json.loads(path.read_text())["claims"]:
            command = claim["command"]
            assert int(command[command.index("--slots") + 1]) > 1
            assert int(command[command.index("--workers") + 1]) > int(
                command[command.index("--slots") + 1]
            )
