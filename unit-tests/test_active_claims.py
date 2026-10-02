import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from claims.active import Active
from framework.checkpoints import write_json
from orchestrator.context import load_context
from orchestrator.discovery import discover
from orchestrator.graph import Graph
from orchestrator.runner import DependencyError, Orchestrator
from pipeline._result import ResultBundle
from pipeline.prepare import Prepare

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = (
    "continuous_communication",
    "learned_protocol",
    "correspondence_bootstrap",
    "independent_clocks",
    "planning_ahead",
    "receiver_replacement",
    "remote_hypothesis",
)


@pytest.fixture
def context(tmp_path):
    return load_context(
        ROOT / "config/smoke.json",
        tmp_path / "output",
        tmp_path / "train.parquet",
        tmp_path / "selection.parquet",
        "cpu",
        "selection",
        None,
    )


@pytest.mark.parametrize("name", (*CLAIMS, "active"))
def test_every_claim_is_discoverable_and_has_real_dependencies(name, context):
    node = discover(f"claims/{name}.py")(context)
    assert node.unavailable_reason is None
    graph = Graph((node,))
    assert node.dependencies()
    assert any(isinstance(task, Prepare) for task in graph.nodes.values())
    assert node.outputs()


def test_union_shares_prerequisites_and_does_not_include_visualization(context):
    active = Active(context)
    roots = tuple(discover(f"claims/{name}.py")(context) for name in CLAIMS)
    union = Graph((active, *roots))
    assert sum(isinstance(task, Prepare) for task in union.nodes.values()) == 1
    assert all(
        not type(task).__module__.startswith("postprocess.")
        for task in union.nodes.values()
    )
    assert len(
        {path for task in union.nodes.values() for path in task.outputs()}
    ) == sum(len(task.outputs()) for task in union.nodes.values())


def test_missing_prerequisite_does_not_start_work_or_create_result(context):
    node = discover("claims/continuous_communication.py")(context)
    runner = Orchestrator([node], context.output / ".orchestrator", context=context)
    runner.inspect()
    with pytest.raises(DependencyError):
        runner.run(mode="require-complete", allow_remote=False, interactive=False)
    assert not node.outputs()[0].exists()


@pytest.mark.parametrize("name", (*CLAIMS, "active"))
def test_claim_script_runs_from_another_directory(name, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "claims" / f"{name}.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-missing" in result.stdout


def test_result_rejects_undeclared_measurement(context, tmp_path):
    path = tmp_path / "untracked.json"
    write_json(path, {"score": 1})

    class Untracked(ResultBundle):
        result_name = "untracked"

        def measurements(self):
            return {"untracked": path}

    node = Untracked(context)
    with pytest.raises(ValueError, match="declared JSON"):
        node.run({})
    assert not node.outputs()[0].exists()


def test_claims_carry_no_verdict_registry():
    for path in (ROOT / "claims").glob("*.py"):
        source = path.read_text()
        assert "ClaimDefinition" not in source
        assert "established-locally" not in source
        assert "NotImplementedError" not in source


def test_every_seed_trains_and_only_declared_baseline_seeds_exist(context):
    union = Graph((Active(context),))
    labels = {task.label for task in union.nodes.values()}
    for replicate in range(context.settings["seeds"]):
        assert f"TrainStage[seed {replicate}:primary:2:trunk]" in labels
    baselines = {label for label in labels if label.startswith("BPEReference")}
    assert {label.split(":")[0] for label in baselines} == {
        f"BPEReference[seed {replicate}"
        for replicate in range(context.settings["baseline_seeds"])
    }
    sweeps = [label for label in labels if label.startswith("Sweep")]
    assert len(sweeps) == 4


def test_selection_and_reporting_claims_never_share_an_identity(context):
    reporting = replace(context, split="reporting", reporting_file=context.train_file)
    for name in (*CLAIMS, "active"):
        claim = discover(f"claims/{name}.py")
        selection, final = claim(context), claim(reporting)
        assert selection.key != final.key
        assert selection.outputs() != final.outputs()
