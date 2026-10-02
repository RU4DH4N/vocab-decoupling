import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest

import orchestrator.context as context_module
import orchestrator.graph as graph_module
import pipeline._base as base_module
from framework.checkpoints import write_json
from orchestrator.artifacts import Artifacts
from orchestrator.context import load_context
from orchestrator.contract import Contract
from orchestrator.discovery import discover
from orchestrator.runner import DependencyError, Orchestrator
from pipeline.prepare import Prepare
from postprocess.graph import render


class Probe(Contract):
    def __init__(
        self,
        name,
        root,
        dependencies=(),
        action=None,
        resource=None,
        inputs=(),
        source=None,
        load=0,
    ):
        self.name, self.root, self.parents = name, root, dependencies
        self.load = load
        self.action, self.resource, self.files, self.source = (
            action,
            resource,
            inputs,
            source,
        )

    @property
    def label(self):
        return self.name

    def parameters(self):
        return {"name": self.name, "root": str(self.root)}

    def dependencies(self):
        return self.parents

    def inputs(self):
        return self.files

    def sources(self):
        return super().sources() + ((self.source,) if self.source else ())

    def outputs(self):
        return (self.root / (self.name + ".txt"),)

    def resources(self):
        return frozenset({self.resource}) if self.resource else frozenset()

    def weight(self):
        return self.load

    def run(self, dependencies):
        assert set(dependencies) == {p.key for p in self.parents}
        if self.action:
            self.action()
        self.outputs()[0].write_text(self.name)


def runner(roots, tmp_path, workers=2, slots=1):
    return Orchestrator(roots, tmp_path / "state", workers=workers, slots=slots)


def run(run):
    return run.run(mode="run-missing", tell=lambda _: None)


def test_shared_dependencies_execute_once_and_second_run_is_cached(tmp_path):
    calls = []
    shared1 = Probe("shared", tmp_path, action=lambda: calls.append(1))
    shared2 = Probe("shared", tmp_path, action=lambda: calls.append(2))
    a = Probe("claim-a", tmp_path, (shared1,))
    b = Probe("claim-b", tmp_path, (shared2,))
    first = runner([a, b], tmp_path)
    assert len(first.graph.nodes) == 3
    run(first)
    assert len(calls) == 1
    again = runner([a, b], tmp_path)
    run(again)
    assert len(calls) == 1
    assert all(s["state"] == "cached" for s in again.states.values())


def test_cycle_fails_before_any_execution(tmp_path):
    a, b = Probe("a", tmp_path), Probe("b", tmp_path)
    a.parents, b.parents = (b,), (a,)
    with pytest.raises(ValueError, match="a -> b -> a"):
        runner([a], tmp_path)
    assert not a.outputs()[0].exists()


def test_invalid_dependency_fails_before_execution(tmp_path):
    with pytest.raises(TypeError, match="Contract instances"):
        runner([Probe("a", tmp_path, ("unknown",))], tmp_path)


def test_conflicting_shared_identity_is_rejected(tmp_path):
    a, b = Probe("same", tmp_path), Probe("same", tmp_path, resource="gpu")
    with pytest.raises(ValueError, match="conflicting declarations"):
        runner([a, b], tmp_path)


def test_different_identity_cannot_write_same_output(tmp_path):
    class Other(Probe):
        pass

    with pytest.raises(ValueError, match="output collision"):
        runner([Probe("same", tmp_path), Other("same", tmp_path)], tmp_path)


def test_noninteractive_inspection_never_starts_missing_work(tmp_path):
    node = Probe("a", tmp_path)
    r = runner([node], tmp_path)
    with pytest.raises(DependencyError, match="non-interactive"):
        r.run(tell=lambda _: None)
    assert not node.outputs()[0].exists()
    with pytest.raises(DependencyError, match="missing/stale"):
        r.run(mode="require-complete", tell=lambda _: None)
    assert not node.outputs()[0].exists()


def test_declining_prompt_never_runs(tmp_path):
    node = Probe("a", tmp_path)
    with pytest.raises(DependencyError, match="declined"):
        runner([node], tmp_path).run(
            interactive=True, ask=lambda _: "no", tell=lambda _: None
        )
    assert not node.outputs()[0].exists()


def test_remote_permission_is_separate(tmp_path):
    node = Probe("a", tmp_path)
    node.remote_required = True
    with pytest.raises(DependencyError, match="explicit"):
        run(runner([node], tmp_path))
    runner([node], tmp_path).run(
        mode="run-missing", allow_remote=True, tell=lambda _: None
    )
    assert node.outputs()[0].exists()


def test_independent_cpu_work_really_overlaps(tmp_path):
    barrier = threading.Barrier(2, timeout=3)
    nodes = [Probe(name, tmp_path, action=barrier.wait) for name in ("a", "b")]
    run(runner(nodes, tmp_path, workers=2))
    assert all(n.outputs()[0].exists() for n in nodes)


def test_exclusive_resource_is_serialized(tmp_path):
    active = 0
    maximum = 0

    def work():
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        time.sleep(0.03)
        active -= 1

    nodes = [
        Probe(name, tmp_path, action=work, resource="test-exclusive-device")
        for name in ("a", "b", "c")
    ]
    run(runner(nodes, tmp_path, workers=3))
    assert maximum == 1


def test_failure_blocks_descendants_not_unrelated_claim(tmp_path):
    def fail():
        raise RuntimeError("broken")

    bad = Probe("bad", tmp_path, action=fail)
    child = Probe("child", tmp_path, (bad,))
    good = Probe("good", tmp_path)
    r = runner([child, good], tmp_path)
    with pytest.raises(DependencyError, match="incomplete roots"):
        run(r)
    assert r.states[bad.key]["state"] == "failed"
    assert r.states[child.key]["state"] == "blocked"
    assert r.states[good.key]["state"] == "done"


def test_missing_output_never_gets_success_manifest(tmp_path):
    class Broken(Probe):
        def run(self, dependencies):
            pass

    node = Broken("broken", tmp_path)
    r = runner([node], tmp_path)
    with pytest.raises(DependencyError):
        run(r)
    assert not r.artifacts.path(node.key).exists()


def test_code_change_reruns_only_steps_whose_inputs_change(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# v1")
    a = Probe("a", tmp_path, source=source)
    b = Probe("b", tmp_path, (a,))
    c = Probe("c", tmp_path, (b,))
    run(runner([c], tmp_path))
    source.write_text("# v2")
    changed = runner([c], tmp_path)
    changed.inspect()
    assert changed.states[a.key]["state"] == "pending"
    run(changed)
    assert changed.states[a.key]["state"] == "done"
    assert changed.states[b.key]["state"] == "cached"
    assert changed.states[c.key]["state"] == "cached"


def test_changed_outputs_rerun_consumers_until_outputs_match(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# v1")
    marker = tmp_path / "marker"
    marker.write_text("one")

    def write():
        (tmp_path / "a.txt").write_text(marker.read_text())

    class Writer(Probe):
        def run(self, dependencies):
            write()

    a = Writer("a", tmp_path, source=source)
    b = Probe("b", tmp_path, (a,))
    c = Probe("c", tmp_path, (b,))
    run(runner([c], tmp_path))
    source.write_text("# v2")
    marker.write_text("two")
    changed = runner([c], tmp_path)
    run(changed)
    assert changed.states[a.key]["state"] == "done"
    assert changed.states[b.key]["state"] == "done"
    assert changed.states[c.key]["state"] == "cached"


def test_tampered_inputs_and_outputs_invalidate_cache(tmp_path):
    input_file = tmp_path / "input"
    input_file.write_text("old")
    a = Probe("a", tmp_path, inputs=(input_file,))
    r = runner([a], tmp_path)
    run(r)
    input_file.write_text("new")
    assert not r.artifacts.inspect()[0][a.key][0]
    run(r)
    a.outputs()[0].write_text("tampered")
    assert not r.artifacts.inspect()[0][a.key][0]


def test_input_mutation_during_execution_never_commits(tmp_path):
    source = tmp_path / "input"
    source.write_text("before")
    node = Probe(
        "a", tmp_path, inputs=(source,), action=lambda: source.write_text("during")
    )
    r = runner([node], tmp_path)
    with pytest.raises(DependencyError):
        run(r)
    assert "changed during execution" in r.states[node.key]["reason"]
    assert not r.artifacts.path(node.key).exists()


def test_config_snapshot_is_not_replaced_without_consent(tmp_path):
    config = tmp_path / "source.json"
    config.write_text('{"seed": 2}')
    context = load_context(
        config,
        tmp_path,
        tmp_path / "train",
        tmp_path / "selection",
        "cpu",
        "selection",
        None,
    )
    context.config_snapshot.parent.mkdir(parents=True)
    context.config_snapshot.write_text('{"seed": 1}')
    r = Orchestrator(
        [Probe("a", tmp_path)], tmp_path / ".orchestrator", context=context
    )
    with pytest.raises(DependencyError, match="declined"):
        r.run(interactive=True, ask=lambda _: "no", tell=lambda _: None)
    assert json.loads(context.config_snapshot.read_text()) == {"seed": 1}


@pytest.mark.parametrize("when", ["before", "during"])
def test_modified_configuration_snapshot_never_commits(tmp_path, monkeypatch, when):
    config = tmp_path / "config.json"
    config.write_text('{"seed": 1}')
    context = load_context(
        config,
        tmp_path,
        tmp_path / "train",
        tmp_path / "selection",
        "cpu",
        "selection",
        None,
    )

    def tamper():
        context.config_snapshot.write_text('{"seed": 999}')

    node = Probe("config-reader", tmp_path, action=tamper if when == "during" else None)
    r = Orchestrator([node], tmp_path / ".orchestrator", context=context)
    original_execute = r._execute

    def execute(key):
        if when == "before":
            tamper()
        original_execute(key)

    monkeypatch.setattr(r, "_execute", execute)
    with pytest.raises(DependencyError, match="incomplete roots"):
        run(r)
    assert "configuration snapshot differs" in r.states[node.key]["reason"]
    assert not r.artifacts.path(node.key).exists()


def test_collected_cache_survives_repository_and_data_path_changes(
    tmp_path, monkeypatch
):

    config = tmp_path / "config.json"
    config.write_text(
        (Path(__file__).resolve().parents[1] / "config/smoke.json").read_text()
    )
    prior = None
    for host in ("remote", "local"):
        root = tmp_path / host
        root.mkdir()
        monkeypatch.setattr(graph_module, "ROOT", root)
        monkeypatch.setattr(base_module, "ROOT", root)
        train, selection = root / "train", root / "selection"
        train.write_text("train bytes")
        selection.write_text("selection bytes")
        context = load_context(
            config,
            root / "artifacts" / "pilot",
            train,
            selection,
            "cuda" if host == "remote" else "cpu",
            "selection",
            None,
        )
        node = Prepare(context)
        graph = graph_module.Graph((node,))
        store = Artifacts(graph, context.output / ".orchestrator")
        for output in node.outputs():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("same collected output")
        if prior is None:
            prior = (node.key, store.commit(node.key, {}))
        else:
            assert node.key == prior[0]
            write_json(store.path(node.key), prior[1])
            assert store.inspect()[0][node.key] == (True, "ready")


def test_configuration_loaded_once_and_not_shared_mutably(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"seed": 1, "seeds": 1, "model": {"width": 32}}')
    context = load_context(
        config,
        tmp_path / "out",
        tmp_path / "train",
        tmp_path / "selection",
        "cpu",
        "selection",
        None,
    )
    config.write_text('{"seed": 999}')
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *args, **kwargs: pytest.fail("contract tried to read configuration"),
    )
    node = Prepare(context)
    assert node.parameters()["config"]["seed"] == 1
    settings = context.settings
    settings["model"]["width"] = 999
    assert context.settings["model"]["width"] == 32


def test_discovery_loads_one_contract_per_file():
    assert discover("claims/planning_ahead.py").__name__ == "PlanningAhead"
    assert discover("claims.active").__name__ == "Active"
    with pytest.raises(ValueError):
        discover("../execution/receiver_experiment.py")
    with pytest.raises(ValueError):
        discover("pipeline/bpe_reference.py")


def test_unavailable_contract_cannot_manufacture_a_result(tmp_path):
    class Unavailable(Probe):
        unavailable_reason = "required numerical experiment is not implemented"

    claim = Unavailable("unavailable", tmp_path)
    r = runner([claim], tmp_path)
    with pytest.raises(DependencyError):
        run(r)
    assert r.states[claim.key]["state"] == "blocked"
    assert not claim.outputs()[0].exists()


def test_graph_export_and_render_are_separate(tmp_path):
    a = Probe("a<&", tmp_path)
    b = Probe("b", tmp_path, (a,))
    r = runner([b], tmp_path)
    r.inspect()
    export = tmp_path / "graph.json"
    r.export(export)
    graph = json.loads(export.read_text())
    assert len(graph["edges"]) == 1
    svg = render(graph)
    ElementTree.fromstring(svg)
    assert "a&lt;&amp;" in svg
    assert not a.outputs()[0].exists()


def test_graph_and_missing_file_diagnostics_use_relative_paths(tmp_path, monkeypatch):

    class PortableProbe(Probe):
        def parameters(self):
            return {"name": self.name}

    monkeypatch.setattr(graph_module, "ROOT", tmp_path)
    output = tmp_path / "artifacts" / "run"
    node = PortableProbe("a", output)
    r = runner([node], output)
    r.inspect()
    pending = r.snapshot()
    assert pending["path_base"] == "repository"
    assert pending["nodes"][0]["outputs"] == ["artifacts/run/a.txt"]
    assert str(tmp_path.resolve()) not in json.dumps(pending)
    run(r)
    assert str(tmp_path.resolve()) not in json.dumps(r.snapshot())
    cached = runner([node], output)
    cached.inspect()
    assert cached.states[node.key]["state"] == "cached"
    assert str(tmp_path.resolve()) not in json.dumps(cached.snapshot())


def test_missing_repo_inputs_have_portable_context_identities(tmp_path, monkeypatch):

    monkeypatch.setattr(context_module, "ROOT", tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}")
    context = load_context(
        config,
        tmp_path / "artifacts" / "run",
        tmp_path / "datasets" / "train.parquet",
        tmp_path / "datasets" / "selection.parquet",
        "cpu",
        "selection",
        None,
    )
    assert context.data_identity == (
        "missing:datasets/train.parquet",
        "missing:datasets/selection.parquet",
    )


@pytest.mark.parametrize("script", ["claims/active.py", "claims/planning_ahead.py"])
def test_direct_file_entrypoint_from_another_working_directory(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-missing" in result.stdout


def test_step_timings_survive_a_later_cached_run(tmp_path):
    node = Probe("timed", tmp_path, action=lambda: time.sleep(0.01))
    run(runner([node], tmp_path))
    timings = json.loads((tmp_path / "state" / "timings.json").read_text())
    assert timings[node.key]["label"] == "timed"
    assert timings[node.key]["seconds"] >= 0.01
    run(runner([node], tmp_path))
    assert json.loads((tmp_path / "state" / "timings.json").read_text()) == timings


def overlap(tmp_path, weights, slots):
    running, snapshots = {}, []
    lock = threading.Lock()

    def work(name, weight):
        with lock:
            running[name] = min(weight, slots)
            snapshots.append(dict(running))
        time.sleep(0.05)
        with lock:
            del running[name]

    nodes = [
        Probe(f"n{i}", tmp_path, action=lambda i=i, w=w: work(f"n{i}", w), load=w)
        for i, w in enumerate(weights)
    ]
    run(runner(nodes, tmp_path, workers=len(nodes), slots=slots))
    return snapshots


def test_light_steps_share_the_accelerator_up_to_its_slots(tmp_path):
    snapshots = overlap(tmp_path, [1, 1, 1, 1], 3)
    assert max(len(s) for s in snapshots) == 3


def test_steps_never_exceed_the_slots_and_exclusive_steps_run_alone(tmp_path):
    snapshots = overlap(tmp_path, [2, 2, 1, 5, 1, 0], 3)
    assert all(sum(s.values()) <= 3 for s in snapshots)
    assert all(
        not ("n3" in s and any(w for name, w in s.items() if name != "n3"))
        for s in snapshots
    )
    assert any(len([w for w in s.values() if w]) > 1 for s in snapshots)


def test_a_step_that_fails_while_sharing_is_retried_alone(tmp_path):
    attempts = []
    company = []

    def flaky():
        attempts.append(len(company))
        if len(attempts) == 1:
            time.sleep(0.05)
            raise RuntimeError("out of memory")

    def neighbour():
        company.append(1)
        time.sleep(0.05)
        company.pop()

    a = Probe("a", tmp_path, action=flaky, load=1)
    b = Probe("b", tmp_path, action=neighbour, load=1)
    runs = runner([a, b], tmp_path, workers=2, slots=2)
    run(runs)
    assert len(attempts) == 2 and attempts[1] == 0
    assert runs.states[a.key]["state"] == "done"


def test_a_step_that_fails_alone_is_not_retried(tmp_path):
    attempts = []

    def broken():
        attempts.append(1)
        raise RuntimeError("bug")

    a = Probe("a", tmp_path, action=broken, load=1)
    runs = runner([a], tmp_path, workers=2, slots=2)
    with pytest.raises(DependencyError):
        run(runs)
    assert attempts == [1] and runs.states[a.key]["state"] == "failed"
