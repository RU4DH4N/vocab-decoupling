import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from corpora import documents
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

import execution.baseline_experiment as baseline
import execution.baseline_training as baseline_training
import execution.bootstrap as bootstrap
import execution.clocks as clocks
import execution.external as external
import execution.fitted_controls as controls
import execution.flops as flops
import execution.interfaces as interfaces
import execution.planning_experiment as planning
import execution.planning_training as planning_training
import execution.preflight as preflight
import execution.preparation as preparation
import execution.receiver_experiment as experiment
import execution.sweep as sweep
import execution.training as training
import execution.training_loop as training_loop
from claims.active import Active
from data.corpus import split_words
from execution.layout import Layout
from execution.protocol_io import load_protocol
from execution.streams import generate_streams, prompt_count
from framework.checkpoints import load_checkpoint
from orchestrator.context import load_context
from orchestrator.runner import Orchestrator
from pipeline._base import PipelineContract
from postprocess.analysis import analyse

ROOT = Path(__file__).resolve().parents[1]


def smoke_context(tmp_path):
    train, selection = tmp_path / "train.parquet", tmp_path / "selection.parquet"
    pq.write_table(pa.table({"text": documents(1, 60, (30, 90))}), train)
    pq.write_table(pa.table({"text": documents(2, 24, (30, 90))}), selection)
    inputs = json.loads((ROOT / "config/smoke.json").read_text())
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({**inputs, "passes": 20, "mauve": False}))
    return load_context(
        config, tmp_path / "output", train, selection, "cpu", "selection", None
    )


def in_process_workers(monkeypatch):

    modules = {
        "execution.training": training,
        "execution.receiver_experiment": experiment,
        "execution.planning_experiment": planning,
        "execution.baseline_experiment": baseline,
        "execution.baseline_training": baseline_training,
        "execution.planning_training": planning_training,
        "execution.preparation": preparation,
        "execution.fitted_controls": controls,
        "execution.sweep": sweep,
        "execution.preflight": preflight,
        "execution.flops": flops,
        "execution.clocks": clocks,
        "execution.bootstrap": bootstrap,
        "execution.interfaces": interfaces,
        "execution.external": external,
    }

    def load_external(options, device):
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        layout = Layout(output, 0)
        tokenizer = Tokenizer.from_file(str(layout.baseline_tokenizer))
        prompts = [
            example["prompt"]
            for split in ("selection", "reporting")
            if layout.evaluation_prompts(split).exists()
            for example in json.loads(layout.evaluation_prompts(split).read_text())
        ]
        context = 1 + max(len(tokenizer.encode(p).ids) for p in prompts)
        torch.manual_seed(0)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=tokenizer.get_vocab_size(),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=256,
            )
        )
        return model.to(device).eval(), tokenizer, context

    monkeypatch.setattr(external, "load_external", load_external)

    def worker(self, *arguments):
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", [self.module, *map(str, arguments)])
            modules[self.module].main()

    monkeypatch.setattr(PipelineContract, "worker", worker)
    return training, baseline_training, planning_training, controls


def test_complete_workflow_and_reuse(tmp_path, monkeypatch):
    context = smoke_context(tmp_path)
    training, baseline, planning, controls = in_process_workers(monkeypatch)
    runner = Orchestrator(
        [Active(context)], context.output / ".orchestrator", context=context
    )
    runner.inspect()
    runner.run(mode="run-missing", allow_remote=False, interactive=False)
    active = json.loads((context.output / "claims/active.json").read_text())
    assert len(active["measurements"]) == 7
    design = json.loads((context.output / "design.json").read_text())
    shared = design["shared"]
    assert shared["events"] > shared["interface"]["candidate_window"]
    cap = shared["corpus"]["max_unit_bytes"]
    assert any(
        len(word.encode()) > cap
        for text in pq.read_table(context.train_file)["text"].to_pylist()
        for word in split_words(text)
    )
    external = json.loads((context.output / "external/evaluation.json").read_text())
    assert external["chunked_windows"] > 0
    for path in context.output.glob("seed-*/*/generation.json"):
        if path.parent.name == "baseline":
            continue
        samples = json.loads(path.read_text())["samples"]
        assert {s["stop_reason"] for s in samples} <= {
            "byte-budget",
            "sender-context-limit",
        }
        assert all(s["forced_word_breaks"] >= 0 for s in samples)
    assert active["evidence_status"] == "partial"
    assert active["scientific_verdict"] is None
    assert "independent-seed-replication" not in active["missing_experiments"]
    for claim in Active(context).dependencies():
        bundle = json.loads(claim.outputs()[0].read_text())
        assert bundle["measurements"]
        assert bundle["scientific_verdict"] is None
        for measurement in bundle["measurements"].values():
            assert Path(measurement["path"]).is_file()
    for replicate in range(context.settings["seeds"]):
        assert (context.output / f"seed-{replicate}/primary/evaluation.json").is_file()
    for name in ("native", "baseline", "adapter", "trunk"):
        sweep = json.loads((context.output / f"sweep/{name}.json").read_text())
        assert sweep["selected"]["learning_rate"] in sweep_grid(context)
    layout = Layout(context.output, 0)
    config = layout.config()
    assert config["native_training"]["learning_rate"] is not None
    assert config["stages"][2]["alignment_weight"] is not None
    run = layout.run
    fitted = {
        arm: json.loads(
            (run / "controls/fresh-byte" / arm / "model.metrics.json").read_text()
        )
        for arm in ("correct", "shuffled", "native-only")
    }
    assert len({v["trainable_parameters"] for v in fitted.values()}) == 1
    assert len({v["target_bytes"] for v in fitted.values()}) == 1
    aligned = load_checkpoint(run / "fresh-byte/stage-0.pt")["model"]
    for arm in fitted:
        fitted_state = load_checkpoint(run / "controls/fresh-byte" / arm / "model.pt")[
            "model"
        ]
        for name, value in aligned.items():
            if not name.startswith("receiver.channels."):
                torch.testing.assert_close(
                    value, fitted_state["model." + name], rtol=0, atol=0
                )
    totals = sum(
        json.loads((run / "primary" / name).read_text())["target_bytes"]
        for name in (
            "native.metrics.json",
            "stage-0.metrics.json",
            "stage-1.metrics.json",
            "stage-2.metrics.json",
            "stage-3.metrics.json",
        )
    )
    assert (
        json.loads((run / "baseline/model.metrics.json").read_text())["target_bytes"]
        == totals
    )
    exposure = json.loads(layout.exposure.read_text())
    assert exposure["jobs"]["baseline/model"]["target_bytes"] == totals
    primary, _ = load_protocol(
        run / "primary/stage-3.pt", torch.device("cpu"), table=True
    )
    for variant in ("fresh-byte", "fresh-bpe"):
        replacement, _ = load_protocol(
            run / variant / "stage-2.pt", torch.device("cpu"), table=True
        )
        for name, value in primary.sender.state_dict().items():
            torch.testing.assert_close(
                value, replacement.sender.state_dict()[name], rtol=0, atol=0
            )
        native = load_checkpoint(run / variant / "native.pt")
        for name, value in native["model"].items():
            torch.testing.assert_close(
                value, replacement.receiver.native.state_dict()[name], rtol=0, atol=0
            )
        scores = json.loads((run / variant / "evaluation.json").read_text())["scores"]
        assert scores["correct"]["ce"] > 0
    assert not list(context.output.rglob("*.resume.pt"))
    written = json.loads((run / "primary/generation.json").read_text())["samples"]
    parallel, _ = generate_streams(
        experiment.generate_receiver,
        (context.output, 0, "primary", "selection", "cpu"),
        len(written),
        1,
        2,
    )
    for first, second in zip(written, parallel, strict=True):
        assert {k: v for k, v in first.items() if k != "seconds"} == {
            k: v for k, v in second.items() if k != "seconds"
        }
    fresh = Orchestrator(
        [Active(context)], context.output / ".orchestrator", context=context
    )
    fresh.inspect()
    fresh.run(mode="require-complete", allow_remote=False, interactive=False)

    reporting = tmp_path / "reporting.parquet"
    pq.write_table(pa.table({"text": documents(3, 12, (30, 90))}), reporting)
    report = load_context(
        tmp_path / "inputs.json",
        context.output,
        context.train_file,
        context.selection_file,
        "cpu",
        "reporting",
        reporting,
    )
    runner = Orchestrator(
        [Active(report)], report.output / ".orchestrator", context=report
    )
    runner.inspect()
    pending = {
        runner.graph.nodes[key].label
        for key in runner.graph.order
        if runner.states[key]["state"] != "cached"
    }
    assert pending and not any(
        label == "Prepare"
        or label.startswith(("TrainNative", "TrainStage", "Sweep"))
        or label.endswith((":train]", ":planner]", ":fit]", ":oracle-fit]"))
        for label in pending
    )
    runner.run(mode="run-missing", allow_remote=False, interactive=False)
    reported = json.loads((report.output / "claims-reporting/active.json").read_text())
    reporting_prompts = prompt_count(
        Layout(report.output, 0).evaluation_prompts("reporting")
    )
    design = json.loads((report.output / "design.json").read_text())
    assert reporting_prompts < design["shared"]["generation"]["samples"]
    for path in report.output.glob("seed-*/reporting/*/generation.json"):
        assert len(json.loads(path.read_text())["samples"]) == reporting_prompts

    assert reported["split"] == "reporting"
    assert "reporting-split-evaluation" not in reported["missing_experiments"]
    evaluation = run / "reporting/primary/evaluation.json"
    assert json.loads(evaluation.read_text())["split"] == "reporting"
    for split in ("selection", "reporting"):
        results, markdown = analyse(context.output, split)
        assert {"C1", "C2", "C3", "C4", "C5", "C6", "C7"} <= set(results)
        assert results["seeds"] == context.settings["seeds"]
        estimate = results["C5"]["shuffled_minus_real"]
        assert estimate["low"] <= estimate["estimate"] <= estimate["high"]
        assert "## C4" in markdown
        assert (
            results["C7"]["gap_to_external"].keys()
            == results["C7"]["gap_to_bpe"].keys()
        )
        assert results["C7"]["mauve"]["external"] == [None]

    cpu = torch.device("cpu")
    jobs = [
        (
            training,
            lambda layout: training.native(layout, "primary", cpu),
            "primary/native.pt",
        ),
        (
            baseline,
            lambda layout: baseline.train(layout.config(), layout, cpu),
            "baseline/model.pt",
        ),
        (
            planning,
            lambda layout: planning.fit_planner(layout.config(), layout, cpu),
            "planning/planner.pt",
        ),
        (
            planning,
            lambda layout: planning.fit_future(layout.config(), layout, cpu, "planner"),
            "planning/lookahead.pt",
        ),
        (
            controls,
            lambda layout: controls.train(layout, "fresh-bpe", "correct", cpu),
            "controls/fresh-bpe/correct/model.pt",
        ),
    ]
    for index, (module, fit, relative) in enumerate(jobs):
        root = tmp_path / f"resume-{index}"
        shutil.copytree(context.output, root)
        copy = Layout(root, 0)
        output = copy.run / relative
        resume = output.with_name(f"{output.stem}.resume.pt")
        assert not resume.exists()
        expected = load_checkpoint(output)["model"]
        expected_metrics = json.loads(output.with_suffix(".metrics.json").read_text())
        real_save = training_loop.save

        saves = []

        def interrupt(path, state):
            real_save(path, state)
            saves.append(state["step"])
            if len(saves) == 1:
                raise OSError("simulated preemption after durable checkpoint")

        with monkeypatch.context() as patch:
            patch.setattr(training_loop, "save", interrupt)
            with pytest.raises(OSError, match="simulated preemption"):
                fit(copy)
        assert resume.exists()
        fit(copy)
        assert not resume.exists()
        actual = load_checkpoint(output)["model"]
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        metrics = json.loads(output.with_suffix(".metrics.json").read_text())
        assert metrics["target_bytes"] == expected_metrics["target_bytes"]
        assert metrics["steps"] == expected_metrics["steps"]


def sweep_grid(context):
    design = json.loads((context.output / "design.json").read_text())
    return design["grids"]["learning_rate"]
