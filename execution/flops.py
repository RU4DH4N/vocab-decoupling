import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from data.communication import ProtocolBatch, ReceiverUnits
from data.packed import PackedUnits
from data.protocol_corpus import EventCorpus
from data.tokenizers import load_tokenizer
from execution.baseline_training import baseline_loss, bpe_batch, build_baseline
from execution.design import LEARNING_RATE_GRID, PRIMARY_STAGES, REPLACEMENT_STAGES
from execution.fitted_controls import FittedControl
from execution.layout import Layout
from execution.planning_training import future_inputs, make_plan_batch
from execution.protocol_io import build_protocol
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import build_native, encoded, native_loss
from execution.windows import protocol_batch
from framework.checkpoints import write_json
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel, TrainingStage, gather_events
from models.protocol.planner import EventPlanner
from models.protocol.receiver import Receiver
from models.shared.symbols import BYTE_IGNORE_INDEX

SAMPLED_WINDOWS = 8


def counted(step: Callable[[], object]) -> int:
    with FlopCounterMode(display=False) as counter:
        step()
    return counter.get_total_flops()


def _stage(
    model: ProtocolModel[Receiver],
    name: TrainingStage,
    alignment_weight: float,
) -> StageTrainer:
    settings = StageConfig(
        name=name,
        steps=1,
        learning_rate=LEARNING_RATE_GRID[0],
        weight_decay=0.0,
        alignment_weight=alignment_weight,
        grad_clip=1.0,
        warmup_steps=0,
        precision="auto",
    )
    return StageTrainer(model, settings, {"flops": name}, np.random.default_rng(0))


def _backward(module: nn.Module, loss: Tensor) -> None:
    loss.backward()
    module.zero_grad(set_to_none=True)


def _loss(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    return F.cross_entropy(
        logits.float().flatten(0, 1), targets.flatten(), ignore_index=ignore_index
    )


def _protocol(
    config: dict,
    corpus: EventCorpus,
    table: ReceiverUnits,
    native: dict,
    device: torch.device,
) -> ProtocolModel[Receiver]:
    options = {
        "sender": config["sender"],
        "native": native,
        "interface": {**config["interface"], "output_symbols": table.inventory.outputs},
    }
    return build_protocol(options, corpus.vocab, device)


def _receiver_flops(
    config: dict,
    corpus: EventCorpus,
    kind: str,
    table: ReceiverUnits,
    rows: Tensor,
    device: torch.device,
) -> tuple[dict[str, int], ProtocolModel[Receiver], ProtocolBatch]:
    batch = protocol_batch(config, table, rows, device)
    native, native_options = build_native(config, table, device)
    flops = {
        f"native/{kind}": counted(
            lambda: _backward(native, native_loss(native, table, batch, config))
        )
    }
    model = _protocol(config, corpus, table, native_options, device)
    for name in PRIMARY_STAGES if kind == "bytes" else REPLACEMENT_STAGES:
        trainer = _stage(model, name, 1.0 if name == "alignment" else 0.0)
        flops[f"stage/{kind}/{name}"] = counted(lambda: trainer.step(batch))
    for arm in ("correct", "shuffled", "native-only"):
        control = FittedControl(model, arm, 0).to(device)
        flops[f"control/{kind}/{arm}"] = counted(
            lambda: _backward(control, _loss(control(batch), batch.targets))
        )
    return flops, model, batch


def _planning_flops(
    config: dict,
    corpus: EventCorpus,
    model: ProtocolModel[Receiver],
    batch: ProtocolBatch,
    rows: Tensor,
    device: torch.device,
) -> dict[str, int]:
    model.eval().requires_grad_(False)
    planner = EventPlanner(**config["planner"]["model"]).to(device)
    with torch.no_grad():
        messages = model.sender(rows[:, :-1].to(device)).flatten(0, 1)
    packed = PackedUnits.from_units(corpus.vocab, device)
    previous, targets, _ = packed.teacher_forcing(
        rows[:, 1:].reshape(-1), width=corpus.max_bytes + 1
    )
    flops = {
        "planner": counted(
            lambda: _backward(
                planner,
                _loss(
                    planner(messages, previous),
                    targets,
                    ignore_index=BYTE_IGNORE_INDEX,
                ),
            )
        )
    }
    planner.eval()
    flops["rollout"] = counted(
        lambda: make_plan_batch(config, model, planner, rows, device)
    )
    record = make_plan_batch(config, model, planner, rows, device)
    lookahead = LookaheadReceiver(
        model.receiver, config["lookahead"]["horizon"], config["lookahead"]["rank"]
    ).to(device)
    future, log_probs, available = future_inputs(record, "real")
    candidates = gather_events(record["memory"], batch.candidate_event_ids)
    flops["lookahead"] = counted(
        lambda: _backward(
            lookahead,
            _loss(
                lookahead(
                    batch.receiver_ids,
                    candidates,
                    batch.candidate_event_ids,
                    batch.frontier,
                    future,
                    log_probs,
                    available,
                    batch.frontier,
                    hard=False,
                ).logits,
                batch.targets,
            ),
        )
    )
    return flops


def per_window(
    config: dict,
    layout: Layout,
    corpus: EventCorpus,
    windows: list[Tensor],
    device: torch.device,
) -> dict[str, float]:
    tables = {
        kind: encoded(config, layout, variant, corpus)
        for kind, variant in (("bytes", "primary"), ("bpe", "fresh-bpe"))
    }
    tokenizer = load_tokenizer(layout.baseline_tokenizer)
    baseline, _ = build_baseline(config, tokenizer, device)
    totals: dict[str, list[int]] = {}
    for rows in windows:
        flops = {}
        for kind, table in tables.items():
            counts, model, batch = _receiver_flops(
                config, corpus, kind, table, rows, device
            )
            flops |= counts
            if kind == "bytes":
                flops |= _planning_flops(config, corpus, model, batch, rows, device)
        ids, targets = bpe_batch(corpus, rows, tokenizer, device)
        flops["baseline"] = counted(
            lambda: _backward(baseline, baseline_loss(baseline, ids, targets, config))
        )
        for name, value in flops.items():
            totals.setdefault(name, []).append(value)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def account(layout: Layout, device: torch.device) -> dict:
    config = layout.config()
    design = json.loads(layout.design.read_text())
    corpus = EventCorpus.load(layout.corpus)
    rng = np.random.default_rng(config["evaluation_seed"])
    corpus, windows = corpus.restricted(
        [
            corpus.windows("train", config["events"], 1, rng)
            for _ in range(SAMPLED_WINDOWS)
        ]
    )
    window = per_window(config, layout, corpus, windows, device)
    batch = config["batch_size"]

    def job(name: str, steps: int) -> float:
        return window[name] * batch * steps

    stages = {s["name"]: s["steps"] for s in config["stages"]}
    replacement = {s["name"]: s["steps"] for s in config["replacement_stages"]}
    native_steps = config["native_training"]["steps"]
    system = job("native/bytes", native_steps) + sum(
        job(f"stage/bytes/{name}", steps) for name, steps in stages.items()
    )
    fresh = {
        kind: job(f"native/{kind}", native_steps)
        + sum(job(f"stage/{kind}/{name}", steps) for name, steps in replacement.items())
        for kind in ("bytes", "bpe")
    }
    fresh_interface = {
        kind: sum(
            job(f"stage/{kind}/{name}", steps) for name, steps in replacement.items()
        )
        for kind in ("bytes", "bpe")
    }
    baseline_steps = native_steps + sum(stages.values())
    retrain = job("baseline", baseline_steps)
    control_steps = config["control_training"]["steps"]
    sweep_steps = design["steps"]["sweep"]
    candidates = len(design["grids"]["learning_rate"])
    trunk_candidates = candidates + len(design["grids"]["alignment_weight"]) - 1
    return {
        "method": "FlopCounterMode on one step per job type, averaged over sampled windows, times windows per step and steps",
        "sampled_windows": SAMPLED_WINDOWS,
        "per_window_step": window,
        "totals": {
            "protocol_system": system,
            "replacement_receiver": fresh,
            "replacement_interface_only": fresh_interface,
            "bpe_retrain": retrain,
            "planning": job("planner", config["planner"]["steps"])
            + 2 * job("lookahead", config["lookahead"]["steps"])
            + job("rollout", config["lookahead"]["steps"]),
            "controls": sum(
                job(f"control/{kind}/{arm}", control_steps)
                for kind in ("bytes", "bpe")
                for arm in ("correct", "shuffled", "native-only")
            ),
            "sweeps": candidates
            * sweep_steps
            * (
                window["native/bytes"]
                + window["baseline"]
                + window["stage/bytes/alignment"]
                + window["stage/bytes/communication"]
            )
            * batch
            + trunk_candidates * sweep_steps * window["stage/bytes/trunk"] * batch,
        },
        "ratios": {
            f"replacement_{kind}_to_bpe_retrain": fresh[kind] / retrain
            for kind in ("bytes", "bpe")
        }
        | {
            f"replacement_{kind}_to_protocol_system": fresh[kind] / system
            for kind in ("bytes", "bpe")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, 0)
    write_json(args.output / "flops.json", account(layout, torch.device(args.device)))


if __name__ == "__main__":
    main()
