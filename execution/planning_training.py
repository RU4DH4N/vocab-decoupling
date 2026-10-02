import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from data.packed import PackedUnits
from data.protocol_corpus import EventCorpus
from execution.exposure_design import verify_exposure
from execution.layout import Layout, required
from execution.protocol_io import load_protocol
from execution.training import encoded, identity, stage_path
from execution.training_loop import Job, train_steps
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import (
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from framework.resume import (
    configuration_fingerprint,
    resume_path,
)
from framework.runtime import (
    autocast_context,
    device_synchronize,
    set_seed,
)
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.planner import EventPlanner, roll_ahead_batch
from models.protocol.receiver import Receiver
from models.shared.symbols import BYTE_IGNORE_INDEX


def primary_path(config: dict, layout: Layout) -> Path:
    return stage_path(layout, "primary", len(config["stages"]) - 1)


def load_planner(config: dict, layout: Layout, device: torch.device) -> EventPlanner:
    path = layout.run / "planning" / "planner.pt"
    payload = load_checkpoint(path, map_location=device)
    if payload["metadata"]["sender_sha256"] != file_sha256(
        primary_path(config, layout)
    ):
        raise ValueError("planner belongs to another sender checkpoint")
    planner = EventPlanner(**config["planner"]["model"]).to(device)
    planner.load_state_dict(payload["model"], strict=True)
    return planner.eval()


def fit_planner(config: dict, layout: Layout, device: torch.device) -> None:
    options = config["planner"]
    learning_rate = required(options["learning_rate"], "planner learning rate")
    set_seed(options["seed"])
    model, _ = load_protocol(primary_path(config, layout), device, table=True)
    model.requires_grad_(False).eval()
    corpus = EventCorpus.load(layout.corpus)
    packed = PackedUnits.from_units(corpus.vocab, device)
    planner = EventPlanner(**options["model"]).to(device)
    optimizer = torch.optim.AdamW(
        planner.parameters(),
        lr=learning_rate,
        weight_decay=options["weight_decay"],
    )
    rng = np.random.default_rng(options["data_seed"])
    provenance = identity(
        config, layout, [primary_path(config, layout)], "execution.planning_training"
    )
    fingerprint = configuration_fingerprint(
        {**provenance, "job": "planner", "learning_rate": learning_rate}
    )
    output = layout.run / "planning" / "planner.pt"

    def step(index: int) -> tuple[Tensor, dict[str, float], str]:
        rows = corpus.windows("train", config["events"], config["batch_size"], rng)
        with torch.no_grad():
            messages = model.sender(rows[:, :-1].to(device)).flatten(0, 1)
        previous, targets, lengths = packed.teacher_forcing(
            rows[:, 1:].reshape(-1), width=corpus.max_bytes + 1
        )
        with autocast_context(device, config["precision"]):
            loss = F.cross_entropy(
                planner(messages, previous).float().flatten(0, 1),
                targets.flatten(),
                ignore_index=BYTE_IGNORE_INDEX,
            )
        return loss, {"target_bytes": int(lengths.sum().item())}, ""

    planner.train()
    outcome = train_steps(
        Job(
            label="planner",
            steps=options["steps"],
            learning_rate=learning_rate,
            warmup_steps=options["warmup_steps"],
            grad_clip=options["grad_clip"],
            interval=config["checkpoint_interval"],
            resume=resume_path(output),
            fingerprint=fingerprint,
            modules={"planner": planner},
            parameters=list(planner.parameters()),
            optimizer=optimizer,
            rng=rng,
            device=device,
            counters=("target_bytes",),
        ),
        step,
    )
    completed = outcome.completed
    byte_count = int(outcome.counters["target_bytes"])
    verify_exposure(layout.exposure, "planning/planner", completed, byte_count)
    save_checkpoint(
        output,
        planner,
        {
            "architecture": "reset-byte-event-planner",
            "model_config": options["model"],
            "identity": provenance,
            "sender_sha256": file_sha256(primary_path(config, layout)),
        },
    )
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "steps": completed,
            "seconds": outcome.seconds,
            "target_bytes": byte_count,
            "parameters": sum(p.numel() for p in planner.parameters()),
        },
    )
    resume_path(output).unlink(missing_ok=True)


@torch.no_grad()
def make_plan_batch(
    config: dict,
    model: ProtocolModel[Receiver],
    planner: EventPlanner,
    rows: Tensor,
    device: torch.device,
) -> dict:
    ids = rows[:, :-1].to(device)
    plan = roll_ahead_batch(
        model.sender,
        planner,
        model.sender.trunk_table[ids],
        horizon=config["lookahead"]["horizon"],
        width=config["lookahead"]["hypotheses"],
        max_symbols=config["planner"]["max_symbols"],
    )
    return {
        "rows": rows.cpu(),
        "memory": plan.memory,
        "future": plan.messages,
        "log_probs": plan.log_probs,
        "available": plan.available,
        "proposals": plan.proposals,
    }


@torch.no_grad()
def oracle_future(
    model: ProtocolModel[Receiver],
    rows: Tensor,
    horizon: int,
    device: torch.device,
) -> dict[str, Tensor]:
    full = model.sender(rows.to(device))
    batch, events = rows.shape[0], rows.shape[1] - 1
    future = full.new_zeros(batch, events, horizon, 1, full.shape[-1])
    available = torch.zeros(batch, events, horizon, 1, dtype=torch.bool, device=device)
    for step in range(horizon):
        reach = events - step
        if reach <= 0:
            break
        future[:, :reach, step, 0] = full[
            :, 1 + step + torch.arange(reach, device=device)
        ]
        available[:, :reach, step, 0] = True
    return {
        "future": future,
        "log_probs": torch.where(available, 0.0, -torch.inf),
        "available": available,
    }


def future_inputs(record: dict, condition: str) -> tuple[Tensor, ...]:
    source = record["oracle"] if condition == "oracle" else record
    future, log_probs, available = (
        source["future"],
        source["log_probs"],
        source["available"],
    )
    if condition == "shuffled":
        future, log_probs, available = (
            v.roll(1, 0) for v in (future, log_probs, available)
        )
    elif condition == "repeated":
        available = available.any(-1, keepdim=True)
        future = record["memory"][:, :, None, None].expand(
            *available.shape, future.shape[-1]
        )
        log_probs = torch.where(available, 0.0, -torch.inf)
    elif condition == "zero":
        available = torch.zeros_like(available)
    elif condition not in ("real", "oracle"):
        raise ValueError("unknown future condition")
    return future, log_probs, available


def lookahead_path(layout: Layout, source: str) -> Path:
    name = "lookahead.pt" if source == "planner" else "lookahead-oracle.pt"
    return layout.run / "planning" / name


def fit_future(config: dict, layout: Layout, device: torch.device, source: str) -> None:
    options = config["lookahead"]
    learning_rate = required(options["learning_rate"], "lookahead learning rate")
    set_seed(options["seed"])
    model, _ = load_protocol(primary_path(config, layout), device, table=True)
    model.requires_grad_(False).eval()
    receiver = LookaheadReceiver(
        model.receiver, options["horizon"], options["rank"]
    ).to(device)
    planner = load_planner(config, layout, device) if source == "planner" else None
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "primary", corpus)
    parameters = [p for p in receiver.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=options["weight_decay"]
    )
    rng = np.random.default_rng(options["data_seed"])
    parents = [primary_path(config, layout)]
    if source == "planner":
        parents.append(layout.run / "planning/planner.pt")
    provenance = {
        **identity(config, layout, parents, "execution.planning_training"),
        "source": source,
    }
    fingerprint = configuration_fingerprint(
        {**provenance, "learning_rate": learning_rate}
    )
    output = lookahead_path(layout, source)

    def step(index: int) -> tuple[Tensor, dict[str, float], str]:
        rows = corpus.windows("train", config["events"], config["batch_size"], rng)
        planned = time.perf_counter()
        if planner is None:
            with torch.no_grad():
                record = {
                    "rows": rows,
                    "memory": model.sender(rows[:, :-1].to(device)),
                    "oracle": oracle_future(model, rows, options["horizon"], device),
                }
        else:
            record = make_plan_batch(config, model, planner, rows, device)
        device_synchronize(device)
        rollout = time.perf_counter() - planned
        batch = protocol_batch(config, table, record["rows"], device)
        future, log_probs, available = future_inputs(
            record, "real" if planner is not None else "oracle"
        )
        with autocast_context(device, config["precision"]):
            result = receiver(
                batch.receiver_ids,
                gather_events(record["memory"], batch.candidate_event_ids),
                batch.candidate_event_ids,
                batch.frontier,
                future,
                log_probs,
                available,
                batch.frontier,
                hard=False,
            )
            loss = F.cross_entropy(
                result.logits.float().flatten(0, 1), batch.targets.flatten()
            )
        counters = {
            "target_bytes": int(batch.target_bytes.sum().item()),
            "rollout_seconds": rollout,
        }
        return loss, counters, f" availability={available.float().mean().item():.4f}"

    receiver.train()
    outcome = train_steps(
        Job(
            label="lookahead",
            steps=options["steps"],
            learning_rate=learning_rate,
            warmup_steps=options["warmup_steps"],
            grad_clip=options["grad_clip"],
            interval=config["checkpoint_interval"],
            resume=resume_path(output),
            fingerprint=fingerprint,
            modules={"receiver": receiver},
            parameters=parameters,
            optimizer=optimizer,
            rng=rng,
            device=device,
            counters=("target_bytes", "rollout_seconds"),
        ),
        step,
    )
    save_checkpoint(
        output,
        receiver.channels,
        {
            "architecture": "hypothesis-attention-futures",
            "identity": provenance,
            "primary_sha256": file_sha256(primary_path(config, layout)),
            "horizon": options["horizon"],
            "rank": options["rank"],
        },
    )
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "seconds": outcome.seconds,
            "steps": outcome.completed,
            "target_bytes": int(outcome.counters["target_bytes"]),
            "rollout_seconds": outcome.counters["rollout_seconds"],
            "trainable_parameters": sum(p.numel() for p in parameters),
            "sender_and_current_receiver_frozen": True,
        },
    )
    resume_path(output).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("planner", "fit", "oracle-fit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    config = layout.config()
    validate(config)
    device = torch.device(args.device)
    if args.task == "planner":
        fit_planner(config, layout, device)
    else:
        fit_future(
            config, layout, device, "planner" if args.task == "fit" else "oracle"
        )


if __name__ == "__main__":
    main()
