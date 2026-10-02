import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus, selection_windows
from execution.design import SWEEP_FRACTION, seed_for
from execution.layout import Layout, required
from execution.protocol_evaluation import RowScores, evaluate_controls
from execution.protocol_io import evaluation_model, interface_state
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import encoded, initial_protocol, stage_path
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import write_json
from framework.runtime import autocast_context, cosine_schedule, set_seed
from models.protocol.correspondence import alignment_diagnostics
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.receiver import Receiver

Phase = Literal["alignment", "communication", "joint", "latent"]

ARMS: dict[str, tuple[Phase, ...]] = {
    "likelihood": ("joint", "joint", "joint"),
    "latent": ("latent", "latent", "latent"),
    "curriculum": ("alignment", "joint", "joint"),
    "bootstrap": ("alignment", "communication", "joint"),
    "supervised": ("alignment", "alignment", "alignment"),
}


def _settings(name: Phase, config: dict, steps: int) -> StageConfig:
    return StageConfig(
        name="joint" if name == "latent" else name,
        steps=steps,
        learning_rate=required(config["stages"][0]["learning_rate"], "adapter rate"),
        weight_decay=config["stages"][0]["weight_decay"],
        alignment_weight=1.0 if name == "alignment" else 0.0,
        grad_clip=config["stages"][0]["grad_clip"],
        warmup_steps=config["stages"][0]["warmup_steps"],
        precision=config["precision"],
    )


def _reads(
    model: ProtocolModel[Receiver], batch: ProtocolBatch, memory: Tensor
) -> tuple[Tensor, Tensor]:
    window = model.receiver.candidate_window
    frontier = batch.frontier
    full = model.receiver(
        batch.receiver_ids,
        gather_events(memory, batch.candidate_event_ids),
        batch.candidate_event_ids,
        frontier,
        hard=False,
    )
    prior = full.correspondence[0]
    likelihoods, priors = [], []
    targets = batch.targets.clamp_min(0)

    def read(values: Tensor, event: Tensor, ids: Tensor) -> Tensor:
        readout = model.receiver(
            batch.receiver_ids,
            gather_events(values, event.unsqueeze(-1)),
            ids,
            frontier,
            hard=True,
        )
        log_probs = readout.logits.float().log_softmax(-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    for offset in range(window):
        event = frontier - offset
        valid = event.ge(0) & frontier.ge(0)
        event = torch.where(valid, event, -1)
        ids = torch.where(valid, frontier, -1).unsqueeze(-1)
        likelihood = checkpoint(read, memory, event, ids, use_reentrant=False)
        likelihoods.append(cast(Tensor, likelihood))
        match = batch.candidate_event_ids.eq(event.unsqueeze(-1))
        chosen = prior.masked_fill(~match, -torch.inf).logsumexp(-1)
        priors.append(torch.where(valid, chosen, -torch.inf))
    return torch.stack(likelihoods, -1), torch.stack(priors, -1)


def latent_nll(
    model: ProtocolModel[Receiver], batch: ProtocolBatch, memory: Tensor
) -> tuple[Tensor, Tensor]:
    likelihoods, priors = _reads(model, batch, memory)
    live = batch.targets.ne(-100) & priors.isfinite().any(-1)
    marginal = torch.where(
        priors.isfinite(), priors + likelihoods, torch.full_like(priors, -torch.inf)
    ).logsumexp(-1)
    return -marginal, live


class LatentTrainer:
    def __init__(
        self,
        model: ProtocolModel[Receiver],
        settings: StageConfig,
        rng: np.random.Generator,
    ) -> None:
        self.model, self.settings, self.rng = model, settings, rng
        model.configure_stage("joint")
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        self.completed = 0

    def step(self, batch: ProtocolBatch) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = self.settings.learning_rate * cosine_schedule(
                self.completed, self.settings.steps, self.settings.warmup_steps
            )
        with autocast_context(batch.sender_ids.device, self.settings.precision):
            memory = self.model.sender(batch.sender_ids)
            nll, live = latent_nll(self.model, batch, memory)
            loss = nll[live].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("latent-path loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters, self.settings.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.completed += 1


def _first_layer_accuracy(
    model: ProtocolModel[Receiver], batch: ProtocolBatch, memory: Tensor
) -> tuple[int, int]:
    prior = model.receiver(
        batch.receiver_ids,
        gather_events(memory, batch.candidate_event_ids),
        batch.candidate_event_ids,
        batch.frontier,
        hard=False,
    ).correspondence[0]
    diagnostics = alignment_diagnostics(
        prior, batch.candidate_event_ids, batch.alignment_targets
    )
    valid = diagnostics.valid & batch.targets.ne(-100)
    return int((diagnostics.hard_correct & valid).sum().item()), int(valid.sum().item())


@torch.no_grad()
def _held_out_batches(
    table: ReceiverUnits,
    corpus: EventCorpus,
    config: dict,
    device: torch.device,
    batches: int,
) -> Iterator[ProtocolBatch]:
    rng = np.random.default_rng(config["evaluation_seed"])
    for _ in range(batches):
        rows, _ = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        yield protocol_batch(config, table, rows, device)


def _accuracy(aligned: list[int]) -> float | None:
    return aligned[0] / aligned[1] if aligned[1] else None


def _soft_scores(
    model: ProtocolModel[Receiver],
    batches: Iterator[ProtocolBatch],
    permutation: Tensor,
) -> dict:
    combined: dict[str, list[RowScores]] = {"correct": [], "shuffled": []}
    aligned = [0, 0]
    for batch in batches:
        scores = evaluate_controls(model, batch, permutation=permutation, hard=False)
        for name, rows in combined.items():
            rows.append(scores[name])
        correct, count = _first_layer_accuracy(
            model, batch, model.sender(batch.sender_ids)
        )
        aligned = [aligned[0] + correct, aligned[1] + count]
    return {
        name: {
            **RowScores.concatenate(rows).aggregate(),
            "first_layer_event_accuracy": _accuracy(aligned),
            "likelihood": "soft read",
        }
        for name, rows in combined.items()
    }


def _latent_scores(
    model: ProtocolModel[Receiver],
    batches: Iterator[ProtocolBatch],
    permutation: Tensor,
    device: torch.device,
) -> dict:
    totals = {"correct": [0.0, 0], "shuffled": [0.0, 0]}
    aligned = [0, 0]
    for batch in batches:
        memory = model.sender(batch.sender_ids)
        reads = {"correct": memory, "shuffled": memory[permutation.to(device)]}
        for name, values in reads.items():
            nll, live = latent_nll(model, batch, values)
            totals[name][0] += float(nll[live].double().sum().item())
            totals[name][1] += int(live.sum().item())
        correct, count = _first_layer_accuracy(model, batch, memory)
        aligned = [aligned[0] + correct, aligned[1] + count]
    return {
        name: {
            "ce": nats / count if count else None,
            "first_layer_event_accuracy": _accuracy(aligned),
            "likelihood": "marginal over hard reads",
        }
        for name, (nats, count) in totals.items()
    }


def held_out(
    model: ProtocolModel[Receiver],
    table: ReceiverUnits,
    corpus: EventCorpus,
    config: dict,
    device: torch.device,
    batches: int,
    latent: bool,
) -> dict:
    model.eval()
    permutation = torch.arange(config["batch_size"]).roll(1)
    windows = _held_out_batches(table, corpus, config, device, batches)
    if latent:
        return _latent_scores(model, windows, permutation, device)
    return _soft_scores(model, windows, permutation)


def _fit_arm(
    config: dict,
    layout: Layout,
    corpus: EventCorpus,
    table: ReceiverUnits,
    arm: str,
    device: torch.device,
) -> tuple[list[dict], dict[str, Tensor]]:
    steps = config["stages"][0]["steps"]
    quick = max(1, round(config["evaluation_batches"] * SWEEP_FRACTION))
    root_seed = json.loads(layout.design.read_text())["inputs"]["root_seed"]
    set_seed(seed_for(root_seed, layout.replicate, "bootstrap"))
    model, _, _ = initial_protocol(config, layout, "fresh-byte", corpus, table, device)
    rng = np.random.default_rng(config["data_seed"] + 1)
    trajectory = []
    for phase, name in enumerate(ARMS[arm]):
        settings = _settings(name, config, steps)
        trainer = (
            LatentTrainer(model, settings, rng)
            if name == "latent"
            else StageTrainer(model, settings, {"bootstrap": arm, "phase": phase}, rng)
        )
        for _ in range(steps):
            rows = corpus.windows("train", config["events"], config["batch_size"], rng)
            trainer.step(protocol_batch(config, table, rows, device))
        scores = held_out(model, table, corpus, config, device, quick, name == "latent")
        trajectory.append({"phase": phase, "stage": name, "selection": scores})
        print(f"bootstrap arm={arm} phase={phase}:{name}", flush=True)
    return trajectory, interface_state(model)


def train(layout: Layout, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "fresh-byte", corpus)
    fitted = {arm: _fit_arm(config, layout, corpus, table, arm, device) for arm in ARMS}
    trajectories = {arm: trajectory for arm, (trajectory, _) in fitted.items()}
    states = {arm: state for arm, (_, state) in fitted.items()}
    steps = config["stages"][0]["steps"]
    quick = max(1, round(config["evaluation_batches"] * SWEEP_FRACTION))
    folder = layout.run / "bootstrap"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(states, folder / "interfaces.pt")
    write_json(
        folder / "trajectories.json",
        {
            "arms": {arm: list(phases) for arm, phases in ARMS.items()},
            "steps_per_phase": steps,
            "trajectory_evaluation_batches": quick,
            "trajectories": trajectories,
        },
    )


@torch.no_grad()
def measure(layout: Layout, device: torch.device, split: str) -> None:
    config = layout.config()
    validate(config)
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    table = encoded(config, layout, "fresh-byte", corpus)
    states = torch.load(layout.run / "bootstrap" / "interfaces.pt", weights_only=True)
    results = {}
    for arm, phases in ARMS.items():
        model = evaluation_model(
            stage_path(layout, "fresh-byte", 0), corpus, split, device
        )
        model.receiver.load_state_dict(
            {**model.receiver.state_dict(), **states[arm]}, strict=True
        )
        results[arm] = held_out(
            model,
            table,
            corpus,
            config,
            device,
            layout.evaluation_batches(config, split),
            phases[-1] == "latent",
        )
    write_json(
        layout.measured(split) / "bootstrap" / "evaluation.json",
        {"split": split, "arms": results},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"))
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    device = torch.device(args.device)
    if args.task == "train":
        train(layout, device)
    elif args.split is None:
        parser.error("measure requires a split")
    else:
        measure(layout, device, args.split)


if __name__ == "__main__":
    main()
