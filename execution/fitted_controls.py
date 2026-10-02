import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.communication import ProtocolBatch
from data.protocol_corpus import EventCorpus, selection_windows
from execution.layout import Layout, required
from execution.protocol_io import load_protocol
from execution.training import encoded, identity, stage_path
from execution.training_loop import Job, train_steps
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import load_checkpoint, save_checkpoint, write_json
from framework.resume import (
    configuration_fingerprint,
    resume_path,
)
from framework.runtime import (
    autocast_context,
    set_seed,
)
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.receiver import Receiver

ARMS = ("correct", "shuffled", "native-only")


class FittedControl(nn.Module):
    native_projection: Tensor

    def __init__(
        self, model: ProtocolModel[Receiver], arm: str, projection_seed: int
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError("unknown fitted control")
        self.model, self.arm = model, arm
        model.configure_stage("communication")
        hidden = model.receiver.native.trunk.config.d_model
        message = model.receiver.sender_dimensions
        projection = (
            torch.randn(
                hidden,
                message,
                generator=torch.Generator().manual_seed(projection_seed),
            )
            / hidden**0.5
        )
        self.register_buffer(
            "native_projection", projection.to(model.sender.trunk.input_reference)
        )

    def forward(self, batch: ProtocolBatch) -> Tensor:
        receiver = self.model.receiver
        if self.arm == "native-only":

            def communicate(layer: int, hidden: Tensor) -> Tensor:
                return receiver.channels[layer](hidden, hidden @ self.native_projection)

            return receiver.native(batch.receiver_ids, layer_transform=communicate)[
                ..., : receiver.output_symbols
            ]
        with torch.no_grad():
            memory = self.model.sender(batch.sender_ids)
            if self.arm == "shuffled":
                memory = memory.roll(1, 0)
            candidates = gather_events(memory, batch.candidate_event_ids)
        return receiver(
            batch.receiver_ids,
            candidates,
            batch.candidate_event_ids,
            batch.frontier,
            hard=False,
        ).logits


def train(layout: Layout, variant: str, arm: str, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    options = config["control_training"]
    learning_rate = required(options["learning_rate"], "control learning rate")
    set_seed(options["seed"])
    parent = stage_path(layout, variant, 0)
    model, _ = load_protocol(parent, device, table=True)
    control = FittedControl(model, arm, options["projection_seed"])
    parameters = tuple(p for p in control.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=options["weight_decay"]
    )
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, variant, corpus)
    provenance = {
        **identity(config, layout, [parent], "execution.fitted_controls"),
        "control": arm,
        "variant": variant,
    }
    fingerprint = configuration_fingerprint(
        {**provenance, "learning_rate": learning_rate}
    )
    rng = np.random.default_rng(options["data_seed"])
    output = layout.run / "controls" / variant / arm / "model.pt"

    def step(index: int) -> tuple[Tensor, dict[str, float], str]:
        rows, _ = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="train"
        )
        batch = protocol_batch(config, table, rows, device)
        with autocast_context(device, config["precision"]):
            loss = F.cross_entropy(
                control(batch).float().flatten(0, 1), batch.targets.flatten()
            )
        return loss, {"target_bytes": int(batch.target_bytes.sum().item())}, ""

    control.train()
    outcome = train_steps(
        Job(
            label=f"control={arm}",
            steps=options["steps"],
            learning_rate=learning_rate,
            warmup_steps=options["warmup_steps"],
            grad_clip=options["grad_clip"],
            interval=config["checkpoint_interval"],
            resume=resume_path(output),
            fingerprint=fingerprint,
            modules={"control": control},
            parameters=parameters,
            optimizer=optimizer,
            rng=rng,
            device=device,
            counters=("target_bytes",),
        ),
        step,
    )
    save_checkpoint(output, control, {"identity": provenance, "arm": arm})
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "steps": outcome.completed,
            "target_bytes": int(outcome.counters["target_bytes"]),
            "seconds": outcome.seconds,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "fixed_parameters": "sender, native receiver and correspondence scorers",
            "donors": "different input groups; no original-document guarantee",
        },
    )
    resume_path(output).unlink(missing_ok=True)


@torch.no_grad()
def measure(
    layout: Layout, variant: str, arm: str, device: torch.device, split: str
) -> None:
    config = layout.config()
    validate(config)
    options = config["control_training"]
    model, _ = load_protocol(stage_path(layout, variant, 0), device, table=True)
    control = FittedControl(model, arm, options["projection_seed"])
    payload = load_checkpoint(
        layout.run / "controls" / variant / arm / "model.pt", map_location=device
    )
    expected = {
        **identity(
            config,
            layout,
            [stage_path(layout, variant, 0)],
            "execution.fitted_controls",
        ),
        "control": arm,
        "variant": variant,
    }
    if payload["metadata"]["identity"] != expected:
        raise ValueError("control checkpoint identity differs from this experiment")
    control.load_state_dict(payload["model"], strict=True)
    control.eval()
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    if split == "reporting":
        model.sender.build_table(corpus.vocab)
    table = encoded(config, layout, variant, corpus)
    rng = np.random.default_rng(config["evaluation_seed"])
    nats, tokens, sizes, groups = [], [], [], []
    for _ in range(layout.evaluation_batches(config, split)):
        rows, group_ids = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        batch = protocol_batch(config, table, rows, device)
        logits = control(batch)
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), batch.targets.flatten(), reduction="none"
        ).reshape_as(batch.targets)
        nats.extend(loss.double().sum(-1).tolist())
        tokens.extend(batch.targets.ne(-100).sum(-1).tolist())
        sizes.extend(batch.target_bytes.tolist())
        groups.extend(group_ids.tolist())
    write_json(
        layout.measured(split) / "controls" / variant / arm / "evaluation.json",
        {
            "split": split,
            "arm": arm,
            "ce": sum(nats) / sum(tokens),
            "nll": nats,
            "tokens": tokens,
            "utf8_bytes": sizes,
            "group_ids": groups,
            "comparison": "separately fitted R with matched initialization, parameters, windows and updates",
        },
    )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--variant", choices=("fresh-byte", "fresh-bpe"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"))
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    device = torch.device(args.device)
    if args.task == "train":
        train(layout, args.variant, args.arm, device)
    elif args.split is None:
        parser.error("measure requires a split")
    else:
        measure(layout, args.variant, args.arm, device, args.split)


if __name__ == "__main__":
    main()
