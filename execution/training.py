import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus
from data.tokenizers import load_tokenizer
from execution.design import unswept
from execution.exposure_design import verify_exposure
from execution.layout import Layout, required
from execution.protocol_io import ARCHITECTURE, build_protocol, load_protocol
from execution.protocol_stages import StageConfig, StageTrainer
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
    capture,
    configuration_fingerprint,
    current,
    resume_path,
    save,
)
from framework.runtime import (
    autocast_context,
    device_synchronize,
    set_seed,
)
from framework.sources import closure
from models.bpe.bpe import BPEModel
from models.protocol.model import ProtocolModel
from models.protocol.receiver import Receiver

ROOT = Path(__file__).resolve().parents[1]


def identity(config: dict, layout: Layout, parents: list[Path], module: str) -> dict:
    sources = {
        path.relative_to(ROOT).as_posix(): file_sha256(path) for path in closure(module)
    }
    return {
        "config": unswept(config),
        "sources": sources,
        "corpus": file_sha256(layout.corpus / "metadata.json"),
        "exposure": file_sha256(layout.exposure),
        "tokenizer": file_sha256(layout.receiver_tokenizer),
        "parents": {
            p.relative_to(layout.shared).as_posix(): file_sha256(p) for p in parents
        },
    }


def encoded(
    config: dict, layout: Layout, variant: str, corpus: EventCorpus
) -> ReceiverUnits:
    if config["receivers"][variant]["kind"] == "bytes":
        return ReceiverUnits.bytes(corpus.vocab)
    return ReceiverUnits.bpe(corpus.vocab, load_tokenizer(layout.receiver_tokenizer))


def native_path(layout: Layout, variant: str) -> Path:
    return layout.run / variant / "native.pt"


def stage_path(layout: Layout, variant: str, index: int) -> Path:
    return layout.run / variant / f"stage-{index}.pt"


def stages(config: dict, variant: str) -> list[dict]:
    return config["stages" if variant == "primary" else "replacement_stages"]


def native_options(config: dict, table: ReceiverUnits) -> dict:
    inventory = table.inventory
    word = config["receiver_context"] == "word"
    return {
        **config["native"],
        "vocab_size": inventory.inputs,
        "boundaries": [inventory.stop, inventory.start] if word else None,
    }


def build_native(
    config: dict, table: ReceiverUnits, device: torch.device
) -> tuple[BPEModel, dict]:
    options = native_options(config, table)
    return BPEModel(**options).to(device), options


def native_loss(
    model: BPEModel, table: ReceiverUnits, batch: ProtocolBatch, config: dict
) -> Tensor:
    with autocast_context(batch.receiver_ids.device, config["precision"]):
        logits = model(batch.receiver_ids)[..., : table.inventory.outputs]
        return F.cross_entropy(logits.float().flatten(0, 1), batch.targets.flatten())


def native(layout: Layout, variant: str, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    schedule = config["native_training"]
    learning_rate = required(schedule["learning_rate"], "native learning rate")
    set_seed(config["receivers"][variant]["seed"])
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, variant, corpus)
    model, options = build_native(config, table, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=schedule["weight_decay"],
    )
    rng = np.random.default_rng(config["data_seed"])
    provenance = {
        **identity(config, layout, [], "execution.training"),
        "variant": variant,
        "job": "native",
    }
    fingerprint = configuration_fingerprint(
        {**provenance, "learning_rate": learning_rate}
    )
    output = native_path(layout, variant)

    def step(index: int) -> tuple[Tensor, dict[str, float], str]:
        rows = corpus.windows("train", config["events"], config["batch_size"], rng)
        batch = protocol_batch(config, table, rows, device)
        loss = native_loss(model, table, batch, config)
        return loss, {"target_bytes": int(batch.target_bytes.sum().item())}, ""

    model.train()
    outcome = train_steps(
        Job(
            label=f"native={variant}",
            steps=schedule["steps"],
            learning_rate=learning_rate,
            warmup_steps=schedule["warmup_steps"],
            grad_clip=schedule["grad_clip"],
            interval=config["checkpoint_interval"],
            resume=resume_path(output),
            fingerprint=fingerprint,
            modules={"native": model},
            parameters=list(model.parameters()),
            optimizer=optimizer,
            rng=rng,
            device=device,
            counters=("target_bytes",),
        ),
        step,
    )
    completed = outcome.completed
    target_bytes = int(outcome.counters["target_bytes"])
    metadata = {
        "architecture": "autonomous-fine-clock-receiver",
        "native_config": options,
        "outputs": table.inventory.outputs,
        "kind": config["receivers"][variant]["kind"],
        "identity": provenance,
    }
    verify_exposure(layout.exposure, f"{variant}/native", completed, target_bytes)
    save_checkpoint(output, model, metadata)
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "steps": completed,
            "target_bytes": target_bytes,
            "seconds": outcome.seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
            "independent_native": True,
        },
    )
    resume_path(output).unlink(missing_ok=True)


def stage_schedule(config: dict, variant: str, index: int) -> StageConfig:
    settings = stages(config, variant)[index]
    required(settings["learning_rate"], f"{settings['name']} learning rate")
    required(settings["alignment_weight"], f"{settings['name']} alignment weight")
    return StageConfig(**settings)


def initial_protocol(
    config: dict,
    layout: Layout,
    variant: str,
    corpus: EventCorpus,
    table: ReceiverUnits,
    device: torch.device,
) -> tuple[ProtocolModel[Receiver], dict, list[Path]]:
    parents = [native_path(layout, variant)]
    payload = load_checkpoint(parents[0], map_location=device)
    metadata = payload["metadata"]
    if metadata["identity"] != {
        **identity(config, layout, [], "execution.training"),
        "variant": variant,
        "job": "native",
    }:
        raise ValueError("native checkpoint belongs to another experiment")
    options = {
        "sender": config["sender"],
        "native": metadata["native_config"],
        "interface": {
            **config["interface"],
            "output_symbols": table.inventory.outputs,
        },
    }
    model = build_protocol(options, corpus.vocab, device)
    model.receiver.native.load_state_dict(payload["model"], strict=True)
    if variant != "primary":
        parent = stage_path(layout, "primary", len(config["stages"]) - 1)
        parents.append(parent)
        shared, shared_metadata = load_protocol(parent, device, table=True)
        if shared_metadata["vocab"] != corpus.vocab:
            raise ValueError("replacement sender inventory changed")
        model.sender.load_state_dict(shared.sender.state_dict(), strict=True)
        del shared
    return model, options, parents


def train_stage(layout: Layout, variant: str, index: int, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    schedule = stage_schedule(config, variant, index)
    set_seed(config["receivers"][variant]["seed"] + index + 1)
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, variant, corpus)
    if index:
        parents = [stage_path(layout, variant, index - 1)]
        model, metadata = load_protocol(parents[0], device, table=True)
        if metadata["vocab"] != corpus.vocab:
            raise ValueError("stage input vocabulary changed")
        options = metadata["model_config"]
    else:
        model, options, parents = initial_protocol(
            config, layout, variant, corpus, table, device
        )
    provenance = {
        **identity(config, layout, parents, "execution.training"),
        "variant": variant,
        "stage_index": index,
    }
    rng = np.random.default_rng(config["data_seed"] + index + 1)
    trainer = StageTrainer(model, schedule, provenance, rng)
    output = stage_path(layout, variant, index)
    resume = resume_path(output)
    byte_count, elapsed = 0, 0.0
    state = current(resume, trainer.fingerprint)
    if state is not None:
        trainer.resume(state)
        byte_count = state["extra"]["target_bytes"]
        elapsed = state["extra"]["seconds"]
    device_synchronize(device)
    started = time.perf_counter()
    for _ in range(trainer.completed, schedule.steps):
        rows = corpus.windows("train", config["events"], config["batch_size"], rng)
        batch = protocol_batch(config, table, rows, device)
        losses = trainer.step(batch)
        byte_count += int(batch.target_bytes.sum().item())
        print(
            f"receiver={variant} stage={index}:{schedule.name} step={trainer.completed}/{schedule.steps} ce={losses.language.item():.6f} alignment={losses.alignment.item():.6f}",
            flush=True,
        )
        if (
            trainer.completed % config["checkpoint_interval"] == 0
            or trainer.completed == schedule.steps
        ):
            device_synchronize(device)
            save(
                resume,
                capture(
                    trainer.completed,
                    trainer.fingerprint,
                    {"model": model},
                    trainer.optimizer,
                    rng,
                    {
                        "target_bytes": byte_count,
                        "seconds": elapsed + time.perf_counter() - started,
                    },
                ),
            )
    device_synchronize(device)
    verify_exposure(
        layout.exposure, f"{variant}/stage-{index}", trainer.completed, byte_count
    )
    save_checkpoint(
        output,
        model,
        {
            "architecture": ARCHITECTURE,
            "model_config": options,
            "vocab": corpus.vocab,
            "identity": provenance,
            "variant": variant,
            "stage": index,
        },
    )
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "stage": schedule.name,
            "steps": trainer.completed,
            "target_bytes": byte_count,
            "seconds": elapsed + time.perf_counter() - started,
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "native_frozen": schedule.name != "trunk",
            "sender_frozen": schedule.name != "trunk",
        },
    )
    resume.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("native", "stage"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument(
        "--variant", choices=("primary", "fresh-byte", "fresh-bpe"), required=True
    )
    parser.add_argument("--stage", type=int)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    if args.task == "native":
        native(layout, args.variant, torch.device(args.device))
    else:
        if args.stage is None or not 0 <= args.stage < len(
            stages(layout.config(), args.variant)
        ):
            parser.error("stage index is required and must index the declared schedule")
        train_stage(layout, args.variant, args.stage, torch.device(args.device))


if __name__ == "__main__":
    main()
