import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from data.protocol_corpus import EventCorpus
from data.tokenizers import load_tokenizer
from execution.exposure_design import verify_exposure
from execution.layout import Layout, required
from execution.training import identity
from execution.training_loop import Job, train_steps
from execution.validation import validate
from framework.checkpoints import (
    file_sha256,
    save_checkpoint,
    write_json,
)
from framework.resume import (
    configuration_fingerprint,
    resume_path,
)
from framework.runtime import (
    autocast_context,
    set_seed,
)
from models.bpe.bpe import BPEModel


def bpe_batch(
    corpus: EventCorpus, rows: Tensor, tokenizer: Tokenizer, device: torch.device
) -> tuple[Tensor, Tensor]:
    windows = corpus.decode(rows)
    prefixes = tokenizer.encode_batch([units[0] for units in windows])
    continuations = tokenizer.encode_batch(["".join(units[1:]) for units in windows])
    sequences, labels = [], []
    for prefix_encoding, target_encoding in zip(prefixes, continuations, strict=True):
        prefix, target = prefix_encoding.ids, target_encoding.ids
        ids = prefix + target
        sequences.append(ids[:-1])
        labels.append([-100] * (len(prefix) - 1) + target)
    width = max(map(len, sequences))
    x = torch.tensor([x + [0] * (width - len(x)) for x in sequences], device=device)
    y = torch.tensor([y + [-100] * (width - len(y)) for y in labels], device=device)
    return x, y


def schedule(config: dict) -> list[tuple[int, int]]:
    return [
        (config["data_seed"], config["native_training"]["steps"]),
        *(
            (config["data_seed"] + i + 1, s["steps"])
            for i, s in enumerate(config["stages"])
        ),
    ]


def build_baseline(
    config: dict, tokenizer: Tokenizer, device: torch.device
) -> tuple[BPEModel, dict]:
    options = {**config["baseline"]["model"], "vocab_size": tokenizer.get_vocab_size()}
    return BPEModel(**options).to(device), options


def baseline_loss(
    model: BPEModel, ids: Tensor, targets: Tensor, config: dict
) -> Tensor:
    with autocast_context(ids.device, config["precision"]):
        return F.cross_entropy(model(ids).float().flatten(0, 1), targets.flatten())


def train(config: dict, layout: Layout, device: torch.device) -> None:
    options = config["baseline"]
    learning_rate = required(options["learning_rate"], "baseline learning rate")
    set_seed(options["seed"])
    corpus = EventCorpus.load(layout.corpus)
    tokenizer = load_tokenizer(layout.baseline_tokenizer)
    model, model_options = build_baseline(config, tokenizer, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=options["weight_decay"],
    )
    phases = schedule(config)
    steps = sum(count for _, count in phases)
    provenance = {
        **identity(config, layout, [], "execution.baseline_training"),
        "baseline_tokenizer": file_sha256(layout.baseline_tokenizer),
    }
    fingerprint = configuration_fingerprint(
        {**provenance, "learning_rate": learning_rate}
    )
    output = layout.run / "baseline/model.pt"
    rng = np.random.default_rng(phases[0][0])
    starts = {}
    position = 0
    for data_seed, count in phases:
        starts[position] = data_seed
        position += count

    def step(index: int) -> tuple[Tensor, dict[str, float], str]:
        if index in starts:
            rng.bit_generator.state = np.random.default_rng(
                starts[index]
            ).bit_generator.state
        rows = corpus.windows("train", config["events"], config["batch_size"], rng)
        ids, targets = bpe_batch(corpus, rows, tokenizer, device)
        loss = baseline_loss(model, ids, targets, config)
        size = sum(len("".join(units[1:]).encode()) for units in corpus.decode(rows))
        return loss, {"target_bytes": size}, ""

    model.train()
    outcome = train_steps(
        Job(
            label="baseline",
            steps=steps,
            learning_rate=learning_rate,
            warmup_steps=options["warmup_steps"],
            grad_clip=options["grad_clip"],
            interval=config["checkpoint_interval"],
            resume=resume_path(output),
            fingerprint=fingerprint,
            modules={"model": model},
            parameters=list(model.parameters()),
            optimizer=optimizer,
            rng=rng,
            device=device,
            counters=("target_bytes",),
        ),
        step,
    )
    completed = outcome.completed
    byte_count = int(outcome.counters["target_bytes"])
    verify_exposure(layout.exposure, "baseline/model", completed, byte_count)
    save_checkpoint(
        output,
        model,
        {
            "architecture": "whole-text-bpe",
            "model_config": model_options,
            "identity": provenance,
        },
    )
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "steps": completed,
            "target_bytes": byte_count,
            "seconds": outcome.seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
            "data_matching": "primary L=0 including native pretraining; not FLOP matching",
        },
    )
    resume_path(output).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    config = layout.config()
    validate(config)
    train(config, layout, torch.device(args.device))


if __name__ == "__main__":
    main()
