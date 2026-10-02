import argparse
import json
import math
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus, selection_windows
from data.tokenizers import load_tokenizer
from execution.baseline_training import baseline_loss, bpe_batch, build_baseline
from execution.design import SWEEP_FRACTION, warmup
from execution.layout import Layout
from execution.protocol_io import load_protocol
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import (
    build_native,
    encoded,
    initial_protocol,
    native_loss,
    stage_path,
)
from execution.windows import protocol_batch
from framework.checkpoints import write_json
from framework.runtime import cosine_schedule, set_seed
from models import conventions
from models.protocol.model import ProtocolModel, TrainingStage
from models.protocol.receiver import Receiver


class Diverged(Exception):
    pass


def _diverged(error: Exception) -> bool:
    return isinstance(error, FloatingPointError) or (
        isinstance(error, RuntimeError) and "non-finite" in str(error)
    )


def _evaluation(config: dict, corpus: EventCorpus) -> list[Tensor]:
    batches = max(1, math.ceil(config["evaluation_batches"] * SWEEP_FRACTION))
    rng = np.random.default_rng(config["evaluation_seed"])
    return [
        selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )[0]
        for _ in range(batches)
    ]


@torch.no_grad()
def _ce(pairs: Iterable[tuple[Tensor, Tensor]]) -> float:
    nats, count = 0.0, 0
    for logits, targets in pairs:
        nats += F.cross_entropy(
            logits.float().flatten(0, 1),
            targets.flatten(),
            reduction="sum",
            ignore_index=-100,
        ).item()
        count += int(targets.ne(-100).sum().item())
    return nats / count


def _fit(
    model: nn.Module,
    parameters: list[nn.Parameter],
    learning_rate: float,
    steps: int,
    loss_fn: Callable[[], Tensor],
) -> None:
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=conventions.WEIGHT_DECAY
    )
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * cosine_schedule(step, steps, warmup(steps))
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise Diverged
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters, conventions.GRAD_CLIP, error_if_nonfinite=True
            )
        except RuntimeError as error:
            if _diverged(error):
                raise Diverged from None
            raise
        optimizer.step()
    model.eval()


def _stage(
    model: ProtocolModel[Receiver],
    name: TrainingStage,
    learning_rate: float,
    alignment_weight: float,
    steps: int,
    batches: Iterator[ProtocolBatch],
) -> None:
    settings = StageConfig(
        name=name,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=conventions.WEIGHT_DECAY,
        alignment_weight=alignment_weight,
        grad_clip=conventions.GRAD_CLIP,
        warmup_steps=warmup(steps),
        precision="auto",
    )
    trainer = StageTrainer(model, settings, {"sweep": name}, np.random.default_rng(0))
    try:
        for _ in range(steps):
            trainer.step(next(batches))
    except (FloatingPointError, RuntimeError) as error:
        if _diverged(error):
            raise Diverged from None
        raise


def _score(candidates: list[dict], run: Callable[..., float]) -> list[dict]:
    results = []
    for candidate in candidates:
        started = time.perf_counter()
        try:
            value = run(**candidate)
        except Diverged:
            value = math.inf
        results.append(
            {
                **candidate,
                "selection_ce": value,
                "seconds": time.perf_counter() - started,
            }
        )
        print(f"sweep {candidate} selection_ce={value:.6f}", flush=True)
    return results


def _best(results: list[dict], keys: tuple[str, ...]) -> dict:
    finite = [r for r in results if math.isfinite(r["selection_ce"])]
    if not finite:
        raise ValueError("every sweep candidate diverged")
    best = min(finite, key=lambda r: r["selection_ce"])
    return {key: best[key] for key in keys}


def _protocol_ce(
    model: ProtocolModel[Receiver],
    table: ReceiverUnits,
    rows: list[Tensor],
    config: dict,
    device: torch.device,
) -> float:
    def pairs() -> Iterator[tuple[Tensor, Tensor]]:
        for window in rows:
            batch = protocol_batch(config, table, window, device)
            readout = model(
                batch.sender_ids,
                batch.receiver_ids,
                batch.candidate_event_ids,
                batch.frontier,
                hard=False,
            )
            yield readout.logits, batch.targets

    return _ce(pairs())


def _training_batches(
    corpus: EventCorpus,
    table: ReceiverUnits,
    config: dict,
    device: torch.device,
    seed: int,
) -> Iterator[ProtocolBatch]:
    rng = np.random.default_rng(seed)
    while True:
        yield protocol_batch(
            config,
            table,
            corpus.windows("train", config["events"], config["batch_size"], rng),
            device,
        )


def sweep_native(
    layout: Layout, design: dict, config: dict, device: torch.device
) -> tuple[list[dict], dict]:
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "primary", corpus)
    rows = _evaluation(config, corpus)
    steps = design["steps"]["sweep"]

    def run(learning_rate: float) -> float:
        set_seed(config["receivers"]["primary"]["seed"])
        model, _ = build_native(config, table, device)
        batches = _training_batches(corpus, table, config, device, config["data_seed"])

        def loss() -> Tensor:
            return native_loss(model, table, next(batches), config)

        _fit(model, list(model.parameters()), learning_rate, steps, loss)

        def pairs() -> Iterator[tuple[Tensor, Tensor]]:
            for window in rows:
                batch = protocol_batch(config, table, window, device)
                logits = model(batch.receiver_ids)[..., : table.inventory.outputs]
                yield logits, batch.targets

        return _ce(pairs())

    results = _score(
        [{"learning_rate": lr} for lr in design["grids"]["learning_rate"]], run
    )
    return results, _best(results, ("learning_rate",))


def sweep_baseline(
    layout: Layout, design: dict, config: dict, device: torch.device
) -> tuple[list[dict], dict]:
    corpus = EventCorpus.load(layout.corpus)
    tokenizer = load_tokenizer(layout.baseline_tokenizer)
    rows = _evaluation(config, corpus)
    steps = design["steps"]["sweep"]

    def run(learning_rate: float) -> float:
        set_seed(config["baseline"]["seed"])
        model, _ = build_baseline(config, tokenizer, device)
        rng = np.random.default_rng(config["data_seed"])

        def loss() -> Tensor:
            window = corpus.windows(
                "train", config["events"], config["batch_size"], rng
            )
            ids, targets = bpe_batch(corpus, window, tokenizer, device)
            return baseline_loss(model, ids, targets, config)

        _fit(model, list(model.parameters()), learning_rate, steps, loss)

        def pairs() -> Iterator[tuple[Tensor, Tensor]]:
            for window in rows:
                ids, targets = bpe_batch(corpus, window, tokenizer, device)
                yield model(ids), targets

        return _ce(pairs())

    results = _score(
        [{"learning_rate": lr} for lr in design["grids"]["learning_rate"]], run
    )
    return results, _best(results, ("learning_rate",))


def sweep_adapter(
    layout: Layout, design: dict, config: dict, device: torch.device
) -> tuple[list[dict], dict]:
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "primary", corpus)
    rows = _evaluation(config, corpus)
    steps = design["steps"]["sweep"]

    def run(learning_rate: float) -> float:
        set_seed(config["receivers"]["primary"]["seed"] + 1)
        model, _, _ = initial_protocol(config, layout, "primary", corpus, table, device)
        batches = _training_batches(
            corpus, table, config, device, config["data_seed"] + 1
        )
        _stage(model, "alignment", learning_rate, 1.0, steps, batches)
        _stage(model, "communication", learning_rate, 0.0, steps, batches)
        model.eval()
        return _protocol_ce(model, table, rows, config, device)

    results = _score(
        [{"learning_rate": lr} for lr in design["grids"]["learning_rate"]], run
    )
    return results, _best(results, ("learning_rate",))


def sweep_trunk(
    layout: Layout, design: dict, config: dict, device: torch.device
) -> tuple[list[dict], dict]:
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "primary", corpus)
    rows = _evaluation(config, corpus)
    steps = design["steps"]["sweep"]
    parent = stage_path(layout, "primary", 1)

    def run(learning_rate: float, alignment_weight: float) -> float:
        set_seed(config["receivers"]["primary"]["seed"] + 3)
        model, _ = load_protocol(parent, device, table=True)
        batches = _training_batches(
            corpus, table, config, device, config["data_seed"] + 3
        )
        _stage(model, "trunk", learning_rate, alignment_weight, steps, batches)
        model.eval()
        return _protocol_ce(model, table, rows, config, device)

    grids = design["grids"]
    rates = _score(
        [
            {"learning_rate": lr, "alignment_weight": 0.0}
            for lr in grids["learning_rate"]
        ],
        run,
    )
    rate = _best(rates, ("learning_rate",))["learning_rate"]
    weights = _score(
        [
            {"learning_rate": rate, "alignment_weight": weight}
            for weight in grids["alignment_weight"]
            if weight != 0.0
        ],
        run,
    )
    results = rates + weights
    return results, _best(results, ("learning_rate", "alignment_weight"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("native", "baseline", "adapter", "trunk"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, 0)
    design = json.loads(layout.design.read_text())
    config = layout.config()
    device = torch.device(args.device)
    started = time.perf_counter()
    results, selected = {
        "native": sweep_native,
        "baseline": sweep_baseline,
        "adapter": sweep_adapter,
        "trunk": sweep_trunk,
    }[args.task](layout, design, config, device)
    write_json(
        layout.sweep(args.task),
        {
            "sweep": args.task,
            "replicate": 0,
            "split": "selection",
            "steps": design["steps"]["sweep"],
            "results": results,
            "selected": selected,
            "seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
