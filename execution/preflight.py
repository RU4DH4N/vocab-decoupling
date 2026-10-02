import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor, nn

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus, window_sums
from data.tokenizers import load_tokenizer
from execution.baseline_training import bpe_batch
from execution.design import LEARNING_RATE_GRID
from execution.layout import Layout
from execution.protocol_io import build_protocol
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import native_options
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import write_json
from framework.memory import CudaMeter, Meter
from framework.runtime import autocast_context, device_synchronize, set_seed
from models.bpe.bpe import BPEModel
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.planner import EventPlanner
from models.protocol.receiver import Receiver


def update(module: nn.Module, loss: Tensor, optimizer: torch.optim.Optimizer) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("preflight produced a nonfinite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in module.parameters() if p.requires_grad],
        1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()


def longest_window(corpus: EventCorpus, events: int, lengths: np.ndarray) -> Tensor:
    ids, offsets = corpus.ids["train"], corpus.offsets["train"]
    span = events + 1
    sums, valid = window_sums(lengths[ids], np.diff(offsets), span)
    if not valid.any():
        raise ValueError(f"train has no document with {span} units")
    start = int(np.where(valid, sums, -1).argmax())
    return torch.from_numpy(np.array(ids[start : start + span]))


def _stages(
    config: dict,
    model: ProtocolModel[Receiver],
    batch: ProtocolBatch,
    kind: str,
    meter: Meter,
) -> list[dict]:
    losses = []
    for settings in config["stages"]:
        trainer = StageTrainer(
            model,
            StageConfig(
                **{
                    **settings,
                    "learning_rate": LEARNING_RATE_GRID[0],
                    "alignment_weight": settings["alignment_weight"] or 0.0,
                }
            ),
            {"preflight": True},
            np.random.default_rng(0),
        )
        with meter.phase(f"{kind} {settings['name']}"):
            loss = trainer.step(batch)
        losses.append({"stage": settings["name"], "loss": loss.total.item()})
    return losses


def _lookahead(
    config: dict,
    model: ProtocolModel[Receiver],
    batch: ProtocolBatch,
    device: torch.device,
) -> float:
    options = config["lookahead"]
    model.eval().requires_grad_(False)
    with torch.no_grad():
        memory = model.sender(batch.sender_ids)
        candidates = gather_events(memory, batch.candidate_event_ids)
    future = LookaheadReceiver(model.receiver, options["horizon"], options["rank"]).to(
        device
    )
    optimizer = torch.optim.AdamW(
        [p for p in future.parameters() if p.requires_grad], lr=LEARNING_RATE_GRID[0]
    )
    shape = (*batch.sender_ids.shape, options["horizon"], options["hypotheses"])
    plans = torch.zeros(*shape, config["sender"]["d_meaning"], device=device)
    with autocast_context(device, config["precision"]):
        output = future(
            batch.receiver_ids,
            candidates,
            batch.candidate_event_ids,
            batch.frontier,
            plans,
            torch.ones(shape, device=device),
            torch.ones(shape, device=device, dtype=torch.bool),
            batch.frontier,
            hard=False,
        )
        loss = F.cross_entropy(
            output.logits.float().flatten(0, 1), batch.targets.flatten()
        )
    update(future, loss, optimizer)
    return loss.item()


def _receiver(
    config: dict,
    corpus: EventCorpus,
    kind: str,
    table: ReceiverUnits,
    device: torch.device,
    meter: Meter,
) -> dict:
    lengths = np.array([len(piece) + 1 for piece in table.pieces])
    rows = longest_window(corpus, config["events"], lengths)
    batch = protocol_batch(config, table, rows.expand(config["batch_size"], -1), device)
    options = {
        "sender": config["sender"],
        "native": native_options(config, table),
        "interface": {**config["interface"], "output_symbols": table.inventory.outputs},
    }
    with meter.phase(f"{kind} build"):
        model = build_protocol(options, corpus.vocab, device)
    receiver = model.receiver
    counts = {
        "sender": sum(p.numel() for p in model.sender.parameters()),
        "native": sum(p.numel() for p in receiver.native.parameters()),
        "correspondence": sum(p.numel() for p in receiver.scorers.parameters()),
        "communication": sum(p.numel() for p in receiver.channels.parameters()),
    }
    stages = _stages(config, model, batch, kind, meter)
    with meter.phase(f"{kind} lookahead"):
        future = _lookahead(config, model, batch, device)
    return {
        "parameters": counts,
        "stages": stages,
        "future_loss": future,
        "receiver_steps": batch.receiver_ids.shape[1],
    }


def _planner(config: dict, device: torch.device) -> dict:
    planner = EventPlanner(**config["planner"]["model"]).to(device)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=LEARNING_RATE_GRID[0])
    width = config["corpus"]["max_unit_bytes"] + 1
    rows = config["batch_size"] * config["events"]
    ids = torch.zeros(rows, width, device=device, dtype=torch.long)
    messages = torch.zeros(rows, config["sender"]["d_meaning"], device=device)
    with autocast_context(device, config["precision"]):
        loss = F.cross_entropy(
            planner(messages, ids).float().flatten(0, 1), ids.flatten()
        )
    update(planner, loss, optimizer)
    return {
        "loss": loss.item(),
        "parameters": sum(p.numel() for p in planner.parameters()),
    }


def _baseline(
    config: dict, corpus: EventCorpus, tokenizer: Tokenizer, device: torch.device
) -> dict:
    baseline = BPEModel(
        **config["baseline"]["model"], vocab_size=tokenizer.get_vocab_size()
    ).to(device)
    optimizer = torch.optim.AdamW(baseline.parameters(), lr=LEARNING_RATE_GRID[0])
    lengths = np.array([len(e.ids) for e in tokenizer.encode_batch(corpus.vocab)])
    rows = longest_window(corpus, config["events"], lengths)
    ids, targets = bpe_batch(
        corpus, rows.expand(config["batch_size"], -1), tokenizer, device
    )
    with autocast_context(device, config["precision"]):
        loss = F.cross_entropy(baseline(ids).float().flatten(0, 1), targets.flatten())
    update(baseline, loss, optimizer)
    return {
        "loss": loss.item(),
        "parameters": sum(p.numel() for p in baseline.parameters()),
        "steps": ids.shape[1],
    }


def check(
    config: dict,
    corpus: EventCorpus,
    receiver_tokenizer: Tokenizer,
    baseline_tokenizer: Tokenizer,
    device: torch.device,
    meter: Meter,
) -> dict:
    validate(config)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("remote preflight requires CUDA BF16 support")
    set_seed(config["receivers"]["primary"]["seed"])
    started = time.perf_counter()
    tables = {
        "bytes": ReceiverUnits.bytes(corpus.vocab),
        "bpe": ReceiverUnits.bpe(corpus.vocab, receiver_tokenizer),
    }
    results = {
        kind: _receiver(config, corpus, kind, table, device, meter)
        for kind, table in tables.items()
    }
    with meter.phase("planner"):
        results["planner"] = _planner(config, device)
    with meter.phase("baseline"):
        results["baseline"] = _baseline(config, corpus, baseline_tokenizer, device)
    device_synchronize(device)
    gpu = (
        {
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": max(meter.peaks.values()),
        }
        if device.type == "cuda"
        else {}
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "checks": results,
        "seconds": time.perf_counter() - started,
        "compile_enabled": False,
        "worst_case": "the longest training window for each receiver and the baseline, repeated across the batch",
        "excludes": ["rollout cache storage", "MAUVE encoder"],
        "phase_peak_bytes": meter.peaks,
        **gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, 0)
    device = torch.device(args.device)
    write_json(
        args.output / "preflight.json",
        check(
            layout.config(),
            EventCorpus.load(layout.corpus),
            load_tokenizer(layout.receiver_tokenizer),
            load_tokenizer(layout.baseline_tokenizer),
            device,
            CudaMeter(device) if device.type == "cuda" else Meter(),
        ),
    )


if __name__ == "__main__":
    main()
