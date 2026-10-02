import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from data.communication import ProtocolBatch, ReceiverUnits, candidate_events
from data.protocol_corpus import EventCorpus, selection_windows
from execution.design import seed_for
from execution.layout import Layout, required
from execution.protocol_evaluation import RowScores, evaluate_controls
from execution.protocol_io import interface_state
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import encoded, initial_protocol
from execution.validation import validate
from execution.windows import protocol_batch
from framework.checkpoints import write_json
from framework.runtime import set_seed
from models.protocol.model import ProtocolModel
from models.protocol.receiver import Receiver

LEVELS = (1, 2, 3)
ARMS = ("learned", "latest", "fixed", "oracle")


def window_events(config: dict) -> int:
    events = config["sender"]["max_seq_len"] // max(LEVELS) - 1
    if events < 1:
        raise ValueError("the sender context is too short for the clock distortions")
    return events


def split_word(word: str, pieces: int, rng: np.random.Generator) -> list[str]:
    count = min(pieces, len(word))
    cuts = sorted(rng.choice(np.arange(1, len(word)), size=count - 1, replace=False))
    bounds = [0, *cuts, len(word)]
    return [word[start:end] for start, end in zip(bounds, bounds[1:], strict=False)]


def distort(
    model: ProtocolModel[Receiver],
    table: ReceiverUnits,
    corpus: EventCorpus,
    rows: Tensor,
    level: int,
    rng: np.random.Generator,
    config: dict,
    device: torch.device,
) -> tuple[ProtocolBatch, dict[str, Tensor]]:
    base = protocol_batch(config, table, rows, "cpu")
    words = corpus.decode(rows)
    pieces, keys = [], []
    closed = 0
    for row in words:
        row_pieces, row_keys = [], []
        for word_index, word in enumerate(row[:-1]):
            parts = split_word(word, int(rng.integers(1, level + 1)), rng)
            end = 0
            for position, part in enumerate(parts):
                end += len(part.encode())
                final = position == len(parts) - 1
                row_pieces.append(part)
                row_keys.append((word_index, end, final))
        pieces.append(row_pieces)
        keys.append(row_keys)
        closed = max(closed, len(row_pieces))
    inventory = list(dict.fromkeys(piece for row in pieces for piece in row))
    model.sender.build_table(inventory)
    index = {piece: position for position, piece in enumerate(inventory)}
    boundary_steps = base.receiver_ids.eq(table.inventory.start) | base.receiver_ids.eq(
        table.inventory.stop
    )
    steps = torch.arange(base.receiver_ids.shape[1]).expand_as(base.receiver_ids)
    consumed = (steps - torch.where(boundary_steps, steps, 0).cummax(-1).values).numpy()
    frontier = base.frontier.numpy()
    sender_ids = np.zeros((len(words), closed), dtype=np.int64)
    latest = np.full(frontier.shape, -1, dtype=np.int64)
    boundary = np.full(frontier.shape, -1, dtype=np.int64)
    scale = config["corpus"]["max_unit_bytes"] + 2
    for row, (row_pieces, row_keys) in enumerate(zip(pieces, keys, strict=True)):
        count = len(row_pieces)
        sender_ids[row, :count] = [index[piece] for piece in row_pieces]
        order = np.array(
            [
                word * scale + (scale - 1 if final else end)
                for word, end, final in row_keys
            ]
        )
        live = frontier[row] >= 0
        threshold = (frontier[row] + 1) * scale + consumed[row]
        latest[row] = np.where(
            live, np.searchsorted(order, threshold, side="right") - 1, -1
        )
        boundary[row] = np.where(
            live,
            np.searchsorted(order, (frontier[row] + 1) * scale, side="left") - 1,
            -1,
        )
    targets = base.targets
    fixed = np.where(frontier >= 0, np.minimum(frontier, latest), -1)
    batch = replace(
        base,
        sender_ids=torch.from_numpy(sender_ids),
        candidate_event_ids=candidate_events(
            torch.from_numpy(latest), config["interface"]["candidate_window"]
        ),
        frontier=torch.from_numpy(latest),
        alignment_targets=torch.from_numpy(boundary).masked_fill(targets.eq(-100), -1),
    )
    rules = {
        "latest": torch.from_numpy(latest),
        "fixed": torch.from_numpy(fixed),
        "oracle": torch.from_numpy(boundary),
    }
    return to_device(batch, device), {k: v.to(device) for k, v in rules.items()}


def to_device(batch: ProtocolBatch, device: torch.device) -> ProtocolBatch:
    return ProtocolBatch(
        *(
            getattr(batch, field).to(device)
            for field in ProtocolBatch.__dataclass_fields__
        )
    )


def reading(batch: ProtocolBatch, rules: dict[str, Tensor], arm: str) -> ProtocolBatch:
    if arm == "learned":
        return batch
    slot = rules[arm]
    return replace(
        batch,
        candidate_event_ids=slot.unsqueeze(-1),
        frontier=slot,
        alignment_targets=slot.masked_fill(batch.targets.eq(-100), -1),
    )


def schedule(config: dict, arm: str) -> list[StageConfig]:
    alignment, communication = config["replacement_stages"][:2]
    phases = [alignment, communication] if arm == "learned" else [communication] * 2
    return [
        StageConfig(
            **{
                **stage,
                "learning_rate": required(stage["learning_rate"], "adapter rate"),
            }
        )
        for stage in phases
    ]


def _fit_arm(
    config: dict,
    layout: Layout,
    corpus: EventCorpus,
    table: ReceiverUnits,
    level: int,
    arm: str,
    device: torch.device,
) -> dict[str, Tensor]:
    root_seed = json.loads(layout.design.read_text())["inputs"]["root_seed"]
    events = window_events(config)
    set_seed(seed_for(root_seed, layout.replicate, "clocks"))
    model, _, _ = initial_protocol(config, layout, "fresh-byte", corpus, table, device)
    rng = np.random.default_rng(seed_for(root_seed, layout.replicate, level))
    for phase, settings in enumerate(schedule(config, arm)):
        trainer = StageTrainer(
            model, settings, {"clocks": arm, "level": level, "phase": phase}, rng
        )
        for _ in range(settings.steps):
            rows = corpus.windows("train", events, config["batch_size"], rng)
            batch, rules = distort(
                model, table, corpus, rows, level, rng, config, device
            )
            trainer.step(reading(batch, rules, arm))
    print(f"clocks level={level} arm={arm}", flush=True)
    return interface_state(model)


def train(layout: Layout, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "fresh-byte", corpus)
    states = {
        f"{level}/{arm}": _fit_arm(config, layout, corpus, table, level, arm, device)
        for level in LEVELS
        for arm in ARMS
    }
    folder = layout.run / "clocks"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(states, folder / "interfaces.pt")


def _score_arm(
    config: dict,
    layout: Layout,
    corpora: tuple[EventCorpus, EventCorpus],
    table: ReceiverUnits,
    state: dict[str, Tensor],
    level: int,
    arm: str,
    split: str,
    device: torch.device,
) -> dict:
    training, corpus = corpora
    model, _, _ = initial_protocol(
        config, layout, "fresh-byte", training, table, device
    )
    model.receiver.load_state_dict(
        {**model.receiver.state_dict(), **state}, strict=True
    )
    model.eval()
    events = window_events(config)
    permutation = torch.arange(config["batch_size"]).roll(1)
    rng = np.random.default_rng(config["evaluation_seed"])
    distortion = np.random.default_rng(config["evaluation_seed"] + level)
    combined: dict[str, list[RowScores]] = {}
    agree, live = 0, 0
    for _ in range(layout.evaluation_batches(config, split)):
        rows, _ = selection_windows(
            corpus, events, config["batch_size"], rng, split="selection"
        )
        batch, rules = distort(
            model, table, corpus, rows, level, distortion, config, device
        )
        valid = batch.alignment_targets.ge(0)
        if arm != "learned":
            agree += int((rules[arm].eq(rules["oracle"]) & valid).sum().item())
            live += int(valid.sum().item())
        scores = evaluate_controls(
            model, reading(batch, rules, arm), permutation=permutation, hard=False
        )
        for name, value in scores.items():
            combined.setdefault(name, []).append(value)
    rows_by_arm = {name: RowScores.concatenate(rows) for name, rows in combined.items()}
    aggregated = {name: value.aggregate() for name, value in rows_by_arm.items()}
    if arm != "learned":
        for value in aggregated.values():
            value["hard_event_accuracy"] = agree / live if live else None
    return {
        "scores": aggregated,
        "paired_nll": {name: value.nll.tolist() for name, value in rows_by_arm.items()},
        "utf8_bytes": rows_by_arm["correct"].utf8_bytes.tolist(),
    }


@torch.no_grad()
def measure(layout: Layout, device: torch.device, split: str) -> None:
    config = layout.config()
    validate(config)
    corpora = (
        EventCorpus.load(layout.corpus),
        EventCorpus.load(layout.evaluation_corpus(split)),
    )
    table = encoded(config, layout, "fresh-byte", corpora[1])
    states = torch.load(layout.run / "clocks" / "interfaces.pt", weights_only=True)
    events = window_events(config)
    results = {
        str(level): {
            arm: _score_arm(
                config,
                layout,
                corpora,
                table,
                states[f"{level}/{arm}"],
                level,
                arm,
                split,
                device,
            )
            for arm in ARMS
        }
        for level in LEVELS
    }
    write_json(
        layout.measured(split) / "clocks" / "evaluation.json",
        {
            "split": split,
            "levels": list(LEVELS),
            "window_events": events,
            "distortion": "each word split into 1..level pieces at random character boundaries; the sender reads pieces",
            "availability": "sender states of completed pieces, including non-final pieces of the word being spelled",
            "target": "the sender state at the end of the previous word",
            "arms": {
                "learned": "scorer bootstrapped by alignment, then communication",
                "latest": "reads the most recent available state",
                "fixed": "reads slot index equal to word index (t == u)",
                "oracle": "reads the target state; a diagnostic ceiling",
            },
            "results": results,
        },
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
