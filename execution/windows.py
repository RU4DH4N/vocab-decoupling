import numpy as np
import torch
from torch import Tensor

from data.communication import ProtocolBatch, ReceiverUnits
from data.protocol_corpus import EventCorpus, selection_windows
from execution.protocol_evaluation import RowScores, evaluate_controls
from models.protocol.model import ProtocolModel


def protocol_batch(
    config: dict, table: ReceiverUnits, rows: Tensor, device: torch.device | str
) -> ProtocolBatch:
    return table.batch(
        rows,
        device=device,
        max_receiver_steps=config["receiver_steps"],
        candidate_window=config["interface"]["candidate_window"],
    )


@torch.no_grad()
def score_windows(
    model: ProtocolModel,
    table: ReceiverUnits,
    corpus: EventCorpus,
    config: dict,
    batches: int,
    device: torch.device,
) -> tuple[dict[str, RowScores], list[int]]:
    rng = np.random.default_rng(config["evaluation_seed"])
    permutation = torch.arange(config["batch_size"]).roll(1)
    parts: dict[str, list[RowScores]] = {}
    groups: list[int] = []
    for _ in range(batches):
        rows, group_ids = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        groups.extend(group_ids.tolist())
        batch = protocol_batch(config, table, rows, device)
        scores = evaluate_controls(model, batch, permutation=permutation, hard=False)
        for name, value in scores.items():
            parts.setdefault(name, []).append(value)
    return {name: RowScores.concatenate(rows) for name, rows in parts.items()}, groups
