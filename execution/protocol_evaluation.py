import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from data.communication import ProtocolBatch, ReceiverInventory
from models.protocol.correspondence import alignment_diagnostics
from models.protocol.model import ProtocolModel, gather_events
from models.protocol.receiver import ReceiverReadout


@dataclass(frozen=True)
class RowScores:
    nll: Tensor
    tokens: Tensor
    correct: Tensor
    utf8_bytes: Tensor
    alignment_count: Tensor
    correct_event_mass: Tensor
    hard_event_correct: Tensor
    event_error: Tensor

    @classmethod
    def concatenate(cls, parts: list["RowScores"]) -> "RowScores":
        return cls(
            *(
                torch.cat([getattr(part, field).cpu() for part in parts])
                for field in cls.__dataclass_fields__
            )
        )

    def aggregate(self) -> dict[str, float | int | None]:
        nll = float(self.nll.double().sum().item())
        tokens = int(self.tokens.sum().item())
        size = int(self.utf8_bytes.sum().item())
        aligned = int(self.alignment_count.sum().item())
        return {
            "nll_nats": nll,
            "tokens_including_stop": tokens,
            "utf8_bytes": size,
            "ce": nll / tokens if tokens else None,
            "canonical_path_bpb": nll / (math.log(2) * size) if size else None,
            "top1": float(self.correct.sum().item()) / tokens if tokens else None,
            "alignment_count": aligned,
            "correct_event_mass": float(self.correct_event_mass.sum().item()) / aligned
            if aligned
            else None,
            "hard_event_accuracy": float(self.hard_event_correct.sum().item()) / aligned
            if aligned
            else None,
            "expected_event_error": float(self.event_error.sum().item()) / aligned
            if aligned
            else None,
        }


def score_rows(readout: ReceiverReadout, batch: ProtocolBatch) -> RowScores:
    logits = readout.logits.float()
    if logits.shape[:2] != batch.targets.shape or logits.ndim != 3:
        raise ValueError("logits must match reference target rows")
    if batch.target_bytes.shape != batch.targets.shape[:1]:
        raise ValueError("byte counts must have one value per prefix")
    live = batch.targets.ne(-100)
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        batch.targets.flatten(),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(batch.targets)
    alignment_count = torch.zeros_like(batch.target_bytes)
    mass = torch.zeros_like(batch.target_bytes, dtype=torch.float64)
    correct = torch.zeros_like(batch.target_bytes)
    error = torch.zeros_like(mass)
    for log_probs in readout.correspondence:
        diagnostics = alignment_diagnostics(
            log_probs, batch.candidate_event_ids, batch.alignment_targets
        )
        valid = diagnostics.valid & live
        alignment_count += valid.sum(-1)
        mass += diagnostics.correct_mass.masked_fill(~valid, 0).double().sum(-1)
        correct += (diagnostics.hard_correct & valid).sum(-1)
        error += (
            diagnostics.expected_event_error.masked_fill(~valid, 0).double().sum(-1)
        )
    return RowScores(
        losses.double().sum(-1),
        live.sum(-1),
        (logits.argmax(-1).eq(batch.targets) & live).sum(-1),
        batch.target_bytes,
        alignment_count,
        mass,
        correct,
        error,
    )


@torch.no_grad()
def evaluate_controls(
    model: ProtocolModel,
    batch: ProtocolBatch,
    *,
    permutation: Tensor,
    hard: bool,
) -> dict[str, RowScores]:
    rows = batch.receiver_ids.shape[0]
    expected = torch.arange(rows)
    if (
        permutation.device.type != "cpu"
        or permutation.dtype not in (torch.int32, torch.int64)
        or permutation.shape != expected.shape
        or not torch.equal(permutation.sort().values, expected)
        or torch.any(permutation == expected)
    ):
        raise ValueError(
            "shuffled control requires a CPU permutation with no fixed rows"
        )
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        memory = model.sender(batch.sender_ids)
        results: dict[str, RowScores] = {}
        for name, values in (
            ("correct", memory),
            ("shuffled", memory[permutation.to(memory.device)]),
            ("zero_message", torch.zeros_like(memory)),
        ):
            readout = model.receiver(
                batch.receiver_ids,
                gather_events(values, batch.candidate_event_ids),
                batch.candidate_event_ids,
                batch.frontier,
                hard=hard,
            )
            results[name] = score_rows(readout, batch)
        native = model.receiver.native(batch.receiver_ids)[
            ..., : model.receiver.output_symbols
        ]
        results["native"] = score_rows(ReceiverReadout(native, ()), batch)
        return results
    finally:
        for module, training in modes:
            module.training = training


def position_scores(
    readout: ReceiverReadout,
    batch: ProtocolBatch,
    inventory: ReceiverInventory,
    width: int,
) -> Tensor:
    logits = readout.logits.float()
    live = batch.targets.ne(-100)
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        batch.targets.flatten(),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(batch.targets)
    boundary = batch.receiver_ids.eq(inventory.start) | batch.receiver_ids.eq(
        inventory.stop
    )
    steps = torch.arange(boundary.shape[-1], device=boundary.device).expand_as(boundary)
    starts = torch.where(boundary, steps, 0).cummax(-1).values
    position = (steps - starts).clamp_max(width - 1)[live]
    totals = torch.zeros(2, width, dtype=torch.float64, device=logits.device)
    totals[0].scatter_add_(0, position, losses[live].double())
    totals[1].scatter_add_(0, position, torch.ones_like(position, dtype=torch.float64))
    return totals


def prefix_ambiguity(vocab: list[str], width: int) -> np.ndarray:

    raw = [unit.encode() for unit in vocab]
    lengths = np.array([len(unit) for unit in raw])
    table = np.zeros((len(raw), width), dtype=np.int64)
    for size in range(width):
        eligible = np.flatnonzero(lengths >= size)
        keys = np.array([raw[i][:size] for i in eligible], dtype=object)
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        table[eligible, size] = counts[inverse]
    return table


def ambiguity_scores(
    readout: ReceiverReadout,
    batch: ProtocolBatch,
    rows: Tensor,
    table: Tensor,
    inventory: ReceiverInventory,
    bins: int,
) -> Tensor:
    logits = readout.logits.float()
    live = batch.targets.ne(-100)
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        batch.targets.flatten(),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(batch.targets)
    boundary = batch.receiver_ids.eq(inventory.start) | batch.receiver_ids.eq(
        inventory.stop
    )
    steps = torch.arange(boundary.shape[-1], device=boundary.device).expand_as(boundary)
    position = steps - torch.where(boundary, steps, 0).cummax(-1).values
    event = (batch.frontier + 1).clamp(0, rows.shape[1] - 1)
    units = rows.to(event.device).gather(1, event)
    counts = table.to(event.device)[
        units, position.clamp_max(table.shape[1] - 1)
    ].clamp_min(1)
    decade = counts.double().log10().floor().long().clamp_max(bins - 1)[live]
    totals = torch.zeros(2, bins, dtype=torch.float64, device=logits.device)
    totals[0].scatter_add_(0, decade, losses[live].double())
    totals[1].scatter_add_(0, decade, torch.ones_like(decade, dtype=torch.float64))
    return totals
