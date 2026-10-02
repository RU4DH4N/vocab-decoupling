from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ProtocolBatch:
    sender_ids: Tensor
    receiver_ids: Tensor
    targets: Tensor
    candidate_event_ids: Tensor
    frontier: Tensor
    alignment_targets: Tensor
    target_bytes: Tensor


@dataclass(frozen=True)
class ReceiverInventory:
    symbols: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.symbols, bool)
            or not isinstance(self.symbols, int)
            or self.symbols <= 0
        ):
            raise ValueError("symbols must be a positive integer")

    @property
    def stop(self) -> int:
        return self.symbols

    @property
    def start(self) -> int:
        return self.symbols + 1

    @property
    def inputs(self) -> int:
        return self.symbols + 2

    @property
    def outputs(self) -> int:
        return self.symbols + 1


def candidate_events(frontier: Tensor, window: int) -> Tensor:
    if window <= 0:
        raise ValueError("candidate_window must be positive")
    ids = frontier[..., None] + torch.arange(1 - window, 1)
    return ids.masked_fill(ids.lt(0), -1)


class ReceiverUnits:
    def __init__(
        self,
        units: Sequence[str],
        pieces: Sequence[Sequence[int]],
        inventory: ReceiverInventory,
    ) -> None:
        if not units or len(units) != len(pieces):
            raise ValueError("units and pieces must be nonempty and have equal length")
        if any(not unit for unit in units):
            raise ValueError("empty coarse units are not supported")
        self.inventory = inventory
        self.pieces = tuple(tuple(row) for row in pieces)
        self.byte_lengths = tuple(len(unit.encode("utf-8")) for unit in units)
        for row in self.pieces:
            if not row or any(
                isinstance(piece, bool)
                or not isinstance(piece, int)
                or not 0 <= piece < inventory.symbols
                for piece in row
            ):
                raise ValueError(
                    "pieces must be nonempty sequences of ordinary inventory IDs"
                )
        self._lengths = np.array([len(row) + 1 for row in self.pieces], dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(self._lengths)[:-1]))
        self._symbols = np.array(
            [symbol for row in self.pieces for symbol in (*row, inventory.stop)],
            dtype=np.int64,
        )
        self._bytes = np.array(self.byte_lengths, dtype=np.int64)

    @classmethod
    def bytes(cls, units: Sequence[str]) -> "ReceiverUnits":
        return cls(
            units,
            [list(unit.encode("utf-8")) for unit in units],
            ReceiverInventory(256),
        )

    @classmethod
    def bpe(cls, units: Sequence[str], tokenizer: Any) -> "ReceiverUnits":
        pieces = [list(tokenizer.encode(unit).ids) for unit in units]
        for unit, row in zip(units, pieces, strict=True):
            if tokenizer.decode(row) != unit:
                raise ValueError(
                    "receiver tokenizer does not losslessly encode an event"
                )
        return cls(units, pieces, ReceiverInventory(tokenizer.get_vocab_size()))

    def batch(
        self,
        rows: Tensor,
        *,
        device: torch.device | str,
        max_receiver_steps: int,
        candidate_window: int,
    ) -> ProtocolBatch:
        if rows.device.type != "cpu" or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("coarse rows must be integer CPU tensors")
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
            raise ValueError("rows require a nonempty batch and at least two events")
        if rows.min().item() < 0 or rows.max().item() >= len(self.pieces):
            raise ValueError("coarse row IDs are outside the encoded inventory")
        batch_size, window = rows.shape
        units = rows.numpy()
        lengths = self._lengths[units]
        ends = lengths.cumsum(1)
        totals = ends[:, -1]
        width = int(totals.max())
        if width > max_receiver_steps:
            raise ValueError(
                f"receiver width={width} exceeds max_receiver_steps={max_receiver_steps}; "
                "the limit includes the context prefix and every event STOP"
            )
        steps = np.arange(width)
        live = steps[None, :] < totals[:, None]
        event = np.minimum(
            (steps[None, :, None] >= ends[:, None, :]).sum(-1), window - 1
        )
        unit = np.take_along_axis(units, event, 1)
        start = np.take_along_axis(ends - lengths, event, 1)
        symbols = self._symbols[self._offsets[unit] + np.where(live, steps - start, 0)]
        previous = np.zeros((batch_size, width), dtype=np.int64)
        previous[:, 0] = self.inventory.start
        previous[:, 1:] = np.where(live[:, 1:], symbols[:, :-1], 0)
        scored = live & (steps[None, :] >= lengths[:, :1])
        targets = torch.from_numpy(np.where(scored, symbols, -100))
        frontier = torch.from_numpy(np.where(live, event - 1, -1))
        previous = torch.from_numpy(previous)
        byte_counts = self._bytes[units[:, 1:]].sum(1)
        return ProtocolBatch(
            sender_ids=rows[:, :-1].to(device),
            receiver_ids=previous.to(device),
            targets=targets.to(device),
            candidate_event_ids=candidate_events(frontier, candidate_window).to(device),
            frontier=frontier.to(device),
            alignment_targets=frontier.masked_fill(targets.eq(-100), -1).to(device),
            target_bytes=torch.from_numpy(byte_counts).to(device),
        )
