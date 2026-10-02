from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from models.shared.symbols import BYTE_IGNORE_INDEX, START, STOP


@dataclass(frozen=True)
class PackedUnits:
    blob: Tensor
    offsets: Tensor
    lengths: Tensor
    lengths_cpu: Tensor

    @classmethod
    def from_units(
        cls, units: Sequence[str], device: torch.device | str
    ) -> "PackedUnits":
        if not units:
            raise ValueError("packed units require at least one unit")
        encoded = [unit.encode("utf-8") for unit in units]
        lengths = torch.tensor([len(raw) for raw in encoded], dtype=torch.long)
        offsets = lengths.cumsum(0) - lengths
        joined = b"".join(encoded)
        blob = (
            torch.frombuffer(bytearray(joined), dtype=torch.uint8)
            if joined
            else torch.zeros(1, dtype=torch.uint8)
        )
        return cls(blob.to(device), offsets.to(device), lengths.to(device), lengths)

    def __len__(self) -> int:
        return len(self.lengths_cpu)

    def teacher_forcing(
        self,
        rows_cpu: Tensor,
        width: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if rows_cpu.device.type != "cpu" or rows_cpu.ndim != 1 or not len(rows_cpu):
            raise ValueError("rows must be a non-empty CPU index vector")
        required = int(self.lengths_cpu[rows_cpu].max()) + 1
        if width is None:
            width = required
        elif width < required:
            raise ValueError(
                f"teacher-forcing width={width} is below required width={required}"
            )
        rows = rows_cpu.to(self.blob.device)
        lengths = self.lengths[rows]
        positions = torch.arange(width, device=rows.device)
        inside = positions[None, :] < lengths[:, None]
        cursor = (self.offsets[rows][:, None] + positions[None, :]).clamp(
            max=len(self.blob) - 1
        )
        pieces = self.blob[cursor].long()
        targets = torch.where(
            inside,
            pieces,
            torch.where(
                positions[None, :] == lengths[:, None], STOP, BYTE_IGNORE_INDEX
            ),
        )
        shifted = torch.where(inside, pieces, START)
        previous = torch.cat(
            (shifted.new_full((len(rows), 1), START), shifted[:, :-1]), dim=1
        )
        return previous, targets, lengths
