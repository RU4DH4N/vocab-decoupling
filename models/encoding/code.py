import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from models.encoding.walsh import hadamard_rows
from models.shared.symbols import N_BYTES

BLANK = N_BYTES

N_ROWS = N_BYTES + 1
VOCABULARY_TABLE_BUDGET_BYTES = 64_000_000
_POSITION_CODE_DOMAIN = b"vocab-decoupling-position-code-v1\0"
_WORKING_BYTES_PER_VALUE = 12


def _byte_matrix_from_bytes(
    raw: list[bytes],
    width: int | None = None,
) -> NDArray[np.int32]:
    if width is None:
        width = max((len(value) for value in raw), default=1)

    out = np.full(
        (len(raw), width),
        BLANK,
        dtype=np.int32,
    )

    for i, value in enumerate(raw):
        if len(value) > width:
            raise ValueError(f"unit={i} needs bytes={len(value)} but width={width}")
        out[i, : len(value)] = np.frombuffer(
            value,
            dtype=np.uint8,
        ).astype(np.int32)

    return out


def byte_matrix(vocab: list[str], width: int | None = None) -> NDArray[np.int32]:

    return _byte_matrix_from_bytes([unit.encode("utf-8") for unit in vocab], width)


def _position_codes(
    start: int,
    stop: int,
    dims: int,
    seed: int,
) -> NDArray[np.float32]:
    rows = []
    n_bytes = (dims + 7) // 8
    for position in range(start, stop):
        identity = f"{seed}:{position}".encode("ascii")
        digest = hashlib.shake_256(_POSITION_CODE_DOMAIN + identity).digest(n_bytes)
        bits = np.unpackbits(
            np.frombuffer(digest, dtype=np.uint8),
            count=dims,
            bitorder="little",
        )
        rows.append(bits.astype(np.float32) * 2.0 - 1.0)
    if not rows:
        return np.zeros((0, dims), dtype=np.float32)
    return np.stack(rows)


def position_codes(n: int, dims: int, seed: int) -> NDArray[np.float32]:

    return _position_codes(0, max(0, n), dims, seed)


class Code(nn.Module):
    P: Tensor
    C: Tensor

    def __init__(
        self,
        dims: int,
        seed: int,
    ) -> None:
        super().__init__()

        if dims <= N_BYTES:
            raise ValueError(
                f"dims={dims} must exceed N_BYTES={N_BYTES} to draw one row per byte"
            )

        self.dims = int(dims)
        self.seed = int(seed)

        C = np.zeros((N_ROWS, dims), dtype=np.float32)
        C[:N_BYTES] = hadamard_rows(
            dims,
            np.arange(1, N_BYTES + 1),
            dtype=np.float32,
        )

        self.register_buffer(
            "P",
            torch.zeros(0, self.dims, dtype=torch.float32),
            persistent=False,
        )

        self.register_buffer(
            "C",
            torch.from_numpy(C),
            persistent=False,
        )

    def _positions(self, n: int, device: torch.device) -> Tensor:
        if self.P.shape[0] < n:
            extension = _position_codes(
                self.P.shape[0],
                n,
                self.dims,
                self.seed,
            )
            added = torch.from_numpy(extension).to(
                device=device,
                dtype=self.C.dtype,
            )
            self.P = torch.cat((self.P.to(device), added), dim=0)
        elif self.P.device != device:
            self.P = self.P.to(device)

        return self.P[:n]

    def forward(self, b: Tensor) -> Tensor:
        return (self.C[b] * self._positions(b.shape[-1], b.device)).sum(dim=-2)

    @torch.no_grad()
    def table(
        self,
        bmat: Tensor,
        budget: int,
    ) -> Tensor:

        width = max(1, bmat.shape[1])
        chunk = max(
            1,
            budget // (width * self.dims * _WORKING_BYTES_PER_VALUE),
        )

        out = torch.empty(
            bmat.shape[0], self.dims, device=bmat.device, dtype=self.C.dtype
        )
        for start in range(0, bmat.shape[0], chunk):
            out[start : start + chunk] = self(bmat[start : start + chunk])
        return out

    @torch.no_grad()
    def _long_unit(self, raw: bytes, budget: int) -> Tensor:
        positions_per_chunk = max(
            1,
            budget // (self.dims * _WORKING_BYTES_PER_VALUE),
        )
        out = torch.zeros(self.dims, device=self.C.device, dtype=self.C.dtype)
        for start in range(0, len(raw), positions_per_chunk):
            stop = min(len(raw), start + positions_per_chunk)
            byte_ids = torch.from_numpy(
                np.frombuffer(raw[start:stop], dtype=np.uint8).astype(np.int64)
            ).to(self.C.device)
            positions = torch.from_numpy(
                _position_codes(start, stop, self.dims, self.seed)
            ).to(device=self.C.device, dtype=self.C.dtype)
            out += (self.C[byte_ids] * positions).sum(dim=0)
        return out

    @torch.no_grad()
    def vocabulary_table(
        self,
        vocab: Sequence[str | bytes],
        budget: int,
    ) -> Tensor:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if len(vocab) == 0:
            return torch.empty(
                0,
                self.dims,
                device=self.C.device,
                dtype=self.C.dtype,
            )

        raw = [
            unit if isinstance(unit, bytes) else unit.encode("utf-8") for unit in vocab
        ]
        table = torch.empty(
            len(raw), self.dims, device=self.C.device, dtype=self.C.dtype
        )
        start = 0
        while start < len(raw):
            stop = start
            width = 1
            while stop < len(raw):
                candidate_width = max(width, len(raw[stop]))
                candidate_rows = stop - start + 1
                working_bytes = (
                    candidate_rows
                    * candidate_width
                    * self.dims
                    * _WORKING_BYTES_PER_VALUE
                )
                if stop > start and working_bytes > budget:
                    break
                width = candidate_width
                stop += 1

            row_bytes = len(raw[start]) * self.dims * _WORKING_BYTES_PER_VALUE
            if row_bytes > budget:
                table[start] = self._long_unit(raw[start], budget)
                start = stop
                continue

            bmat = torch.from_numpy(_byte_matrix_from_bytes(raw[start:stop], width)).to(
                device=self.C.device, dtype=torch.long
            )
            table[start:stop] = self(bmat)
            start = stop

        return table
