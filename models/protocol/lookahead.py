from collections.abc import Sequence
from typing import Self, cast

import torch
from torch import Tensor, nn

from models.protocol.receiver import Receiver, ReceiverReadout
from models.shared.kv_cache import KVCache


def at_steps(values: Tensor, positions: Tensor) -> Tensor:
    batch, steps = positions.shape
    tail = values.shape[2:]
    shape = (batch, steps, *([1] * len(tail)))
    indices = positions.clamp_min(0).reshape(shape).expand(batch, steps, *tail)
    return values.gather(1, indices).masked_fill(positions.lt(0).reshape(shape), 0)


class FutureChannel(nn.Module):
    def __init__(self, hidden: int, message: int, rank: int) -> None:
        super().__init__()
        if not 0 < rank <= min(hidden, message):
            raise ValueError("future rank must fit message and native widths")
        self.rank = rank
        self.query = nn.Linear(hidden, rank, bias=False)
        self.key = nn.Linear(message, rank, bias=False)
        self.down = nn.Linear(message, rank, bias=False)
        self.gate = nn.Linear(hidden + message + 1, hidden, bias=True)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self,
        hidden: Tensor,
        current: Tensor,
        futures: Tensor,
        log_probs: Tensor,
        available: Tensor,
        positions: Tensor,
    ) -> Tensor:
        futures = futures.masked_fill(~available[..., None], 0)
        keys = at_steps(self.key(futures), positions)
        values = at_steps(self.down(futures), positions)
        log_probs = at_steps(log_probs, positions)
        available = at_steps(available, positions)
        scores = (self.query(hidden)[..., None, :] * keys).sum(-1)
        scores = scores.float() / self.rank**0.5 + log_probs.float()
        scores = scores.masked_fill(~available, -torch.inf)
        live = available.any(-1, keepdim=True)
        weights = torch.where(live, scores, 0).softmax(-1).masked_fill(~available, 0)
        read = (weights.to(values.dtype)[..., None] * values).sum(-2)
        mass = (
            torch.where(available, log_probs.float(), -torch.inf)
            .logsumexp(-1, keepdim=True)
            .exp()
        )
        gate = self.gate(
            torch.cat((hidden, current, mass.nan_to_num(0).to(hidden.dtype)), -1)
        ).sigmoid()
        delta = gate * self.up(nn.functional.silu(read))
        return hidden + delta.masked_fill(~live, 0)


class LookaheadReceiver(nn.Module):
    def __init__(self, current: Receiver, horizon: int, rank: int) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("lookahead horizon must be positive")
        self.current = current.requires_grad_(False).eval()
        self.horizon = horizon
        self.channels = nn.ModuleList(
            nn.ModuleList(
                FutureChannel(
                    current.native.trunk.config.d_model, current.sender_dimensions, rank
                )
                for _ in range(horizon)
            )
            for _ in current.channels
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.current.eval()
        return self

    def forward(
        self,
        ids: Tensor,
        candidates: Tensor,
        event_ids: Tensor,
        frontier: Tensor,
        future: Tensor,
        log_probs: Tensor,
        available: Tensor,
        positions: Tensor,
        *,
        hard: bool,
        caches: Sequence[KVCache] | None = None,
    ) -> ReceiverReadout:
        if (
            future.ndim != 5
            or future.shape[0] != ids.shape[0]
            or future.shape[2] != self.horizon
            or positions.shape != ids.shape
            or future.shape[-1] != self.current.sender_dimensions
            or log_probs.shape != future.shape[:-1]
            or available.shape != log_probs.shape
            or available.dtype != torch.bool
        ):
            raise ValueError(
                "lookahead needs batch/event/horizon/hypothesis/message axes"
            )

        def add_future(layer: int, hidden: Tensor, current: Tensor) -> Tensor:
            for index, channel in enumerate(cast(nn.ModuleList, self.channels[layer])):
                hidden = channel(
                    hidden,
                    current,
                    future[..., index, :, :],
                    log_probs[..., index, :],
                    available[..., index, :],
                    positions,
                )
            return hidden

        return self.current(
            ids,
            candidates,
            event_ids,
            frontier,
            hard=hard,
            caches=caches,
            communication_transform=add_future,
        )
