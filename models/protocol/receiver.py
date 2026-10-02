from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from models.bpe.bpe import BPEModel
from models.protocol.correspondence import (
    CompatibilityScorer,
    causal_event_mask,
    correspondence_log_probs,
    select_message,
)
from models.shared.kv_cache import KVCache
from models.shared.validate import require_positive


class LowRankMessageChannel(nn.Module):
    def __init__(
        self,
        hidden_dimensions: int,
        message_dimensions: int,
        rank: int,
    ) -> None:
        super().__init__()
        require_positive(
            hidden_dimensions=hidden_dimensions,
            message_dimensions=message_dimensions,
            rank=rank,
        )
        if rank > min(hidden_dimensions, message_dimensions):
            raise ValueError(
                "rank must not exceed either interface width, "
                f"rank={rank}, hidden_dimensions={hidden_dimensions}, "
                f"message_dimensions={message_dimensions}"
            )
        self.gate = nn.Linear(
            hidden_dimensions + message_dimensions,
            hidden_dimensions,
            bias=True,
        )
        self.down = nn.Linear(message_dimensions, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dimensions, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor, message: Tensor) -> Tensor:
        if hidden.shape[:-1] != message.shape[:-1]:
            raise ValueError(
                "hidden and message leading dimensions differ, "
                f"hidden={tuple(hidden.shape)}, message={tuple(message.shape)}"
            )
        gate = torch.sigmoid(self.gate(torch.cat((hidden, message), dim=-1)))
        residual = self.up(nn.functional.silu(self.down(message)))
        return hidden + gate * residual


@dataclass(frozen=True)
class ReceiverReadout:
    logits: Tensor
    correspondence: tuple[Tensor, ...]


class Receiver(nn.Module):
    def __init__(
        self,
        native: BPEModel,
        sender_dimensions: int,
        score_dimensions: int,
        communication_rank: int,
        candidate_window: int,
        output_symbols: int,
    ) -> None:
        super().__init__()
        if candidate_window <= 0:
            raise ValueError("candidate_window must be positive")
        if not 0 < output_symbols <= native.lm_head.out_features:
            raise ValueError("output_symbols must fit the native output projection")
        self.native = native.requires_grad_(False).eval()
        self.output_symbols = output_symbols
        self.candidate_window = candidate_window
        self.sender_dimensions = sender_dimensions
        width = native.trunk.config.d_model
        layers = native.trunk.config.n_layers
        self.scorers = nn.ModuleList(
            CompatibilityScorer(sender_dimensions, width, score_dimensions)
            for _ in range(layers)
        )
        self.channels = nn.ModuleList(
            LowRankMessageChannel(width, sender_dimensions, communication_rank)
            for _ in range(layers)
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.native.eval()
        return self

    def forward(
        self,
        ids: Tensor,
        candidates: Tensor,
        event_ids: Tensor,
        frontier: Tensor,
        *,
        hard: bool,
        caches: Sequence[KVCache] | None = None,
        communication_transform: Callable[[int, Tensor, Tensor], Tensor] | None = None,
    ) -> ReceiverReadout:
        if ids.ndim != 2 or candidates.ndim != 4 or candidates.shape[:2] != ids.shape:
            raise ValueError(
                "candidates must be [batch, receiver steps, candidates, width]"
            )
        if candidates.shape[-1] != self.sender_dimensions:
            raise ValueError("candidate width differs from the sender interface")
        if event_ids.shape != candidates.shape[:-1] or frontier.shape != ids.shape:
            raise ValueError(
                "event IDs and frontier must match receiver/candidate axes"
            )
        allowed = causal_event_mask(event_ids, frontier, self.candidate_window)
        candidates = candidates.masked_fill(~allowed.unsqueeze(-1), 0)
        correspondence: list[Tensor] = []

        def communicate(layer: int, hidden: Tensor) -> Tensor:
            log_probs = correspondence_log_probs(
                self.scorers[layer](hidden, candidates), allowed
            )
            correspondence.append(log_probs)
            selected = select_message(candidates, log_probs, hard=hard)
            hidden = self.channels[layer](hidden, selected)
            return (
                hidden
                if communication_transform is None
                else communication_transform(layer, hidden, selected)
            )

        logits = self.native(ids, caches=caches, layer_transform=communicate)
        logits = logits[..., : self.output_symbols]
        return ReceiverReadout(logits, tuple(correspondence))
