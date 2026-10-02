import math
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
from models.protocol.receiver import LowRankMessageChannel, ReceiverReadout

KINDS = ("input", "linear", "attention")


class LinearChannel(nn.Module):
    def __init__(self, hidden: int, message: int) -> None:
        super().__init__()
        self.map = nn.Linear(message, hidden, bias=False)
        nn.init.zeros_(self.map.weight)

    def forward(self, hidden: Tensor, message: Tensor) -> Tensor:
        return hidden + self.map(message)


class HeadScorer(nn.Module):
    def __init__(self, sender: int, receiver: int, heads: int, width: int) -> None:
        super().__init__()
        self.heads, self.width = heads, width
        self.query = nn.Linear(receiver, heads * width, bias=False)
        self.key = nn.Linear(sender, heads * width, bias=False)

    def forward(self, receiver: Tensor, candidates: Tensor) -> Tensor:
        query = self.query(receiver).unflatten(-1, (self.heads, self.width))
        key = self.key(candidates).unflatten(-1, (self.heads, self.width))
        return torch.einsum("...hd,...khd->...hk", query, key) / math.sqrt(self.width)


class HeadValue(nn.Module):
    def __init__(self, hidden: int, message: int, heads: int, width: int) -> None:
        super().__init__()
        self.heads, self.width = heads, width
        self.value = nn.Linear(message, heads * width, bias=False)
        self.out = nn.Linear(heads * width, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, hidden: Tensor, probs: Tensor, candidates: Tensor) -> Tensor:
        value = self.value(candidates).unflatten(-1, (self.heads, self.width))
        read = torch.einsum("...hk,...khd->...hd", probs.to(value.dtype), value)
        return hidden + self.out(read.flatten(-2))


def layerwise_parameters(width: int, message: int, score: int, rank: int) -> int:
    layer = nn.ModuleList(
        (
            CompatibilityScorer(message, width, score),
            LowRankMessageChannel(width, message, rank),
        )
    )
    return sum(p.numel() for p in layer.parameters())


class InterfaceReceiver(nn.Module):
    def __init__(
        self,
        native: BPEModel,
        sender_dimensions: int,
        score_dimensions: int,
        communication_rank: int,
        candidate_window: int,
        output_symbols: int,
        kind: str,
    ) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"interface kind must be one of {KINDS}")
        if not 0 < output_symbols <= native.lm_head.out_features:
            raise ValueError("output_symbols must fit the native output projection")
        self.native = native.requires_grad_(False).eval()
        self.kind = kind
        self.output_symbols = output_symbols
        self.candidate_window = candidate_window
        self.sender_dimensions = sender_dimensions
        width = native.trunk.config.d_model
        layers = native.trunk.config.n_layers
        message = sender_dimensions
        if kind == "attention":
            heads = native.trunk.config.n_heads
            budget = layerwise_parameters(
                width, message, score_dimensions, communication_rank
            )
            head_width = max(1, round(budget / (heads * 2 * (width + message))))
            self.scorers = nn.ModuleList(
                HeadScorer(message, width, heads, head_width) for _ in range(layers)
            )
            self.channels = nn.ModuleList(
                HeadValue(width, message, heads, head_width) for _ in range(layers)
            )
            return
        self.scorers = nn.ModuleList(
            (CompatibilityScorer(message, width, score_dimensions),)
        )
        self.channels = nn.ModuleList(
            (
                LowRankMessageChannel(width, message, min(width, message))
                if kind == "input"
                else LinearChannel(width, message),
            )
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
    ) -> ReceiverReadout:
        allowed = causal_event_mask(event_ids, frontier, self.candidate_window)
        candidates = candidates.masked_fill(~allowed.unsqueeze(-1), 0)
        correspondence: list[Tensor] = []

        def communicate(layer: int, hidden: Tensor) -> Tensor:
            if layer >= len(self.scorers):
                return hidden
            if self.kind != "attention":
                log_probs = correspondence_log_probs(
                    self.scorers[layer](hidden, candidates), allowed
                )
                correspondence.append(log_probs)
                selected = select_message(candidates, log_probs, hard=hard)
                return self.channels[layer](hidden, selected)
            scores = self.scorers[layer](hidden, candidates)
            heads = correspondence_log_probs(
                scores, allowed.unsqueeze(-2).expand_as(scores)
            )
            probs = heads.exp()
            if hard:
                probs = torch.zeros_like(probs).scatter_(
                    -1, heads.argmax(-1, keepdim=True), 1.0
                ) * heads.isfinite().any(-1, keepdim=True)
                pooled = probs.mean(-2).log()
            else:
                live = heads.isfinite()
                pooled = torch.where(live, heads, 0.0).logsumexp(-2) - math.log(
                    heads.shape[-2]
                )
            correspondence.append(pooled.masked_fill(~allowed, -torch.inf))
            return self.channels[layer](hidden, probs, candidates)

        logits = self.native(ids, layer_transform=communicate)
        return ReceiverReadout(
            logits[..., : self.output_symbols], tuple(correspondence)
        )
