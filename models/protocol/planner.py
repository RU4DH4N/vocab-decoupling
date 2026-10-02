from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from models.protocol.receiver import LowRankMessageChannel
from models.protocol.sender import CoarseSender
from models.shared.block import Block
from models.shared.kv_cache import KVCache
from models.shared.rope import rotate_half
from models.shared.symbols import N_BYTES, START, STOP
from models.shared.validate import require_positive


class ByteReceiver(nn.Module):
    def __init__(
        self,
        message_dimensions: int,
        embedding_dimensions: int,
        hidden_dimensions: int,
        communication_rank: int,
    ) -> None:
        super().__init__()
        require_positive(
            message_dimensions=message_dimensions,
            embedding_dimensions=embedding_dimensions,
            hidden_dimensions=hidden_dimensions,
            communication_rank=communication_rank,
        )
        self.hidden_dimensions = hidden_dimensions
        self.embedding = nn.Embedding(N_BYTES + 2, embedding_dimensions)
        self.recurrent = nn.GRU(
            embedding_dimensions,
            hidden_dimensions,
            batch_first=True,
        )
        self.current_channel = LowRankMessageChannel(
            embedding_dimensions,
            message_dimensions,
            communication_rank,
        )
        self.output = nn.Linear(hidden_dimensions, N_BYTES + 1, bias=True)

    def initial_hidden(
        self,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        return reference.new_zeros(batch_size, self.hidden_dimensions)

    def communicate(self, hidden: Tensor, current: Tensor) -> Tensor:
        return self.current_channel(hidden, current)

    def step(
        self,
        previous: Tensor,
        hidden: Tensor,
        current: Tensor,
    ) -> tuple[Tensor, Tensor]:
        conditioned = self.communicate(self.embedding(previous), current)
        output, hidden = self.recurrent(conditioned[:, None], hidden)
        return self.output(output[:, 0]), hidden


class EventPlanner(nn.Module):
    def __init__(
        self,
        message_dimensions: int,
        embedding_dimensions: int,
        hidden_dimensions: int,
        communication_rank: int,
    ) -> None:
        super().__init__()
        self.decoder = ByteReceiver(
            message_dimensions,
            embedding_dimensions,
            hidden_dimensions,
            communication_rank,
        )

    def forward(self, messages: Tensor, previous: Tensor) -> Tensor:
        if (
            messages.ndim != 2
            or previous.ndim != 2
            or messages.shape[0] != previous.shape[0]
        ):
            raise ValueError(
                "planner expects messages [B,D] and previous symbols [B,S]"
            )
        embedded = self.decoder.embedding(previous)
        conditioned = self.decoder.communicate(
            embedded, messages[:, None].expand(-1, previous.shape[1], -1)
        )
        hidden, _ = self.decoder.recurrent(conditioned)
        return self.decoder.output(hidden)


@dataclass(frozen=True)
class Hypotheses:
    raw: list[list[bytes]]
    log_probs: Tensor
    available: Tensor


@dataclass(frozen=True)
class CausalPlan:
    messages: Tensor
    log_probs: Tensor
    available: Tensor


@dataclass(frozen=True)
class BatchPlan:
    memory: Tensor
    messages: Tensor
    log_probs: Tensor
    available: Tensor
    proposals: tuple[tuple[bytes, ...], ...]


@torch.no_grad()
def propose_beams(
    planner: EventPlanner, messages: Tensor, *, width: int, max_symbols: int
) -> Hypotheses:
    count = messages.shape[0]
    device = messages.device
    decoder = planner.decoder
    hidden = (
        decoder.initial_hidden(count, messages)[:, None]
        .expand(-1, width, -1)
        .contiguous()
    )
    conditioning = messages[:, None].expand(-1, width, -1).reshape(count * width, -1)
    scores = torch.full((count, width), -torch.inf, device=device)
    scores[:, 0] = 0.0
    previous = torch.full((count, width), START, device=device)
    symbols = torch.full((count, width, max_symbols), -1, device=device)
    finished = torch.zeros(count, width, dtype=torch.bool, device=device)
    rows = torch.arange(count, device=device)[:, None]
    for step in range(max_symbols):
        logits, state = decoder.step(
            previous.flatten(),
            hidden.reshape(1, count * width, -1),
            conditioning,
        )
        log_probs = logits.float().log_softmax(-1).reshape(count, width, -1)
        if not step:
            log_probs[..., STOP] = -torch.inf
        stay = torch.full_like(log_probs, -torch.inf)
        stay[..., STOP] = 0.0
        extended = scores[..., None] + torch.where(finished[..., None], stay, log_probs)
        scores, index = extended.flatten(1).topk(width, dim=-1)
        beam, symbol = index // log_probs.shape[-1], index % log_probs.shape[-1]
        hidden = state.reshape(count, width, -1)[rows, beam]
        symbols = symbols[rows, beam]
        finished = finished[rows, beam]
        symbols[:, :, step] = torch.where(finished | symbol.eq(STOP), -1, symbol)
        finished = finished | symbol.eq(STOP)
        previous = symbol
        if bool(finished.all()):
            break
    available = finished & scores.isfinite()
    raw = [[bytes(beam[beam >= 0].tolist()) for beam in row] for row in symbols.cpu()]
    return Hypotheses(raw, torch.where(available, scores, -torch.inf), available)


def _branch(
    sender: CoarseSender,
    codes: Tensor,
    truth: list[tuple[Tensor, Tensor]],
    earlier: list[tuple[Tensor, Tensor]],
    positions: Tensor,
    allowed: Tensor,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
    trunk = sender.trunk
    batch, events = codes.shape[:2]
    known = allowed.shape[1]
    x = trunk.in_proj(codes)
    updated = []
    for module, (true_k, true_v), (old_k, old_v) in zip(
        trunk.blocks, truth, earlier, strict=True
    ):
        block = cast(Block, module)
        h = block.ln1(x)
        q, k, v = (
            t.transpose(1, 2)
            for t in block.qkv(h)
            .reshape(batch, events, 3, block.n_heads, block.head_dim)
            .unbind(2)
        )
        cos = block.rope.cos.to(codes.device)[positions]
        sin = block.rope.sin.to(codes.device)[positions]
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(q.dtype)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(k.dtype)
        own_k = torch.cat((old_k, k[..., None, :]), dim=-2)
        own_v = torch.cat((old_v, v[..., None, :]), dim=-2)
        scale = block.head_dim**-0.5
        prefix = torch.einsum("bnqd,bnkd->bnqk", q.float(), true_k.float()) * scale
        prefix = prefix.masked_fill(~allowed, -torch.inf)
        own = torch.einsum("bnqd,bnqhd->bnqh", q.float(), own_k.float()) * scale
        weights = torch.cat((prefix, own), dim=-1).softmax(-1)
        attended = torch.einsum(
            "bnqk,bnkd->bnqd", weights[..., :known], true_v.float()
        ) + torch.einsum("bnqh,bnqhd->bnqd", weights[..., known:], own_v.float())
        attended = attended.to(x.dtype).transpose(1, 2).reshape(batch, events, -1)
        x = x + block.out_proj(attended)
        x = x + block.mlp(block.ln2(x))
        updated.append((own_k, own_v))
    return sender.meaning_norm(trunk.out(trunk.ln_f(x))), updated


def _roll(
    sender: CoarseSender,
    planner: EventPlanner,
    memory: Tensor,
    truth: list[tuple[Tensor, Tensor]],
    first: Tensor,
    allowed: Tensor,
    *,
    horizon: int,
    width: int,
    max_symbols: int,
) -> BatchPlan:
    batch, events, dimensions = memory.shape
    earlier = [
        (
            k.new_zeros(*k.shape[:-2], events, 0, k.shape[-1]),
            v.new_zeros(*v.shape[:-2], events, 0, v.shape[-1]),
        )
        for k, v in truth
    ]
    messages = memory.new_zeros(batch, events, horizon, width, dimensions)
    log_probs = torch.full(
        (batch, events, horizon, width), -torch.inf, device=memory.device
    )
    available = torch.zeros(
        batch, events, horizon, width, device=memory.device, dtype=torch.bool
    )
    proposed_first = propose_beams(
        planner,
        memory.reshape(batch * events, -1),
        width=width,
        max_symbols=max_symbols,
    )
    proposals = tuple(tuple(row) for row in proposed_first.raw)
    raw = [guess for row in proposed_first.raw for guess in row]
    score = proposed_first.log_probs.reshape(batch, events, width)
    live = proposed_first.available.reshape(batch, events, width)
    for step in range(horizon):
        proposed, dead = sender.encode_trunk_vocab(raw)
        live = live & ~dead.reshape(batch, events, width)
        chains = (
            proposed.reshape(batch, events, width, -1)
            .transpose(1, 2)
            .reshape(batch * width, events, -1)
        )
        current, earlier = _branch(
            sender, chains, truth, earlier, first + step, allowed
        )
        current = current.reshape(batch, width, events, dimensions).transpose(1, 2)
        messages[:, :, step] = torch.where(live[..., None], current, 0)
        log_probs[:, :, step] = torch.where(live, score, -torch.inf)
        available[:, :, step] = live
        if step + 1 == horizon:
            break
        following = propose_beams(
            planner,
            current.transpose(1, 2).reshape(batch * width * events, -1),
            width=1,
            max_symbols=max_symbols,
        )
        order = (
            following.log_probs.reshape(batch, width, events).transpose(1, 2),
            following.available.reshape(batch, width, events).transpose(1, 2),
        )
        score = score + order[0]
        live = live & order[1]
        raw = [
            following.raw[(b * width + k) * events + e][0]
            for b in range(batch)
            for e in range(events)
            for k in range(width)
        ]
    return BatchPlan(memory, messages, log_probs, available, proposals)


def _truth(
    caches: Sequence[KVCache], known: int, width: int
) -> list[tuple[Tensor, Tensor]]:
    truth = []
    for cache in caches:
        if cache.k is None or cache.v is None or cache.pos < known:
            raise RuntimeError("the sender caches do not hold the known events")
        truth.append(
            (
                cache.k[:, :, :known].repeat_interleave(width, 0),
                cache.v[:, :, :known].repeat_interleave(width, 0),
            )
        )
    return truth


@torch.no_grad()
def roll_ahead_batch(
    sender: CoarseSender,
    planner: EventPlanner,
    codes: Tensor,
    *,
    horizon: int,
    width: int,
    max_symbols: int,
) -> BatchPlan:
    if sender.training or planner.training:
        raise ValueError("causal rollout requires frozen evaluation-mode modules")
    events = codes.shape[1]
    if horizon < 1 or events + horizon > sender.trunk.config.max_seq_len:
        raise ValueError("sender context cannot fit the declared speculative horizon")
    caches = sender.new_caches(events)
    memory = sender.state(codes, caches)
    order = torch.arange(events, device=codes.device)
    return _roll(
        sender,
        planner,
        memory,
        _truth(caches, events, width),
        order + 1,
        order[None, :] <= order[:, None],
        horizon=horizon,
        width=width,
        max_symbols=max_symbols,
    )


@torch.no_grad()
def roll_ahead_last(
    sender: CoarseSender,
    planner: EventPlanner,
    caches: Sequence[KVCache],
    memory: Tensor,
    *,
    horizon: int,
    width: int,
    max_symbols: int,
) -> CausalPlan:
    if sender.training or planner.training:
        raise ValueError("causal rollout requires frozen evaluation-mode modules")
    known = caches[0].pos
    if horizon < 1 or known + horizon > sender.trunk.config.max_seq_len:
        raise ValueError("sender context cannot fit the declared speculative horizon")
    plan = _roll(
        sender,
        planner,
        memory,
        _truth(caches, known, width),
        torch.tensor([known], device=memory.device),
        torch.ones(1, known, dtype=torch.bool, device=memory.device),
        horizon=horizon,
        width=width,
        max_symbols=max_symbols,
    )
    return CausalPlan(plan.messages[0, 0], plan.log_probs[0, 0], plan.available[0, 0])
