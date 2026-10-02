import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from data.communication import ReceiverInventory
from execution.sampling import GenerationLimits, generation_record, sample
from framework.runtime import device_synchronize
from models.protocol.lookahead import LookaheadReceiver
from models.protocol.model import ProtocolModel
from models.protocol.planner import (
    BatchPlan,
    CausalPlan,
    EventPlanner,
    roll_ahead_batch,
    roll_ahead_last,
)


@dataclass(frozen=True)
class SymbolBytes:
    values: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not self.values or any(
            not isinstance(v, bytes) or not v for v in self.values
        ):
            raise ValueError("every ordinary symbol must encode nonempty bytes")

    @property
    def inventory(self) -> ReceiverInventory:
        return ReceiverInventory(len(self.values))

    @classmethod
    def bytes(cls) -> "SymbolBytes":
        return cls(tuple(bytes([i]) for i in range(256)))

    @classmethod
    def bpe(cls, tokenizer: Any) -> "SymbolBytes":
        if tokenizer.model.__class__.__name__.lower() != "bpe":
            raise ValueError("expected a ByteLevel BPE tokenizer")
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        mapping = {chr(b): b for b in visible}
        mapping.update(
            {
                chr(256 + i): b
                for i, b in enumerate(b for b in range(256) if b not in visible)
            }
        )
        values = []
        for index in range(tokenizer.get_vocab_size()):
            token = tokenizer.id_to_token(index)
            if not token or any(c not in mapping for c in token):
                raise ValueError("BPE vocabulary contains a non-byte symbol")
            values.append(bytes(mapping[c] for c in token))
        return cls(tuple(values))


class SessionLimitError(ValueError):
    pass


class GenerationSession:
    def __init__(
        self,
        model: ProtocolModel,
        symbols: SymbolBytes,
        *,
        hard: bool,
        lookahead: LookaheadReceiver | None = None,
        planner: EventPlanner | None = None,
        planner_max_symbols: int | None = None,
        planner_width: int | None = None,
    ) -> None:
        if model.training or model.sender.training or model.receiver.training:
            raise ValueError("generation requires an evaluation-mode model")
        inventory = symbols.inventory
        if (
            model.receiver.output_symbols != inventory.outputs
            or model.receiver.native.embedding.num_embeddings != inventory.inputs
        ):
            raise ValueError("generation alphabet differs from the receiver inventory")
        self.model, self.symbols, self.hard = model, symbols, hard
        if lookahead is not None and (
            lookahead.current is not model.receiver
            or planner is None
            or planner_max_symbols is None
            or planner_max_symbols < 1
            or planner_width is None
            or planner_width < 1
        ):
            raise ValueError(
                "lookahead requires the same current receiver and a causal planner"
            )
        if lookahead is not None and (
            lookahead.training or planner is None or planner.training
        ):
            raise ValueError("lookahead and planner must be in evaluation mode")
        self.lookahead, self.planner = lookahead, planner
        self.planner_max_symbols, self.planner_width = (
            planner_max_symbols,
            planner_width,
        )
        self.plan: CausalPlan | None = None
        self.events: list[bytes] = []
        self.device = model.receiver.native.embedding.weight.device
        self.receiver_caches = model.receiver.native.new_caches(
            model.receiver.native.trunk.config.max_seq_len
        )
        self.sender_caches = model.sender.new_caches(
            model.sender.trunk.config.max_seq_len
        )
        self.messages: list[Tensor] = []
        self.local = model.receiver.native.boundaries is not None
        self.partial = bytearray()
        self.started = False

    def rollout(self, codes: Tensor) -> BatchPlan:
        if (
            self.lookahead is None
            or self.planner is None
            or self.planner_max_symbols is None
            or self.planner_width is None
        ):
            raise ValueError("planning requires a lookahead receiver and planner")
        return roll_ahead_batch(
            self.model.sender,
            self.planner,
            codes,
            horizon=self.lookahead.horizon,
            width=self.planner_width,
            max_symbols=self.planner_max_symbols,
        )

    def causal_plan(self) -> CausalPlan | None:
        assert self.lookahead is not None
        assert self.planner is not None
        assert self.planner_max_symbols is not None and self.planner_width is not None
        if (
            len(self.events) + self.lookahead.horizon
            > self.model.sender.trunk.config.max_seq_len
        ):
            return None
        return roll_ahead_last(
            self.model.sender,
            self.planner,
            self.sender_caches,
            self.messages[-1],
            horizon=self.lookahead.horizon,
            width=self.planner_width,
            max_symbols=self.planner_max_symbols,
        )

    @torch.no_grad()
    def push(self, symbol: int) -> Tensor:
        inventory = self.symbols.inventory
        if (
            isinstance(symbol, bool)
            or not isinstance(symbol, int)
            or not 0 <= symbol < inventory.inputs
        ):
            raise ValueError("input symbol is outside the declared inventory")
        if (not self.started and symbol != inventory.start) or (
            self.started and symbol == inventory.start
        ):
            raise ValueError("START must occur exactly once at the beginning")
        if not self.receiver_caches[0].remaining:
            raise SessionLimitError("receiver-context-limit")
        if symbol == inventory.stop:
            if not self.partial:
                raise SessionLimitError("empty-event")
            if not self.sender_caches[0].remaining:
                raise SessionLimitError("sender-context-limit")
            table, dead = self.model.sender.encode_trunk_vocab([bytes(self.partial)])
            if dead.any():
                raise SessionLimitError("directionless-event")
            message = self.model.sender.state(table[None], self.sender_caches)
            self.messages.append(message)
            self.events.append(bytes(self.partial))
            self.partial.clear()
            if self.lookahead is not None:
                self.plan = self.causal_plan()
        elif symbol != inventory.start:
            self.partial.extend(self.symbols.values[symbol])
        self.started = True
        if self.local and symbol in (inventory.stop, inventory.start):
            for cache in self.receiver_caches:
                cache.reset()
        frontier = len(self.messages) - 1
        count = min(len(self.messages), self.model.receiver.candidate_window)
        if count:
            candidates = torch.cat(self.messages[-count:], dim=1)[:, None]
            events = torch.arange(
                frontier - count + 1, frontier + 1, device=self.device
            )[None, None]
        else:
            candidates = self.model.receiver.native.embedding.weight.new_zeros(
                1, 1, 1, self.model.receiver.sender_dimensions
            )
            events = torch.full((1, 1, 1), -1, device=self.device)
        inputs = (
            torch.tensor([[symbol]], device=self.device),
            candidates,
            events,
            torch.tensor([[frontier]], device=self.device),
        )
        if self.lookahead is None:
            return self.model.receiver(
                *inputs, hard=self.hard, caches=self.receiver_caches
            ).logits[0, 0]
        if self.plan is None:
            future = candidates.new_zeros(
                1, 1, self.lookahead.horizon, 1, candidates.shape[-1]
            )
            log_probs = torch.zeros(1, 1, self.lookahead.horizon, 1, device=self.device)
            available = torch.zeros_like(log_probs, dtype=torch.bool)
        else:
            future = self.plan.messages[None, None]
            log_probs = self.plan.log_probs[None, None]
            available = self.plan.available[None, None]
        return self.lookahead(
            *inputs,
            future,
            log_probs,
            available,
            torch.zeros(1, 1, dtype=torch.long, device=self.device),
            hard=self.hard,
            caches=self.receiver_caches,
        ).logits[0, 0]

    def prime(self, events: Sequence[Sequence[int]]) -> Tensor:
        if not events or any(not event for event in events):
            raise ValueError("prompt must contain nonempty encoded events")
        inventory = self.symbols.inventory
        for event in events:
            if any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < inventory.symbols
                for i in event
            ):
                raise ValueError("prompt events may contain only ordinary symbols")
        required = 1 if self.local else 1 + sum(len(event) + 1 for event in events)
        if (
            required > self.receiver_caches[0].remaining
            or len(events) > self.sender_caches[0].remaining
        ):
            raise SessionLimitError("prompt-context-limit")
        if (
            self.lookahead is not None
            and len(events) + self.lookahead.horizon
            > self.model.sender.trunk.config.max_seq_len
        ):
            raise SessionLimitError("prompt-context-limit")
        raw = [b"".join(self.symbols.values[s] for s in event) for event in events]
        codes, dead = self.model.sender.encode_trunk_vocab(raw)
        if dead.any():
            raise SessionLimitError("directionless-event")
        memory = self.model.sender.state(codes[None], self.sender_caches)
        self.messages = [memory[:, index : index + 1] for index in range(len(events))]
        self.events = raw
        self.started = True
        ids = torch.tensor(
            [
                [inventory.start]
                + [symbol for event in events for symbol in (*event, inventory.stop)]
            ],
            device=self.device,
        )
        frontier = ids.eq(inventory.stop).cumsum(-1) - 1
        count = len(events)
        event_ids = torch.arange(count, device=self.device).expand(
            1, ids.shape[1], count
        )
        candidates = memory[:, None].expand(1, ids.shape[1], count, memory.shape[-1])
        if self.local:
            ids, frontier = ids[:, -1:], frontier[:, -1:]
            event_ids, candidates = event_ids[:, -1:], candidates[:, -1:]
        if self.lookahead is None:
            return self.model.receiver(
                ids,
                candidates,
                event_ids,
                frontier,
                hard=self.hard,
                caches=self.receiver_caches,
            ).logits[0, -1]
        plan = self.rollout(codes[None])
        self.plan = CausalPlan(
            plan.messages[0, -1], plan.log_probs[0, -1], plan.available[0, -1]
        )
        return self.lookahead(
            ids,
            candidates,
            event_ids,
            frontier,
            plan.messages,
            plan.log_probs,
            plan.available,
            frontier,
            hard=self.hard,
            caches=self.receiver_caches,
        ).logits[0, -1]


@torch.no_grad()
def generate_protocol(
    model: ProtocolModel,
    symbols: SymbolBytes,
    prompt: Sequence[Sequence[int]],
    limits: GenerationLimits,
    *,
    seed: int,
    hard: bool,
    lookahead: LookaheadReceiver | None = None,
    planner: EventPlanner | None = None,
    planner_max_symbols: int | None = None,
    planner_width: int | None = None,
) -> dict:
    session = GenerationSession(
        model,
        symbols,
        hard=hard,
        lookahead=lookahead,
        planner=planner,
        planner_max_symbols=planner_max_symbols,
        planner_width=planner_width,
    )
    device_synchronize(session.device)
    start = time.perf_counter()
    logits = session.prime(prompt)
    generator = torch.Generator(device=session.device).manual_seed(seed)
    stop = symbols.inventory.stop
    sizes = torch.tensor(
        [*map(len, symbols.values), 0], device=session.device, dtype=torch.long
    )
    raw = bytearray()
    event_bytes = 0
    forced = 0
    reason = "symbol-limit"
    for _ in range(limits.symbols):
        allowed = sizes <= limits.event_bytes - event_bytes
        allowed[stop] = event_bytes > 0
        forced += int(event_bytes >= limits.event_bytes)
        symbol = sample(
            logits.masked_fill(~allowed, -torch.inf),
            generator,
            limits.temperature,
            limits.top_p,
        )
        if symbol != stop:
            value = symbols.values[symbol]
            raw.extend(value)
            event_bytes += len(value)
            if len(raw) >= limits.bytes:
                reason = "byte-budget"
                break
        else:
            event_bytes = 0
        try:
            logits = session.push(symbol)
        except SessionLimitError as error:
            reason = str(error)
            break
    device_synchronize(session.device)
    return {
        **generation_record(
            bytes(raw[: limits.bytes]),
            time.perf_counter() - start,
            reason != "byte-budget",
        ),
        "stop_reason": reason,
        "forced_word_breaks": forced,
        "sender_events": len(session.messages),
        "receiver_steps": session.receiver_caches[0].pos,
        "communication": "current-only" if lookahead is None else "causal-lookahead",
    }
