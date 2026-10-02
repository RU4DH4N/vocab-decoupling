from typing import Generic, Literal, Self, TypeVar

import torch
from torch import Tensor, nn

from models.protocol.interfaces import InterfaceReceiver
from models.protocol.receiver import Receiver, ReceiverReadout
from models.protocol.sender import CoarseSender

AnyReceiver = TypeVar("AnyReceiver", Receiver, InterfaceReceiver)

TrainingStage = Literal["alignment", "communication", "joint", "trunk"]


def gather_events(memory: Tensor, event_ids: Tensor) -> Tensor:
    if memory.ndim != 3 or event_ids.ndim != 3 or memory.shape[0] != event_ids.shape[0]:
        raise ValueError("expected memory [B,T,D] and event IDs [B,U,K]")
    if event_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("event IDs must be integer tensors")
    batch, steps, candidates = event_ids.shape
    indices = event_ids.clamp_min(0).long().reshape(batch, -1)
    values = memory.gather(1, indices[..., None].expand(-1, -1, memory.shape[-1]))
    values = values.reshape(batch, steps, candidates, memory.shape[-1])
    return values.masked_fill(event_ids.lt(0).unsqueeze(-1), 0)


class ProtocolModel(nn.Module, Generic[AnyReceiver]):
    def __init__(self, sender: CoarseSender, receiver: AnyReceiver) -> None:
        super().__init__()
        if sender.trunk.config.out_dims != receiver.sender_dimensions:
            raise ValueError("sender output and receiver communication widths differ")
        self.sender = sender
        self.receiver: AnyReceiver = receiver
        self.stage: TrainingStage | None = None

    def configure_stage(self, stage: TrainingStage) -> None:
        if stage not in ("alignment", "communication", "joint", "trunk"):
            raise ValueError(f"unknown training stage: {stage}")
        self.requires_grad_(False)
        self.zero_grad(set_to_none=True)
        if stage in ("alignment", "joint", "trunk"):
            self.receiver.scorers.requires_grad_(True)
        if stage in ("communication", "joint", "trunk"):
            self.receiver.channels.requires_grad_(True)
        if stage == "trunk":
            self.sender.requires_grad_(True)
            self.receiver.native.requires_grad_(True)
        self.stage = stage
        self.train(self.training)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.stage != "trunk":
            self.sender.eval()
        return self

    def forward(
        self,
        sender_ids: Tensor,
        receiver_ids: Tensor,
        candidate_event_ids: Tensor,
        frontier: Tensor,
        *,
        hard: bool,
    ) -> ReceiverReadout:
        memory = self.sender(sender_ids)
        candidates = gather_events(memory, candidate_event_ids)
        return self.receiver(
            receiver_ids, candidates, candidate_event_ids, frontier, hard=hard
        )
