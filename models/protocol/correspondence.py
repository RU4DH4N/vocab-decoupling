from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.shared.validate import require_positive


class CompatibilityScorer(nn.Module):
    def __init__(
        self,
        sender_dimensions: int,
        receiver_dimensions: int,
        score_dimensions: int,
    ) -> None:
        super().__init__()
        require_positive(
            sender_dimensions=sender_dimensions,
            receiver_dimensions=receiver_dimensions,
            score_dimensions=score_dimensions,
        )
        self.sender = nn.Linear(sender_dimensions, score_dimensions, bias=False)
        self.receiver = nn.Linear(receiver_dimensions, score_dimensions, bias=False)
        self.scale = score_dimensions**-0.5

    def forward(self, receiver: Tensor, candidates: Tensor) -> Tensor:
        if receiver.ndim < 2 or candidates.ndim != receiver.ndim + 1:
            raise ValueError("candidates need one extra candidate axis before width")
        if receiver.shape[:-1] != candidates.shape[:-2]:
            raise ValueError("receiver and candidates must share leading dimensions")
        if (
            receiver.shape[-1] != self.receiver.in_features
            or candidates.shape[-1] != self.sender.in_features
        ):
            raise ValueError("receiver or sender width differs from scorer dimensions")
        return (self.receiver(receiver).unsqueeze(-2) * self.sender(candidates)).sum(
            -1
        ) * self.scale


def causal_event_mask(event_ids: Tensor, frontier: Tensor, window: int) -> Tensor:
    if window <= 0:
        raise ValueError("window must be positive")
    if event_ids.dtype not in (torch.int32, torch.int64) or frontier.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("event IDs and frontier must be integer tensors")
    if event_ids.ndim < 2 or event_ids.shape[:-1] != frontier.shape:
        raise ValueError("frontier shape must match the candidate leading dimensions")
    return (
        (event_ids >= 0)
        & (event_ids <= frontier.unsqueeze(-1))
        & (event_ids > frontier.unsqueeze(-1) - window)
    )


def correspondence_log_probs(scores: Tensor, allowed: Tensor) -> Tensor:
    if scores.ndim < 2 or scores.shape[-1] == 0:
        raise ValueError("scores must have a nonempty candidate axis")
    if allowed.dtype != torch.bool or allowed.shape != scores.shape:
        raise ValueError("allowed must be a boolean tensor with the scores' shape")
    live = allowed.any(-1, keepdim=True)
    masked = scores.float().masked_fill(~allowed, -torch.inf)
    safe = torch.where(live, masked, torch.zeros_like(masked))
    return safe.log_softmax(-1).masked_fill(~allowed, -torch.inf)


def select_message(candidates: Tensor, log_probs: Tensor, *, hard: bool) -> Tensor:
    if candidates.shape[:-1] != log_probs.shape:
        raise ValueError("candidate axes must match correspondence probabilities")
    if hard:
        index = log_probs.argmax(-1)
        selected = candidates.gather(
            -2, index[..., None, None].expand(*index.shape, 1, candidates.shape[-1])
        ).squeeze(-2)
        available = log_probs.isfinite().any(-1, keepdim=True)
        return torch.where(available, selected, torch.zeros_like(selected))
    return (log_probs.exp().to(candidates.dtype).unsqueeze(-1) * candidates).sum(-2)


def alignment_nll(
    log_probs: Tensor, event_ids: Tensor, targets: Tensor, live: Tensor
) -> Tensor:
    if event_ids.shape != log_probs.shape or targets.shape != log_probs.shape[:-1]:
        raise ValueError("event IDs and target shapes must match log probabilities")
    if live.dtype != torch.bool or live.shape != targets.shape:
        raise ValueError("live must be a boolean tensor with the targets' shape")
    positive = (
        event_ids.eq(targets.unsqueeze(-1)) & event_ids.ge(0) & log_probs.isfinite()
    )
    present = positive.any(-1)
    terms = log_probs.masked_fill(~positive, -torch.inf)
    terms = torch.where(present.unsqueeze(-1), terms, torch.zeros_like(terms))
    loss = -terms.logsumexp(-1)
    loss = torch.where(present, loss, torch.full_like(loss, torch.inf))
    return torch.where(live, loss, torch.zeros_like(loss))


@dataclass(frozen=True)
class AlignmentDiagnostics:
    valid: Tensor
    correct_mass: Tensor
    hard_correct: Tensor
    expected_event_error: Tensor


def alignment_diagnostics(
    log_probs: Tensor, event_ids: Tensor, targets: Tensor
) -> AlignmentDiagnostics:
    if event_ids.shape != log_probs.shape or targets.shape != log_probs.shape[:-1]:
        raise ValueError("event IDs and target shapes must match log probabilities")
    weights = log_probs.exp()
    positive = event_ids.eq(targets.unsqueeze(-1)) & event_ids.ge(0)
    selected_ids = event_ids.gather(-1, log_probs.argmax(-1, keepdim=True)).squeeze(-1)
    valid = log_probs.isfinite().any(-1) & targets.ge(0)
    error = (event_ids - targets.unsqueeze(-1)).abs().to(weights.dtype)
    return AlignmentDiagnostics(
        valid=valid,
        correct_mass=torch.where(valid, (weights * positive).sum(-1), 0),
        hard_correct=valid & selected_ids.eq(targets),
        expected_event_error=torch.where(valid, (weights * error).sum(-1), 0),
    )
