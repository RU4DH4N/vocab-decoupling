import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from data.communication import ProtocolBatch
from framework.resume import capture, configuration_fingerprint, restore, save
from framework.runtime import autocast_context, cosine_schedule
from models.protocol.correspondence import alignment_nll
from models.protocol.model import ProtocolModel, TrainingStage
from models.shared.numerics import masked_cross_entropy


@dataclass(frozen=True)
class StageConfig:
    name: TrainingStage
    steps: int
    learning_rate: float
    weight_decay: float
    alignment_weight: float
    grad_clip: float
    warmup_steps: int
    precision: str

    def __post_init__(self) -> None:
        if self.name not in ("alignment", "communication", "joint", "trunk"):
            raise ValueError(f"unknown stage: {self.name}")
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps <= 0
        ):
            raise ValueError("steps must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or not 0 <= self.warmup_steps < self.steps
        ):
            raise ValueError("warmup_steps must be in [0, steps)")
        for name in ("learning_rate", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "alignment_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.name == "alignment" and self.alignment_weight <= 0:
            raise ValueError("alignment bootstrap requires positive alignment_weight")
        if self.name == "communication" and self.alignment_weight != 0:
            raise ValueError(
                "communication stage freezes the scorer; alignment_weight must be zero"
            )
        if self.precision not in ("fp32", "bf16", "auto"):
            raise ValueError("precision must be fp32, bf16 or auto")


@dataclass(frozen=True)
class StageLoss:
    total: Tensor
    language: Tensor
    alignment: Tensor
    token_count: Tensor
    alignment_count: Tensor


class StageTrainer:
    def __init__(
        self,
        model: ProtocolModel,
        config: StageConfig,
        identity: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        if not identity:
            raise ValueError("an explicit experiment identity is required")
        self.model, self.config, self.rng = model, config, rng
        model.configure_stage(config.name)
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            self.parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.fingerprint = configuration_fingerprint(
            {"stage": asdict(config), "experiment": identity}
        )
        self.completed = 0

    def objective(self, batch: ProtocolBatch) -> StageLoss:
        if (
            batch.targets.shape != batch.receiver_ids.shape
            or batch.alignment_targets.shape != batch.targets.shape
        ):
            raise ValueError("token and alignment targets must match receiver IDs")
        result = self.model(
            batch.sender_ids,
            batch.receiver_ids,
            batch.candidate_event_ids,
            batch.frontier,
            hard=False,
        )
        live = batch.targets.ne(-100)
        alignment_live = live & batch.alignment_targets.ge(0)
        language = masked_cross_entropy(
            result.logits.float().reshape(-1, result.logits.shape[-1]),
            batch.targets.reshape(-1),
            ignore=-100,
        )
        losses = [
            alignment_nll(
                p, batch.candidate_event_ids, batch.alignment_targets, alignment_live
            )
            for p in result.correspondence
        ]
        alignment = torch.stack(losses).sum() / (
            alignment_live.sum().clamp_min(1) * len(losses)
        )
        total = language if self.config.name != "alignment" else language * 0
        if self.config.alignment_weight:
            total = total + self.config.alignment_weight * alignment
        return StageLoss(total, language, alignment, live.sum(), alignment_live.sum())

    def step(self, batch: ProtocolBatch) -> StageLoss:
        if self.model.stage != self.config.name:
            raise RuntimeError("model stage changed; construct a fresh stage trainer")
        if self.completed >= self.config.steps:
            raise RuntimeError("stage is already complete")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.config.learning_rate * cosine_schedule(
            self.completed, self.config.steps, self.config.warmup_steps
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        with autocast_context(batch.sender_ids.device, self.config.precision):
            losses = self.objective(batch)
        if not torch.isfinite(losses.total.detach()):
            raise FloatingPointError(
                "non-finite stage loss; optimizer step was not taken"
            )
        if not losses.token_count or (
            self.config.name == "alignment" and not losses.alignment_count
        ):
            raise ValueError(
                "training batch contains no live supervision for this stage"
            )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.parameters, self.config.grad_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.completed += 1
        return StageLoss(
            *(
                value.detach()
                for value in (
                    losses.total,
                    losses.language,
                    losses.alignment,
                    losses.token_count,
                    losses.alignment_count,
                )
            )
        )

    def save_resume(self, path: Path) -> None:
        save(
            path,
            capture(
                self.completed,
                self.fingerprint,
                {"model": self.model},
                self.optimizer,
                self.rng,
                {"stage": asdict(self.config)},
            ),
        )

    def resume(self, state: dict) -> None:
        step = state["step"]
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not 0 <= step <= self.config.steps
        ):
            raise ValueError("resume step falls outside the declared stage")
        self.completed = restore(state, {"model": self.model}, self.optimizer, self.rng)
