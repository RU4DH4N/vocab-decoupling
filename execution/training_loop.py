import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from framework.resume import capture, current, restore, save
from framework.runtime import cosine_schedule, device_synchronize

Step = Callable[[int], tuple[Tensor, dict[str, float], str]]


@dataclass(frozen=True)
class Job:
    label: str
    steps: int
    learning_rate: float
    warmup_steps: int
    grad_clip: float
    interval: int
    resume: Path
    fingerprint: str
    modules: dict[str, nn.Module]
    parameters: Sequence[nn.Parameter]
    optimizer: torch.optim.Optimizer
    rng: np.random.Generator
    device: torch.device
    counters: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    completed: int
    counters: dict[str, float]
    seconds: float


def train_steps(job: Job, step: Step) -> Outcome:
    completed, elapsed = 0, 0.0
    totals = {name: 0.0 for name in job.counters}
    state = current(job.resume, job.fingerprint)
    if state is not None:
        completed = restore(state, job.modules, job.optimizer, job.rng)
        totals = {name: state["extra"][name] for name in job.counters}
        elapsed = state["extra"]["seconds"]
    if not 0 <= completed <= job.steps:
        raise ValueError(f"{job.label} resume step outside declared schedule")
    device_synchronize(job.device)
    started = time.perf_counter()
    for index in range(completed, job.steps):
        job.optimizer.zero_grad(set_to_none=True)
        for group in job.optimizer.param_groups:
            group["lr"] = job.learning_rate * cosine_schedule(
                index, job.steps, job.warmup_steps
            )
        loss, added, note = step(index)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{job.label} loss is nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            job.parameters, job.grad_clip, error_if_nonfinite=True
        )
        job.optimizer.step()
        completed = index + 1
        for name, value in added.items():
            totals[name] += value
        print(
            f"{job.label} step={completed}/{job.steps} ce={loss.item():.6f}{note}",
            flush=True,
        )
        if completed % job.interval == 0 or completed == job.steps:
            device_synchronize(job.device)
            seconds = elapsed + time.perf_counter() - started
            save(
                job.resume,
                capture(
                    completed,
                    job.fingerprint,
                    job.modules,
                    job.optimizer,
                    job.rng,
                    {**totals, "seconds": seconds},
                ),
            )
    device_synchronize(job.device)
    return Outcome(completed, totals, elapsed + time.perf_counter() - started)
