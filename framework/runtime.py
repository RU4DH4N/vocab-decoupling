import math
import random
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype:
    selected = "bf16" if precision == "auto" and device.type == "cuda" else precision
    if selected in ("auto", "fp32"):
        return torch.float32
    if selected != "bf16":
        raise ValueError(f"unknown precision={precision!r}")
    if device.type != "cuda":
        raise ValueError(
            f"precision={selected!r} requires CUDA, device={device.type!r}"
        )
    return torch.bfloat16


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[Any]:
    if autocast_dtype(device, precision) is torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def adamw(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    device: torch.device,
    undecayed: Iterable[torch.nn.Parameter],
) -> torch.optim.AdamW:
    exempt = {id(p) for p in undecayed}
    live = [p for p in parameters if p.requires_grad]
    groups = [
        {
            "params": [p for p in live if p.ndim >= 2 and id(p) not in exempt],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in live if p.ndim < 2 or id(p) in exempt],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        [group for group in groups if group["params"]],
        lr=lr,
        fused=device.type == "cuda",
    )


def device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_schedule(step: int, steps: int, warmup: int) -> float:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if step < 0:
        raise ValueError("step cannot be negative")
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
