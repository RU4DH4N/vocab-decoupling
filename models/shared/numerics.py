import torch
import torch.nn.functional as F
from torch import Tensor


def normalise(
    x: Tensor,
    eps: float,
) -> Tensor:
    if not x.is_floating_point():
        raise TypeError("normalise() expects a floating-point tensor")

    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x

    inv_norm = torch.rsqrt(work.square().sum(dim=-1, keepdim=True) + eps)

    return (work * inv_norm).to(dtype=x.dtype)


def masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore: int,
) -> Tensor:
    if targets.dtype != torch.long:
        targets = targets.long()

    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore,
        reduction="sum",
    )

    count = targets.ne(ignore).sum().clamp_min(1)

    return loss / count.to(dtype=loss.dtype)


TABLE_ROWS = 1 << 16


def normalise_table(
    codes: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    if codes.ndim != 2:
        raise ValueError(
            f"codes must have shape (vocabulary, code_dims), got {tuple(codes.shape)}"
        )

    table = codes.clone() if codes.requires_grad else codes
    dead = torch.empty(codes.shape[0], dtype=torch.bool, device=codes.device)
    for start in range(0, codes.shape[0], TABLE_ROWS):
        block = table[start : start + TABLE_ROWS]
        dead[start : start + TABLE_ROWS] = block.square().sum(dim=-1) == 0
        block.copy_(normalise(block, eps=eps))
    return table, dead
