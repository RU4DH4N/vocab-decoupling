import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.shared.linear import ResidualLinear


class SwiGLU(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        mlp_ratio: float,
        multiple_of: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = swiglu_hidden_dim(d_model, mlp_ratio, multiple_of)

        self.gate_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, self.hidden_dim, bias=False)
        self.down_proj = ResidualLinear(self.hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)

        return self.down_proj(hidden)


def swiglu_hidden_dim(
    d_model: int,
    mlp_ratio: float,
    multiple_of: int,
) -> int:
    for name, value in {
        "d_model": d_model,
        "mlp_ratio": mlp_ratio,
        "multiple_of": multiple_of,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, {name}={value}")

    target = d_model * mlp_ratio * 2 / 3
    return multiple_of * math.ceil(target / multiple_of)
