import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from models.shared.block import Block
from models.shared.kv_cache import KVCache
from models.shared.linear import ResidualLinear
from models.shared.rope import RotaryEmbedding


@dataclass(frozen=True)
class TrunkConfig:
    in_dims: int
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    out_dims: int
    initialiser_range: float
    dropout: float
    multiple_of: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float

    def __post_init__(self) -> None:
        positive = {
            "in_dims": self.in_dims,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "out_dims": self.out_dims,
            "multiple_of": self.multiple_of,
            "max_seq_len": self.max_seq_len,
            "mlp_ratio": self.mlp_ratio,
            "initialiser_range": self.initialiser_range,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, {name}={value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), dropout={self.dropout}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        head_dim = self.d_model // self.n_heads
        if head_dim % 2:
            raise ValueError(
                f"RoPE requires an even head dimension, head_dim={head_dim}, "
                f"d_model={self.d_model}, n_heads={self.n_heads}"
            )


def local_mask(segments: Tensor) -> Tensor:
    steps = segments.shape[-1]
    causal = torch.ones(steps, steps, dtype=torch.bool, device=segments.device).tril()
    same = segments[:, :, None].eq(segments[:, None, :])
    return (causal & same)[:, None]


class Trunk(nn.Module):
    def __init__(self, config: TrunkConfig) -> None:
        super().__init__()
        self.config = config

        self.in_proj = (
            nn.Identity()
            if config.in_dims == config.d_model
            else nn.Linear(config.in_dims, config.d_model, bias=False)
        )

        rope = RotaryEmbedding(
            config.d_model // config.n_heads,
            config.max_seq_len,
            config.rope_theta,
        )

        self.blocks = nn.ModuleList(
            Block(
                d_model=config.d_model,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                multiple_of=config.multiple_of,
                rope=rope,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        )

        self.ln_f = nn.RMSNorm(config.d_model, eps=config.norm_eps)

        self.out = (
            nn.Identity()
            if config.out_dims == config.d_model
            else nn.Linear(config.d_model, config.out_dims, bias=False)
        )

        residual_std = config.initialiser_range / math.sqrt(2 * config.n_layers)
        n_flagged = sum(
            1 for module in self.modules() if isinstance(module, ResidualLinear)
        )
        if n_flagged != 2 * config.n_layers:
            raise RuntimeError(
                f"expected {2 * config.n_layers} residual-output projections, "
                f"n_flagged={n_flagged}"
            )
        for module in self.modules():
            self._init_weights(module, config.initialiser_range, residual_std)

    @property
    def input_reference(self) -> Tensor:
        if isinstance(self.in_proj, nn.Linear):
            return self.in_proj.weight
        return cast(Block, self.blocks[0]).qkv.weight

    @staticmethod
    def _init_weights(
        module: nn.Module,
        std: float,
        residual_std: float,
    ) -> None:
        if isinstance(module, nn.Linear):
            if isinstance(module, ResidualLinear):
                nn.init.normal_(module.weight, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def new_caches(self, max_seq_len: int) -> list[KVCache]:
        return [KVCache(max_seq_len) for _ in self.blocks]

    def forward(
        self,
        x: Tensor,
        caches: Sequence[KVCache] | None = None,
        *,
        layer_transform: Callable[[int, Tensor], Tensor] | None = None,
        segments: Tensor | None = None,
    ) -> Tensor:
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} caches, got {len(caches)}")
        offset = 0 if caches is None else caches[0].pos
        if offset + x.shape[-2] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {offset + x.shape[-2]} exceeds configured "
                f"maximum {self.config.max_seq_len}"
            )
        h = self.in_proj(x)
        mask = None if segments is None else local_mask(segments)
        for index, block in enumerate(self.blocks):
            h = block(
                h,
                mask=mask,
                is_causal=mask is None,
                cache=None if caches is None else caches[index],
            )
            if layer_transform is not None:
                h = layer_transform(index, h)

        return self.out(self.ln_f(h))
