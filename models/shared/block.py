import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.shared.kv_cache import KVCache
from models.shared.linear import ResidualLinear
from models.shared.rope import RotaryEmbedding
from models.shared.swiglu import SwiGLU, swiglu_hidden_dim


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        multiple_of: int,
        rope: RotaryEmbedding,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, n_heads={n_heads}")
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        if rope.head_dim != self.head_dim:
            raise ValueError(
                f"rope_head_dim={rope.head_dim} does not match head_dim={self.head_dim}"
            )

        self.ln1 = nn.RMSNorm(d_model, eps=norm_eps)
        self.ln2 = nn.RMSNorm(d_model, eps=norm_eps)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = ResidualLinear(d_model, d_model, bias=False)

        self.mlp = SwiGLU(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            multiple_of=multiple_of,
        )

        self.rope = rope

        self.p_drop = float(dropout)
        self.drop = nn.Dropout(self.p_drop) if self.p_drop > 0 else None

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        is_causal: bool,
        cache: KVCache | None,
    ) -> Tensor:
        if mask is not None and is_causal:
            raise ValueError("fold causality into mask, or pass is_causal=False")

        B, T, C = x.shape

        offset = 0
        if cache is not None:
            offset = cache.pos
            if offset and T > 1:
                raise ValueError(
                    "cached forward takes a full prefill or one token, not a chunk"
                )

        h = self.ln1(x)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, k = self.rope(q, k, offset)

        if cache is not None:
            k, v = cache.update(k, v)
            is_causal = is_causal and T > 1
        if mask is not None and mask.shape[-1] != k.shape[-2]:
            raise ValueError(
                f"mask_key_length={mask.shape[-1]} does not match "
                f"key_length={k.shape[-2]}"
            )

        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=is_causal,
            dropout_p=self.p_drop if self.training else 0.0,
        )

        a = a.transpose(1, 2).contiguous().view(B, T, C)
        a = self.out_proj(a)

        if self.drop is not None:
            a = self.drop(a)
        x = x + a

        f = self.mlp(self.ln2(x))

        if self.drop is not None:
            f = self.drop(f)
        x = x + f

        return x


def block_flops(d_model: int, mlp_ratio: float, multiple_of: int) -> int:
    hidden_dim = swiglu_hidden_dim(d_model, mlp_ratio, multiple_of)

    return 2 * (4 * d_model * d_model + 3 * d_model * hidden_dim)
