import torch
import torch.nn as nn
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)

    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim={head_dim} must be even for rotary embedding")

        self.head_dim = head_dim
        self.configured_max_seq_len = max_seq_len
        self.theta = float(theta)

        cos, sin = self._tables(max_seq_len, torch.device("cpu"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _tables(
        self,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.theta ** (exponent / self.head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        offset: int,
    ) -> tuple[Tensor, Tensor]:
        if offset < 0:
            raise ValueError(f"offset={offset} must be non-negative")
        if q.shape[-1] != self.head_dim:
            raise ValueError(
                f"q_head_dim={q.shape[-1]} does not match head_dim={self.head_dim}"
            )
        if k.shape[-1] != self.head_dim:
            raise ValueError(
                f"k_head_dim={k.shape[-1]} does not match head_dim={self.head_dim}"
            )
        if q.shape[-2] != k.shape[-2]:
            raise ValueError(
                f"q_sequence_length={q.shape[-2]} does not match "
                f"k_sequence_length={k.shape[-2]}"
            )

        end = offset + q.shape[-2]

        if end > self.cos.shape[0]:
            grown = max(end, 2 * self.cos.shape[0])
            self.cos, self.sin = self._tables(grown, q.device)
        elif self.cos.device != q.device:
            self.cos = self.cos.to(q.device)
            self.sin = self.sin.to(q.device)

        cos = self.cos[offset:end]
        sin = self.sin[offset:end]
        rotated_q = q.float() * cos + rotate_half(q.float()) * sin
        rotated_k = k.float() * cos + rotate_half(k.float()) * sin

        return rotated_q.to(q.dtype), rotated_k.to(k.dtype)
