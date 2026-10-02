import torch
from torch import Tensor


class KVCache:
    def __init__(self, max_seq_len: int) -> None:
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
            raise TypeError(
                f"max_seq_len must be an integer, "
                f"max_seq_len_type={type(max_seq_len).__name__}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, max_seq_len={max_seq_len}")
        self.max_seq_len = max_seq_len
        self.k: Tensor | None = None
        self.v: Tensor | None = None
        self.pos = 0

    @torch.no_grad()
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if k.ndim != 4:
            raise ValueError(f"expected k shape=(B, H, T, D), k_shape={tuple(k.shape)}")
        if k.shape != v.shape:
            raise ValueError(
                f"k_shape={tuple(k.shape)} does not match v_shape={tuple(v.shape)}"
            )
        if k.dtype != v.dtype or k.device != v.device:
            raise ValueError(
                f"k_dtype={k.dtype}, k_device={k.device}, "
                f"v_dtype={v.dtype}, v_device={v.device}"
            )

        B, H, T, D = k.shape
        if T == 0:
            raise ValueError("update requires at least one position, T=0")
        if self.pos + T > self.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.max_seq_len}, required_positions={self.pos + T}"
            )

        if self.k is None:
            self.k = torch.zeros(
                B, H, self.max_seq_len, D, dtype=k.dtype, device=k.device
            )
            self.v = torch.zeros_like(self.k)
        elif self.v is None:
            raise RuntimeError("cache has keys without values")
        elif (B, H, D) != (self.k.shape[0], self.k.shape[1], self.k.shape[3]):
            allocated = (self.k.shape[0], self.k.shape[1], self.k.shape[3])
            raise ValueError(
                f"cache_shape={allocated} does not match input_shape={(B, H, D)}; "
                "call free() before changing shape"
            )
        elif k.dtype != self.k.dtype or k.device != self.k.device:
            raise ValueError(
                f"cache_dtype={self.k.dtype}, cache_device={self.k.device}, "
                f"input_dtype={k.dtype}, input_device={k.device}"
            )

        cache_k = self.k
        cache_v = self.v

        cache_k[:, :, self.pos : self.pos + T] = k
        cache_v[:, :, self.pos : self.pos + T] = v
        self.pos += T

        return cache_k[:, :, : self.pos], cache_v[:, :, : self.pos]

    def reset(self) -> None:
        self.pos = 0

    def free(self) -> None:
        self.k = None
        self.v = None
        self.pos = 0

    @property
    def remaining(self) -> int:
        return self.max_seq_len - self.pos
