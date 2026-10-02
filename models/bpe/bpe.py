from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from models.shared.kv_cache import KVCache
from models.shared.trunk import Trunk, TrunkConfig


class BPEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        multiple_of: int,
        max_seq_len: int,
        rope_theta: float,
        norm_eps: float,
        initialiser_range: float,
        tie_word_embeddings: bool,
        boundaries: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.boundaries = None if boundaries is None else tuple(boundaries)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.trunk = Trunk(
            TrunkConfig(
                in_dims=d_model,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                out_dims=d_model,
                dropout=dropout,
                multiple_of=multiple_of,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                initialiser_range=initialiser_range,
            )
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initialiser_range)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=initialiser_range)

    def forward(
        self,
        ids: Tensor,
        *,
        caches: Sequence[KVCache] | None = None,
        layer_transform: Callable[[int, Tensor], Tensor] | None = None,
    ) -> Tensor:
        return self.lm_head(self.state(ids, caches, layer_transform=layer_transform))

    def state(
        self,
        ids: Tensor,
        caches: Sequence[KVCache] | None = None,
        *,
        layer_transform: Callable[[int, Tensor], Tensor] | None = None,
    ) -> Tensor:
        segments = None
        if self.boundaries is not None and (not caches or caches[0].pos == 0):
            segments = torch.isin(ids, ids.new_tensor(self.boundaries)).cumsum(-1)
        return self.trunk(
            self.embedding(ids),
            caches,
            layer_transform=layer_transform,
            segments=segments,
        )

    def new_caches(self, max_seq_len: int) -> list[KVCache]:
        return self.trunk.new_caches(max_seq_len)

    def loss(self, ids: Tensor, targets: Tensor) -> Tensor:
        logits = self(ids)
        return nn.functional.cross_entropy(
            logits.flatten(0, -2),
            targets.flatten(),
        )

    def backward_chunked_loss(
        self,
        states: Tensor,
        targets: Tensor,
        chunk_rows: int,
        weight: float,
    ) -> Tensor:
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, chunk_rows={chunk_rows}")
        if not 0 < weight <= 1:
            raise ValueError(f"weight must be in (0, 1], weight={weight}")
        if states.shape[:-1] != targets.shape:
            raise ValueError(
                f"state prefix shape {states.shape[:-1]} does not match "
                f"targets shape {targets.shape}"
            )
        if not states.requires_grad:
            raise ValueError("states must require gradients for chunked backward")

        detached_states = states.detach().requires_grad_(True)
        flat_states = detached_states.reshape(-1, detached_states.shape[-1])
        flat_targets = targets.reshape(-1)
        denominator = flat_targets.ne(-100).sum().clamp_min(1).to(torch.float32)
        detached_sum = torch.zeros((), dtype=torch.float32, device=states.device)
        for start in range(0, flat_states.shape[0], chunk_rows):
            stop = min(start + chunk_rows, flat_states.shape[0])
            logits = self.lm_head(flat_states[start:stop])
            chunk_sum = nn.functional.cross_entropy(
                logits,
                flat_targets[start:stop],
                reduction="sum",
            )
            detached_sum = detached_sum + chunk_sum.detach()
            (chunk_sum * (weight / denominator)).backward()
        if detached_states.grad is None:
            raise RuntimeError("vocabulary chunks did not produce state gradients")
        states.backward(detached_states.grad)
        return detached_sum / denominator
