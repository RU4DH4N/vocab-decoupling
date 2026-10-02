from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.encoding.code import VOCABULARY_TABLE_BUDGET_BYTES, Code
from models.shared.kv_cache import KVCache
from models.shared.numerics import normalise_table
from models.shared.trunk import Trunk, TrunkConfig


class CoarseSender(nn.Module):
    trunk_table: Tensor

    def __init__(
        self,
        in_dims: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float,
        d_meaning: int,
        codebook_seed: int,
        dropout: float,
        multiple_of: int,
        max_seq_len: int,
        rope_theta: float,
        norm_eps: float,
        initialiser_range: float,
    ) -> None:
        super().__init__()
        self.trunk_code = Code(dims=in_dims, seed=codebook_seed)
        self.trunk = Trunk(
            TrunkConfig(
                in_dims=in_dims,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                out_dims=d_meaning,
                dropout=dropout,
                multiple_of=multiple_of,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                initialiser_range=initialiser_range,
            )
        )
        self.meaning_norm = nn.RMSNorm(d_meaning, eps=norm_eps)
        self.norm_eps = norm_eps
        self.register_buffer("trunk_table", torch.empty(0, in_dims), persistent=False)

    @torch.no_grad()
    def encode_trunk_vocab(self, vocab: Sequence[str | bytes]) -> tuple[Tensor, Tensor]:
        consumer = self.trunk.input_reference
        table, dead = normalise_table(
            self.trunk_code.vocabulary_table(
                vocab, budget=VOCABULARY_TABLE_BUDGET_BYTES
            ),
            eps=self.norm_eps,
        )
        return table.to(consumer), dead.to(device=consumer.device)

    @torch.no_grad()
    def build_table(self, vocab: Sequence[str | bytes]) -> None:
        table, dead = self.encode_trunk_vocab(vocab)
        if dead.any():
            rows = dead.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"input inventory contains {len(rows)} directionless entries; "
                f"first_indices={rows[:10]}"
            )
        self.trunk_table = table

    def state(
        self, embeddings: Tensor, caches: Sequence[KVCache] | None = None
    ) -> Tensor:
        return self.meaning_norm(self.trunk(embeddings, caches))

    def new_caches(self, max_seq_len: int) -> list[KVCache]:
        return self.trunk.new_caches(max_seq_len)

    def forward(self, ids: Tensor, caches: Sequence[KVCache] | None = None) -> Tensor:
        if self.trunk_table.numel() == 0:
            raise RuntimeError("no input table; call build_table(vocab) first")
        return self.state(self.trunk_table[ids], caches)
