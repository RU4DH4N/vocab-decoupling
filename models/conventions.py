HEAD_DIMENSIONS = 64
WIDTH_PER_LAYER = 64
MLP_RATIO = 4.0
MULTIPLE_OF = 64
NORM_EPS = 1e-5
ROPE_THETA = 10_000.0
INITIALISER_RANGE = 0.02
DROPOUT = 0.0
TIE_WORD_EMBEDDINGS = True
CODE_DIMENSIONS = 512
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0


def layers_for(width: int) -> int:
    return max(2, round(width / WIDTH_PER_LAYER))


def heads_for(width: int) -> int:
    return max(1, width // HEAD_DIMENSIONS)


def shapes() -> list[tuple[int, int]]:
    widths = [16, 32, 48, *range(64, 4097, HEAD_DIMENSIONS)]
    return [
        (width, layers)
        for width in widths
        for layers in range(max(2, layers_for(width) - 2), layers_for(width) + 3)
    ]


def transformer(width: int, layers: int, max_seq_len: int) -> dict:
    return {
        "d_model": width,
        "n_layers": layers,
        "n_heads": heads_for(width),
        "mlp_ratio": MLP_RATIO,
        "dropout": DROPOUT,
        "multiple_of": MULTIPLE_OF,
        "max_seq_len": max_seq_len,
        "rope_theta": ROPE_THETA,
        "norm_eps": NORM_EPS,
        "initialiser_range": INITIALISER_RANGE,
    }
