import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    resamples: int
    seed: int


def bits_per_byte(nats: np.ndarray, utf8_bytes: np.ndarray) -> float:
    return float(nats.sum() / (math.log(2) * utf8_bytes.sum()))


def bootstrap_bits_per_byte(
    nats: np.ndarray,
    utf8_bytes: np.ndarray,
    resamples: int,
    seed: int,
    level: float,
) -> Interval:
    _validate(nats, utf8_bytes, resamples, level)
    rng = np.random.default_rng(seed)
    count = len(nats)
    draws = np.empty(resamples)
    for index in range(resamples):
        pick = rng.integers(0, count, size=count)
        draws[index] = bits_per_byte(nats[pick], utf8_bytes[pick])
    low, high = _percentiles(draws, level)
    return Interval(bits_per_byte(nats, utf8_bytes), low, high, resamples, seed)


def paired_bootstrap_difference(
    left_nats: np.ndarray,
    right_nats: np.ndarray,
    utf8_bytes: np.ndarray,
    resamples: int,
    seed: int,
    level: float,
) -> Interval:
    _validate(left_nats, utf8_bytes, resamples, level)
    if right_nats.shape != left_nats.shape:
        raise ValueError("paired comparison needs one score per document per system")
    rng = np.random.default_rng(seed)
    count = len(utf8_bytes)
    draws = np.empty(resamples)
    for index in range(resamples):
        pick = rng.integers(0, count, size=count)
        draws[index] = bits_per_byte(left_nats[pick], utf8_bytes[pick]) - bits_per_byte(
            right_nats[pick], utf8_bytes[pick]
        )
    low, high = _percentiles(draws, level)
    estimate = bits_per_byte(left_nats, utf8_bytes) - bits_per_byte(
        right_nats, utf8_bytes
    )
    return Interval(estimate, low, high, resamples, seed)


def _validate(
    nats: np.ndarray, utf8_bytes: np.ndarray, resamples: int, level: float
) -> None:
    if nats.ndim != 1 or utf8_bytes.shape != nats.shape:
        raise ValueError("nats and utf8_bytes must be aligned one-dimensional arrays")
    if len(nats) == 0:
        raise ValueError("at least one document is required")
    if (utf8_bytes <= 0).any():
        raise ValueError("every document must contain at least one scored byte")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, resamples={resamples}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), level={level}")


def _percentiles(draws: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return (
        float(np.quantile(draws, tail)),
        float(np.quantile(draws, 1 - tail)),
    )


def hierarchical_paired_bootstrap(
    left_by_seed: list[np.ndarray],
    right_by_seed: list[np.ndarray],
    bytes_by_seed: list[np.ndarray],
    resamples: int,
    seed: int,
    level: float,
) -> Interval:
    if not left_by_seed or not (
        len(left_by_seed) == len(right_by_seed) == len(bytes_by_seed)
    ):
        raise ValueError("one paired array triple is required per seed")
    for left, right, utf8_bytes in zip(
        left_by_seed, right_by_seed, bytes_by_seed, strict=True
    ):
        _validate(left, utf8_bytes, resamples, level)
        if right.shape != left.shape:
            raise ValueError(
                "paired comparison needs one score per document per system"
            )
    rng = np.random.default_rng(seed)
    seeds = len(left_by_seed)

    def pooled(chosen: list[int], picks: list[np.ndarray]) -> float:
        left = np.concatenate([left_by_seed[k][pick] for k, pick in zip(chosen, picks)])
        right = np.concatenate(
            [right_by_seed[k][pick] for k, pick in zip(chosen, picks)]
        )
        weight = np.concatenate(
            [bytes_by_seed[k][pick] for k, pick in zip(chosen, picks)]
        )
        return bits_per_byte(left, weight) - bits_per_byte(right, weight)

    draws = np.empty(resamples)
    for index in range(resamples):
        chosen = rng.integers(0, seeds, size=seeds).tolist()
        picks = [
            rng.integers(0, len(bytes_by_seed[k]), size=len(bytes_by_seed[k]))
            for k in chosen
        ]
        draws[index] = pooled(chosen, picks)
    low, high = _percentiles(draws, level)
    estimate = pooled(
        list(range(seeds)), [np.arange(len(utf8_bytes)) for utf8_bytes in bytes_by_seed]
    )
    return Interval(estimate, low, high, resamples, seed)
