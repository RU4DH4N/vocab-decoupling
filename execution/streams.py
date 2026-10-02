import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from framework.checkpoints import file_sha256

Shard = list[tuple[int, dict]]


def _shard(
    worker: Callable[..., Shard], arguments: tuple, indices: list[int], streams: int
) -> Shard:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // streams))
    return worker(*arguments, indices)


def generate_streams(
    worker: Callable[..., Shard],
    arguments: tuple,
    count: int,
    latency_samples: int,
    streams: int,
) -> tuple[list[dict], list[dict]]:
    timed = list(range(min(latency_samples, count)))
    rest = list(range(len(timed), count))
    records = dict(worker(*arguments, timed))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shards = [rest[i::streams] for i in range(streams) if rest[i::streams]]
    if streams == 1:
        records.update(worker(*arguments, rest))
    elif shards:
        with ProcessPoolExecutor(len(shards), mp_context=get_context("spawn")) as pool:
            for shard in pool.map(
                _shard,
                [worker] * len(shards),
                [arguments] * len(shards),
                shards,
                [streams] * len(shards),
            ):
                records.update(shard)
    return [records[i] for i in range(count)], [records[i] for i in timed]


def prompt_count(prompts: Path) -> int:
    return len(json.loads(prompts.read_text()))


def device_streams(options: dict, device: torch.device) -> int:
    return options["streams"] if device.type == "cuda" else 1


def summary(
    records: list[dict], timed: list[dict], prompts: Path, options: dict
) -> dict:
    latencies = [row["seconds"] / max(1, row["bytes"]) for row in timed]
    return {
        "samples": records,
        "prompts_sha256": file_sha256(prompts),
        "censored_samples": sum(r["censored"] for r in records),
        "invalid_utf8_samples": sum(not r["valid_utf8"] for r in records),
        "latency_seconds_per_byte": {
            "p50": float(np.quantile(latencies, 0.5)),
            "p95": float(np.quantile(latencies, 0.95)),
        },
        "bytes_per_second": sum(r["bytes"] for r in timed)
        / sum(r["seconds"] for r in timed),
        "latency_definition": "the first latency_samples prompts generated alone on the device; end-to-end seconds including prompt prefill divided by emitted bytes; model loading excluded",
        "decoding": options,
    }
