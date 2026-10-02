import os

import torch

from execution.streams import generate_streams


def seeded(offset: int, indices: list[int]) -> list[tuple[int, dict]]:
    records = []
    for index in indices:
        generator = torch.Generator().manual_seed(offset + index)
        records.append(
            (
                index,
                {
                    "values": torch.rand(4, generator=generator).tolist(),
                    "process": os.getpid(),
                },
            )
        )
    return records


def test_concurrent_streams_return_the_serial_samples_in_order():
    serial, _ = generate_streams(seeded, (7,), 11, 3, 1)
    parallel, timed = generate_streams(seeded, (7,), 11, 3, 2)
    assert [r["values"] for r in parallel] == [r["values"] for r in serial]
    assert [r["values"] for r in timed] == [r["values"] for r in serial[:3]]
    assert {r["process"] for r in timed} == {os.getpid()}
    assert len({r["process"] for r in parallel[3:]} - {os.getpid()}) == 2


def test_latency_samples_never_exceed_the_prompts():
    records, timed = generate_streams(seeded, (0,), 2, 5, 4)
    assert len(records) == len(timed) == 2
