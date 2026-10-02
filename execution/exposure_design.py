import json
from pathlib import Path

import numpy as np

from data.protocol_corpus import EventCorpus


def plan_exposure(config: dict, corpus: EventCorpus) -> dict:
    lengths = np.array([len(unit.encode()) for unit in corpus.vocab], dtype=np.int64)
    totals: dict[tuple[int, int], int] = {}

    def phase(seed: int, steps: int) -> dict:
        key = (seed, steps)
        if key not in totals:
            rng = np.random.default_rng(seed)
            total = 0
            for _ in range(steps):
                rows = corpus.windows(
                    "train", config["events"], config["batch_size"], rng
                )
                total += int(lengths[rows[:, 1:].numpy()].sum())
            totals[key] = total
        return {"steps": steps, "data_seed": seed, "target_bytes": totals[key]}

    jobs = {}
    for variant in config["receivers"]:
        jobs[f"{variant}/native"] = phase(
            config["data_seed"], config["native_training"]["steps"]
        )
        stages = config["stages" if variant == "primary" else "replacement_stages"]
        for index, stage in enumerate(stages):
            jobs[f"{variant}/stage-{index}"] = {
                **phase(config["data_seed"] + index + 1, stage["steps"]),
                "lexical_objective": stage["name"] != "alignment",
            }
    primary = [value for name, value in jobs.items() if name.startswith("primary/")]
    jobs["baseline/model"] = {
        "steps": sum(value["steps"] for value in primary),
        "target_bytes": sum(value["target_bytes"] for value in primary),
    }
    jobs["planning/planner"] = phase(
        config["planner"]["data_seed"], config["planner"]["steps"]
    )
    return {
        "matching": "processed target UTF-8 bytes, excluding context prefix, including repeat exposure",
        "not_matched": [
            "FLOPs",
            "parameters",
            "lexical supervision during alignment bootstrap",
        ],
        "extra_work": [
            "planner",
            "lookahead cache/fitting",
            "replacement natives/interfaces",
            "fitted controls",
            "evaluation/generation",
        ],
        "jobs": jobs,
    }


def verify_exposure(path: Path, job: str, steps: int, target_bytes: int) -> None:
    expected = json.loads(path.read_text())["jobs"][job]
    if (steps, target_bytes) != (expected["steps"], expected["target_bytes"]):
        raise ValueError(
            f"{job} exposure differs from the precomputed design: steps={steps}, target_bytes={target_bytes}"
        )
