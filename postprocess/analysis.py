import argparse
import json
import math
from pathlib import Path

import numpy as np

from framework.checkpoints import write_json
from framework.statistics import hierarchical_paired_bootstrap

RESAMPLES = 1000
LEVEL = 0.95


class Run:
    def __init__(self, root: Path, split: str) -> None:
        self.root, self.split = root, split
        design = json.loads((root / "design.json").read_text())
        self.inputs = design["inputs"]
        self.seed = self.inputs["root_seed"]
        self.seeds = range(self.inputs["seeds"])
        self.baseline_seeds = range(self.inputs["baseline_seeds"])

    def measured(self, replicate: int, path: str) -> dict:
        root = self.root / f"seed-{replicate}"
        folder = root if self.split == "selection" else root / "reporting"
        return json.loads((folder / path).read_text())

    def external(self, path: str) -> dict:
        folder = self.root if self.split == "selection" else self.root / "reporting"
        return json.loads((folder / "external" / path).read_text())

    def trained(self, replicate: int, path: str) -> dict:
        return json.loads((self.root / f"seed-{replicate}" / path).read_text())


def bits(nats: np.ndarray, utf8_bytes: np.ndarray) -> float:
    return float(nats.sum() / (math.log(2) * utf8_bytes.sum()))


def pooled(nats: list[np.ndarray], utf8_bytes: list[np.ndarray]) -> float:
    return bits(np.concatenate(nats), np.concatenate(utf8_bytes))


def difference(
    run: Run,
    left: list[np.ndarray],
    right: list[np.ndarray],
    utf8_bytes: list[np.ndarray],
) -> dict:
    interval = hierarchical_paired_bootstrap(
        left, right, utf8_bytes, RESAMPLES, run.seed, LEVEL
    )
    return {
        "estimate": interval.estimate,
        "low": interval.low,
        "high": interval.high,
        "per_seed": [
            bits(a, b) - bits(c, b) for a, c, b in zip(left, right, utf8_bytes)
        ],
    }


def same_windows(*documents: dict) -> None:
    groups = [document["group_ids"] for document in documents]
    if any(group != groups[0] for group in groups[1:]):
        raise ValueError("paired results were not scored on the same windows")


def rows(evaluation: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    paired = evaluation["paired_rows"][arm]
    return np.array(paired["nll"]), np.array(paired["utf8_bytes"])


def continuous_communication(run: Run) -> dict:
    arms = ("correct", "shuffled", "native-only")
    fitted = {
        arm: [
            run.measured(r, f"controls/fresh-byte/{arm}/evaluation.json")
            for r in run.seeds
        ]
        for arm in arms
    }
    for r in run.seeds:
        same_windows(*(fitted[arm][r] for arm in arms))
    nats = {arm: [np.array(d["nll"]) for d in fitted[arm]] for arm in arms}
    utf8 = [np.array(d["utf8_bytes"]) for d in fitted["correct"]]
    inference = [run.measured(r, "fresh-byte/evaluation.json") for r in run.seeds]
    ablation = {
        arm: [rows(d, arm)[0] for d in inference]
        for arm in ("correct", "shuffled", "zero_message", "native")
    }
    ablation_bytes = [rows(d, "correct")[1] for d in inference]
    return {
        "fitted_bits_per_byte": {arm: pooled(nats[arm], utf8) for arm in arms},
        "fitted_shuffled_minus_correct": difference(
            run, nats["shuffled"], nats["correct"], utf8
        ),
        "fitted_native_only_minus_correct": difference(
            run, nats["native-only"], nats["correct"], utf8
        ),
        "inference_bits_per_byte": {
            arm: pooled(values, ablation_bytes) for arm, values in ablation.items()
        },
        "inference_shuffled_minus_correct": difference(
            run, ablation["shuffled"], ablation["correct"], ablation_bytes
        ),
    }


def learned_protocol(run: Run) -> dict:
    evaluations = [run.measured(r, "interfaces/evaluation.json") for r in run.seeds]
    sizes = run.trained(0, "interfaces/interfaces.json")["trainable_parameters"]
    result = {}
    for kind in evaluations[0]["interfaces"]:
        entries = [e["interfaces"][kind] for e in evaluations]
        utf8 = [np.array(e["utf8_bytes"]) for e in entries]
        nats = {
            arm: [np.array(e["paired_nll"][arm]) for e in entries]
            for arm in ("correct", "shuffled", "zero_message")
        }
        result[kind] = {
            "trainable_parameters": sizes[kind],
            "correct_bits_per_byte": pooled(nats["correct"], utf8),
            "shuffled_minus_correct": difference(
                run, nats["shuffled"], nats["correct"], utf8
            ),
            "zero_minus_correct": difference(
                run, nats["zero_message"], nats["correct"], utf8
            ),
        }
    return result


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def correspondence_bootstrap(run: Run) -> dict:
    evaluations = [run.measured(r, "bootstrap/evaluation.json") for r in run.seeds]
    trajectories = [run.trained(r, "bootstrap/trajectories.json") for r in run.seeds]
    result = {}
    for arm in evaluations[0]["arms"]:
        scores = [e["arms"][arm] for e in evaluations]
        accuracy = [s["correct"]["first_layer_event_accuracy"] for s in scores]
        gain = [s["shuffled"]["ce"] - s["correct"]["ce"] for s in scores]
        phases = []
        for phase in range(len(trajectories[0]["trajectories"][arm])):
            points = [t["trajectories"][arm][phase]["selection"] for t in trajectories]
            phases.append(
                {
                    "stage": trajectories[0]["trajectories"][arm][phase]["stage"],
                    "first_layer_event_accuracy": mean(
                        [p["correct"]["first_layer_event_accuracy"] for p in points]
                    ),
                    "shuffled_minus_correct_ce": mean(
                        [p["shuffled"]["ce"] - p["correct"]["ce"] for p in points]
                    ),
                }
            )
        result[arm] = {
            "likelihood": scores[0]["correct"]["likelihood"],
            "first_layer_event_accuracy": mean(accuracy),
            "first_layer_event_accuracy_per_seed": accuracy,
            "shuffled_minus_correct_ce": mean(gain),
            "shuffled_minus_correct_ce_per_seed": gain,
            "trajectory": phases,
        }
    return result


def independent_clocks(run: Run) -> dict:
    evaluations = [run.measured(r, "clocks/evaluation.json") for r in run.seeds]
    result = {}
    for level in evaluations[0]["results"]:
        result[level] = {}
        for arm in evaluations[0]["arms"]:
            entries = [e["results"][level][arm] for e in evaluations]
            utf8 = [np.array(e["utf8_bytes"]) for e in entries]
            nats = {
                condition: [np.array(e["paired_nll"][condition]) for e in entries]
                for condition in ("correct", "shuffled")
            }
            result[level][arm] = {
                "correct_bits_per_byte": pooled(nats["correct"], utf8),
                "shuffled_minus_correct": difference(
                    run, nats["shuffled"], nats["correct"], utf8
                ),
                "event_accuracy": mean(
                    [e["scores"]["correct"]["hard_event_accuracy"] for e in entries]
                ),
            }
    return result


def breakdown(evaluations: list[dict], key: str) -> dict:
    result = {}
    for condition in evaluations[0][key]:
        nats = np.sum([e[key][condition]["nll"] for e in evaluations], axis=0)
        tokens = np.sum([e[key][condition]["tokens"] for e in evaluations], axis=0)
        result[condition] = [
            float(n / t / math.log(2)) if t else None for n, t in zip(nats, tokens)
        ]
    return result


def planning_ahead(run: Run) -> dict:
    evaluations = [run.measured(r, "planning/evaluation.json") for r in run.seeds]
    utf8 = [np.array(e["utf8_bytes"]) for e in evaluations]
    nats = {
        condition: [np.array(e["paired_nll"][condition]) for e in evaluations]
        for condition in evaluations[0]["paired_nll"]
    }
    return {
        "bits_per_byte": {c: pooled(v, utf8) for c, v in nats.items()},
        "shuffled_minus_real": difference(run, nats["shuffled"], nats["real"], utf8),
        "repeated_minus_real": difference(run, nats["repeated"], nats["real"], utf8),
        "zero_minus_real": difference(run, nats["zero"], nats["real"], utf8),
        "real_minus_oracle": difference(run, nats["real"], nats["oracle"], utf8),
        "real_minus_oracle_fitted": difference(
            run, nats["real"], nats["oracle_fitted"], utf8
        ),
        "zero_minus_oracle_fitted": difference(
            run, nats["zero"], nats["oracle_fitted"], utf8
        ),
        "planner": {
            key: mean([e["planner"][key] for e in evaluations])
            for key in evaluations[0]["planner"]
        },
        "bits_per_symbol_by_position": breakdown(evaluations, "by_position"),
        "bits_per_symbol_by_ambiguity_decade": breakdown(
            evaluations, "by_prefix_ambiguity"
        ),
    }


def receiver_replacement(run: Run) -> dict:
    evaluations = {
        variant: [run.measured(r, f"{variant}/evaluation.json") for r in run.seeds]
        for variant in ("primary", "fresh-byte", "fresh-bpe")
    }
    result = {}
    for variant, documents in evaluations.items():
        utf8 = [rows(d, "correct")[1] for d in documents]
        nats = {
            arm: [rows(d, arm)[0] for d in documents]
            for arm in ("correct", "shuffled", "native")
        }
        result[variant] = {
            "bits_per_byte": {arm: pooled(v, utf8) for arm, v in nats.items()},
            "shuffled_minus_correct": difference(
                run, nats["shuffled"], nats["correct"], utf8
            ),
            "native_minus_correct": difference(
                run, nats["native"], nats["correct"], utf8
            ),
        }
    primary = result["primary"]["native_minus_correct"]["estimate"]
    for variant in ("fresh-byte", "fresh-bpe"):
        gain = result[variant]["native_minus_correct"]["estimate"]
        result[variant]["recovery_of_primary_gain"] = (
            gain / primary if primary else None
        )
    controls = {
        arm: [
            run.measured(r, f"controls/fresh-bpe/{arm}/evaluation.json")
            for r in run.seeds
        ]
        for arm in ("correct", "shuffled", "native-only")
    }
    utf8 = [np.array(d["utf8_bytes"]) for d in controls["correct"]]
    nats = {arm: [np.array(d["nll"]) for d in docs] for arm, docs in controls.items()}
    result["fresh-bpe"]["fitted_shuffled_minus_correct"] = difference(
        run, nats["shuffled"], nats["correct"], utf8
    )
    result["fresh-bpe"]["fitted_native_only_minus_correct"] = difference(
        run, nats["native-only"], nats["correct"], utf8
    )
    return result


def remote_hypothesis(run: Run) -> dict:
    flops = json.loads((run.root / "flops.json").read_text())
    baseline = [run.measured(r, "baseline/evaluation.json") for r in run.baseline_seeds]
    utf8 = [np.array(d["utf8_bytes"]) for d in baseline]
    baseline_nats = [np.array(d["nats"]) for d in baseline]
    external = run.external("evaluation.json")
    external_nats = np.array(external["nats"])
    result = {
        "bpe_bits_per_byte": pooled(baseline_nats, utf8),
        "external_bits_per_byte": bits(external_nats, np.array(external["utf8_bytes"])),
        "external_nfkc_changed_windows": external["nfkc_changed_windows"],
        "gap_to_bpe": {},
        "gap_to_external": {},
        "mauve": {},
        "latency_seconds_per_byte": {},
        "training_flops": flops["totals"],
        "flop_ratios": flops["ratios"],
    }
    for variant in ("primary", "fresh-byte", "fresh-bpe"):
        documents = [
            run.measured(r, f"{variant}/evaluation.json") for r in run.baseline_seeds
        ]
        for document, reference in zip(documents, baseline, strict=True):
            same_windows(document, reference)
        result["gap_to_bpe"][variant] = difference(
            run, [rows(d, "correct")[0] for d in documents], baseline_nats, utf8
        )
        documents = [run.measured(r, f"{variant}/evaluation.json") for r in run.seeds]
        for document in documents:
            same_windows(document, external)
        result["gap_to_external"][variant] = difference(
            run,
            [rows(d, "correct")[0] for d in documents],
            [external_nats for _ in documents],
            [np.array(external["utf8_bytes"]) for _ in documents],
        )
    generators = {
        "primary": "primary",
        "fresh-byte": "fresh-byte",
        "fresh-bpe": "fresh-bpe",
        "lookahead": "planning",
    }
    for name, folder in generators.items():
        result["mauve"][name] = [
            run.measured(r, f"{folder}/mauve.json").get("mauve") for r in run.seeds
        ]
        result["latency_seconds_per_byte"][name] = [
            run.measured(r, f"{folder}/generation.json").get("latency_seconds_per_byte")
            for r in run.seeds
        ]
    result["mauve"]["bpe"] = [
        run.measured(r, "baseline/mauve.json").get("mauve") for r in run.baseline_seeds
    ]
    result["latency_seconds_per_byte"]["bpe"] = [
        run.measured(r, "baseline/generation.json")["latency_seconds_per_byte"]
        for r in run.baseline_seeds
    ]
    result["mauve"]["external"] = [run.external("mauve.json").get("mauve")]
    result["latency_seconds_per_byte"]["external"] = [
        run.external("generation.json")["latency_seconds_per_byte"]
    ]
    return result


def interval(value: dict) -> str:
    return f"{value['estimate']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in body])


def _c1(c1: dict) -> list[str]:
    comparisons = (
        ("fitted shuffled − correct", "fitted_shuffled_minus_correct"),
        ("fitted native-only − correct", "fitted_native_only_minus_correct"),
        ("inference shuffled − correct", "inference_shuffled_minus_correct"),
    )
    rows = [[label, interval(c1[key])] for label, key in comparisons]
    return ["## C1", table(["comparison", "difference"], rows)]


def _c2(c2: dict) -> list[str]:
    rows = [
        [
            kind,
            f"{v['trainable_parameters']:,}",
            number(v["correct_bits_per_byte"]),
            interval(v["shuffled_minus_correct"]),
        ]
        for kind, v in c2.items()
    ]
    header = ["interface", "parameters", "correct bpb", "shuffled − correct"]
    return ["## C2", table(header, rows)]


def _c3(c3: dict) -> list[str]:
    rows = [
        [
            arm,
            v["likelihood"],
            number(v["first_layer_event_accuracy"]),
            number(v["shuffled_minus_correct_ce"]),
        ]
        for arm, v in c3.items()
    ]
    header = ["arm", "likelihood", "first-layer accuracy", "shuffled − correct (CE)"]
    return ["## C3", table(header, rows)]


def _c4(c4: dict) -> list[str]:
    rows = [
        [
            level,
            arm,
            number(v["correct_bits_per_byte"]),
            interval(v["shuffled_minus_correct"]),
            number(v["event_accuracy"]),
        ]
        for level, arms in c4.items()
        for arm, v in arms.items()
    ]
    header = ["level", "arm", "correct bpb", "shuffled − correct", "event accuracy"]
    return ["## C4", table(header, rows)]


def _c5(c5: dict) -> list[str]:
    comparisons = (
        "shuffled_minus_real",
        "repeated_minus_real",
        "zero_minus_real",
        "real_minus_oracle",
        "real_minus_oracle_fitted",
        "zero_minus_oracle_fitted",
    )
    by_position = c5["bits_per_symbol_by_position"]
    positions = [
        [str(position), *(number(v[position]) for v in by_position.values())]
        for position in range(len(by_position["real"]))
    ]
    return [
        "## C5",
        table(
            ["comparison", "difference"],
            [[name.replace("_", " "), interval(c5[name])] for name in comparisons],
        ),
        table(
            ["planner", "rate"],
            [[name.replace("_", " "), number(v)] for name, v in c5["planner"].items()],
        ),
        table(["position", *by_position], positions),
    ]


def _c6(c6: dict) -> list[str]:
    rows = [
        [
            variant,
            number(v["bits_per_byte"]["correct"]),
            interval(v["shuffled_minus_correct"]),
            interval(v["native_minus_correct"]),
            number(v.get("recovery_of_primary_gain")),
        ]
        for variant, v in c6.items()
    ]
    header = [
        "receiver",
        "correct bpb",
        "shuffled − correct",
        "native − correct",
        "recovery",
    ]
    return ["## C6", table(header, rows)]


def _c7(h: dict) -> list[str]:
    summary = (
        f"BPE baseline: {number(h['bpe_bits_per_byte'])} bpb. "
        f"External BabyLlama: {number(h['external_bits_per_byte'])} bpb "
        f"({h['external_nfkc_changed_windows']} windows change under NFKC)."
    )
    gaps = [
        [variant, interval(v), interval(h["gap_to_external"][variant])]
        for variant, v in h["gap_to_bpe"].items()
    ]
    mauve = [
        [name, ", ".join(number(v) for v in values)]
        for name, values in h["mauve"].items()
    ]
    ratios = [
        [name.replace("_", " "), number(v)] for name, v in h["flop_ratios"].items()
    ]
    return [
        "## C7",
        summary,
        table(["receiver", "receiver − BPE (bpb)", "receiver − external (bpb)"], gaps),
        table(["generator", "MAUVE per seed"], mauve),
        table(["ratio", "value"], ratios),
    ]


def report(results: dict, run: Run) -> str:
    sections = [
        f"# Results ({run.split} split, {len(run.seeds)} seeds, "
        f"{int(LEVEL * 100)}% hierarchical paired bootstrap)",
        "Differences are in bits per byte; positive means the second arm is better.",
        *_c1(results["C1"]),
        *_c2(results["C2"]),
        *_c3(results["C3"]),
        *_c4(results["C4"]),
        *_c5(results["C5"]),
        *_c6(results["C6"]),
        *_c7(results["C7"]),
    ]
    return "\n\n".join(sections) + "\n"


def analyse(root: Path, split: str) -> tuple[dict, str]:
    run = Run(root, split)
    results = {
        "split": split,
        "seeds": len(run.seeds),
        "bootstrap": {"resamples": RESAMPLES, "level": LEVEL, "seed": run.seed},
        "C1": continuous_communication(run),
        "C2": learned_protocol(run),
        "C3": correspondence_bootstrap(run),
        "C4": independent_clocks(run),
        "C5": planning_ahead(run),
        "C6": receiver_replacement(run),
        "C7": remote_hypothesis(run),
    }
    return results, report(results, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    args = parser.parse_args()
    results, markdown = analyse(args.output, args.split)
    folder = args.output / "analysis"
    write_json(folder / f"{args.split}.json", results)
    (folder / f"{args.split}.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
