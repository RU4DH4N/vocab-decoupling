import argparse
import json
from collections import defaultdict
from pathlib import Path

from claims.active import Active
from orchestrator.context import load_context
from orchestrator.graph import Graph

IGNORED = ("config", "output_namespace", "data_identity", "replicate")
TRAINING = {
    "pipeline.train_native.TrainNative",
    "pipeline.train_stage.TrainStage",
    "pipeline.sweep.Sweep",
    "pipeline.bootstrap.Bootstrap",
    "pipeline.interfaces.Interfaces",
    "pipeline.fitted_control.FittedControl",
    "pipeline.bpe_reference.BPEReference",
    "pipeline.planning.Planning",
}
MEASURING = ("measure",)


def signature(contract: str, parameters: dict) -> str:
    kept = {k: v for k, v in parameters.items() if k not in IGNORED}
    return json.dumps([contract, kept], sort_keys=True)


def generation_seconds(outputs: list[str], root: Path) -> float:
    total = 0.0
    for output in outputs:
        path = root / output
        if path.name != "generation.json" or not path.exists():
            continue
        report = json.loads(path.read_text())
        seconds = [s["seconds"] for s in report["samples"]]
        decoding = report["decoding"]
        alone = decoding.get("latency_samples", len(seconds))
        streams = decoding.get("streams", 1)
        total += sum(seconds[:alone]) + sum(seconds[alone:]) / streams
    return total


def loop_seconds(outputs: list[str], root: Path) -> float | None:
    total = None
    for output in outputs:
        path = root / output
        if path.name.endswith(".metrics.json") and path.exists():
            seconds = json.loads(path.read_text()).get("seconds")
            if seconds is not None:
                total = (total or 0.0) + seconds
    return total


def forecast(
    pilot: Path, paper_design: dict, paper_config: Path, repository: Path
) -> tuple[dict, dict[str, float], list[str]]:
    pilot_design = json.loads((pilot / "design.json").read_text())
    ratios = {
        "steps": paper_design["steps"]["full"] / pilot_design["steps"]["full"],
        "evaluation": paper_design["shared"]["evaluation_batches"]
        / pilot_design["shared"]["evaluation_batches"],
        "data": paper_design["statistics"]["train_bytes"]
        / pilot_design["statistics"]["train_bytes"],
    }
    measured = {}
    timings = json.loads((pilot / ".orchestrator/timings.json").read_text())
    for node in timings.values():
        seconds = node["seconds"]
        measured[signature(node["contract"], node["parameters"])] = (
            seconds,
            generation_seconds(node["outputs"], repository),
            loop_seconds(node["outputs"], repository),
        )
    by_contract: dict[str, float] = defaultdict(float)
    unmatched = []
    for split in ("selection", "reporting"):
        context = load_context(
            paper_config,
            pilot.parent / "forecast",
            None,
            None,
            "cuda",
            split,
            None if split == "selection" else pilot.parent / "forecast.parquet",
        )
        graph = Graph((Active(context),))
        for node in graph.nodes.values():
            contract = f"{type(node).__module__}.{type(node).__qualname__}"
            parameters = node.parameters()
            if split == "reporting" and "split" not in parameters:
                continue
            key = signature(contract, parameters)
            if key not in measured:
                key = signature(contract, {**parameters, "split": "selection"})
            if key not in measured:
                unmatched.append(node.label)
                continue
            seconds, generation, loop = measured[key]
            task = parameters.get("task", "")
            if contract.endswith("MauveReceiver") or task == "mauve":
                scaled = seconds
            elif contract.endswith(("Prepare", "PrepareReporting")):
                scaled = seconds * ratios["data"]
            elif (
                contract.endswith(("MeasureReceiver", "ClockDistortion"))
                or task in MEASURING
            ):
                scaled = (seconds - generation) * ratios["evaluation"] + generation
            elif contract in TRAINING and loop is not None:
                scaled = seconds - loop + loop * ratios["steps"]
            elif contract in TRAINING:
                scaled = seconds * ratios["steps"]
            else:
                scaled = seconds
            by_contract[f"{contract.rsplit('.', 1)[-1]}:{split}"] += scaled
    return ratios, dict(by_contract), unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--paper-design", type=Path, required=True)
    parser.add_argument("--paper-config", type=Path, required=True)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    ratios, by_contract, unmatched = forecast(
        args.pilot.resolve(),
        json.loads(args.paper_design.read_text()),
        args.paper_config,
        repository,
    )
    hours = sum(by_contract.values()) / 3600
    print(json.dumps({"ratios": ratios}, indent=2))
    for name, seconds in sorted(by_contract.items(), key=lambda item: -item[1]):
        print(f"{name:>20}  {seconds / 3600:6.2f} h")
    print(f"{'total':>20}  {hours:6.2f} h  ~${hours * args.usd_per_hour:.2f}")
    if unmatched:
        print(f"no pilot timing for {len(unmatched)} steps, e.g. {unmatched[:3]}")


if __name__ == "__main__":
    main()
