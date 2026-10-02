from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE
from pipeline._measure import MeasureContract
from pipeline.sweep import Sweep


class BPEReference(MeasureContract):
    tasks = ("train", "measure", "mauve")

    def __init__(self, context: Context, replicate: int, task: str) -> None:
        super().__init__(context, replicate)
        if task not in self.tasks:
            raise ValueError("unknown baseline task")
        if replicate >= context.settings["baseline_seeds"]:
            raise ValueError("replicate outside the declared baseline seeds")
        self.task = task
        self.module = (
            "execution.baseline_training"
            if task == "train"
            else "execution.baseline_experiment"
        )

    @property
    def label(self) -> str:
        split = "" if self.task == "train" else f":{self.context.split}"
        return f"BPEReference[seed {self.replicate}:{self.task}{split}]"

    def parameters(self) -> dict:
        parameters = {**super().parameters(), "task": self.task}
        if self.task == "train":
            del parameters["split"]
        return parameters

    def dependencies(self) -> tuple[Contract, ...]:
        if self.task == "train":
            return (Sweep(self.context, "baseline"),)
        previous = BPEReference(
            self.context, self.replicate, self.tasks[self.tasks.index(self.task) - 1]
        )
        return (previous, *self.split_dependencies())

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "train":
            root = self.run_root / "baseline"
            return tuple(
                root / f"model{suffix}" for suffix in (".pt", ".json", ".metrics.json")
            )
        root = self.measured_root / "baseline"
        return (
            (root / "evaluation.json", root / "generation.json")
            if self.task == "measure"
            else (root / "mauve.json",)
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.task == "train":
            self.seed_worker("train")
        else:
            self.measure_worker(self.task)

    def accelerator_weight(self) -> int:
        return 1 if self.task == "mauve" else EXCLUSIVE
