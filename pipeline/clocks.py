from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._measure import MeasureContract
from pipeline.sweep import Sweep
from pipeline.train_native import TrainNative
from pipeline.train_stage import TrainStage


class ClockDistortion(MeasureContract):
    module = "execution.clocks"
    tasks = ("train", "measure")

    def __init__(self, context: Context, replicate: int, task: str) -> None:
        super().__init__(context, replicate)
        if task not in self.tasks:
            raise ValueError("unknown clock task")
        self.task = task

    @property
    def label(self) -> str:
        split = "" if self.task == "train" else f":{self.context.split}"
        return f"ClockDistortion[seed {self.replicate}:{self.task}{split}]"

    def parameters(self) -> dict:
        parameters = {**super().parameters(), "task": self.task}
        if self.task == "train":
            del parameters["split"]
        return parameters

    def dependencies(self) -> tuple[Contract, ...]:
        if self.task == "train":
            return (
                TrainNative(self.context, self.replicate, "fresh-byte"),
                TrainStage.final(self.context, self.replicate, "primary"),
                Sweep(self.context, "adapter"),
            )
        return (
            ClockDistortion(self.context, self.replicate, "train"),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "train":
            return (self.run_root / "clocks" / "interfaces.pt",)
        return (self.measured_root / "clocks" / "evaluation.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.task == "train":
            self.seed_worker("train")
        else:
            self.measure_worker("measure")
