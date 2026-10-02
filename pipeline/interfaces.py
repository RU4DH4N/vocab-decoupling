from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._measure import MeasureContract
from pipeline.sweep import Sweep
from pipeline.train_native import TrainNative
from pipeline.train_stage import TrainStage


class Interfaces(MeasureContract):
    module = "execution.interfaces"
    tasks = ("train", "measure")

    def __init__(self, context: Context, replicate: int, task: str) -> None:
        super().__init__(context, replicate)
        if task not in self.tasks:
            raise ValueError("unknown interface task")
        self.task = task

    @property
    def label(self) -> str:
        split = "" if self.task == "train" else f":{self.context.split}"
        return f"Interfaces[seed {self.replicate}:{self.task}{split}]"

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
            Interfaces(self.context, self.replicate, "train"),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "train":
            folder = self.run_root / "interfaces"
            return folder / "interfaces.pt", folder / "interfaces.json"
        return (self.measured_root / "interfaces" / "evaluation.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.task == "train":
            self.seed_worker("train")
        else:
            self.measure_worker("measure")
