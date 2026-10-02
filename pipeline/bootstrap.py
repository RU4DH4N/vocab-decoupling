from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE
from pipeline._measure import MeasureContract
from pipeline.train_stage import TrainStage


class Bootstrap(MeasureContract):
    module = "execution.bootstrap"
    tasks = ("train", "measure")

    def __init__(self, context: Context, replicate: int, task: str) -> None:
        super().__init__(context, replicate)
        if task not in self.tasks:
            raise ValueError("unknown bootstrap task")
        self.task = task

    @property
    def label(self) -> str:
        split = "" if self.task == "train" else f":{self.context.split}"
        return f"Bootstrap[seed {self.replicate}:{self.task}{split}]"

    def parameters(self) -> dict:
        parameters = {**super().parameters(), "task": self.task}
        if self.task == "train":
            del parameters["split"]
        return parameters

    def dependencies(self) -> tuple[Contract, ...]:
        if self.task == "train":
            return (
                TrainStage(self.context, self.replicate, "fresh-byte", 0),
                TrainStage.final(self.context, self.replicate, "primary"),
            )
        return (
            Bootstrap(self.context, self.replicate, "train"),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "train":
            folder = self.run_root / "bootstrap"
            return folder / "interfaces.pt", folder / "trajectories.json"
        return (self.measured_root / "bootstrap" / "evaluation.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.task == "train":
            self.seed_worker("train")
        else:
            self.measure_worker("measure")

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.task == "train" else 1
