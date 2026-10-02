from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._measure import MeasureContract
from pipeline.train_stage import TrainStage


class Planning(MeasureContract):
    tasks = ("planner", "fit", "oracle-fit", "measure", "mauve")

    def __init__(self, context: Context, replicate: int, task: str) -> None:
        super().__init__(context, replicate)
        if task not in self.tasks:
            raise ValueError("unknown planning task")
        self.task = task
        self.module = (
            "execution.planning_experiment"
            if self.measures
            else "execution.planning_training"
        )

    @property
    def measures(self) -> bool:
        return self.task in ("measure", "mauve")

    @property
    def label(self) -> str:
        split = f":{self.context.split}" if self.measures else ""
        return f"Planning[seed {self.replicate}:{self.task}{split}]"

    def parameters(self) -> dict:
        parameters = {**super().parameters(), "task": self.task}
        if not self.measures:
            del parameters["split"]
        return parameters

    def dependencies(self) -> tuple[Contract, ...]:
        index = self.tasks.index(self.task)
        if not index:
            return (TrainStage.final(self.context, self.replicate, "primary"),)
        previous = Planning(self.context, self.replicate, self.tasks[index - 1])
        if self.task == "measure":
            return (previous, *self.split_dependencies())
        return (previous,)

    def outputs(self) -> tuple[Path, ...]:
        if not self.measures:
            root = self.run_root / "planning"
            name = {
                "planner": "planner",
                "fit": "lookahead",
                "oracle-fit": "lookahead-oracle",
            }[self.task]
            return tuple(
                root / f"{name}{suffix}" for suffix in (".pt", ".json", ".metrics.json")
            )
        root = self.measured_root / "planning"
        if self.task == "measure":
            return root / "evaluation.json", root / "generation.json"
        return (root / "mauve.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.measures:
            self.measure_worker(self.task)
        else:
            self.seed_worker(self.task)

    def accelerator_weight(self) -> int:
        return 1
