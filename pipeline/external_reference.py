from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import PipelineContract
from pipeline.prepare import Prepare
from pipeline.prepare_reporting import PrepareReporting


class ExternalReference(PipelineContract):
    module = "execution.external"
    tasks = ("measure", "mauve")

    def __init__(self, context: Context, task: str) -> None:
        super().__init__(context)
        if task not in self.tasks:
            raise ValueError("unknown external task")
        self.task = task

    @property
    def label(self) -> str:
        return f"ExternalReference[{self.task}:{self.context.split}]"

    @property
    def root(self) -> Path:
        output = self.context.output
        if self.context.split == "selection":
            return output / "external"
        return output / "reporting" / "external"

    def parameters(self) -> dict:
        return {**super().parameters(), "split": self.context.split, "task": self.task}

    def dependencies(self) -> tuple[Contract, ...]:
        if self.task == "mauve":
            return (ExternalReference(self.context, "measure"),)
        if self.context.split == "selection":
            return (Prepare(self.context),)
        return (PrepareReporting(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "measure":
            return (self.root / "evaluation.json", self.root / "generation.json")
        return (self.root / "mauve.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        c = self.context
        self.worker(
            self.task,
            "--output",
            c.output,
            "--device",
            c.device,
            "--split",
            c.split,
        )

    def accelerator_weight(self) -> int:
        return 1
