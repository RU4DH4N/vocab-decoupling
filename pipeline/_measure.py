from pathlib import Path

from orchestrator.contract import Contract
from pipeline._base import SeedContract
from pipeline.prepare_reporting import PrepareReporting


class MeasureContract(SeedContract):
    @property
    def measured_root(self) -> Path:
        if self.context.split == "selection":
            return self.run_root
        return self.run_root / "reporting"

    def parameters(self) -> dict:
        return {**super().parameters(), "split": self.context.split}

    def split_dependencies(self) -> tuple[Contract, ...]:
        if self.context.split == "selection":
            return ()
        return (PrepareReporting(self.context),)

    def measure_worker(self, task: str, *arguments: object) -> None:
        self.seed_worker(task, "--split", self.context.split, *arguments)
