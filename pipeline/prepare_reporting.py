from collections.abc import Mapping
from pathlib import Path

from orchestrator.contract import Contract
from pipeline._base import PipelineContract
from pipeline.prepare import Prepare


class PrepareReporting(PipelineContract):
    module = "execution.preparation"
    uses_device = False

    def inputs(self) -> tuple[Path, ...]:
        if self.context.reporting_file is None:
            raise ValueError("the reporting split needs a reporting file")
        return (
            self.context.reporting_file,
            self.context.train_file,
            self.context.selection_file,
        )

    def dependencies(self) -> tuple[Contract, ...]:
        return (Prepare(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        root = self.context.output / "reporting"
        return (
            root / "design.json",
            root / "prompts.json",
            *(
                root / "corpus" / name
                for name in (
                    "metadata.json",
                    "train-ids.npy",
                    "train-offsets.npy",
                    "selection-ids.npy",
                    "selection-offsets.npy",
                )
            ),
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        c = self.context
        self.worker(
            "report",
            "--output",
            c.output,
            "--reporting-file",
            c.reporting_file,
            "--train-file",
            c.train_file,
            "--selection-file",
            c.selection_file,
        )
