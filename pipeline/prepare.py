from collections.abc import Mapping
from pathlib import Path

from orchestrator.contract import Contract
from pipeline._base import PipelineContract


class Prepare(PipelineContract):
    module = "execution.preparation"
    uses_device = False

    def inputs(self) -> tuple[Path, ...]:
        return self.context.train_file, self.context.selection_file

    def outputs(self) -> tuple[Path, ...]:
        root = self.context.output
        return (
            root / "design.json",
            root / "prompts.json",
            root / "receiver-tokenizer.json",
            root / "baseline-tokenizer.json",
            *(
                root / "exposure" / f"seed-{replicate}.json"
                for replicate in range(self.context.settings["seeds"])
            ),
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
            "prepare",
            "--inputs",
            c.config_snapshot,
            "--output",
            c.output,
            "--train-file",
            c.train_file,
            "--selection-file",
            c.selection_file,
        )
