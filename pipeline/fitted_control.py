from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE
from pipeline._measure import MeasureContract
from pipeline.train_stage import TrainStage


class FittedControl(MeasureContract):
    module = "execution.fitted_controls"

    def __init__(
        self, context: Context, replicate: int, variant: str, arm: str, task: str
    ) -> None:
        super().__init__(context, replicate)
        if (
            variant not in ("fresh-byte", "fresh-bpe")
            or arm not in ("correct", "shuffled", "native-only")
            or task not in ("train", "measure")
        ):
            raise ValueError("invalid control job")
        self.variant, self.arm, self.task = variant, arm, task

    @property
    def label(self) -> str:
        split = "" if self.task == "train" else f":{self.context.split}"
        return (
            f"FittedControl[seed {self.replicate}:{self.variant}:{self.arm}:"
            f"{self.task}{split}]"
        )

    def parameters(self) -> dict:
        parameters = {
            **super().parameters(),
            "receiver": self.variant,
            "control": self.arm,
            "task": self.task,
        }
        if self.task == "train":
            del parameters["split"]
        return parameters

    def dependencies(self) -> tuple[Contract, ...]:
        if self.task == "train":
            return (TrainStage(self.context, self.replicate, self.variant, 0),)
        return (
            FittedControl(
                self.context, self.replicate, self.variant, self.arm, "train"
            ),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        if self.task == "train":
            folder = self.run_root / "controls" / self.variant / self.arm
            return tuple(
                folder / name
                for name in ("model.pt", "model.json", "model.metrics.json")
            )
        return (
            self.measured_root
            / "controls"
            / self.variant
            / self.arm
            / "evaluation.json",
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        if self.task == "train":
            self.seed_worker(
                "train",
                "--variant",
                self.variant,
                "--arm",
                self.arm,
            )
        else:
            self.measure_worker(
                "measure",
                "--variant",
                self.variant,
                "--arm",
                self.arm,
            )

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.variant == "fresh-bpe" else 1
