from collections.abc import Mapping
from pathlib import Path

from execution.design import PRIMARY_STAGES, REPLACEMENT_STAGES
from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE, SeedContract
from pipeline.sweep import Sweep
from pipeline.train_native import TrainNative


def schedule(variant: str) -> tuple[str, ...]:
    return PRIMARY_STAGES if variant == "primary" else REPLACEMENT_STAGES


class TrainStage(SeedContract):
    module = "execution.training"

    def __init__(
        self, context: Context, replicate: int, variant: str, index: int
    ) -> None:
        super().__init__(context, replicate)
        if not 0 <= index < len(schedule(variant)):
            raise ValueError("stage index outside the declared schedule")
        self.variant = variant
        self.index = index
        self.stage_name = schedule(variant)[index]

    @classmethod
    def final(cls, context: Context, replicate: int, variant: str) -> "TrainStage":
        return cls(context, replicate, variant, len(schedule(variant)) - 1)

    @property
    def label(self) -> str:
        return (
            f"TrainStage[seed {self.replicate}:{self.variant}:"
            f"{self.index}:{self.stage_name}]"
        )

    def parameters(self) -> dict:
        return {
            **super().parameters(),
            "receiver": self.variant,
            "stage_index": self.index,
        }

    def dependencies(self) -> tuple[Contract, ...]:
        sweeps = (Sweep(self.context, "adapter"),)
        if self.stage_name == "trunk":
            sweeps += (Sweep(self.context, "trunk"),)
        if self.index:
            previous = TrainStage(
                self.context, self.replicate, self.variant, self.index - 1
            )
            return (previous, *sweeps)
        native = TrainNative(self.context, self.replicate, self.variant)
        if self.variant == "primary":
            return (native, *sweeps)
        primary = TrainStage.final(self.context, self.replicate, "primary")
        return (native, primary, *sweeps)

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.run_root / self.variant / f"stage-{self.index}{suffix}"
            for suffix in (".pt", ".json", ".metrics.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.seed_worker(
            "stage",
            "--variant",
            self.variant,
            "--stage",
            self.index,
        )

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.variant == "fresh-bpe" else 1
