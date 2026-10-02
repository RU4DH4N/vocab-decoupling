from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE
from pipeline._measure import MeasureContract
from pipeline.train_stage import TrainStage


class MeasureReceiver(MeasureContract):
    module = "execution.receiver_experiment"

    def __init__(self, context: Context, replicate: int, variant: str) -> None:
        super().__init__(context, replicate)
        self.variant = variant

    @property
    def label(self) -> str:
        return f"MeasureReceiver[seed {self.replicate}:{self.variant}:{self.context.split}]"

    def parameters(self) -> dict:
        return {**super().parameters(), "receiver": self.variant}

    def dependencies(self) -> tuple[Contract, ...]:
        return (
            TrainStage.final(self.context, self.replicate, self.variant),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.measured_root / self.variant / name
            for name in ("evaluation.json", "generation.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.measure_worker("measure", "--variant", self.variant)

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.variant == "fresh-bpe" else 1
