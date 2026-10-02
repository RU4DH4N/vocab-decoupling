from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._measure import MeasureContract
from pipeline.measure_receiver import MeasureReceiver


class MauveReceiver(MeasureContract):
    module = "execution.receiver_experiment"

    def __init__(self, context: Context, replicate: int, variant: str) -> None:
        super().__init__(context, replicate)
        self.variant = variant

    @property
    def label(self) -> str:
        return (
            f"MauveReceiver[seed {self.replicate}:{self.variant}:{self.context.split}]"
        )

    def parameters(self) -> dict:
        return {**super().parameters(), "receiver": self.variant}

    def dependencies(self) -> tuple[Contract, ...]:
        return (MeasureReceiver(self.context, self.replicate, self.variant),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.measured_root / self.variant / "mauve.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.measure_worker("mauve", "--variant", self.variant)
