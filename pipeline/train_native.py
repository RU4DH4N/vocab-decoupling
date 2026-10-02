from collections.abc import Mapping
from pathlib import Path

from execution.design import RECEIVERS
from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE, SeedContract
from pipeline.sweep import Sweep


class TrainNative(SeedContract):
    module = "execution.training"

    def __init__(self, context: Context, replicate: int, variant: str) -> None:
        super().__init__(context, replicate)
        if variant not in RECEIVERS:
            raise ValueError("receiver is absent from the declared matrix")
        self.variant = variant

    @property
    def label(self) -> str:
        return f"TrainNative[seed {self.replicate}:{self.variant}]"

    def parameters(self) -> dict:
        return {**super().parameters(), "receiver": self.variant}

    def dependencies(self) -> tuple[Contract, ...]:
        return (Sweep(self.context, "native"),)

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.run_root / self.variant / f"native{suffix}"
            for suffix in (".pt", ".json", ".metrics.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.seed_worker("native", "--variant", self.variant)

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.variant == "fresh-bpe" else 1
