from collections.abc import Mapping
from pathlib import Path

from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE, PipelineContract
from pipeline.preflight import Preflight


class Sweep(PipelineContract):
    module = "execution.sweep"

    def __init__(self, context: Context, name: str) -> None:
        super().__init__(context)
        if name not in ("native", "baseline", "adapter", "trunk"):
            raise ValueError("unknown sweep")
        self.name = name

    @property
    def label(self) -> str:
        return f"Sweep[{self.name}]"

    def parameters(self) -> dict:
        return {**super().parameters(), "sweep": self.name}

    def dependencies(self) -> tuple[Contract, ...]:
        from pipeline.train_native import TrainNative
        from pipeline.train_stage import TrainStage

        if self.name == "adapter":
            return (TrainNative(self.context, 0, "primary"),)
        if self.name == "trunk":
            return (TrainStage(self.context, 0, "primary", 1),)
        return (Preflight(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.context.output / "sweep" / f"{self.name}.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        c = self.context
        self.worker(
            self.name,
            "--output",
            c.output,
            "--device",
            c.device,
        )

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.name == "baseline" else 1
