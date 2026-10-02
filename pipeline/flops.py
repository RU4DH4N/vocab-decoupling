from collections.abc import Mapping
from pathlib import Path

from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE, PipelineContract
from pipeline.prepare import Prepare


class FlopAccounting(PipelineContract):
    module = "execution.flops"

    def dependencies(self) -> tuple[Contract, ...]:
        return (Prepare(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.context.output / "flops.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        c = self.context
        self.worker("--output", c.output, "--device", c.device)

    def accelerator_weight(self) -> int:
        return EXCLUSIVE
