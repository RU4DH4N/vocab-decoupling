import subprocess
import sys
from pathlib import Path

from framework.paths import portable_path
from framework.sources import closure
from orchestrator.context import Context
from orchestrator.contract import Contract

ROOT = Path(__file__).resolve().parents[1]
EXCLUSIVE = sys.maxsize


class PipelineContract(Contract):
    uses_device = True
    module: str | None = None

    def __init__(self, context: Context) -> None:
        self.context = context

    @property
    def label(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict:
        c = self.context
        namespace = portable_path(c.output, root=ROOT)
        return {
            "config": c.settings,
            "output_namespace": namespace,
            "data_identity": list(c.data_identity),
        }

    def sources(self) -> tuple[Path, ...]:
        files = set(super().sources())
        if self.module is not None:
            files.update(closure(self.module))
        return tuple(sorted(files))

    def weight(self) -> int:
        if not self.uses_device or self.context.device == "cpu":
            return 0
        return self.accelerator_weight()

    def accelerator_weight(self) -> int:
        return 1

    def worker(self, *arguments: object) -> None:
        if self.module is None:
            raise RuntimeError(f"{self.label} declares no worker module")
        subprocess.run(
            [sys.executable, "-m", self.module, *map(str, arguments)],
            cwd=ROOT,
            check=True,
        )


class SeedContract(PipelineContract):
    def __init__(self, context: Context, replicate: int) -> None:
        super().__init__(context)
        if not 0 <= replicate < context.settings["seeds"]:
            raise ValueError("replicate outside the declared seeds")
        self.replicate = replicate

    @property
    def run_root(self) -> Path:
        return self.context.output / f"seed-{self.replicate}"

    @property
    def label(self) -> str:
        return f"{type(self).__name__}[seed {self.replicate}]"

    def parameters(self) -> dict:
        return {**super().parameters(), "replicate": self.replicate}

    def seed_worker(self, task: str, *arguments: object) -> None:
        c = self.context
        self.worker(
            task,
            "--output",
            c.output,
            "--replicate",
            self.replicate,
            "--device",
            c.device,
            *arguments,
        )
