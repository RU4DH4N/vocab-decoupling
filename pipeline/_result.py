import json
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path

from framework.checkpoints import file_sha256, write_json
from framework.paths import portable_path
from orchestrator.contract import Contract
from pipeline._base import ROOT, PipelineContract


class ResultBundle(PipelineContract):
    uses_device = False
    result_name: str
    missing_experiments: tuple[str, ...] = ()

    @abstractmethod
    def measurements(self) -> Mapping[str, Path]:
        pass

    def parameters(self) -> dict:
        return {**super().parameters(), "split": self.context.split}

    @property
    def replicates(self) -> range:
        return range(self.context.settings["seeds"])

    def seed_root(self, replicate: int) -> Path:
        return self.context.output / f"seed-{replicate}"

    def measured_root(self, replicate: int) -> Path:
        root = self.seed_root(replicate)
        return root if self.context.split == "selection" else root / "reporting"

    def per_seed(self, paths: Mapping[str, str]) -> dict[str, Path]:
        return {
            f"seed-{replicate}/{name}": self.seed_root(replicate) / path
            for replicate in self.replicates
            for name, path in paths.items()
        }

    def measured(self, paths: Mapping[str, str]) -> dict[str, Path]:
        return {
            f"seed-{replicate}/{name}": self.measured_root(replicate) / path
            for replicate in self.replicates
            for name, path in paths.items()
        }

    def claim_path(self, name: str) -> Path:
        folder = "claims" if self.context.split == "selection" else "claims-reporting"
        return self.context.output / folder / f"{name}.json"

    def outputs(self) -> tuple[Path, ...]:
        return (self.claim_path(self.result_name),)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        declared = set()
        visited = set()

        def visit(node: Contract) -> None:
            if node.key in visited:
                return
            visited.add(node.key)
            declared.update(path.resolve() for path in node.outputs())
            for parent in node.dependencies():
                visit(parent)

        for dependency in dependencies.values():
            visit(dependency)
        evidence = {}
        missing = set(self.missing_experiments)
        if self.context.settings["seeds"] < 2:
            missing.add("independent-seed-replication")
        missing.discard(
            "reporting-split-evaluation" if self.context.split == "reporting" else ""
        )
        for name, path in self.measurements().items():
            if path.resolve() not in declared or path.suffix != ".json":
                raise ValueError(f"{name} is not a declared JSON dependency output")
            payload = json.loads(path.read_text())
            item: dict[str, object] = {
                "path": portable_path(path, root=ROOT),
                "sha256": file_sha256(path),
            }
            if (
                isinstance(payload, dict)
                and payload.get("schema") == "contract-result-v1"
            ):
                missing.update(payload["missing_experiments"])
            else:
                item["data"] = payload
            evidence[name] = item
        write_json(
            self.outputs()[0],
            {
                "schema": "contract-result-v1",
                "result": self.result_name,
                "path_base": "repository",
                "dependencies": sorted(dependencies),
                "measurements": evidence,
                "split": self.context.split,
                "evidence_status": "partial" if missing else "available",
                "missing_experiments": sorted(missing),
                "scientific_verdict": None,
            },
        )
