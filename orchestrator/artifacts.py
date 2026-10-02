import json
from pathlib import Path

from framework.checkpoints import file_sha256, write_json
from orchestrator.contract import digest
from orchestrator.graph import Graph, portable


class Artifacts:
    def __init__(self, graph: Graph, directory: Path) -> None:
        self.graph, self.directory = graph, directory

    def path(self, key: str) -> Path:
        return self.directory / "cache" / f"{key}.json"

    def record(self, key: str, parents: dict[str, dict]) -> dict:
        node = self.graph.nodes[key]
        return {
            "fingerprint": self.graph.fingerprints[key],
            "inputs": [file_sha256(p) for p in node.inputs()],
            "outputs": {portable(p): file_sha256(p) for p in node.outputs()},
            "dependencies": {
                p: digest(parents[p]["outputs"]) for p in self.graph.parents[key]
            },
        }

    def inspect(self) -> tuple[dict[str, tuple[bool, str]], dict[str, dict]]:
        states, records = {}, {}
        for key in self.graph.order:
            node = self.graph.nodes[key]
            if node.unavailable_reason:
                states[key] = (False, node.unavailable_reason)
                continue
            if any(p not in records for p in self.graph.parents[key]):
                states[key] = (False, "dependency missing or stale")
                continue
            missing = [
                portable(p)
                for p in (*node.inputs(), *node.outputs())
                if not p.is_file()
            ]
            if missing:
                states[key] = (False, f"missing files: {missing}")
                continue
            record = self.reusable(key, records)
            if record is not None:
                records[key] = record
                states[key] = (True, "ready")
            else:
                states[key] = (
                    False,
                    "missing manifest or changed code/configuration/inputs/outputs",
                )
        return states, records

    def reusable(self, key: str, parents: dict[str, dict]) -> dict | None:
        node = self.graph.nodes[key]
        if not all(p.is_file() for p in (*node.inputs(), *node.outputs())):
            return None
        actual = self.record(key, parents)
        try:
            saved = json.loads(self.path(key).read_text())
        except (OSError, ValueError):
            return None
        return actual if actual == saved else None

    def commit(self, key: str, parents: dict[str, dict]) -> dict:
        record = self.record(key, parents)
        write_json(self.path(key), record)
        return record
