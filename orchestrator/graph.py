from pathlib import Path

from framework.checkpoints import file_sha256
from framework.paths import portable_path
from orchestrator.contract import Contract, digest

ROOT = Path(__file__).resolve().parents[1]


def portable(path: Path) -> str:
    return portable_path(path, root=ROOT)


class Graph:
    def __init__(self, roots: tuple[Contract, ...]) -> None:
        if not roots:
            raise ValueError("at least one contract is required")
        self.nodes: dict[str, Contract] = {}
        self.parents: dict[str, tuple[str, ...]] = {}
        self.fingerprints: dict[str, str] = {}
        self.descriptors: dict[str, dict] = {}
        self.order: list[str] = []
        visiting: list[str] = []
        owners: dict[Path, str] = {}

        def visit(node: Contract) -> None:
            key = node.key
            if key in visiting:
                cycle = visiting[visiting.index(key) :] + [key]
                raise ValueError(
                    "dependency cycle: "
                    + " -> ".join(self.nodes[k].label for k in cycle)
                )
            dependencies = node.dependencies()
            if not all(isinstance(dep, Contract) for dep in dependencies):
                raise TypeError(
                    f"{node.label}: dependencies must be Contract instances"
                )
            parents = tuple(dict.fromkeys(dep.key for dep in dependencies))
            outputs = tuple(path.resolve() for path in node.outputs())
            if not outputs or len(set(outputs)) != len(outputs):
                raise ValueError(f"{node.label}: declare distinct output files")
            source_hashes = {
                str(path.resolve()): file_sha256(path) for path in node.sources()
            }
            descriptor = {
                "parameters": node.parameters(),
                "dependencies": parents,
                "inputs": [str(p.resolve()) for p in node.inputs()],
                "outputs": [str(p) for p in outputs],
                "resources": sorted(node.resources()),
                "sources": source_hashes,
                "remote_required": node.remote_required,
                "unavailable_reason": node.unavailable_reason,
            }
            if key in self.nodes:
                if descriptor != self.descriptors[key]:
                    raise ValueError(
                        f"conflicting declarations for the same contract identity: {node.label}"
                    )
                return
            for output in outputs:
                if output in owners:
                    raise ValueError(
                        f"output collision: {output}; use separate output directories for different configurations"
                    )
                owners[output] = key
            self.nodes[key] = node
            self.parents[key] = parents
            self.descriptors[key] = descriptor
            self.fingerprints[key] = digest(
                {
                    "identity": key,
                    "dependencies": parents,
                    "sources": {portable(p): file_sha256(p) for p in node.sources()},
                    "outputs": [portable(p) for p in outputs],
                    "input_count": len(node.inputs()),
                }
            )
            visiting.append(key)
            for dependency in dependencies:
                visit(dependency)
            visiting.pop()
            self.order.append(key)

        for root in roots:
            visit(root)
        self.roots = tuple(dict.fromkeys(root.key for root in roots))
