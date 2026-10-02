import json
import os
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path

from framework.checkpoints import file_sha256, write_json
from orchestrator.artifacts import Artifacts
from orchestrator.context import Context
from orchestrator.contract import Contract
from orchestrator.graph import Graph, portable
from orchestrator.locks import Lock, resource_lock


class DependencyError(RuntimeError):
    pass


class Orchestrator:
    def __init__(
        self,
        roots: list[Contract],
        state_dir: Path,
        *,
        workers: int = 1,
        slots: int = 1,
        run_name: str = "contracts",
        context: Context | None = None,
    ) -> None:
        if workers < 1 or slots < 1:
            raise ValueError("workers and slots must be positive")
        self.graph = Graph(tuple(roots))
        self.state_dir, self.workers, self.run_name = state_dir, workers, run_name
        self.slots = slots
        self.context = context
        self.artifacts = Artifacts(self.graph, state_dir)
        self.states = {}
        self.started = None
        self.finished = None

    def inspect(self) -> dict[str, dict]:
        statuses, records = self.artifacts.inspect()
        self.states = {
            key: {
                "state": "cached" if ready else "pending",
                "reason": reason,
                "started": None,
                "finished": None,
            }
            for key, (ready, reason) in statuses.items()
        }
        for key, node in self.graph.nodes.items():
            if node.unavailable_reason:
                self.states[key]["state"] = "blocked"
        for key in self.graph.order:
            if any(
                self.states[p]["state"] == "blocked" for p in self.graph.parents[key]
            ):
                self.states[key].update(
                    state="blocked", reason="dependency unavailable"
                )
        return records

    def snapshot(self) -> dict:
        return {
            "schema": "contract-graph-v1",
            "path_base": "repository",
            "roots": list(self.graph.roots),
            "started": self.started,
            "finished": self.finished,
            "nodes": [
                {
                    "id": k,
                    "label": self.graph.nodes[k].label,
                    "contract": f"{type(self.graph.nodes[k]).__module__}.{type(self.graph.nodes[k]).__name__}",
                    "parameters": self.graph.nodes[k].parameters(),
                    "outputs": [portable(p) for p in self.graph.nodes[k].outputs()],
                    "resources": sorted(self.graph.nodes[k].resources()),
                    **self.states[k],
                }
                for k in self.graph.order
            ],
            "edges": [
                {"dependency": p, "consumer": k}
                for k in self.graph.order
                for p in self.graph.parents[k]
            ],
        }

    def export(self, path: Path) -> None:
        write_json(path, self.snapshot())

    def _publish(self) -> None:
        self.export(self.state_dir / "graph.json")
        directory = os.environ.get("VDR_PROGRESS_DIR")
        if directory:
            write_json(
                Path(directory) / f"{self.run_name}.json",
                {
                    "claim": self.run_name,
                    "started": self.started,
                    "updated": time.time(),
                    "finished": self.finished,
                    "error": "; ".join(
                        s["reason"]
                        for s in self.states.values()
                        if s["state"] == "failed"
                    )
                    or None,
                    "tasks": [
                        {
                            "name": self.graph.nodes[k].label,
                            **s,
                        }
                        for k, s in self.states.items()
                    ],
                },
            )

    def _record_timing(self, key: str) -> None:
        path = self.state_dir / "timings.json"
        timings = json.loads(path.read_text()) if path.exists() else {}
        node, state = self.graph.nodes[key], self.states[key]
        timings[key] = {
            "label": node.label,
            "contract": f"{type(node).__module__}.{type(node).__name__}",
            "parameters": node.parameters(),
            "outputs": [portable(p) for p in node.outputs()],
            "seconds": state["finished"] - state["started"],
        }
        write_json(path, timings)

    def _execute(self, key: str) -> None:
        node = self.graph.nodes[key]
        missing = [portable(p) for p in node.inputs() if not p.is_file()]
        if missing:
            raise DependencyError(f"{node.label}: missing inputs {missing}")
        source_hashes = self.graph.descriptors[key]["sources"]
        if any(
            file_sha256(Path(path)) != value for path, value in source_hashes.items()
        ):
            raise DependencyError(
                f"{node.label}: source changed after planning; rebuild the graph"
            )
        inputs_before = [file_sha256(p) for p in node.inputs()]
        self.artifacts.path(key).unlink(missing_ok=True)
        for output in node.outputs():
            output.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            for resource in sorted(node.resources()):
                stack.enter_context(resource_lock(resource))
            if self.context is not None:
                self.context.verify_snapshot()
            node.run({p: self.graph.nodes[p] for p in self.graph.parents[key]})
            if self.context is not None:
                self.context.verify_snapshot()
        if inputs_before != [file_sha256(p) for p in node.inputs()] or any(
            file_sha256(Path(path)) != value for path, value in source_hashes.items()
        ):
            raise DependencyError(
                f"{node.label}: inputs or source changed during execution; result not committed"
            )
        missing = [portable(p) for p in node.outputs() if not p.is_file()]
        if missing:
            raise DependencyError(f"{node.label} returned without producing {missing}")

    def run(
        self,
        *,
        mode: str = "prompt",
        allow_remote: bool = False,
        interactive: bool = False,
        ask: Callable[[str], str] = input,
        tell: Callable[[str], object] = print,
    ) -> dict:
        if mode not in {"prompt", "run-missing", "require-complete"}:
            raise ValueError(f"unknown execution mode: {mode}")
        with Lock(self.state_dir / "run.lock"):
            records = self.inspect()
            self.started = time.time()
            self._publish()
            runnable = self._authorise(mode, allow_remote, interactive, ask, tell)
            if self.context is not None and runnable:
                write_json(self.context.config_snapshot, self.context.settings)
            _Execution(self, records).run()
            self.finished = time.time()
            self._publish()
            failed = [
                self.graph.nodes[k].label
                for k in self.graph.roots
                if self.states[k]["state"] not in {"done", "cached"}
            ]
            if failed:
                raise DependencyError(
                    f"incomplete roots: {failed}; see {self.state_dir / 'graph.json'}"
                )
            return self.snapshot()

    def _authorise(
        self,
        mode: str,
        allow_remote: bool,
        interactive: bool,
        ask: Callable[[str], str],
        tell: Callable[[str], object],
    ) -> list[str]:
        for key in self.graph.order:
            state = self.states[key]
            tell(
                f"{state['state']:>8}  {self.graph.nodes[key].label}: {state['reason']}"
            )
        pending = [k for k in self.graph.order if self.states[k]["state"] != "cached"]
        if pending and mode == "require-complete":
            raise DependencyError(
                "missing/stale/blocked contracts; no computation was run"
            )
        runnable = [k for k in pending if self.states[k]["state"] != "blocked"]
        if runnable and mode == "prompt":
            if not interactive:
                raise DependencyError(
                    "non-interactive invocation requires --run-missing or --require-complete"
                )
            answer = ask(f"Run {len(runnable)} missing/stale contracts? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                raise DependencyError("execution declined")
        if not allow_remote and any(
            self.graph.nodes[k].remote_required for k in runnable
        ):
            raise DependencyError(
                "remote contracts require explicit --allow-remote permission"
            )
        return runnable


class _Execution:
    def __init__(self, orchestrator: Orchestrator, records: dict[str, dict]) -> None:
        self.orchestrator, self.records = orchestrator, records
        self.graph, self.states = orchestrator.graph, orchestrator.states
        self.active: dict[Future[None], str] = {}
        self.held: set[str] = set()
        self.load = 0
        self.alone: set[str] = set()
        self.shared: set[str] = set()

    def weight(self, key: str) -> int:
        slots = self.orchestrator.slots
        if key in self.alone:
            return slots
        return min(self.graph.nodes[key].weight(), slots)

    def run(self) -> None:
        with ThreadPoolExecutor(max_workers=self.orchestrator.workers) as pool:
            while True:
                for key in self.graph.order:
                    if self.states[key]["state"] == "pending":
                        self.advance(key, pool)
                self.orchestrator._publish()
                if not self.active:
                    return
                done, _ = wait(self.active, return_when=FIRST_COMPLETED)
                for future in done:
                    self.settle(future)

    def advance(self, key: str, pool: ThreadPoolExecutor) -> None:
        parents = [self.states[p]["state"] for p in self.graph.parents[key]]
        if any(s in {"failed", "blocked"} for s in parents):
            self.mark(key, "blocked", "dependency failed or unavailable")
            return
        if not self.ready(key, parents):
            return
        reused = self.orchestrator.artifacts.reusable(key, self.records)
        if reused is not None:
            self.records[key] = reused
            self.mark(key, "cached", "dependency outputs unchanged")
            return
        self.start(key, pool)

    def ready(self, key: str, parents: list[str]) -> bool:
        return (
            len(self.active) < self.orchestrator.workers
            and not self.graph.nodes[key].resources() & self.held
            and self.load + self.weight(key) <= self.orchestrator.slots
            and all(s in {"cached", "done"} for s in parents)
        )

    def start(self, key: str, pool: ThreadPoolExecutor) -> None:
        self.held.update(self.graph.nodes[key].resources())
        if self.weight(key):
            company = [k for k in self.active.values() if self.weight(k)]
            if company:
                self.shared.update((key, *company))
        self.load += self.weight(key)
        self.states[key].update(state="running", reason="", started=time.time())
        self.active[pool.submit(self.orchestrator._execute, key)] = key

    def settle(self, future: Future[None]) -> None:
        key = self.active.pop(future)
        self.held.difference_update(self.graph.nodes[key].resources())
        self.load -= self.weight(key)
        try:
            future.result()
            self.records[key] = self.orchestrator.artifacts.commit(key, self.records)
        except Exception as error:
            if key in self.shared and key not in self.alone:
                self.alone.add(key)
                self.states[key].update(
                    state="pending",
                    reason=f"retrying alone after sharing the accelerator: {type(error).__name__}",
                )
                return
            self.mark(key, "failed", f"{type(error).__name__}: {error}")
            return
        self.mark(key, "done", "")
        self.orchestrator._record_timing(key)

    def mark(self, key: str, state: str, reason: str) -> None:
        self.states[key].update(state=state, reason=reason, finished=time.time())
