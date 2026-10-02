import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.mauve_receiver import MauveReceiver
from pipeline.planning import Planning


class PlanningAhead(ResultBundle):
    result_name = "planning-ahead"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(
            contract
            for replicate in self.replicates
            for contract in (
                Planning(self.context, replicate, "mauve"),
                MauveReceiver(self.context, replicate, "primary"),
            )
        )

    def measurements(self) -> dict[str, Path]:
        return {
            **self.measured(
                {
                    "future-interventions": "planning/evaluation.json",
                    "future-generation": "planning/generation.json",
                    "future-mauve": "planning/mauve.json",
                    "current-only": "primary/evaluation.json",
                    "current-only-generation": "primary/generation.json",
                    "current-only-mauve": "primary/mauve.json",
                }
            ),
            **self.per_seed(
                {
                    "planner-fit": "planning/planner.metrics.json",
                    "future-fit": "planning/lookahead.metrics.json",
                    "oracle-future-fit": "planning/lookahead-oracle.metrics.json",
                }
            ),
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
