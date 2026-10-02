import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims.continuous_communication import ContinuousCommunication
from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.interfaces import Interfaces


class LearnedProtocol(ResultBundle):
    result_name = "learned-protocol"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return (ContinuousCommunication(self.context),) + tuple(
            Interfaces(self.context, replicate, "measure")
            for replicate in self.replicates
        )

    def measurements(self) -> dict[str, Path]:
        return {
            "communication-controls": self.claim_path("continuous-communication"),
            **self.per_seed({"interface-sizes": "interfaces/interfaces.json"}),
            **self.measured({"interfaces": "interfaces/evaluation.json"}),
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
