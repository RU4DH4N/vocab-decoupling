import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims.continuous_communication import ContinuousCommunication
from claims.correspondence_bootstrap import CorrespondenceBootstrap
from claims.independent_clocks import IndependentClocks
from claims.learned_protocol import LearnedProtocol
from claims.planning_ahead import PlanningAhead
from claims.receiver_replacement import ReceiverReplacement
from claims.remote_hypothesis import RemoteHypothesis
from pipeline._result import ResultBundle


class Active(ResultBundle):
    result_name = "active"

    def dependencies(self) -> tuple[ResultBundle, ...]:
        return tuple(
            contract(self.context)
            for contract in (
                ContinuousCommunication,
                LearnedProtocol,
                CorrespondenceBootstrap,
                IndependentClocks,
                PlanningAhead,
                ReceiverReplacement,
                RemoteHypothesis,
            )
        )

    def measurements(self) -> dict[str, Path]:
        return {
            dependency.result_name: dependency.outputs()[0]
            for dependency in self.dependencies()
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
