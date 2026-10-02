import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.fitted_control import FittedControl
from pipeline.measure_receiver import MeasureReceiver

ARMS = ("correct", "shuffled", "native-only")


class ContinuousCommunication(ResultBundle):
    result_name = "continuous-communication"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(
            FittedControl(self.context, replicate, "fresh-byte", arm, "measure")
            for replicate in self.replicates
            for arm in ARMS
        ) + tuple(
            MeasureReceiver(self.context, replicate, "fresh-byte")
            for replicate in self.replicates
        )

    def measurements(self) -> dict[str, Path]:
        return {
            **self.measured(
                {
                    **{
                        arm: f"controls/fresh-byte/{arm}/evaluation.json"
                        for arm in ARMS
                    },
                    "native-reference-and-interventions": "fresh-byte/evaluation.json",
                }
            ),
            **self.per_seed(
                {
                    f"{arm}-fit": f"controls/fresh-byte/{arm}/model.metrics.json"
                    for arm in ARMS
                }
            ),
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
