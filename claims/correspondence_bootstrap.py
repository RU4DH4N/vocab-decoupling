import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.design import PRIMARY_STAGES
from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.bootstrap import Bootstrap
from pipeline.measure_receiver import MeasureReceiver
from pipeline.train_stage import TrainStage


class CorrespondenceBootstrap(ResultBundle):
    result_name = "correspondence-bootstrap"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return (
            *(
                TrainStage(self.context, replicate, "primary", index)
                for replicate in self.replicates
                for index in range(len(PRIMARY_STAGES))
            ),
            *(
                MeasureReceiver(self.context, replicate, "primary")
                for replicate in self.replicates
            ),
            *(
                Bootstrap(self.context, replicate, "measure")
                for replicate in self.replicates
            ),
        )

    def measurements(self) -> dict[str, Path]:
        return {
            **self.per_seed(
                {
                    **{
                        f"stage-{index}-training": f"primary/stage-{index}.metrics.json"
                        for index in range(len(PRIMARY_STAGES))
                    },
                    "arm-trajectories": "bootstrap/trajectories.json",
                }
            ),
            **self.measured(
                {
                    "final-held-out-correspondence": "primary/evaluation.json",
                    "arms": "bootstrap/evaluation.json",
                }
            ),
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
