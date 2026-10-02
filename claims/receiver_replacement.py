import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims.continuous_communication import ARMS, ContinuousCommunication
from execution.design import RECEIVERS
from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.fitted_control import FittedControl
from pipeline.mauve_receiver import MauveReceiver
from pipeline.train_stage import schedule


class ReceiverReplacement(ResultBundle):
    result_name = "receiver-replacement"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return (
            ContinuousCommunication(self.context),
            *(
                MauveReceiver(self.context, replicate, variant)
                for replicate in self.replicates
                for variant in RECEIVERS
            ),
            *(
                FittedControl(self.context, replicate, "fresh-bpe", arm, "measure")
                for replicate in self.replicates
                for arm in ARMS
            ),
        )

    def measurements(self) -> dict[str, Path]:
        measured, training = {}, {}
        for variant in RECEIVERS:
            for name in ("evaluation", "generation", "mauve"):
                measured[f"{variant}-{name}"] = f"{variant}/{name}.json"
            training[f"{variant}-native"] = f"{variant}/native.metrics.json"
            for index in range(len(schedule(variant))):
                training[f"{variant}-stage-{index}-fit"] = (
                    f"{variant}/stage-{index}.metrics.json"
                )
        for arm in ARMS:
            measured[f"bpe-control-{arm}"] = f"controls/fresh-bpe/{arm}/evaluation.json"
            training[f"bpe-control-{arm}-fit"] = (
                f"controls/fresh-bpe/{arm}/model.metrics.json"
            )
        return {
            "byte-fitted-controls": self.claim_path("continuous-communication"),
            **self.measured(measured),
            **self.per_seed(training),
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
