import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims.planning_ahead import PlanningAhead
from claims.receiver_replacement import ReceiverReplacement
from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.bpe_reference import BPEReference
from pipeline.external_reference import ExternalReference
from pipeline.flops import FlopAccounting


class RemoteHypothesis(ResultBundle):
    result_name = "remote-hypothesis"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return (
            *(
                BPEReference(self.context, replicate, "mauve")
                for replicate in range(self.context.settings["baseline_seeds"])
            ),
            ExternalReference(self.context, "mauve"),
            ReceiverReplacement(self.context),
            PlanningAhead(self.context),
            FlopAccounting(self.context),
        )

    def measurements(self) -> dict[str, Path]:
        root = self.context.output
        baseline = {}
        for replicate in range(self.context.settings["baseline_seeds"]):
            measured = self.measured_root(replicate) / "baseline"
            for name in ("evaluation", "generation", "mauve"):
                baseline[f"seed-{replicate}/bpe-{name}"] = measured / f"{name}.json"
            baseline[f"seed-{replicate}/bpe-training"] = (
                self.seed_root(replicate) / "baseline/model.metrics.json"
            )
        external = ExternalReference(self.context, "mauve").root
        return {
            "replacement": self.claim_path("receiver-replacement"),
            "planning": self.claim_path("planning-ahead"),
            "design": root / "design.json",
            "training-flops": root / "flops.json",
            **baseline,
            **{
                f"external-{name}": external / f"{name}.json"
                for name in ("evaluation", "generation", "mauve")
            },
            **{
                f"seed-{replicate}/fixed-exposure": root
                / "exposure"
                / f"seed-{replicate}.json"
                for replicate in self.replicates
            },
        }


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
