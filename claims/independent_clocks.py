import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contract import Contract
from pipeline._result import ResultBundle
from pipeline.clocks import ClockDistortion
from pipeline.measure_receiver import MeasureReceiver

VARIANTS = ("primary", "fresh-bpe")


class IndependentClocks(ResultBundle):
    result_name = "independent-clocks"
    missing_experiments = ("reporting-split-evaluation",)

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(
            MeasureReceiver(self.context, replicate, variant)
            for replicate in self.replicates
            for variant in VARIANTS
        ) + tuple(
            ClockDistortion(self.context, replicate, "measure")
            for replicate in self.replicates
        )

    def measurements(self) -> dict[str, Path]:
        return self.measured(
            {
                **{variant: f"{variant}/evaluation.json" for variant in VARIANTS},
                "clock-distortion": "clocks/evaluation.json",
            }
        )


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
