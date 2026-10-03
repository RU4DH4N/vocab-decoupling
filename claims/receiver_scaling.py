import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.contract import Contract
from pipeline.measure_receiver import MeasureReceiver
from pipeline.receiver_variants import ExtensionBundle, MeasureVariant, ReceiverSpec

DESIGN = "design"
POINTS: dict[int, tuple[int | str, ...]] = {
    20_000: (DESIGN, 40_000, 80_000),
    10_000_000: (625_000, DESIGN, 2_500_000, 5_000_000),
    20_000_000: (625_000, 1_250_000, DESIGN, 5_000_000),
    40_000_000: (625_000, 1_250_000, 2_500_000, DESIGN),
}


class ReceiverScaling(ExtensionBundle):
    result_name = "receiver-scaling"

    def points(self) -> tuple[int | str, ...]:
        trunk = self.context.settings["trunk_parameters"]
        if trunk not in POINTS:
            raise ValueError(f"no receiver sizes are declared for a {trunk} trunk")
        return POINTS[trunk]

    def node(self, replicate: int, point: int | str) -> tuple[str, Contract]:
        if isinstance(point, str):
            design = MeasureReceiver(self.context, replicate, "fresh-byte")
            return f"seed-{replicate}/{point}", design
        tokenizer = self.context.output / "receiver-tokenizer.json"
        spec = ReceiverSpec(f"bytes-{point}", point, tokenizer, "fresh-byte", None)
        return f"seed-{replicate}/{spec.label}", MeasureVariant(
            self.context, replicate, spec
        )

    def nodes(self) -> dict[str, Contract]:
        return dict(
            self.node(replicate, point)
            for replicate in self.replicates
            for point in self.points()
        )


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
