import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.receiver_tokenizers import tokenizer_path
from orchestrator.contract import Contract
from pipeline.measure_receiver import MeasureReceiver
from pipeline.receiver_variants import ExtensionBundle, MeasureVariant, ReceiverSpec

VOCABULARIES = (512, 2048, 8192)
DESIGN_BODIES = {
    20_000: 8_000,
    40_000_000: 5_000_000,
}


class ReceiverVocabulary(ExtensionBundle):
    result_name = "receiver-vocabulary"

    def body(self) -> int:
        trunk = self.context.settings["trunk_parameters"]
        if trunk not in DESIGN_BODIES:
            raise ValueError(f"no receiver body is declared for a {trunk} trunk")
        return DESIGN_BODIES[trunk]

    def references(self, replicate: int) -> dict[str, Contract]:
        return {
            f"seed-{replicate}/{name}": MeasureReceiver(
                self.context, replicate, variant
            )
            for name, variant in (("bytes", "fresh-byte"), ("bpe-design", "fresh-bpe"))
        }

    def node(self, replicate: int, vocabulary: int) -> tuple[str, Contract]:
        tokenizer = tokenizer_path(self.context.output, vocabulary)
        label = f"bpe-{vocabulary}"
        spec = ReceiverSpec(label, self.body(), tokenizer, "fresh-bpe", vocabulary)
        return f"seed-{replicate}/{label}", MeasureVariant(
            self.context, replicate, spec
        )

    def nodes(self) -> dict[str, Contract]:
        nodes: dict[str, Contract] = {}
        for replicate in self.replicates:
            nodes.update(self.references(replicate))
            nodes.update(self.node(replicate, size) for size in VOCABULARIES)
        return nodes


if __name__ == "__main__":
    from orchestrator.cli import entrypoint

    entrypoint(__file__)
