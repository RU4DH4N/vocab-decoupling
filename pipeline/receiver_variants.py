from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from execution.receiver_tokenizers import tokenizer_path
from execution.receiver_variants import VARIANTS
from framework.paths import portable_path
from orchestrator.context import Context
from orchestrator.contract import Contract
from pipeline._base import EXCLUSIVE, ROOT, PipelineContract, SeedContract
from pipeline._measure import MeasureContract
from pipeline._result import ResultBundle
from pipeline.preflight import Preflight
from pipeline.prepare import Prepare
from pipeline.sweep import Sweep
from pipeline.train_stage import TrainStage, schedule


@dataclass(frozen=True)
class ReceiverSpec:
    label: str
    target_parameters: int
    tokenizer: Path
    variant: str
    vocabulary: int | None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"extension receivers must be one of {VARIANTS}")
        if (self.vocabulary is not None) != (self.variant == "fresh-bpe"):
            raise ValueError("only BPE receivers declare a vocabulary")
        if not self.label or "/" in self.label:
            raise ValueError("receiver label must be a nonempty folder name")
        if self.target_parameters <= 0:
            raise ValueError("target_parameters must be positive")

    def describe(self) -> dict:
        return {
            "label": self.label,
            "target_parameters": self.target_parameters,
            "tokenizer": portable_path(self.tokenizer, root=ROOT),
            "variant": self.variant,
            "vocabulary": self.vocabulary,
        }

    def arguments(self) -> tuple[object, ...]:
        return (
            "--label",
            self.label,
            "--target-parameters",
            self.target_parameters,
            "--tokenizer",
            self.tokenizer,
            "--variant",
            self.variant,
        )


class TrainReceiverTokenizer(PipelineContract):
    module = "execution.receiver_tokenizers"
    uses_device = False

    def __init__(self, context: Context, vocabulary: int) -> None:
        super().__init__(context)
        if vocabulary <= 256:
            raise ValueError("a byte-level BPE vocabulary must exceed the 256 bytes")
        self.vocabulary = vocabulary

    @property
    def label(self) -> str:
        return f"TrainReceiverTokenizer[{self.vocabulary}]"

    def parameters(self) -> dict:
        return {**super().parameters(), "vocabulary": self.vocabulary}

    def dependencies(self) -> tuple[Contract, ...]:
        return (Prepare(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        return (tokenizer_path(self.context.output, self.vocabulary),)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.worker("--output", self.context.output, "--vocabulary", self.vocabulary)


class VariantContract(SeedContract):
    module = "execution.receiver_variants"

    def __init__(self, context: Context, replicate: int, spec: ReceiverSpec) -> None:
        super().__init__(context, replicate)
        self.spec = spec

    @property
    def extension_root(self) -> Path:
        return self.run_root / "extensions" / self.spec.label

    def parameters(self) -> dict:
        return {**super().parameters(), "extension": self.spec.describe()}

    def accelerator_weight(self) -> int:
        return EXCLUSIVE if self.spec.variant == "fresh-bpe" else 1


class SweepVariantNative(VariantContract):
    @property
    def label(self) -> str:
        return f"SweepVariantNative[seed {self.replicate}:{self.spec.label}]"

    def dependencies(self) -> tuple[Contract, ...]:
        return (Preflight(self.context),)

    def outputs(self) -> tuple[Path, ...]:
        return (self.extension_root / "sweep" / "native.json",)

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.seed_worker("sweep", *self.spec.arguments())


class TrainVariantNative(VariantContract):
    @property
    def label(self) -> str:
        return (
            f"TrainVariantNative[seed {self.replicate}:"
            f"{self.spec.label}:{self.spec.variant}]"
        )

    def dependencies(self) -> tuple[Contract, ...]:
        if self.spec.vocabulary is None:
            return (SweepVariantNative(self.context, self.replicate, self.spec),)
        return (
            Sweep(self.context, "native"),
            TrainReceiverTokenizer(self.context, self.spec.vocabulary),
        )

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.extension_root / self.spec.variant / f"native{suffix}"
            for suffix in (".pt", ".json", ".metrics.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.seed_worker("native", *self.spec.arguments())


class TrainVariantStage(VariantContract):
    def __init__(
        self, context: Context, replicate: int, spec: ReceiverSpec, index: int
    ) -> None:
        super().__init__(context, replicate, spec)
        if not 0 <= index < len(schedule(spec.variant)):
            raise ValueError("stage index outside the declared schedule")
        self.index = index

    @classmethod
    def final(
        cls, context: Context, replicate: int, spec: ReceiverSpec
    ) -> "TrainVariantStage":
        return cls(context, replicate, spec, len(schedule(spec.variant)) - 1)

    @property
    def label(self) -> str:
        return (
            f"TrainVariantStage[seed {self.replicate}:{self.spec.label}:"
            f"{self.spec.variant}:{self.index}:{schedule(self.spec.variant)[self.index]}]"
        )

    def parameters(self) -> dict:
        return {**super().parameters(), "stage_index": self.index}

    def dependencies(self) -> tuple[Contract, ...]:
        sweep = Sweep(self.context, "adapter")
        if self.index:
            previous = TrainVariantStage(
                self.context, self.replicate, self.spec, self.index - 1
            )
            return (previous, sweep)
        return (
            TrainVariantNative(self.context, self.replicate, self.spec),
            TrainStage.final(self.context, self.replicate, "primary"),
            sweep,
        )

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.extension_root / self.spec.variant / f"stage-{self.index}{suffix}"
            for suffix in (".pt", ".json", ".metrics.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.seed_worker("stage", *self.spec.arguments(), "--stage", self.index)


class MeasureVariant(VariantContract, MeasureContract):
    @property
    def label(self) -> str:
        return (
            f"MeasureVariant[seed {self.replicate}:{self.spec.label}:"
            f"{self.spec.variant}:{self.context.split}]"
        )

    @property
    def measured_folder(self) -> Path:
        if self.context.split == "selection":
            return self.extension_root / self.spec.variant
        return self.extension_root / "reporting" / self.spec.variant

    def dependencies(self) -> tuple[Contract, ...]:
        return (
            TrainVariantStage.final(self.context, self.replicate, self.spec),
            *self.split_dependencies(),
        )

    def outputs(self) -> tuple[Path, ...]:
        return tuple(
            self.measured_folder / name
            for name in ("evaluation.json", "generation.json")
        )

    def run(self, dependencies: Mapping[str, Contract]) -> None:
        self.measure_worker("measure", *self.spec.arguments())


class ExtensionBundle(ResultBundle):
    missing_experiments = (
        "reporting-split-evaluation",
        "independent-seed-replication",
    )

    @property
    def replicates(self) -> range:
        return range(1)

    @abstractmethod
    def nodes(self) -> dict[str, Contract]:
        pass

    def dependencies(self) -> tuple[Contract, ...]:
        return tuple(self.nodes().values())

    def measurements(self) -> dict[str, Path]:
        return {
            f"{name}-{output.stem}": output
            for name, node in self.nodes().items()
            for output in node.outputs()
        }
