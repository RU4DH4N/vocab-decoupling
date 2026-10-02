import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from execution.design import SWEEPS, replicate_config

T = TypeVar("T")


@dataclass(frozen=True)
class Layout:
    shared: Path
    replicate: int

    @property
    def run(self) -> Path:
        return self.shared / f"seed-{self.replicate}"

    @property
    def corpus(self) -> Path:
        return self.shared / "corpus"

    @property
    def design(self) -> Path:
        return self.shared / "design.json"

    @property
    def prompts(self) -> Path:
        return self.shared / "prompts.json"

    @property
    def receiver_tokenizer(self) -> Path:
        return self.shared / "receiver-tokenizer.json"

    @property
    def baseline_tokenizer(self) -> Path:
        return self.shared / "baseline-tokenizer.json"

    @property
    def exposure(self) -> Path:
        return self.shared / "exposure" / f"seed-{self.replicate}.json"

    @property
    def reporting(self) -> Path:
        return self.shared / "reporting"

    def evaluation_corpus(self, split: str) -> Path:
        return self.corpus if split == "selection" else self.reporting / "corpus"

    def evaluation_prompts(self, split: str) -> Path:
        return self.prompts if split == "selection" else self.reporting / "prompts.json"

    def evaluation_batches(self, config: dict, split: str) -> int:
        if split == "selection":
            return config["selection_batches"]
        return json.loads((self.reporting / "design.json").read_text())[
            "evaluation_batches"
        ]

    def external(self, split: str) -> Path:
        return (self.shared if split == "selection" else self.reporting) / "external"

    def measured(self, split: str) -> Path:
        return self.run if split == "selection" else self.run / "reporting"

    def sweep(self, name: str) -> Path:
        return self.shared / "sweep" / f"{name}.json"

    def config(self) -> dict:
        design = json.loads(self.design.read_text())
        selected = {
            name: json.loads(self.sweep(name).read_text())["selected"]
            for name in SWEEPS
            if self.sweep(name).exists()
        }
        return replicate_config(design, self.replicate, selected)


def required(value: T | None, name: str) -> T:
    if value is None:
        raise ValueError(f"{name} has not been selected; run its sweep first")
    return value
