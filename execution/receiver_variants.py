import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data.protocol_corpus import EventCorpus
from data.tokenizers import load_tokenizer
from execution.design import _body_parameters, _closest, _language_model
from execution.generation import SymbolBytes, generate_protocol
from execution.layout import Layout
from execution.protocol_io import check_inventory, evaluation_model, load_protocol
from execution.sampling import GenerationLimits
from execution.streams import (
    Shard,
    device_streams,
    generate_streams,
    prompt_count,
    summary,
)
from execution.sweep import sweep_native
from execution.training import encoded, native, stage_path, stages, train_stage
from execution.validation import validate
from execution.windows import score_windows
from framework.checkpoints import file_sha256, write_json
from models import conventions

VARIANTS = ("fresh-byte", "fresh-bpe")


@cache
def shape_for(target_parameters: int, context: int) -> tuple[int, int]:
    return _closest(
        target_parameters,
        lambda *s: _body_parameters(_language_model(s, context)),
    )


@dataclass(frozen=True)
class ReceiverLayout(Layout):
    label: str
    target_parameters: int
    tokenizer: Path
    variant: str

    @property
    def own_sweep(self) -> bool:
        return self.variant == "fresh-byte"

    @property
    def run(self) -> Path:
        return self.shared / f"seed-{self.replicate}" / "extensions" / self.label

    @property
    def receiver_tokenizer(self) -> Path:
        return self.tokenizer

    def sweep(self, name: str) -> Path:
        if name == "native" and self.own_sweep:
            return self.run / "sweep" / "native.json"
        return super().sweep(name)

    def config(self) -> dict:
        config = super().config()
        context = config["native"]["max_seq_len"]
        resized = _language_model(shape_for(self.target_parameters, context), context)
        interface = resized_interface(config["interface"], resized["d_model"])
        return {**config, "native": resized, "interface": interface}

    def describe(self, config: dict) -> dict:
        return {
            "label": self.label,
            "target_parameters": self.target_parameters,
            "native": config["native"],
            "interface": config["interface"],
            "tokenizer_sha256": file_sha256(self.tokenizer),
        }


def resized_interface(interface: dict, width: int) -> dict:
    return {
        **interface,
        "score_dimensions": min(conventions.HEAD_DIMENSIONS, width),
        "communication_rank": max(1, width // 4),
    }


def link_primary(layout: ReceiverLayout) -> None:
    trunk = layout.shared / f"seed-{layout.replicate}" / "primary"
    link = layout.run / "primary"
    if not link.is_symlink() and not link.exists():
        layout.run.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path("..") / ".." / "primary", target_is_directory=True)
    if link.resolve() != trunk.resolve():
        raise ValueError("extension primary link points at another trunk")


def sweep(layout: ReceiverLayout, device: torch.device) -> None:
    design = json.loads(layout.design.read_text())
    config = layout.config()
    validate(config)
    started = time.perf_counter()
    results, selected = sweep_native(layout, design, config, device)
    write_json(
        layout.sweep("native"),
        {
            "sweep": "native",
            "replicate": layout.replicate,
            "split": "selection",
            "steps": design["steps"]["sweep"],
            "receiver": layout.describe(config),
            "results": results,
            "selected": selected,
            "seconds": time.perf_counter() - started,
        },
    )


def final_stage(layout: ReceiverLayout, variant: str) -> Path:
    return stage_path(layout, variant, len(stages(layout.config(), variant)) - 1)


def paired_rows(combined: dict) -> dict:
    return {
        name: {field: value.tolist() for field, value in asdict(scores).items()}
        for name, scores in combined.items()
    }


def measure(
    layout: ReceiverLayout, variant: str, device: torch.device, split: str
) -> None:
    config = layout.config()
    validate(config)
    path = final_stage(layout, variant)
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    check_inventory(path, corpus, split)
    options = config["generation"]
    records, timed = generate_streams(
        generate_variant,
        (
            layout.shared,
            layout.replicate,
            layout.label,
            layout.target_parameters,
            layout.tokenizer,
            variant,
            split,
            str(device),
        ),
        prompt_count(layout.evaluation_prompts(split)),
        options["latency_samples"],
        device_streams(options, device),
    )
    output = layout.measured(split) / variant
    write_json(
        output / "generation.json",
        {
            "checkpoint_sha256": file_sha256(path),
            **summary(records, timed, layout.evaluation_prompts(split), options),
        },
    )
    model = evaluation_model(path, corpus, split, device)
    table = encoded(config, layout, variant, corpus)
    combined, group_ids = score_windows(
        model, table, corpus, config, layout.evaluation_batches(config, split), device
    )
    write_json(
        output / "evaluation.json",
        {
            "checkpoint_sha256": file_sha256(path),
            "likelihood_domain": "canonical-path",
            "split": split,
            "receiver": layout.describe(config),
            "controls": "inference ablations, not separately fitted controls",
            "scores": {name: value.aggregate() for name, value in combined.items()},
            "paired_rows": paired_rows(combined),
            "group_ids": group_ids,
        },
    )


def generation_limits(options: dict) -> GenerationLimits:
    keys = ("bytes", "symbols", "event_bytes", "temperature", "top_p")
    return GenerationLimits(**{key: options[key] for key in keys})


def encode_prompt(units: list[str], tokenizer: Tokenizer | None) -> list[list[int]]:
    if tokenizer is None:
        return [list(unit.encode()) for unit in units]
    return [tokenizer.encode(unit).ids for unit in units]


@torch.no_grad()
def generate_variant(
    shared: Path,
    replicate: int,
    label: str,
    target_parameters: int,
    tokenizer_path: Path,
    variant: str,
    split: str,
    device: str,
    indices: list[int],
) -> Shard:
    layout = ReceiverLayout(
        shared, replicate, label, target_parameters, tokenizer_path, variant
    )
    config = layout.config()
    model, _ = load_protocol(
        final_stage(layout, variant), torch.device(device), table=False
    )
    kind = config["receivers"][variant]["kind"]
    tokenizer = load_tokenizer(layout.receiver_tokenizer) if kind == "bpe" else None
    symbols = SymbolBytes.bytes() if tokenizer is None else SymbolBytes.bpe(tokenizer)
    limits = generation_limits(config["generation"])
    examples = json.loads(layout.evaluation_prompts(split).read_text())
    records = []
    for index in indices:
        example = examples[index]
        prompt = encode_prompt(example["prompt_units"], tokenizer)
        record = generate_protocol(
            model,
            symbols,
            prompt,
            limits,
            seed=config["evaluation_seed"] + index,
            hard=False,
        )
        records.append((index, {**example, **record}))
        print(
            f"receiver={label}/{variant} generation={index + 1}/{len(examples)}",
            flush=True,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sweep", "native", "stage", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--target-parameters", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--split", choices=("selection", "reporting"))
    args = parser.parse_args()
    layout = ReceiverLayout(
        args.output,
        args.replicate,
        args.label,
        args.target_parameters,
        args.tokenizer,
        args.variant,
    )
    schedule = len(stages(layout.config(), args.variant))
    if args.task == "sweep" and not layout.own_sweep:
        parser.error("only byte receivers have a per-size native sweep")
    if args.task == "stage" and (args.stage is None or not 0 <= args.stage < schedule):
        parser.error("stage index is required and must index the declared schedule")
    if args.task == "measure" and args.split is None:
        parser.error("split is required")
    device = torch.device(args.device)
    link_primary(layout)
    {
        "sweep": lambda: sweep(layout, device),
        "native": lambda: native(layout, args.variant, device),
        "stage": lambda: train_stage(layout, args.variant, args.stage, device),
        "measure": lambda: measure(layout, args.variant, device, args.split),
    }[args.task]()


if __name__ == "__main__":
    main()
