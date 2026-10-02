import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from data.protocol_corpus import (
    EventCorpus,
)
from data.tokenizers import load_tokenizer
from execution.generation import SymbolBytes, generate_protocol
from execution.layout import Layout
from execution.mauve import mauve_score
from execution.protocol_io import check_inventory, evaluation_model, load_protocol
from execution.sampling import GenerationLimits
from execution.streams import (
    Shard,
    device_streams,
    generate_streams,
    prompt_count,
    summary,
)
from execution.training import encoded, stage_path, stages
from execution.validation import validate
from execution.windows import score_windows
from framework.checkpoints import file_sha256, write_json


def measure(layout: Layout, variant: str, device: torch.device, split: str) -> None:
    config = layout.config()
    validate(config)
    path = stage_path(layout, variant, len(stages(config, variant)) - 1)
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    check_inventory(path, corpus, split)
    options = config["generation"]
    records, timed = generate_streams(
        generate_receiver,
        (layout.shared, layout.replicate, variant, split, str(device)),
        prompt_count(layout.evaluation_prompts(split)),
        options["latency_samples"],
        device_streams(options, device),
    )
    write_json(
        layout.measured(split) / variant / "generation.json",
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
    aggregated = {name: value.aggregate() for name, value in combined.items()}
    raw_scores = {
        name: {field: value.tolist() for field, value in asdict(scores).items()}
        for name, scores in combined.items()
    }
    output = layout.measured(split) / variant
    write_json(
        output / "evaluation.json",
        {
            "checkpoint_sha256": file_sha256(path),
            "likelihood_domain": "canonical-path",
            "split": split,
            "controls": "inference ablations, not separately fitted controls",
            "scores": aggregated,
            "paired_rows": raw_scores,
            "group_ids": group_ids,
        },
    )


@torch.no_grad()
def generate_receiver(
    shared: Path,
    replicate: int,
    variant: str,
    split: str,
    device: str,
    indices: list[int],
) -> Shard:
    layout = Layout(shared, replicate)
    config = layout.config()
    path = stage_path(layout, variant, len(stages(config, variant)) - 1)
    model, _ = load_protocol(path, torch.device(device), table=False)
    kind = config["receivers"][variant]["kind"]
    tokenizer = load_tokenizer(layout.receiver_tokenizer) if kind == "bpe" else None
    symbols = SymbolBytes.bytes() if tokenizer is None else SymbolBytes.bpe(tokenizer)
    options = config["generation"]
    limits = GenerationLimits(
        **{
            key: options[key]
            for key in ("bytes", "symbols", "event_bytes", "temperature", "top_p")
        }
    )
    examples = json.loads(layout.evaluation_prompts(split).read_text())
    records = []
    for index in indices:
        example = examples[index]
        prompt = [
            list(unit.encode()) if tokenizer is None else tokenizer.encode(unit).ids
            for unit in example["prompt_units"]
        ]
        record = generate_protocol(
            model,
            symbols,
            prompt,
            limits,
            seed=config["evaluation_seed"] + index,
            hard=False,
        )
        records.append((index, {**example, **record}))
        print(f"receiver={variant} generation={index + 1}/{len(examples)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("measure", "mauve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument(
        "--variant", choices=("primary", "fresh-byte", "fresh-bpe"), required=True
    )
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    if args.task == "measure":
        measure(layout, args.variant, torch.device(args.device), args.split)
    else:
        folder = layout.measured(args.split) / args.variant
        mauve_score(
            folder / "generation.json",
            layout.config()["mauve"],
            folder / "mauve.json",
            torch.device(args.device),
        )


if __name__ == "__main__":
    main()
