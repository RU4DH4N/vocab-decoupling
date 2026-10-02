import argparse
import json
import math
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from data.protocol_corpus import EventCorpus, selection_windows
from execution.baseline_training import bpe_batch
from execution.generation import SymbolBytes
from execution.layout import Layout
from execution.mauve import mauve_score
from execution.sampling import generation_record, sample
from execution.streams import (
    Shard,
    device_streams,
    generate_streams,
    prompt_count,
    summary,
)
from execution.validation import validate
from framework.checkpoints import write_json
from framework.runtime import autocast_context, device_synchronize


def snapshot(options: dict) -> Path:
    try:
        path = snapshot_download(
            options["model"], revision=options["revision"], local_files_only=True
        )
    except LocalEntryNotFoundError:
        path = snapshot_download(
            options["model"],
            revision=options["revision"],
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
    return Path(path)


def load_external(
    options: dict, device: torch.device
) -> tuple[PreTrainedModel, Tokenizer, int]:
    path = snapshot(options)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.requires_grad_(False).eval()
    nn.Module.to(model, device)
    declared = json.loads((path / "tokenizer_config.json").read_text())
    context = min(
        int(model.config.max_position_embeddings), int(declared["model_max_length"])
    )
    return model, Tokenizer.from_file(str(path / "tokenizer.json")), context


def special_ids(tokenizer: Tokenizer) -> list[int]:
    return sorted(
        index
        for index, token in tokenizer.get_added_tokens_decoder().items()
        if token.special
    )


def logits(
    model: PreTrainedModel, ids: torch.Tensor, config: dict, **kwargs: object
) -> tuple[torch.Tensor, Any]:
    with autocast_context(ids.device, config["precision"]):
        output = model(input_ids=ids, use_cache=bool(kwargs), **kwargs)
    return output.logits.float(), output.past_key_values


def windowed_nats(
    model: PreTrainedModel,
    ids: torch.Tensor,
    targets: torch.Tensor,
    config: dict,
    context: int,
) -> torch.Tensor:
    width = ids.shape[1]
    kept = context - context // 2
    nats = torch.zeros(targets.shape, dtype=torch.float64, device=ids.device)
    start, scored = 0, 0
    while scored < width:
        end = min(start + context, width)
        scores, _ = logits(model, ids[:, start:end], config)
        target = targets[:, start:end].clone()
        target[:, : scored - start] = -100
        nats[:, start:end] += F.cross_entropy(
            scores.flatten(0, 1), target.flatten(), reduction="none"
        ).reshape_as(target)
        scored, start = end, end - kept
    return nats


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    context: int,
    prompt: str,
    config: dict,
    seed: int,
) -> dict:
    options = config["generation"]
    device = next(model.parameters()).device
    device_synchronize(device)
    start = time.perf_counter()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("external generation prompt is empty")
    truncated = len(ids) >= context
    ids = ids[-(context - 1) :]
    kept = context - context // 2
    symbols = SymbolBytes.bpe(tokenizer)
    masked = [i for i in special_ids(tokenizer) if i < model.config.vocab_size]
    scores, cache = logits(
        model, torch.tensor([ids], device=device), config, past_key_values=None
    )
    rng = torch.Generator(device=device).manual_seed(seed)
    output = bytearray()
    reason = "symbol-limit"
    for _ in range(options["symbols"]):
        final = scores[0, -1]
        final[masked] = -torch.inf
        token = sample(final, rng, options["temperature"], options["top_p"])
        output.extend(symbols.values[token])
        ids.append(token)
        if len(output) >= options["bytes"]:
            reason = "byte-budget"
            break
        if cache.get_seq_length() + 1 > context:
            scores, cache = logits(
                model,
                torch.tensor([ids[-kept:]], device=device),
                config,
                past_key_values=None,
            )
        else:
            scores, cache = logits(
                model,
                torch.tensor([[token]], device=device),
                config,
                past_key_values=cache,
            )
    device_synchronize(device)
    return {
        **generation_record(
            bytes(output[: options["bytes"]]),
            time.perf_counter() - start,
            reason != "byte-budget",
        ),
        "stop_reason": reason,
        "prompt_truncated": truncated,
    }


@torch.no_grad()
def measure(config: dict, layout: Layout, device: torch.device, split: str) -> None:
    options = config["external"]
    generation = config["generation"]
    records, timed = generate_streams(
        generate_external,
        (layout.shared, split, str(device)),
        prompt_count(layout.evaluation_prompts(split)),
        generation["latency_samples"],
        device_streams(generation, device),
    )
    report = summary(records, timed, layout.evaluation_prompts(split), generation)
    write_json(
        layout.external(split) / "generation.json",
        {
            **report,
            "model": options,
            "truncated_prompts": sum(r["prompt_truncated"] for r in records),
            "decoding": {
                **report["decoding"],
                "masked": "special tokens",
                "context": "on reaching the context, re-prime with its latter half",
            },
        },
    )
    model, tokenizer, context = load_external(options, device)
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    rng = np.random.default_rng(config["evaluation_seed"])
    nats, sizes, groups, normalised, chunked = [], [], [], [], 0
    for _ in range(layout.evaluation_batches(config, split)):
        rows, group_ids = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        ids, targets = bpe_batch(corpus, rows, tokenizer, device)
        width = targets.shape[1] - targets.ne(-100).flip(-1).int().argmax(-1)
        chunked += int(width.gt(context).sum())
        nats.extend(
            windowed_nats(model, ids, targets, config, context).sum(-1).tolist()
        )
        for units in corpus.decode(rows):
            text = "".join(units)
            sizes.append(len("".join(units[1:]).encode()))
            normalised.append(unicodedata.normalize("NFKC", text) != text)
        groups.extend(group_ids.tolist())
    folder = layout.external(split)
    write_json(
        folder / "evaluation.json",
        {
            "model": options,
            "context": context,
            "scoring": "windows longer than the context are scored in chunks that keep half the context",
            "split": split,
            "likelihood_domain": "canonical path of the model's own BPE over NFKC-normalised text; not text marginal",
            "nats": nats,
            "utf8_bytes": sizes,
            "group_ids": groups,
            "nfkc_changed_windows": sum(normalised),
            "chunked_windows": chunked,
            "canonical_path_bpb": sum(nats) / sum(sizes) / math.log(2),
        },
    )


@torch.no_grad()
def generate_external(
    shared: Path, split: str, device: str, indices: list[int]
) -> Shard:
    layout = Layout(shared, 0)
    config = layout.config()
    model, tokenizer, context = load_external(config["external"], torch.device(device))
    examples = json.loads(layout.evaluation_prompts(split).read_text())
    records = []
    for index in indices:
        example = examples[index]
        record = generate(
            model,
            tokenizer,
            context,
            example["prompt"],
            config,
            config["evaluation_seed"] + index,
        )
        records.append((index, {**example, **record}))
        print(f"external generation={index + 1}/{len(examples)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("measure", "mauve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    args = parser.parse_args()
    layout = Layout(args.output, 0)
    config = layout.config()
    validate(config)
    device = torch.device(args.device)
    if args.task == "measure":
        measure(config, layout, device, args.split)
    else:
        folder = layout.external(args.split)
        mauve_score(
            folder / "generation.json", config["mauve"], folder / "mauve.json", device
        )


if __name__ == "__main__":
    main()
