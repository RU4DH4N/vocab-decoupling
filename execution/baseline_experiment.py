import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from data.protocol_corpus import EventCorpus, selection_windows
from data.tokenizers import load_tokenizer
from execution.baseline_training import (
    bpe_batch,
)
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
from framework.checkpoints import (
    file_sha256,
    load_checkpoint,
    read_sidecar,
    write_json,
)
from framework.runtime import (
    device_synchronize,
)
from models.bpe.bpe import BPEModel


@torch.no_grad()
def generate(
    model: BPEModel, tokenizer: Tokenizer, prompt: str, options: dict, seed: int
) -> dict:
    device = model.embedding.weight.device
    device_synchronize(device)
    start = time.perf_counter()
    ids = tokenizer.encode(prompt).ids
    if not ids or len(ids) > model.trunk.config.max_seq_len:
        raise ValueError("BPE generation prompt must fit the native context")
    symbols = SymbolBytes.bpe(tokenizer)
    caches = model.new_caches(model.trunk.config.max_seq_len)
    logits = model(torch.tensor([ids], device=device), caches=caches)[0, -1]
    rng = torch.Generator(device=device).manual_seed(seed)
    output = bytearray()
    reason = "symbol-limit"
    for _ in range(options["symbols"]):
        token = sample(logits, rng, options["temperature"], options["top_p"])
        output.extend(symbols.values[token])
        if len(output) >= options["bytes"]:
            reason = "byte-budget"
            break
        if not caches[0].remaining:
            reason = "receiver-context-limit"
            break
        logits = model(torch.tensor([[token]], device=device), caches=caches)[0, 0]
    device_synchronize(device)
    return {
        **generation_record(
            bytes(output[: options["bytes"]]),
            time.perf_counter() - start,
            reason != "byte-budget",
        ),
        "stop_reason": reason,
    }


@torch.no_grad()
def measure(config: dict, layout: Layout, device: torch.device, split: str) -> None:
    path = layout.run / "baseline/model.pt"
    tokenizer_sha = read_sidecar(path)["identity"]["baseline_tokenizer"]
    if tokenizer_sha != file_sha256(layout.baseline_tokenizer):
        raise ValueError("baseline tokenizer changed")
    options = config["generation"]
    records, timed = generate_streams(
        generate_baseline,
        (layout.shared, layout.replicate, split, str(device)),
        prompt_count(layout.evaluation_prompts(split)),
        options["latency_samples"],
        device_streams(options, device),
    )
    write_json(
        layout.measured(split) / "baseline/generation.json",
        {
            "checkpoint_sha256": file_sha256(path),
            **summary(records, timed, layout.evaluation_prompts(split), options),
        },
    )
    payload = load_checkpoint(path, map_location=device)
    tokenizer = load_tokenizer(layout.baseline_tokenizer)
    model = BPEModel(**payload["metadata"]["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    rng = np.random.default_rng(config["evaluation_seed"])
    nats, sizes, groups = [], [], []
    for _ in range(layout.evaluation_batches(config, split)):
        rows, group_ids = selection_windows(
            corpus, config["events"], config["batch_size"], rng, split="selection"
        )
        ids, targets = bpe_batch(corpus, rows, tokenizer, device)
        loss = F.cross_entropy(
            model(ids).float().flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape_as(targets)
        nats.extend(loss.double().sum(-1).tolist())
        sizes.extend(len("".join(units[1:]).encode()) for units in corpus.decode(rows))
        groups.extend(group_ids.tolist())
    write_json(
        layout.measured(split) / "baseline/evaluation.json",
        {
            "checkpoint_sha256": file_sha256(path),
            "split": split,
            "likelihood_domain": "canonical BPE path; not text marginal",
            "nats": nats,
            "utf8_bytes": sizes,
            "group_ids": groups,
            "canonical_path_bpb": sum(nats) / sum(sizes) / math.log(2),
        },
    )


@torch.no_grad()
def generate_baseline(
    shared: Path, replicate: int, split: str, device: str, indices: list[int]
) -> Shard:
    layout = Layout(shared, replicate)
    config = layout.config()
    payload = load_checkpoint(layout.run / "baseline/model.pt", map_location=device)
    tokenizer = load_tokenizer(layout.baseline_tokenizer)
    model = BPEModel(**payload["metadata"]["model_config"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    examples = json.loads(layout.evaluation_prompts(split).read_text())
    records = []
    for index in indices:
        example = examples[index]
        record = generate(
            model,
            tokenizer,
            example["prompt"],
            config["generation"],
            config["evaluation_seed"] + index,
        )
        records.append((index, {**example, **record}))
        print(f"baseline generation={index + 1}/{len(examples)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("measure", "mauve"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"), required=True)
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    config = layout.config()
    validate(config)
    device = torch.device(args.device)
    if args.task == "measure":
        measure(config, layout, device, args.split)
    else:
        folder = layout.measured(args.split) / "baseline"
        mauve_score(
            folder / "generation.json", config["mauve"], folder / "mauve.json", device
        )


if __name__ == "__main__":
    main()
