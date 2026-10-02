import hashlib
import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn

from data.protocol_corpus import window_sums
from models import conventions
from models.bpe.bpe import BPEModel
from models.protocol.receiver import Receiver
from models.protocol.sender import CoarseSender

RECEIVERS = ("primary", "fresh-byte", "fresh-bpe")
PRIMARY_STAGES = ("alignment", "communication", "trunk", "joint")
REPLACEMENT_STAGES = ("alignment", "communication", "joint")
SWEEPS = ("native", "baseline", "adapter", "trunk")

UNIT_COVERAGE = 0.999
CONTINUATION_COVERAGE = 0.95
RECEIVER_FRACTION = 1 / 8
ADAPTER_FRACTION = 0.05
SWEEP_FRACTION = 0.05
SELECTION_FRACTION = 0.25
WARMUP_FRACTION = 0.01
CHECKPOINTS_PER_JOB = 50
CANDIDATE_WINDOW = 8
PLANNER_HYPOTHESES = 8
LEARNING_RATE_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
ALIGNMENT_WEIGHT_GRID = (0.0, 0.01, 0.1, 1.0)
BPE_VOCABULARY = 16_384
MAUVE_SAMPLES = 1000
LATENCY_FRACTION = 0.1
GENERATION_STREAMS = 8
MAUVE_MODEL = "openai-community/gpt2-large"
MAUVE_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
EXTERNAL_MODEL = "babylm/babyllama-100m-2024"
EXTERNAL_REVISION = "9c1aeb3459c892ad28fd56ece6672a223c43ba9c"

INPUTS = {
    "dataset": dict,
    "trunk_parameters": int,
    "context_events": int,
    "batch_windows": int,
    "passes": (int, float),
    "seeds": int,
    "baseline_seeds": int,
    "lookahead_horizon": int,
    "root_seed": int,
    "corpus_characters": (int, type(None)),
    "mauve": bool,
    "receiver_context": str,
}
RECEIVER_CONTEXTS = ("word", "window")

SOURCES = {
    "batch_size": "input batch_windows",
    "receiver_context": "input: word receivers attend within the word being spelled; window receivers see the whole byte window",
    "events": "input context_events",
    "precision": "convention: bf16 autocast on CUDA, fp32 elsewhere",
    "bpe_vocabulary": "convention: BabyLM 2025 baseline vocabulary",
    "corpus.max_unit_bytes": f"data: {UNIT_COVERAGE:.1%} quantile of training word length",
    "receiver_steps": "derived: worst-case window of max_unit_bytes words plus STOPs",
    "sender": "derived: aspect-ratio transformer closest to input trunk_parameters",
    "native": f"derived: aspect-ratio transformer with {RECEIVER_FRACTION} of the trunk's parameters",
    "baseline.model": "derived: parameter-matched to sender + primary receiver + interface",
    "interface": "derived: message width = receiver width; score width = one head; rank = width/4",
    "planner.model": "derived: receiver width, half-width embeddings, rank = width/4",
    "lookahead.hypotheses": "convention: planner beam width",
    "planner.steps": "derived: the full budget; the planner is a next-event language model, not an adapter",
    "native_training.steps": "derived: input passes over the training split",
    "stages": f"derived: trunk = full budget, adapters = {ADAPTER_FRACTION} of it",
    "warmup_steps": f"derived: {WARMUP_FRACTION} of each job",
    "learning_rate": "selected: selection-split sweep over the grid",
    "trunk alignment_weight": "selected: selection-split sweep over the grid",
    "weight_decay, grad_clip, architecture": "convention: models.conventions",
    "evaluation_batches": "derived: one pass over the evaluated split",
    "selection_batches": f"derived: {SELECTION_FRACTION} of a pass; the reporting split is evaluated in full",
    "generation": "derived: half the context as prompt; the continuation byte budget fits the rest of the context for most text",
    "generation.bytes": f"data: bytes that {CONTINUATION_COVERAGE:.0%} of training spans of the continuation's units exceed",
    "generation decoding": "a word ends at max_unit_bytes and is never empty, as in the segmented corpus",
    "generation.samples": f"protocol: {MAUVE_SAMPLES} MAUVE samples (Pillutla et al. 2021)",
    "mauve.buckets": "protocol: MAUVE's default of samples / 10",
    "generation.latency_samples": f"protocol: the first {LATENCY_FRACTION} of prompts, generated alone on the device, define latency",
    "generation.streams": "engineering: concurrent generation processes on an accelerator; per-sample seeds make samples identical to serial generation",
    "external": "comparison: BabyLM 2024 BabyLlama baseline at a pinned revision, inference only",
    "seeds": "derived: hash of root_seed and role",
}


def check_inputs(inputs: dict) -> None:
    missing = set(INPUTS) - set(inputs)
    unknown = set(inputs) - set(INPUTS)
    if missing or unknown:
        raise ValueError(
            f"design inputs: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    for name, kind in INPUTS.items():
        value = inputs[name]
        if not isinstance(value, kind) or (
            isinstance(value, bool) and kind is not bool
        ):
            raise ValueError(f"design input {name} has the wrong type")
    for name in ("trunk_parameters", "context_events", "seeds", "lookahead_horizon"):
        if inputs[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if inputs["batch_windows"] < 2:
        raise ValueError("shuffled controls require at least two windows per batch")
    if not 0 < inputs["passes"]:
        raise ValueError("passes must be positive")
    if not 1 <= inputs["baseline_seeds"] <= inputs["seeds"]:
        raise ValueError("baseline_seeds must be between 1 and seeds")
    if inputs["corpus_characters"] is not None and inputs["corpus_characters"] <= 0:
        raise ValueError("corpus_characters must be positive or null")
    if inputs["receiver_context"] not in RECEIVER_CONTEXTS:
        raise ValueError(f"receiver_context must be one of {RECEIVER_CONTEXTS}")


def seed_for(root_seed: int, *role: object) -> int:
    digest = hashlib.sha256("/".join(map(str, (root_seed, *role))).encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def warmup(steps: int) -> int:
    return math.floor(steps * WARMUP_FRACTION)


def continuation_events(inputs: dict) -> int:
    return inputs["context_events"] - inputs["context_events"] // 2


def prompt_shape(inputs: dict, facts: dict) -> tuple[int, int]:
    return inputs["context_events"] // 2, facts["continuation_bytes"]


def statistics(train_lengths: list[int], selection_events: int) -> dict:
    if not train_lengths:
        raise ValueError("the training split has no words")
    ordered = sorted(train_lengths)
    index = min(len(ordered) - 1, math.ceil(UNIT_COVERAGE * len(ordered)) - 1)
    return {
        "train_events": len(ordered),
        "train_bytes": sum(ordered),
        "selection_events": selection_events,
        "mean_event_bytes": sum(ordered) / len(ordered),
        "max_unit_bytes": max(4, ordered[index]),
    }


def continuation_bytes(unit_bytes: Sequence[Sequence[int]], span: int) -> int:
    sizes = np.array([len(doc) for doc in unit_bytes], dtype=np.int64)
    lengths = np.fromiter((b for doc in unit_bytes for b in doc), dtype=np.int64)
    sums, valid = window_sums(lengths, sizes, span)
    if not valid.any():
        raise ValueError(f"no training document has {span} units")
    return int(
        np.quantile(sums[valid], 1 - CONTINUATION_COVERAGE, method="inverted_cdf")
    )


def _closest(target: float, count: Callable[[int, int], int]) -> tuple[int, int]:
    lines = {}
    for width, _ in conventions.shapes():
        if width not in lines:
            two = count(width, 2)
            lines[width] = (two, count(width, 3) - two)

    def distance(shape: tuple[int, int]) -> float:
        base, block = lines[shape[0]]
        return abs(math.log((base + (shape[1] - 2) * block) / target))

    return min(conventions.shapes(), key=distance)


def _sender(
    shape: tuple[int, int], d_meaning: int, max_seq_len: int, codebook_seed: int
) -> dict:
    options = conventions.transformer(*shape, max_seq_len)
    return {
        "in_dims": conventions.CODE_DIMENSIONS,
        **options,
        "d_meaning": d_meaning,
        "codebook_seed": codebook_seed,
    }


def _language_model(shape: tuple[int, int], max_seq_len: int) -> dict:
    return {
        **conventions.transformer(*shape, max_seq_len),
        "tie_word_embeddings": conventions.TIE_WORD_EMBEDDINGS,
    }


def _count(factory: Callable[[], nn.Module]) -> int:

    with torch.device("meta"):
        return sum(p.numel() for p in factory().parameters())


def _sender_parameters(options: dict) -> int:

    return _count(lambda: CoarseSender(**options))


def _body_parameters(options: dict) -> int:
    return _count(lambda: BPEModel(**options, vocab_size=1)) - options["d_model"]


def _interface_parameters(native: dict, interface: dict) -> int:
    with_interface = _count(
        lambda: Receiver(BPEModel(**native, vocab_size=258), **interface)
    )
    return with_interface - _count(lambda: BPEModel(**native, vocab_size=258))


def _adapter(steps: int, learning_rate: float | None) -> dict:
    return {
        "steps": steps,
        "warmup_steps": warmup(steps),
        "learning_rate": learning_rate,
        "weight_decay": conventions.WEIGHT_DECAY,
        "grad_clip": conventions.GRAD_CLIP,
    }


def _stage(
    name: str,
    steps: int,
    learning_rate: float | None,
    alignment_weight: float | None,
) -> dict:
    return {
        "name": name,
        **_adapter(steps, learning_rate),
        "alignment_weight": alignment_weight,
        "precision": "auto",
    }


def resolve(inputs: dict, facts: dict, prompts: int) -> dict:
    check_inputs(inputs)
    events = inputs["context_events"]
    batch = inputs["batch_windows"]
    horizon = inputs["lookahead_horizon"]
    unit = facts["max_unit_bytes"]
    receiver_steps = (events + 1) * (unit + 1)
    sender_context = events + horizon
    baseline_context = (events + 1) * unit - 1

    receiver_shape = _closest(
        inputs["trunk_parameters"] * RECEIVER_FRACTION,
        lambda *shape: _body_parameters(_language_model(shape, receiver_steps)),
    )
    receiver_width = receiver_shape[0]
    trunk_shape = _closest(
        inputs["trunk_parameters"],
        lambda *shape: _sender_parameters(
            _sender(shape, receiver_width, sender_context, 0)
        ),
    )
    native = _language_model(receiver_shape, receiver_steps)
    sender = _sender(trunk_shape, receiver_width, sender_context, 0)
    interface = {
        "sender_dimensions": receiver_width,
        "score_dimensions": min(conventions.HEAD_DIMENSIONS, receiver_width),
        "communication_rank": max(1, receiver_width // 4),
        "candidate_window": CANDIDATE_WINDOW,
    }
    system = (
        _sender_parameters(sender)
        + _body_parameters(native)
        + _interface_parameters(native, {**interface, "output_symbols": 257})
    )
    baseline_shape = _closest(
        system,
        lambda *shape: _body_parameters(_language_model(shape, baseline_context)),
    )

    full = math.ceil(inputs["passes"] * facts["train_events"] / (batch * events))
    adapter = max(1, math.ceil(full * ADAPTER_FRACTION))
    prompt_events, continuation = prompt_shape(inputs, facts)
    samples = min(MAUVE_SAMPLES, prompts)
    return {
        "inputs": inputs,
        "statistics": facts,
        "sources": SOURCES,
        "parameters": {
            "sender": _sender_parameters(sender),
            "native_body": _body_parameters(native),
            "baseline_body": _body_parameters(
                _language_model(baseline_shape, baseline_context)
            ),
            "matched_system": system,
        },
        "steps": {
            "full": full,
            "adapter": adapter,
            "sweep": max(1, math.ceil(full * SWEEP_FRACTION)),
        },
        "grids": {
            "learning_rate": list(LEARNING_RATE_GRID),
            "alignment_weight": list(ALIGNMENT_WEIGHT_GRID),
        },
        "shared": {
            "batch_size": batch,
            "events": events,
            "receiver_steps": receiver_steps,
            "checkpoint_interval": max(1, full // CHECKPOINTS_PER_JOB),
            "evaluation_batches": max(
                1, math.ceil(facts["selection_events"] / (batch * events))
            ),
            "selection_batches": max(
                1,
                math.ceil(
                    facts["selection_events"] / (batch * events) * SELECTION_FRACTION
                ),
            ),
            "precision": "auto",
            "bpe_vocabulary": BPE_VOCABULARY,
            "corpus": {
                "train_characters": inputs["corpus_characters"],
                "validation_characters": inputs["corpus_characters"],
                "max_unit_bytes": unit,
            },
            "native_training": {**_adapter(full, None)},
            "sender": sender,
            "native": native,
            "interface": interface,
            "baseline": {
                **_adapter(full, None),
                "model": _language_model(baseline_shape, baseline_context),
            },
            "generation": {
                "samples": samples,
                "latency_samples": max(1, math.ceil(samples * LATENCY_FRACTION)),
                "streams": GENERATION_STREAMS,
                "prompt_events": prompt_events,
                "bytes": continuation,
                "symbols": 2 * continuation,
                "event_bytes": unit,
                "temperature": 1.0,
                "top_p": 1.0,
            },
            "planner": {
                "model": {
                    "message_dimensions": receiver_width,
                    "embedding_dimensions": max(1, receiver_width // 2),
                    "hidden_dimensions": receiver_width,
                    "communication_rank": max(1, receiver_width // 4),
                },
                **_adapter(full, None),
                "max_symbols": unit + 1,
            },
            "lookahead": {
                "horizon": horizon,
                "rank": max(1, receiver_width // 4),
                "hypotheses": PLANNER_HYPOTHESES,
                **_adapter(adapter, None),
            },
            "control_training": _adapter(adapter, None),
            "mauve": {
                "enabled": inputs["mauve"],
                "model": MAUVE_MODEL,
                "revision": MAUVE_REVISION,
                "max_tokens": continuation,
                "buckets": max(2, samples // 10),
                "seed": seed_for(inputs["root_seed"], "mauve"),
            },
            "external": {"model": EXTERNAL_MODEL, "revision": EXTERNAL_REVISION},
            "receiver_context": inputs["receiver_context"],
            "evaluation_seed": seed_for(inputs["root_seed"], "evaluation"),
        },
    }


def replicate_config(design: dict, replicate: int, selected: dict) -> dict:
    inputs = design["inputs"]
    if not 0 <= replicate < inputs["seeds"]:
        raise ValueError("replicate outside the declared seeds")

    def seed(*role: object) -> int:
        return seed_for(inputs["root_seed"], replicate, *role)

    rate = {name: selected.get(name, {}).get("learning_rate") for name in SWEEPS}
    trunk_weight = selected.get("trunk", {}).get("alignment_weight")
    steps = design["steps"]
    shared = design["shared"]
    config = {
        **shared,
        "replicate": replicate,
        "data_seed": seed("data"),
        "receivers": {
            "primary": {"kind": "bytes", "seed": seed("receiver", "primary")},
            "fresh-byte": {"kind": "bytes", "seed": seed("receiver", "fresh-byte")},
            "fresh-bpe": {"kind": "bpe", "seed": seed("receiver", "fresh-bpe")},
        },
        "sender": {**shared["sender"], "codebook_seed": seed("codebook")},
        "native_training": {
            **shared["native_training"],
            "learning_rate": rate["native"],
        },
        "baseline": {
            **shared["baseline"],
            "learning_rate": rate["baseline"],
            "seed": seed("baseline"),
        },
        "stages": [
            _stage("alignment", steps["adapter"], rate["adapter"], 1.0),
            _stage("communication", steps["adapter"], rate["adapter"], 0.0),
            _stage("trunk", steps["full"], rate["trunk"], trunk_weight),
            _stage("joint", steps["adapter"], rate["adapter"], 0.0),
        ],
        "replacement_stages": [
            _stage("alignment", steps["adapter"], rate["adapter"], 1.0),
            _stage("communication", steps["adapter"], rate["adapter"], 0.0),
            _stage("joint", steps["adapter"], rate["adapter"], 0.0),
        ],
        "planner": {
            **shared["planner"],
            "learning_rate": rate["adapter"],
            "seed": seed("planner"),
            "data_seed": seed("planner", "data"),
        },
        "lookahead": {
            **shared["lookahead"],
            "learning_rate": rate["adapter"],
            "seed": seed("lookahead"),
            "data_seed": seed("lookahead", "data"),
        },
        "control_training": {
            **shared["control_training"],
            "learning_rate": rate["adapter"],
            "seed": seed("control"),
            "projection_seed": seed("control", "projection"),
            "data_seed": seed("control", "data"),
        },
    }
    return config


def unswept(config: dict) -> dict:
    cleared = {**config}
    for section in (
        "native_training",
        "baseline",
        "planner",
        "lookahead",
        "control_training",
    ):
        cleared[section] = {**config[section], "learning_rate": None}
    for key in ("stages", "replacement_stages"):
        cleared[key] = [
            {**stage, "learning_rate": None, "alignment_weight": None}
            for stage in config[key]
        ]
    return cleared
