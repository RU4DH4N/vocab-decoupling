import argparse
import json
import math
from pathlib import Path

import numpy as np

from data.corpus import split_words, to_units
from data.protocol_corpus import (
    EventCorpus,
    document_ids,
    documents,
)
from data.tokenizers import train_bpe
from execution.design import (
    continuation_bytes,
    continuation_events,
    prompt_shape,
    replicate_config,
    resolve,
    seed_for,
    statistics,
)
from execution.exposure_design import plan_exposure
from execution.layout import Layout
from execution.validation import validate
from framework.checkpoints import file_sha256, write_json


def _prompts(
    inputs: dict,
    facts: dict,
    selection_docs: list[str],
    unit_bytes: int,
    limit: int | None,
    role: str,
) -> list[dict]:
    prompt_events, continuation = prompt_shape(inputs, facts)
    order = np.random.default_rng(seed_for(inputs["root_seed"], role)).permutation(
        len(selection_docs)
    )
    prompts = []
    for index in order.tolist():
        units = to_units(selection_docs[index], unit_bytes)
        if len(units) <= prompt_events:
            continue
        reference = "".join(units[prompt_events:]).encode()[:continuation]
        if len(reference) < continuation:
            continue
        prompts.append(
            {
                "id": index,
                "prompt_units": units[:prompt_events],
                "prompt": "".join(units[:prompt_events]),
                "reference": reference.decode("utf-8", errors="ignore"),
            }
        )
        if len(prompts) == limit:
            break
    return prompts


def prepare(inputs: dict, train: Path, selection: Path, root: Path) -> None:
    hashes = {"train": file_sha256(train), "selection": file_sha256(selection)}
    if hashes["train"] == hashes["selection"]:
        raise ValueError("train and selection must have different contents")
    train_ids, selection_ids = document_ids(train), document_ids(selection)
    if (train_ids is None) != (selection_ids is None):
        raise ValueError("use the same grouping identity scheme in both splits")
    if (
        train_ids is not None
        and selection_ids is not None
        and train_ids & selection_ids
    ):
        raise ValueError("released grouping IDs overlap between splits")
    train_docs = documents(train, inputs["corpus_characters"])
    selection_docs = documents(selection, inputs["corpus_characters"])
    if set(train_docs) & set(selection_docs):
        raise ValueError("identical text groups overlap between splits")
    facts = statistics(
        [len(word.encode()) for doc in train_docs for word in split_words(doc)],
        sum(len(split_words(doc)) for doc in selection_docs),
    )
    unit = facts["max_unit_bytes"]
    train_units = [to_units(doc, unit) for doc in train_docs]
    facts["continuation_bytes"] = continuation_bytes(
        [[len(piece.encode()) for piece in doc] for doc in train_units],
        continuation_events(inputs),
    )
    candidates = _prompts(inputs, facts, selection_docs, unit, None, "prompts")
    design = resolve(inputs, facts, len(candidates))
    shared = design["shared"]
    prompts = candidates[: shared["generation"]["samples"]]
    if not prompts:
        raise ValueError("no selection text group is long enough for a prompt")
    corpus = EventCorpus(train_docs, selection_docs, unit)
    corpus.save(root / "corpus", hashes)
    for replicate in range(inputs["seeds"]):
        config = replicate_config(design, replicate, {})
        validate(config)
        write_json(Layout(root, replicate).exposure, plan_exposure(config, corpus))
    pieces = [piece for doc in train_units for piece in doc]
    tokenizer = train_bpe(shared["bpe_vocabulary"], pieces, False)
    tokenizer.save(str(root / "receiver-tokenizer.json"))
    baseline_tokenizer = train_bpe(shared["bpe_vocabulary"], train_docs, True)
    baseline_tokenizer.save(str(root / "baseline-tokenizer.json"))
    write_json(root / "prompts.json", prompts)
    write_json(
        root / "design.json",
        {
            **design,
            "data_sources": hashes,
            "vocabulary": len(corpus.vocab),
            "likelihood": "canonical-path, including STOP; an upper bound on text-marginal BPB",
            "grouping": "input text groups; original conversation/article disjointness is not claimed",
            "baseline_matching": "identical target-text windows to primary native pretraining plus all primary stages; planner/future fitting is extra, reported separately",
        },
    )


def prepare_reporting(
    root: Path, reporting: Path, train: Path, selection: Path
) -> None:
    design = json.loads((root / "design.json").read_text())
    inputs, facts = design["inputs"], design["statistics"]
    source = file_sha256(reporting)
    if source in design["data_sources"].values():
        raise ValueError("the reporting split duplicates a training or selection file")
    identities = document_ids(reporting)
    for other in (train, selection):
        known = document_ids(other)
        if identities is not None and known is not None and identities & known:
            raise ValueError("released grouping IDs overlap the reporting split")
    docs = documents(reporting, inputs["corpus_characters"])
    for other in (train, selection):
        if set(docs) & set(documents(other, inputs["corpus_characters"])):
            raise ValueError("identical text groups overlap the reporting split")
    unit = facts["max_unit_bytes"]
    corpus = EventCorpus([], docs, unit)
    corpus.save(root / "reporting" / "corpus", {"reporting": source})
    prompts = _prompts(
        inputs,
        facts,
        docs,
        unit,
        design["shared"]["generation"]["samples"],
        "reporting-prompts",
    )
    if not prompts:
        raise ValueError("no reporting text group is long enough for a prompt")
    write_json(root / "reporting" / "prompts.json", prompts)
    shared = design["shared"]
    events = len(corpus.ids["selection"])
    write_json(
        root / "reporting" / "design.json",
        {
            "source": source,
            "events": events,
            "vocabulary": len(corpus.vocab),
            "prompts": len(prompts),
            "evaluation_batches": max(
                1, math.ceil(events / (shared["batch_size"] * shared["events"]))
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--reporting-file", type=Path)
    args = parser.parse_args()
    if args.task == "prepare":
        if args.inputs is None:
            parser.error("prepare requires the design inputs")
        prepare(
            json.loads(args.inputs.read_text()),
            args.train_file,
            args.selection_file,
            args.output,
        )
    elif args.reporting_file is None:
        parser.error("report requires the reporting file")
    else:
        prepare_reporting(
            args.output, args.reporting_file, args.train_file, args.selection_file
        )


if __name__ == "__main__":
    main()
