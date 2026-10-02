import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.communication import ReceiverUnits
from data.protocol_corpus import EventCorpus
from execution.design import seed_for
from execution.layout import Layout, required
from execution.protocol_io import interface_state
from execution.protocol_stages import StageConfig, StageTrainer
from execution.training import encoded, initial_protocol
from execution.validation import validate
from execution.windows import protocol_batch, score_windows
from framework.checkpoints import write_json
from framework.runtime import set_seed
from models.protocol.interfaces import KINDS, InterfaceReceiver
from models.protocol.model import ProtocolModel

ARMS = ("layerwise", *KINDS)


def build(
    config: dict,
    layout: Layout,
    corpus: EventCorpus,
    table: ReceiverUnits,
    device: torch.device,
    kind: str,
) -> ProtocolModel:
    model, _, _ = initial_protocol(config, layout, "fresh-byte", corpus, table, device)
    if kind == "layerwise":
        return model
    receiver = InterfaceReceiver(
        model.receiver.native,
        **config["interface"],
        output_symbols=table.inventory.outputs,
        kind=kind,
    )
    return ProtocolModel(model.sender, receiver).to(device)


def train(layout: Layout, device: torch.device) -> None:
    config = layout.config()
    validate(config)
    corpus = EventCorpus.load(layout.corpus)
    table = encoded(config, layout, "fresh-byte", corpus)
    root_seed = json.loads(layout.design.read_text())["inputs"]["root_seed"]
    states, sizes = {}, {}
    for kind in ARMS:
        set_seed(seed_for(root_seed, layout.replicate, "interface", kind))
        model = build(config, layout, corpus, table, device, kind)
        rng = np.random.default_rng(config["data_seed"] + 1)
        for stage in config["replacement_stages"][:2]:
            settings = StageConfig(
                **{
                    **stage,
                    "learning_rate": required(stage["learning_rate"], "adapter rate"),
                }
            )
            trainer = StageTrainer(model, settings, {"interface": kind}, rng)
            for _ in range(settings.steps):
                rows = corpus.windows(
                    "train", config["events"], config["batch_size"], rng
                )
                trainer.step(protocol_batch(config, table, rows, device))
            print(f"interface={kind} stage={stage['name']}", flush=True)
        states[kind] = interface_state(model)
        sizes[kind] = sum(value.numel() for value in states[kind].values())
    folder = layout.run / "interfaces"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(states, folder / "interfaces.pt")
    write_json(
        folder / "interfaces.json",
        {
            "trainable_parameters": sizes,
            "schedule": [s["name"] for s in config["replacement_stages"][:2]],
            "placement": {
                "layerwise": "gated low-rank read after every block",
                "input": "gated full-rank read after the first block only",
                "linear": "ungated linear map after the first block only",
                "attention": "multi-head cross-attention after every block",
            },
        },
    )


@torch.no_grad()
def measure(layout: Layout, device: torch.device, split: str) -> None:
    config = layout.config()
    validate(config)
    corpus = EventCorpus.load(layout.evaluation_corpus(split))
    table = encoded(config, layout, "fresh-byte", corpus)
    training = EventCorpus.load(layout.corpus)
    states = torch.load(layout.run / "interfaces" / "interfaces.pt", weights_only=True)
    results = {}
    for kind in ARMS:
        model = build(config, layout, training, table, device, kind)
        if split == "reporting":
            model.sender.build_table(corpus.vocab)
        model.receiver.load_state_dict(
            {**model.receiver.state_dict(), **states[kind]}, strict=True
        )
        model.eval()
        combined, _ = score_windows(
            model,
            table,
            corpus,
            config,
            layout.evaluation_batches(config, split),
            device,
        )
        results[kind] = {
            "scores": {name: value.aggregate() for name, value in combined.items()},
            "paired_nll": {
                name: value.nll.tolist() for name, value in combined.items()
            },
            "utf8_bytes": combined["correct"].utf8_bytes.tolist(),
        }
    write_json(
        layout.measured(split) / "interfaces" / "evaluation.json",
        {"split": split, "interfaces": results},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", choices=("selection", "reporting"))
    args = parser.parse_args()
    layout = Layout(args.output, args.replicate)
    device = torch.device(args.device)
    if args.task == "train":
        train(layout, device)
    elif args.split is None:
        parser.error("measure requires a split")
    else:
        measure(layout, device, args.split)


if __name__ == "__main__":
    main()
