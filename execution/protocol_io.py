from pathlib import Path

import torch
from torch import Tensor

from data.protocol_corpus import EventCorpus
from framework.checkpoints import load_checkpoint, read_sidecar
from models.bpe.bpe import BPEModel
from models.protocol.model import ProtocolModel
from models.protocol.receiver import Receiver
from models.protocol.sender import CoarseSender

ARCHITECTURE = "layerwise-protocol"


def build_protocol(
    config: dict, vocab: list[str] | None, device: torch.device
) -> ProtocolModel[Receiver]:
    sender = CoarseSender(**config["sender"]).to(device)
    if vocab is not None:
        sender.build_table(vocab)
    native = BPEModel(**config["native"]).to(device)
    receiver = Receiver(native, **config["interface"]).to(device)
    return ProtocolModel(sender, receiver)


def load_protocol(
    path: Path, device: torch.device, *, table: bool
) -> tuple[ProtocolModel[Receiver], dict]:
    payload = load_checkpoint(
        path,
        map_location=device,
        required_metadata=("architecture", "model_config", "vocab", "identity"),
    )
    metadata = payload["metadata"]
    if metadata["architecture"] != ARCHITECTURE:
        raise ValueError("checkpoint does not contain the layerwise protocol")
    model = build_protocol(
        metadata["model_config"], metadata["vocab"] if table else None, device
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, metadata


@torch.no_grad()
def check_inventory(path: Path, corpus: EventCorpus, split: str) -> None:
    metadata = read_sidecar(path)
    if metadata["architecture"] != ARCHITECTURE:
        raise ValueError("checkpoint does not contain the layerwise protocol")
    if split == "selection" and metadata["vocab"] != corpus.vocab:
        raise ValueError("evaluation inventory differs from checkpoint")


def evaluation_model(
    path: Path, corpus: EventCorpus, split: str, device: torch.device
) -> ProtocolModel[Receiver]:
    check_inventory(path, corpus, split)
    model, _ = load_protocol(path, device, table=True)
    if split == "reporting":
        model.sender.build_table(corpus.vocab)
    return model


def interface_state(model: ProtocolModel) -> dict[str, Tensor]:
    return {
        key: value.cpu()
        for key, value in model.receiver.state_dict().items()
        if key.startswith(("scorers.", "channels."))
    }
