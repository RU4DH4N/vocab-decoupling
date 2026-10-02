import importlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from framework.checkpoints import file_sha256, write_json


def mauve_score(
    generation_path: Path, config: dict, output: Path, device: torch.device
) -> None:
    if not config["enabled"]:
        write_json(
            output,
            {
                "status": "not-computed",
                "reason": "disabled by the design inputs",
            },
        )
        return
    mauve = importlib.import_module("mauve")
    importlib.import_module("faiss").omp_set_num_threads(1)

    data = json.loads(generation_path.read_text())
    samples = data["samples"]
    try:
        encoder_path = snapshot_download(
            config["model"], revision=config["revision"], local_files_only=True
        )
    except LocalEntryNotFoundError:
        encoder_path = snapshot_download(
            config["model"],
            revision=config["revision"],
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "merges.txt",
                "vocab.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    result = mauve.compute_mauve(
        p_text=[s["reference"] for s in samples],
        q_text=[s["text"] for s in samples],
        device_id=(device.index or 0) if device.type == "cuda" else -1,
        featurize_model_name=encoder_path,
        max_text_length=config["max_tokens"],
        num_buckets=config["buckets"],
        seed=config["seed"],
        verbose=False,
    )
    write_json(
        output,
        {
            "status": "computed",
            "mauve": float(result.mauve),
            "config": config,
            "encoder_revision": config["revision"],
            "generation_sha256": file_sha256(generation_path),
        },
    )
