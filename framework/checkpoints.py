import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sidecar(path: Path) -> Path:
    if path.suffix in {".pt", ".pth", ".ckpt"}:
        return path.with_suffix(".json")
    return path.with_name(path.name + ".json")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _sidecar_payload(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"checkpoint_sha256": file_sha256(path), "metadata": metadata}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> None:
    metadata = dict(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model": model.state_dict(),
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    write_json(_checkpoint_sidecar(path), _sidecar_payload(path, metadata))


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    required_metadata: tuple[str, ...] = (),
    verify_sidecar: bool = True,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or set(("model", "metadata")) - payload.keys():
        raise CheckpointError(f"{path} is not a model checkpoint")
    version = payload.get("format_version")
    if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{path} uses unsupported checkpoint format version {version!r}"
        )
    if not isinstance(payload["model"], dict) or not isinstance(
        payload["metadata"], dict
    ):
        raise CheckpointError(f"{path} has invalid model or metadata fields")
    missing = set(required_metadata) - payload["metadata"].keys()
    if missing:
        raise CheckpointError(f"{path} is missing metadata fields: {sorted(missing)}")
    sidecar = _checkpoint_sidecar(path)
    if verify_sidecar:
        if not sidecar.is_file():
            raise CheckpointError(f"checkpoint sidecar is missing: {sidecar}")
        recorded = _read_sidecar(path)
        if recorded["checkpoint_sha256"] != file_sha256(path):
            raise CheckpointError(
                f"{path} does not match the hash its sidecar recorded; "
                "if a save was interrupted, run repair_sidecar after inspecting it"
            )
        if not _json_equal(recorded["metadata"], payload["metadata"]):
            raise CheckpointError(f"checkpoint metadata disagrees with {sidecar}")
    return payload


def repair_sidecar(path: Path) -> None:
    payload = load_checkpoint(path, verify_sidecar=False)
    write_json(_checkpoint_sidecar(path), _sidecar_payload(path, payload["metadata"]))


def _read_sidecar(path: Path) -> dict[str, Any]:
    recorded = json.loads(_checkpoint_sidecar(path).read_text())
    if (
        not isinstance(recorded, dict)
        or set(recorded) != {"checkpoint_sha256", "metadata"}
        or not isinstance(recorded["metadata"], dict)
        or not isinstance(recorded["checkpoint_sha256"], str)
        or len(recorded["checkpoint_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in recorded["checkpoint_sha256"])
    ):
        raise CheckpointError(f"invalid checkpoint sidecar: {path}")
    return recorded


def read_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    recorded = _read_sidecar(path)
    if recorded["checkpoint_sha256"] != file_sha256(path):
        raise CheckpointError(f"{path} does not match the hash its sidecar recorded")
    return cast(dict[str, Any], recorded["metadata"])
