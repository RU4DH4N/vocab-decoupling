import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

RESUME_FORMAT_VERSION = 1


def configuration_fingerprint(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resume_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.resume{output.suffix}")


def capture(
    step: int,
    fingerprint: str,
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    extra: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "format_version": RESUME_FORMAT_VERSION,
        "step": step,
        "fingerprint": fingerprint,
        "models": {name: module.state_dict() for name, module in models.items()},
        "optimizer": optimizer.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "extra": extra,
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps_rng"] = torch.mps.get_rng_state()
    return state


def save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class StaleResume(ValueError):
    pass


def load(path: Path, fingerprint: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format_version") != RESUME_FORMAT_VERSION
    ):
        raise StaleResume(f"unsupported resume state, path={path}")
    if state["fingerprint"] != fingerprint:
        raise StaleResume(
            "resume state was written by a different training configuration; "
            + f"refusing to continue, path={path}"
        )
    return state


def current(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load(path, fingerprint)
    except StaleResume as error:
        print(f"discarding stale resume state: {error}", flush=True)
        path.unlink()
        return None


def restore(
    state: dict[str, Any],
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> int:
    for name, module in models.items():
        module.load_state_dict(state["models"][name])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if "mps_rng" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps_rng"])
    return int(state["step"])
