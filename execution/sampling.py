import base64
import math
from dataclasses import dataclass

import torch
from torch import Tensor


def sample(
    logits: Tensor, generator: torch.Generator, temperature: float, top_p: float
) -> int:
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p must be in (0, 1]")
    if top_p == 1:
        probabilities = (logits.float() / temperature).softmax(-1)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())
    values, order = (logits.float() / temperature).sort(descending=True)
    probabilities = values.softmax(-1)
    remove = probabilities.cumsum(-1) - probabilities >= top_p
    probabilities = probabilities.masked_fill(remove, 0)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(order[selected].item())


def generation_record(raw: bytes, elapsed: float, censored: bool) -> dict:
    try:
        text = raw.decode("utf-8")
        valid = True
    except UnicodeDecodeError:
        text, valid = raw.decode("utf-8", errors="replace"), False
    return {
        "text": text,
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "bytes": len(raw),
        "seconds": elapsed,
        "censored": censored,
        "valid_utf8": valid,
    }


@dataclass(frozen=True)
class GenerationLimits:
    bytes: int
    symbols: int
    event_bytes: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        for value in (self.bytes, self.symbols, self.event_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("generation limits must be positive integers")
        if (
            not math.isfinite(self.temperature)
            or self.temperature <= 0
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("invalid sampling temperature or top_p")
