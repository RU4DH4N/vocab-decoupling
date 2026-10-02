import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class Meter:
    def __init__(self) -> None:
        self.peaks: dict[str, int] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        yield


class CudaMeter(Meter):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        yield
        torch.cuda.synchronize(self.device)
        self.peaks[name] = torch.cuda.max_memory_allocated(self.device)


class TensorMeter(Meter, TorchDispatchMode):
    def __init__(self) -> None:
        Meter.__init__(self)
        TorchDispatchMode.__init__(self)
        self.live: dict[int, list[int]] = {}
        self.current = 0
        self.peak = 0

    def _release(self, key: int) -> None:
        entry = self.live[key]
        entry[0] -= 1
        if not entry[0]:
            del self.live[key]
            self.current -= entry[1]

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        output = func(*args, **(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                continue
            storage = value.untyped_storage()
            key = storage.data_ptr()
            if not key:
                continue
            if key in self.live:
                self.live[key][0] += 1
            else:
                self.live[key] = [1, storage.nbytes()]
                self.current += storage.nbytes()
                self.peak = max(self.peak, self.current)
            weakref.finalize(value, self._release, key)
        return output

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.peak = self.current
        yield
        self.peaks[name] = self.peak
