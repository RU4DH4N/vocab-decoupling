import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class Contract(ABC):
    remote_required = False
    unavailable_reason: str | None = None

    def __init__(self, context: object = None) -> None:
        self.context = context

    @property
    def label(self) -> str:
        return type(self).__name__

    def parameters(self) -> dict:
        return {}

    @property
    def key(self) -> str:
        return digest(
            {
                "contract": f"{type(self).__module__}.{type(self).__qualname__}",
                "parameters": self.parameters(),
            }
        )

    def dependencies(self) -> tuple["Contract", ...]:
        return ()

    def inputs(self) -> tuple[Path, ...]:
        return ()

    @abstractmethod
    def outputs(self) -> tuple[Path, ...]:
        pass

    def resources(self) -> frozenset[str]:
        return frozenset()

    def weight(self) -> int:
        return 0

    def sources(self) -> tuple[Path, ...]:
        return tuple(
            sorted(
                {
                    Path(inspect.getfile(cls)).resolve()
                    for cls in type(self).__mro__
                    if issubclass(cls, Contract) and cls is not ABC
                }
            )
        )

    @abstractmethod
    def run(self, dependencies: Mapping[str, "Contract"]) -> None:
        pass
