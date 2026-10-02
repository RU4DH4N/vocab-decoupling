import importlib
import inspect
from pathlib import Path

from orchestrator.contract import Contract


def discover(name: str) -> type[Contract]:
    module_name = name.removesuffix(".py").replace("/", ".")
    parts = module_name.split(".")
    if (
        len(parts) != 2
        or parts[0] != "claims"
        or not parts[1].isidentifier()
        or parts[1].startswith("_")
    ):
        raise ValueError("name a claim in claims/")
    module = importlib.import_module(module_name)
    matches = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Contract)
        and obj is not Contract
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{name} must define exactly one concrete Contract subclass, got {len(matches)}"
        )
    return matches[0]


def script_name(filename: str) -> str:
    path = Path(filename).resolve()
    return f"{path.parent.name}.{path.stem}"
