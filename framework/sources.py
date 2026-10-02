import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _file(module: str) -> Path | None:
    base = ROOT.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imports(path: Path, module: str) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parent, *([base] if base else [])])
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


@cache
def closure(module: str) -> tuple[Path, ...]:
    seen: dict[str, Path] = {}
    pending = [module]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = _file(name)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        pending.extend(_imports(path, name))
    if module not in seen:
        raise ValueError(f"{module} is not a module in this repository")
    return tuple(sorted(set(seen.values())))
