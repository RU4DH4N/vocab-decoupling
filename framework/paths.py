from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path, *, root: Path = ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
