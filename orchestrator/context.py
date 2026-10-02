import json
from dataclasses import dataclass
from pathlib import Path

from framework.checkpoints import file_sha256
from framework.paths import portable_path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Context:
    configuration_json: str
    config_source: Path
    output: Path
    train_file: Path
    selection_file: Path
    device: str
    data_identity: tuple[str, str]
    split: str
    reporting_file: Path | None

    @property
    def settings(self) -> dict:
        return json.loads(self.configuration_json)

    @property
    def config_snapshot(self) -> Path:
        return self.output / ".orchestrator" / "configuration.json"

    def verify_snapshot(self) -> None:
        actual = json.loads(self.config_snapshot.read_text())
        if (
            json.dumps(actual, sort_keys=True, allow_nan=False)
            != self.configuration_json
        ):
            raise ValueError(
                "configuration snapshot differs from resolved configuration"
            )


def load_context(
    config_source: Path,
    output: Path,
    train_file: Path | None,
    selection_file: Path | None,
    device: str,
    split: str,
    reporting_file: Path | None,
) -> Context:
    if split not in ("selection", "reporting"):
        raise ValueError("split must be selection or reporting")
    settings = json.loads(config_source.read_text())
    if not isinstance(settings, dict):
        raise ValueError("configuration must be a JSON object")
    dataset = settings.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset configuration must be an object")

    def data_path(explicit: Path | None, key: str) -> Path:
        if explicit is not None:
            return explicit
        configured = dataset.get(key)
        if not isinstance(configured, str) or not configured:
            raise ValueError(f"dataset.{key} must be a nonempty path string")
        path = Path(configured)
        return path if path.is_absolute() else ROOT / path

    train_file = data_path(train_file, "train_file")
    selection_file = data_path(selection_file, "selection_file")
    if split == "reporting":
        reporting_file = data_path(reporting_file, "reporting_file").resolve()
    identities = tuple(
        file_sha256(p) if p.is_file() else f"missing:{portable_path(p, root=ROOT)}"
        for p in (train_file, selection_file)
    )
    return Context(
        json.dumps(settings, sort_keys=True, allow_nan=False),
        config_source.resolve(),
        output.resolve(),
        train_file.resolve(),
        selection_file.resolve(),
        device,
        (identities[0], identities[1]),
        split,
        reporting_file,
    )
