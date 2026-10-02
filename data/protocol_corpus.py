import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from data.corpus import to_units
from framework.checkpoints import file_sha256, write_json


def document_ids(path: Path) -> set[str] | None:
    parquet = pq.ParquetFile(path)
    if "document_sha256" not in parquet.schema_arrow.names:
        return None
    ids = set()
    for batch in parquet.iter_batches(columns=["document_sha256"]):
        for identity in batch.column(0).to_pylist():
            if not isinstance(identity, str) or not identity:
                raise ValueError("document_sha256 must contain non-empty identities")
            ids.add(identity)
    return ids


def documents(path: Path, max_characters: int | None) -> list[str]:
    if max_characters is None:
        max_characters = sys.maxsize
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    result = []
    used = 0
    parquet = pq.ParquetFile(path)
    grouped = "document_sha256" in parquet.schema_arrow.names
    columns = ["text", "document_sha256"] if grouped else ["text"]
    current, lines, seen = None, [], set()

    def append_document() -> None:
        nonlocal used
        text = "\n".join(lines)[: max_characters - used]
        if text:
            result.append(text)
            used += len(text)

    for batch in parquet.iter_batches(columns=columns):
        for row in batch.to_pylist():
            identity = row["document_sha256"] if grouped else object()
            if grouped and (not isinstance(identity, str) or not identity):
                raise ValueError("document_sha256 must contain non-empty identities")
            if current is not None and identity != current:
                append_document()
                if used == max_characters:
                    return result
                seen.add(current)
                lines = []
            if identity != current and identity in seen:
                raise ValueError(
                    "document rows are non-contiguous; cannot reconstruct safely"
                )
            current = identity
            if row["text"]:
                lines.append(row["text"])
    append_document()
    return result


class EventCorpus:
    def __init__(self, train: list[str], selection: list[str], max_bytes: int) -> None:
        if max_bytes < 4:
            raise ValueError("max_bytes must accommodate a four-byte UTF-8 character")
        units = {
            "train": [to_units(text, max_bytes) for text in train],
            "selection": [to_units(text, max_bytes) for text in selection],
        }
        self.vocab = sorted(
            {u for split in units.values() for doc in split for u in doc}
        )
        index = {value: row for row, value in enumerate(self.vocab)}
        self.ids = {}
        self.offsets = {}
        for split, docs in units.items():
            self.ids[split] = np.array(
                [index[u] for doc in docs for u in doc], dtype=np.int64
            )
            self.offsets[split] = np.array(
                [0, *np.cumsum([len(doc) for doc in docs])], dtype=np.int64
            )
        self.max_bytes = max_bytes
        self._window_indices: dict[tuple[str, int], np.ndarray] = {}

    def save(self, root: Path, sources: dict) -> None:
        if (root / "metadata.json").exists():
            existing = json.loads((root / "metadata.json").read_text())
            if (
                existing["sources"] != sources
                or existing["vocab"] != self.vocab
                or existing["max_bytes"] != self.max_bytes
            ):
                raise ValueError(
                    "corpus output belongs to another design; choose a fresh output directory"
                )
            loaded = self.load(root)
            if any(
                not np.array_equal(self.ids[s], loaded.ids[s])
                or not np.array_equal(self.offsets[s], loaded.offsets[s])
                for s in self.ids
            ):
                raise ValueError(
                    "corpus byte selection changed; choose a fresh output directory"
                )
            return
        root.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for split in self.ids:
            for name, values in (
                ("ids", self.ids[split]),
                ("offsets", self.offsets[split]),
            ):
                path = root / f"{split}-{name}.npy"
                np.save(path, values, allow_pickle=False)
                hashes[path.name] = file_sha256(path)
        write_json(
            root / "metadata.json",
            {
                "format": "document-events-v1",
                "vocab": self.vocab,
                "max_bytes": self.max_bytes,
                "sources": sources,
                "sha256": hashes,
            },
        )

    @classmethod
    def load(cls, root: Path) -> "EventCorpus":

        metadata = json.loads((root / "metadata.json").read_text())
        if metadata["format"] != "document-events-v1":
            raise ValueError("unknown event corpus format")
        obj = cls.__new__(cls)
        obj.vocab = metadata["vocab"]
        obj.max_bytes = metadata["max_bytes"]
        obj.ids, obj.offsets = {}, {}
        obj._window_indices = {}
        for filename, expected in metadata["sha256"].items():
            if file_sha256(root / filename) != expected:
                raise ValueError(f"corpus hash mismatch: {filename}")
        for split in ("train", "selection"):
            obj.ids[split] = np.load(root / f"{split}-ids.npy", mmap_mode="r")
            obj.offsets[split] = np.load(root / f"{split}-offsets.npy", mmap_mode="r")
        return obj

    def windows(
        self, split: str, events: int, batch: int, rng: np.random.Generator
    ) -> Tensor:
        offsets = self.offsets[split]
        key = (split, events)
        if key not in self._window_indices:
            counts = np.maximum(np.diff(offsets) - events, 0)
            self._window_indices[key] = counts.cumsum()
        cumulative = self._window_indices[key]
        if not len(cumulative) or cumulative[-1] == 0:
            raise ValueError(f"{split} has no document with {events + 1} units")
        draws = rng.integers(0, int(cumulative[-1]), size=batch)
        document = np.searchsorted(cumulative, draws, side="right")
        prior = np.where(document == 0, 0, cumulative[np.maximum(document - 1, 0)])
        starts = offsets[document] + draws - prior
        return torch.from_numpy(
            self.ids[split][starts[:, None] + np.arange(events + 1)]
        )

    def restricted(self, rows: list[Tensor]) -> tuple["EventCorpus", list[Tensor]]:
        units = sorted({int(unit) for row in rows for unit in row.flatten()})
        index = {unit: position for position, unit in enumerate(units)}
        subset = EventCorpus.__new__(EventCorpus)
        subset.vocab = [self.vocab[unit] for unit in units]
        subset.max_bytes = self.max_bytes
        subset.ids, subset.offsets, subset._window_indices = {}, {}, {}
        return subset, [
            row.apply_(index.__getitem__) for row in (r.clone() for r in rows)
        ]

    def decode(self, rows: Tensor) -> list[list[str]]:
        return [[self.vocab[i] for i in row] for row in rows.tolist()]


def window_sums(
    lengths: np.ndarray, sizes: np.ndarray, span: int
) -> tuple[np.ndarray, np.ndarray]:
    total = np.concatenate(([0], lengths.cumsum()))
    sums = total[span:] - total[:-span]
    ends = np.repeat(sizes.cumsum(), sizes)[: len(sums)]
    return sums, np.arange(len(sums)) + span <= ends


def selection_windows(
    corpus: EventCorpus,
    events: int,
    size: int,
    rng: np.random.Generator,
    *,
    split: str,
) -> tuple[Tensor, np.ndarray]:
    offsets = corpus.offsets[split]
    lengths = np.diff(offsets)
    eligible = np.flatnonzero(lengths > events)
    if len(eligible) < size:
        raise ValueError(
            "shuffled evaluation requires one distinct text group per batch row"
        )
    groups = rng.choice(eligible, size=size, replace=False)
    starts = offsets[groups] + np.array(
        [rng.integers(0, lengths[d] - events) for d in groups]
    )
    rows = torch.from_numpy(corpus.ids[split][starts[:, None] + np.arange(events + 1)])
    return rows, groups
