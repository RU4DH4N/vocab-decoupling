import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from framework.checkpoints import file_sha256, write_json

RELEASE = {
    "train": (
        "train_100M.zip",
        "train_100M",
        "train",
        "37a92db26f391e436febd592453084394d0c0830c8c9bdb30be781e440cdc826",
        "https://osf.io/download/ywea7/",
    ),
    "selection": (
        "dev.zip",
        "dev",
        "dev",
        "b5af75aeb1ee5a36f65ee98f316fdae73b16a0ba3767cca3f3dfa5eaac2dc621",
        "https://osf.io/download/uyd7v/",
    ),
    "reporting": (
        "test.zip",
        "test",
        "test",
        "12bd1c9c89f4f82156feb71b287d47f713429ee1292e938c6ab463a882816cef",
        "https://osf.io/download/ftwu3/",
    ),
}

SOURCES = (
    "bnc_spoken",
    "childes",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
    "switchboard",
)


SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("source_identity", pa.string()),
        ("release_split", pa.string()),
        ("first_line", pa.int64()),
        ("last_line", pa.int64()),
        ("group_id", pa.string()),
    ]
)
WRITE_ROWS = 1024


def line_groups(
    stream: Iterable[str], lines_per_group: int
) -> Iterator[tuple[int, int, str]]:
    group: list[str] = []
    first = 0
    index = -1
    for index, line in enumerate(stream):
        if not group:
            first = index
        group.append(line)
        if len(group) == lines_per_group:
            yield first, index, "".join(group)
            group = []
    if group:
        yield first, index, "".join(group)


def write_split(
    archive: Path,
    output: Path,
    *,
    split: str,
    directory: str,
    suffix: str,
    lines_per_group: int,
) -> dict:
    records: list[dict] = []
    counts = {}
    with zipfile.ZipFile(archive) as zipped, pq.ParquetWriter(output, SCHEMA) as writer:
        for source in SOURCES:
            entry = {"lines": 0, "utf8_bytes": 0, "groups": 0}
            counts[source] = entry
            member = f"{directory}/{source}.{suffix}"
            with (
                zipped.open(member) as raw,
                io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream,
            ):
                for first, last, text in line_groups(stream, lines_per_group):
                    identity = hashlib.sha256(
                        f"{split}\0{source}\0{first}\0{last}\0{text}".encode()
                    ).hexdigest()
                    records.append(
                        {
                            "text": text,
                            "source_identity": source,
                            "release_split": split,
                            "first_line": first,
                            "last_line": last,
                            "group_id": identity,
                        }
                    )
                    entry["lines"] = last + 1
                    entry["groups"] += 1
                    entry["utf8_bytes"] += len(text.encode())
                    if len(records) >= WRITE_ROWS:
                        writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                        records.clear()
        if records:
            writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
    return counts


def prepare(archives: dict[str, Path], output: Path, lines_per_group: int) -> dict:
    if type(lines_per_group) is not int or lines_per_group <= 0:
        raise ValueError("lines_per_group must be a positive integer")
    if set(archives) != set(RELEASE):
        raise ValueError("provide all three released archives")
    for split, path in archives.items():
        if file_sha256(path) != RELEASE[split][3]:
            raise ValueError(
                f"{split} archive does not match the pinned official 2025 release"
            )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["lines_per_group"] != lines_per_group:
            raise ValueError(
                "grouping changed; choose a fresh dataset output directory"
            )
        for split in RELEASE:
            if (
                file_sha256(output / f"{split}.parquet")
                != manifest["outputs"][split]["sha256"]
            ):
                raise ValueError("prepared released corpus has changed")
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for split, (_, directory, suffix, digest, url) in RELEASE.items():
        path = output / f"{split}.parquet"
        if path.exists():
            raise ValueError("uncommitted dataset output exists; use a fresh directory")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{split}.", suffix=".parquet", dir=output
        )
        os.close(descriptor)
        temp = Path(temporary)
        try:
            counts = write_split(
                archives[split],
                temp,
                split=split,
                directory=directory,
                suffix=suffix,
                lines_per_group=lines_per_group,
            )
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        results[split] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "archive_sha256": digest,
            "publisher_url": url,
            "sources": counts,
        }
    manifest = {
        "release": "BabyLM-2025-text",
        "lines_per_group": lines_per_group,
        "original_document_disjointness": False,
        "assignment": "official train/dev/test preserved",
        "grouping": "consecutive fixed-line computational contexts; not original documents",
        "outputs": results,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--dev-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lines-per-group", type=int, required=True)
    args = parser.parse_args()
    manifest = prepare(
        {
            "train": args.train_archive,
            "selection": args.dev_archive,
            "reporting": args.test_archive,
        },
        args.output,
        args.lines_per_group,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
