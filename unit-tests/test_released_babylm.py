import json
import zipfile

import pyarrow.parquet as pq
import pytest

from data import released_babylm as released
from framework.checkpoints import file_sha256


@pytest.fixture
def archives(tmp_path, monkeypatch):
    monkeypatch.setattr(released, "SOURCES", ("alpha", "beta"))
    sources = {"alpha": "one\r\ntwó\nthree", "beta": "four\nfive\n"}
    paths, pins = {}, {}
    for split in ("train", "selection", "reporting"):
        path = tmp_path / f"{split}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for source, content in sources.items():
                archive.writestr(f"{split}/{source}.txt", content)
            archive.writestr("../../must-not-extract.txt", "not a corpus member")
        paths[split] = path
        pins[split] = (
            path.name,
            split,
            "txt",
            file_sha256(path),
            "https://example.invalid",
        )
    monkeypatch.setattr(released, "RELEASE", pins)
    return paths, sources


def test_release_preserves_text_sources_and_split_identity(archives, tmp_path):
    paths, sources = archives
    output = tmp_path / "prepared"
    manifest = released.prepare(paths, output, 2)
    assert manifest["original_document_disjointness"] is False
    identities = []
    for split in paths:
        rows = pq.read_table(output / f"{split}.parquet").to_pylist()
        assert {row["release_split"] for row in rows} == {split}
        for source, expected in sources.items():
            assert (
                "".join(row["text"] for row in rows if row["source_identity"] == source)
                == expected
            )
        assert [(r["first_line"], r["last_line"]) for r in rows] == [
            (0, 1),
            (2, 2),
            (0, 1),
        ]
        identities.append({row["group_id"] for row in rows})
    assert not identities[0] & identities[1]
    assert released.prepare(paths, output, 2) == manifest
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert not (tmp_path / "must-not-extract.txt").exists()
    with pytest.raises(ValueError, match="grouping changed"):
        released.prepare(paths, output, 3)


def test_wrong_release_fails_before_writing(archives, tmp_path):
    paths, _ = archives
    with zipfile.ZipFile(paths["reporting"], "a") as archive:
        archive.writestr("extra", "modified")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="pinned official"):
        released.prepare(paths, output, 2)
    assert not output.exists()


def test_changed_prepared_data_is_rejected(archives, tmp_path):
    paths, _ = archives
    output = tmp_path / "prepared"
    released.prepare(paths, output, 2)
    (output / "train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared released corpus has changed"):
        released.prepare(paths, output, 2)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_group_size(archives, tmp_path, value):
    paths, _ = archives
    with pytest.raises(ValueError, match="positive integer"):
        released.prepare(paths, tmp_path / "prepared", value)
