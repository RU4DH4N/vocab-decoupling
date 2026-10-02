from pathlib import Path

from framework.paths import portable_path


def test_artifact_reference_is_repository_relative(tmp_path):
    assert (
        portable_path(tmp_path / "artifacts/run/model.pt", root=tmp_path)
        == "artifacts/run/model.pt"
    )


def test_relative_and_absolute_inputs_have_the_same_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert portable_path(Path("artifacts/run/../model.pt"), root=tmp_path) == (
        portable_path(tmp_path / "artifacts/model.pt", root=tmp_path)
    )


def test_reference_survives_repository_relocation(tmp_path):
    references = [
        portable_path(tmp_path / host / "artifacts/run/model.pt", root=tmp_path / host)
        for host in ("laptop", "remote")
    ]
    assert references == ["artifacts/run/model.pt"] * 2


def test_external_path_is_not_disguised_as_a_repo_file(tmp_path):
    external = tmp_path / "repo-other/model.pt"
    assert (
        portable_path(external, root=tmp_path / "repo") == external.resolve().as_posix()
    )
