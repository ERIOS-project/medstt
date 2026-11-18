"""Import publication and failures without network access."""

import pytest

from medstt.data import hub


class FakeDataset:
    def __init__(self, fail=False):
        self.fail = fail

    def cast_column(self, *args):
        return self

    def __iter__(self):
        yield {"audio": {"bytes": b"audio", "path": "sample.wav"}, "text": "bonjour"}
        if self.fail:
            raise OSError("interrupted stream")


def import_into(path):
    return hub.import_dataset(
        "fake",
        path,
        config=None,
        split="train",
        audio_column="audio",
        text_column="text",
        max_samples=None,
        streaming=True,
    )


def test_import_refuses_existing_dataset_before_downloading(tmp_path, monkeypatch):
    (tmp_path / "metadata.csv").write_text("old metadata")
    (tmp_path / "audio.wav").write_bytes(b"old audio")

    def unexpected(*args, **kwargs):
        pytest.fail("Existing output must be rejected before loading the source")

    monkeypatch.setattr(hub, "load_dataset", unexpected)
    with pytest.raises(FileExistsError, match="new output name"):
        import_into(tmp_path)
    assert (tmp_path / "metadata.csv").read_text() == "old metadata"
    assert (tmp_path / "audio.wav").read_bytes() == b"old audio"


@pytest.mark.parametrize("empty_destination", [False, True])
def test_interrupted_import_publishes_nothing_and_can_retry(
    tmp_path, monkeypatch, empty_destination
):
    output = tmp_path / "dataset"
    if empty_destination:
        output.mkdir()
    monkeypatch.setattr(hub, "load_dataset", lambda *a, **kw: FakeDataset(fail=True))
    with pytest.raises(OSError, match="interrupted stream"):
        import_into(output)
    assert not output.exists() or not list(output.iterdir())
    assert not list(tmp_path.glob(".dataset-*"))
    monkeypatch.setattr(hub, "load_dataset", lambda *a, **kw: FakeDataset())
    assert import_into(output) == 1
    assert (output / "audio/00000000.wav").read_bytes() == b"audio"
    assert "bonjour" in (output / "metadata.csv").read_text()


def test_cli_imports_official_splits_under_one_output(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from medstt.cli import app
    from medstt.io import read_records
    from medstt.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    calls = []

    class SplitDataset(FakeDataset):
        def __init__(self, split):
            self.split = split

        def __iter__(self):
            for i in range(2 if self.split == "train" else 1):
                yield {
                    "audio": {"bytes": b"audio", "path": "sample.wav"},
                    "text": f"{self.split} sentence {i}",
                }

    def load(source, config, *, split, streaming):
        calls.append(split)
        return SplitDataset(split)

    monkeypatch.setattr(hub, "load_dataset", load)
    args = ["dataset", "import", "fake", "-o", "real", "--workdir", str(ws.root)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == ["train", "test"]
    root = ws.root / "datasets/real"
    assert not (root / "metadata.csv").exists()
    train = read_records(root / "train/metadata.csv")
    test = read_records(root / "test/metadata.csv")
    assert len(train) == 2 and len(test) == 1
    assert all(row["text"].startswith("train") for row in train)
    assert test[0]["text"] == "test sentence 0"
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert calls == ["train", "test"]  # preflight prevents redownloading


def test_cli_can_retry_only_failed_test_split(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from medstt.cli import app
    from medstt.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    calls = []
    fail = True

    def load(source, config, *, split, streaming):
        calls.append(split)
        return FakeDataset(fail=fail and split == "test")

    monkeypatch.setattr(hub, "load_dataset", load)
    args = ["dataset", "import", "fake", "-o", "real", "--workdir", str(ws.root)]
    assert CliRunner().invoke(app, args).exit_code != 0
    root = ws.root / "datasets/real"
    train = (root / "train/metadata.csv").read_bytes()
    assert not (root / "test").exists()
    fail = False
    result = CliRunner().invoke(app, [*args, "--split", "test"])
    assert result.exit_code == 0, result.output
    assert calls == ["train", "test", "test"]
    assert (root / "train/metadata.csv").read_bytes() == train
    assert (root / "test/metadata.csv").is_file()


@pytest.mark.parametrize("split", ["../outside", ".", "/absolute"])
def test_cli_rejects_unsafe_split_paths(tmp_path, monkeypatch, split):
    from typer.testing import CliRunner

    from medstt.cli import app
    from medstt.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws")
    monkeypatch.setattr(hub, "load_dataset", lambda *a, **kw: pytest.fail("Unexpected load"))
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "import",
            "fake",
            "-o",
            "real",
            "--split",
            split,
            "--workdir",
            str(ws.root),
        ],
    )
    assert result.exit_code != 0
    assert "single directory name" in result.output
