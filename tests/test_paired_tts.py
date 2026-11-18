"""Paired synthetic datasets keep matching transcripts despite rejects and retries."""

from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from medstt.cli import app
from medstt.io import read_records
from medstt.tts.synthesize import (
    SynthesisSettings,
    finalize_paired_datasets,
    synthesize_dataset,
)
from medstt.workspace import Workspace


class FakeTTS:
    name = "fake"

    async def close(self):
        pass

    def __init__(self, reject=False, fail=False):
        self.reject = reject
        self.fail = fail

    async def synthesize(self, text, voice):
        if self.fail and voice == "female":
            raise RuntimeError("temporary error")
        length = 10 if self.reject and voice == "male" else 1600
        return np.ones(length, dtype=np.float32) * 0.1, 16000


SETTINGS = SynthesisSettings(16000, 0.05, 1)
SENTENCES = [{"id": "s1", "text": "Une phrase médicale."}]


def test_pair_rejection_excludes_same_slot_from_both_datasets(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), SETTINGS)
    synthesize_dataset(SENTENCES, "text", test, FakeTTS(reject=True), SETTINGS)
    assert finalize_paired_datasets(train, test) == 1
    left, right = [read_records(d / "metadata.csv") for d in (train, test)]
    assert [r["id"] for r in left] == [r["id"] for r in right] == ["s1-female"]
    assert left[0]["text"] == right[0]["text"]


def test_pair_failure_blocks_finalization_until_retry(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), SETTINGS)
    synthesize_dataset(SENTENCES, "text", test, FakeTTS(fail=True), SETTINGS)
    with pytest.raises(ValueError, match="unfinished"):
        finalize_paired_datasets(train, test)
    assert len(read_records(train / "metadata.csv")) == 2
    assert synthesize_dataset(SENTENCES, "text", test, FakeTTS(), SETTINGS) == (1, 0, 0)
    assert finalize_paired_datasets(train, test) == 2
    (test / "audio/s1-female.wav").unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        finalize_paired_datasets(train, test)


def test_pair_rejects_mismatched_texts_and_identical_directories(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), SETTINGS)
    synthesize_dataset([{"id": "s1", "text": "Autre phrase."}], "text", test, FakeTTS(), SETTINGS)
    with pytest.raises(ValueError, match="text mismatch"):
        finalize_paired_datasets(train, test)
    assert len(read_records(train / "metadata.csv")) == 2
    with pytest.raises(ValueError, match="must differ"):
        finalize_paired_datasets(train, train)


def test_cli_generates_both_splits_with_separate_prompts(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "sentences/cardio.txt").write_text("Une phrase médicale.\n")
    prompts = []

    class Backend(FakeTTS):
        def __init__(self, voice_prompts, **kwargs):
            super().__init__()
            prompts.append(voice_prompts)

        def with_voice_prompts(self, voice_prompts):
            return Backend(voice_prompts)

    def load_module(name):
        if name == "medstt.tts.backends":
            return SimpleNamespace(
                QwenVoiceDesign=Backend,
                MODEL_ID="fake",
                DEFAULT_VOICE_PROMPTS={},
                DEFAULT_TEST_VOICE_PROMPTS={},
            )
        return import_module(name)

    monkeypatch.setattr("medstt.cli.import_module", load_module)
    args = ["--workdir", str(ws.root)]
    result = CliRunner().invoke(
        app,
        [
            "tts",
            "synthesize",
            "cardio.txt",
            "-o",
            "paired",
            "--min-duration",
            "0.05",
            *args,
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert not (ws.root / "datasets/paired/metadata.csv").exists()
    for split in ("train", "test"):
        assert len(read_records(ws.root / f"datasets/paired/{split}/metadata.csv")) == 2
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "finalize",
            "paired",
            *args,
        ],
    )
    assert result.exit_code == 0, result.output


def test_paired_shards_finalize_with_identical_ids(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    sentences = [{"id": str(i), "text": f"Phrase numéro {i}."} for i in range(3)]
    for shard in range(4):
        settings = SynthesisSettings(16000, 0.05, 1, shard_index=shard, num_shards=4)
        for directory in (train, test):
            synthesize_dataset(sentences, "text", directory, FakeTTS(), settings)
    assert finalize_paired_datasets(train, test) == 6
    for directory in (train, test):
        assert synthesize_dataset(sentences, "text", directory, FakeTTS(), SETTINGS) == (0, 0, 0)
    left, right = [read_records(d / "metadata.csv") for d in (train, test)]
    assert [(r["id"], r["text"]) for r in left] == [(r["id"], r["text"]) for r in right]


def test_metadata_grows_after_each_shard_and_is_repaired_on_resume(tmp_path):
    train = tmp_path / "train"
    first = replace(SETTINGS, num_shards=2, shard_index=0)
    second = replace(SETTINGS, num_shards=2, shard_index=1)
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), first)
    assert len(read_records(train / "metadata.csv")) == 1
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), second)
    assert len(read_records(train / "metadata.csv")) == 2
    (train / "metadata.csv").unlink()
    assert synthesize_dataset(SENTENCES, "text", train, FakeTTS(), second) == (0, 0, 0)
    assert len(read_records(train / "metadata.csv")) == 2


def test_train_metadata_available_before_first_test_request(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "sentences/in.txt").write_text("Une phrase médicale.\n")
    train = ws.root / "datasets/synth-medical-words/train"
    test = ws.root / "datasets/synth-medical-words/test"
    checked = []

    class Backend(FakeTTS):
        def __init__(self, **kwargs):
            super().__init__()
            self.is_test = False

        def with_voice_prompts(self, prompts):
            backend = Backend()
            backend.is_test = True
            return backend

        async def synthesize(self, text, voice):
            if self.is_test:
                assert len(read_records(train / "metadata.csv")) == 2
                assert not (test / "metadata.csv").exists()
                checked.append(voice)
                raise RuntimeError("test interrupted")
            return await super().synthesize(text, voice)

    monkeypatch.setattr("medstt.tts.backends.QwenVoiceDesign", Backend)
    result = CliRunner().invoke(
        app, ["tts", "synthesize", "in.txt", "--workdir", str(ws.root), "--min-duration", "0.05"]
    )
    assert result.exit_code == 1
    assert sorted(checked) == ["female", "male"]
    assert len(read_records(train / "metadata.csv")) == 2
    assert read_records(test / "metadata.csv") == []


def test_split_publication_explains_incomplete_resume(tmp_path, caplog):
    train = tmp_path / "train"
    first = replace(SETTINGS, num_shards=2, shard_index=0)
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), first)
    assert "missing progress: 1" in caplog.text
    assert "s1-female" in caplog.text
    caplog.clear()
    synthesize_dataset(SENTENCES, "text", train, FakeTTS(), SETTINGS)
    changed = [{"id": "s1", "text": "Une autre phrase."}]
    synthesize_dataset(changed, "text", train, FakeTTS(), SETTINGS)
    assert "text mismatch: 2" in caplog.text
    assert "publishing partial metadata.csv" in caplog.text
    assert read_records(train / "metadata.csv") == []
