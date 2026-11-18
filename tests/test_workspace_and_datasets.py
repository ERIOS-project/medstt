import numpy as np
import pytest
import soundfile as sf

from medstt.audio import decode_audio
from medstt.data.loading import DatasetSpec, load_datasets
from medstt.io import write_csv
from medstt.train.common import build_splits
from medstt.train.config import MODEL_ID, TrainConfig
from medstt.tts.base import TTSBackend
from medstt.tts.synthesize import SynthesisSettings, finalize_dataset, synthesize_dataset
from medstt.workspace import Workspace, WorkspaceError


class SineTTS(TTSBackend):
    """Deterministic fake backend: duration grows with the sentence length."""

    name = "fake:sine"

    async def synthesize(self, text, voice):
        sr = 22050
        t = np.arange(int(sr * 0.1 * len(text.split()))) / sr
        return 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


@pytest.fixture
def ws(tmp_path):
    return Workspace.init(tmp_path / "ws")


def test_open_requires_initialized_workspace(tmp_path):
    with pytest.raises(WorkspaceError):
        Workspace.open(tmp_path)


def test_resolve_prefers_workspace_subdirectory(ws):
    (ws.root / "words" / "list.txt").write_text("a\n")
    assert ws.resolve("list.txt", "words") == ws.root / "words" / "list.txt"
    assert ws.resolve("new.jsonl", "sentences") == ws.root / "sentences" / "new.jsonl"


def test_synthesize_shards_finalize_and_load(ws):
    sentences = [
        {"id": f"s{i}", "text": " ".join(["mot"] * n), "word": "mot"}
        for i, n in enumerate([3, 12, 15, 40])  # 0.3 s is too short, 4 s is too long
    ]
    out = ws.root / "datasets" / "fake"
    totals = [0, 0]
    for shard in range(2):
        settings = SynthesisSettings(
            sampling_rate=16000,
            min_duration=1.0,
            max_duration=2.0,
            shard_index=shard,
            num_shards=2,
        )
        written, rejected, failed = synthesize_dataset(sentences, "text", out, SineTTS(), settings)
        totals[0] += written
        totals[1] += rejected
        assert failed == 0
    assert totals == [4, 4]
    # Resuming does nothing once every sentence has been handled.
    settings = SynthesisSettings(sampling_rate=16000, min_duration=1.0, max_duration=2.0)
    assert synthesize_dataset(sentences, "text", out, SineTTS(), settings) == (0, 0, 0)
    assert finalize_dataset(out) == 4

    ds = load_datasets(ws, [DatasetSpec(path="fake", repeat=2)])
    assert len(ds) == 8
    assert set(ds.column_names) == {"audio", "text", "word", "source"}
    assert {path.name for path in (out / "audio").glob("*.wav")} == {
        "s1-male.wav",
        "s1-female.wav",
        "s2-male.wav",
        "s2-female.wav",
    }
    audio = decode_audio(ds[0]["audio"], 16000)
    assert audio.dtype == np.float32 and 1.0 <= len(audio) / 16000 <= 2.0


def test_decode_audio_resamples_and_downmixes(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.zeros((44100, 2), dtype=np.float32), 44100)
    assert decode_audio(str(path), 16000).shape == (16000,)


def test_tts_resume_retries_only_missing_voice(ws):
    class FailFemaleOnce(SineTTS):
        def __init__(self):
            self.calls = []

        async def synthesize(self, text, voice):
            self.calls.append(voice)
            if voice == "female" and self.calls.count("female") == 1:
                raise RuntimeError("temporary failure")
            return await super().synthesize(text, voice)

    backend = FailFemaleOnce()
    records = [{"id": "s0", "text": " ".join(["mot"] * 12)}]
    settings = SynthesisSettings(sampling_rate=16000, min_duration=1.0, max_duration=2.0)
    output = ws.root / "datasets" / "retry"

    assert synthesize_dataset(records, "text", output, backend, settings) == (1, 0, 1)
    assert synthesize_dataset(records, "text", output, backend, settings) == (1, 0, 0)
    assert backend.calls == ["male", "female", "female"]
    assert finalize_dataset(output) == 2


def test_train_config_parsing_and_validation():
    raw = {
        "run_name": "r",
        "data": {"train": ["a", {"path": "b", "max_samples": 10}]},
    }
    cfg = TrainConfig.from_dict(raw)
    assert cfg.data.train[1].max_samples == 10
    assert MODEL_ID == "CohereLabs/cohere-transcribe-03-2026"
    with pytest.raises(ValueError, match="lora"):
        TrainConfig.from_dict({**raw, "method": {"type": "lora"}})
    with pytest.raises(ValueError, match="fixed"):
        TrainConfig.from_dict({**raw, "model": {"name_or_path": "other/model"}})
    with pytest.raises(ValueError, match="16000"):
        TrainConfig.from_dict({**raw, "data": {"train": ["a"], "sampling_rate": 8000}})
    with pytest.raises(ValueError):
        DatasetSpec.parse({"path": "a", "typo": 1})


def test_init_writes_full_training_settings(ws):
    cfg = TrainConfig.from_dict(ws.section("train"))
    assert cfg.run_name == "cohere-medical"
    assert cfg.data.train[0].path == "synth-medical-words/train"
    assert cfg.training["learning_rate"] == 1e-5
    assert set(ws.section("tts")["voice_prompts"]) == {"male", "female"}


def test_multiple_training_paths_and_seeded_eval_split(ws):
    for name in ("synthetic", "existing"):
        dataset_dir = ws.root / "datasets" / name
        (dataset_dir / "audio").mkdir(parents=True)
        records = []
        for index in range(4):
            file_name = f"audio/{index}.wav"
            sf.write(dataset_dir / file_name, np.zeros(1600), 16000)
            records.append({"file_name": file_name, "text": f"{name} sample {index}"})
        write_csv(dataset_dir / "metadata.csv", records, first_columns=("file_name", "text"))

    cfg = TrainConfig.from_dict(
        {
            "run_name": "smoke",
            "seed": 42,
            "data": {"train": ["synthetic", {"path": "existing"}], "eval_split": 0.25},
        }
    )
    train, evaluation = build_splits(ws, cfg)
    _, evaluation_again = build_splits(ws, cfg)

    assert len(train) == 6 and len(evaluation) == 2
    assert {row["source"] for row in [*train, *evaluation]} == {"synthetic", "existing"}
    assert evaluation["text"] == evaluation_again["text"]


def test_duration_mix_and_no_validation(ws):
    for name, duration in (("synthetic", 1), ("real", 2)):
        folder = ws.root / "datasets" / name
        folder.mkdir()
        sf.write(folder / "sample.wav", np.zeros(16000 * duration), 16000)
        write_csv(
            folder / "metadata.csv",
            [{"file_name": "sample.wav", "text": "bonjour"}],
            first_columns=("file_name", "text"),
        )
    specs = [DatasetSpec(path="synthetic", weight=0.3), DatasetSpec(path="real", weight=0.7)]
    ds = load_datasets(ws, specs)
    assert ds["text"] == load_datasets(ws, specs)["text"]
    from medstt.data.loading import audio_duration

    seconds = {
        name: sum(audio_duration(row["audio"]) for row in ds if row["source"] == name)
        for name in ("synthetic", "real")
    }
    assert 0.9 <= seconds["synthetic"] <= 1.9
    assert 2.1 <= seconds["real"] <= 4.1
    cfg = TrainConfig.from_dict(ws.section("train"))
    cfg.data.train = specs
    train, evaluation = build_splits(ws, cfg)
    assert evaluation is None and len(train) == 2  # mixing happens after token filtering
    with pytest.raises(ValueError, match="positive weight"):
        load_datasets(ws, [specs[0], DatasetSpec(path="real")])


@pytest.mark.parametrize("damage", ["missing", "corrupt", "empty"])
def test_tts_resume_repairs_damaged_success_and_finalizer_refuses_it(ws, damage):
    class CountingTTS(SineTTS):
        def __init__(self):
            self.voices = []

        async def synthesize(self, text, voice):
            self.voices.append(voice)
            return await super().synthesize(text, voice)

    backend = CountingTTS()
    records = [{"id": "s0", "text": " ".join(["mot"] * 12)}]
    settings = SynthesisSettings(sampling_rate=16000, min_duration=1.0, max_duration=2.0)
    output = ws.root / "datasets" / "repair"
    assert synthesize_dataset(records, "text", output, backend, settings) == (2, 0, 0)
    original_metadata = (output / "metadata.csv").read_bytes()
    damaged = output / "audio/s0-female.wav"
    intact = (output / "audio/s0-male.wav").read_bytes()
    if damage == "missing":
        damaged.unlink()
    elif damage == "corrupt":
        damaged.write_bytes(b"broken")
    else:
        sf.write(damaged, np.zeros(0), 16000)
    with pytest.raises(ValueError, match="Rerun synthesis"):
        finalize_dataset(output)
    assert (output / "metadata.csv").read_bytes() == original_metadata
    backend.voices.clear()
    assert synthesize_dataset(records, "text", output, backend, settings) == (1, 0, 0)
    assert backend.voices == ["female"]
    assert (output / "audio/s0-male.wav").read_bytes() == intact
    assert finalize_dataset(output) == 2


def test_init_bundles_local_lists_and_preserves_user_edits(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from medstt.cli import app

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    ws = Workspace.init(tmp_path / "workspace")
    files = sorted((ws.root / "local").glob("*.txt"))
    assert len(files) == 4
    assert ws.section("nomenclatures")["local"] == {"format": "local", "path": "local"}
    assert "local" in ws.section("words")["sources"]
    result = CliRunner().invoke(
        app,
        [
            "nomenclature",
            "import",
            "local",
            "--workdir",
            str(ws.root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (ws.root / "nomenclature/normalized/local.jsonl").stat().st_size > 0
    files[0].write_text("My custom list\n")
    config_path = ws.root / "medstt.yaml"
    config_path.write_text("words: {sources: [cim10]}\n")
    Workspace.init(ws.root)
    assert files[0].read_text() == "My custom list\n"
    assert config_path.read_text() == "words: {sources: [cim10]}\n"


@pytest.mark.parametrize("synthetic_count,real_count", [(8, 100), (100, 30)])
def test_mix_keeps_all_synthetic_and_samples_real_by_relative_duration(
    ws, synthetic_count, real_count
):
    from medstt.data.loading import audio_duration

    for name, count in (("synthetic", synthetic_count), ("voxpopuli", real_count)):
        folder = ws.root / "datasets" / name / "train"
        folder.mkdir(parents=True)
        sf.write(folder / "sample.wav", np.zeros(16000), 16000)
        write_csv(
            folder / "metadata.csv",
            [{"file_name": "sample.wav", "text": f"{name}-{i}"} for i in range(count)],
            first_columns=("file_name", "text"),
        )
    specs = [
        DatasetSpec(path="synthetic/train", weight=0.8),
        DatasetSpec(path="voxpopuli/train", weight=0.2),
    ]
    mixed = load_datasets(ws, specs, seed=42)
    synthetic = [row for row in mixed if row["source"] == "synthetic/train"]
    real = [row for row in mixed if row["source"] == "voxpopuli/train"]
    assert len(synthetic) == synthetic_count
    assert len({row["text"] for row in synthetic}) == synthetic_count
    assert len(real) == synthetic_count // 4
    assert len({row["text"] for row in real}) == len(real)
    assert sum(audio_duration(row["audio"]) for row in real) == synthetic_count / 4
    assert mixed["text"] == load_datasets(ws, specs, seed=42)["text"]
    other = load_datasets(ws, specs, seed=43)
    assert {row["text"] for row in other if row["source"] == "voxpopuli/train"} != {
        row["text"] for row in real
    }
