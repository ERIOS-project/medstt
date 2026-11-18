"""Standalone evaluation with a fake transcriber: metrics, report, cache, and CLI."""

import json

import pytest
from typer.testing import CliRunner

from medstt.cli import app
from medstt.eval import evaluate as evaluation
from medstt.eval.evaluate import (
    BASE_MODEL,
    EvalSettings,
    evaluate,
    model_name,
    parse_models,
    resolve_model,
)
from medstt.io import read_records, write_csv
from medstt.metrics import term_hits, text_metrics, word_errors
from medstt.workspace import Workspace

VOICES = ("male", "female")
WORDS = [f"varioliforme{i}" for i in range(40)]


@pytest.fixture
def ws(tmp_path):
    """A paired synthetic dataset as `tts synthesize` writes it, and a general set."""
    ws = Workspace.init(tmp_path / "ws")
    rows = [
        {
            "file_name": f"audio/s{i}-{voice}.wav",
            "text": f"Le patient présente une acné {word} étendue.",
            "id": f"s{i}-{voice}",
            "word": word,
            "speaker": voice,
            "source": "cim10" if i % 2 else "orphanet",
            "source_id": f"s{i}",
        }
        for i, word in enumerate(WORDS)
        for voice in VOICES
    ]
    for split in ("train", "test"):
        path = ws.root / "datasets" / "synth-medical-words" / split / "metadata.csv"
        write_csv(path, rows, ("file_name", "text"))
    general = [
        {"file_name": f"audio/{i:08d}.wav", "text": f"Les parlements nationaux ont voté {i} lois."}
        for i in range(30)
    ]
    write_csv(ws.root / "datasets" / "vox" / "metadata.csv", general, ("file_name", "text"))
    (ws.root / "runs" / "cohere-medical" / "final").mkdir(parents=True)
    return ws


class FakeTranscriber:
    """Base: misses every medical word. Run: finds it in train/ (training voices), and in
    test/ (test voices) for the first half of the words; one extra error on general speech."""

    def __init__(self, model, calls):
        self.run = model != BASE_MODEL
        self.calls = calls

    def transcribe(self, paths):
        self.calls.append(len(paths))
        return [self._one(str(path)) for path in paths]

    def _one(self, path):
        name = path.rsplit("/", 1)[-1].removesuffix(".wav")
        if name.startswith("s"):
            index = int(name[1:].split("-", 1)[0])
            training_voice = "/train/" in path
            found = self.run and (training_voice or index < 20)
            heard = WORDS[index] if found else "variole forme"
            return f"Le patient présente une acné {heard} étendue."
        number = int(name)
        adjective = "" if number % 5 == 0 else " nationaux"
        extra = " euh" if self.run and number % 3 == 0 else ""
        return f"Les parlements{adjective} ont voté {number} lois.{extra}"


def _factory(calls):
    return lambda model, _settings: FakeTranscriber(model, calls)


def _settings(**overrides):
    values = {
        "models": {"base": BASE_MODEL, "run": "cohere-medical"},
        "synthetic": "synth-medical-words",
        "general": ("vox",),
    }
    return EvalSettings(**{**values, **overrides})


def test_evaluation_measures_the_three_objectives(ws):
    out = evaluate(ws, "first-run", _settings(), transcriber_factory=_factory([]))

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["reference"] == "base"
    assert metrics["models"]["run"].endswith("runs/cohere-medical/final")

    (words,) = metrics["objectives"]["new_words"]
    assert words["dataset"] == "synth-medical-words/train" and words["n"] == 80
    assert words["term_recall"] == {"base": 0.0, "run": 1.0}
    assert words["term_recall_gain"] == 1.0
    assert words["wer_change"] < 0

    (voices,) = metrics["objectives"]["voice_timbres"]
    assert voices["recordings"] == 80
    assert voices["train_voice_gain"] == pytest.approx(1.0)
    assert voices["test_voice_gain"] == pytest.approx(0.5)
    assert voices["gain_ratio"] == pytest.approx(0.5)
    assert voices["per_voice"]["test voices, male"] == {"n": 40, "base": 0.0, "run": 0.5}
    assert voices["per_voice"]["train voices, female"] == {"n": 40, "base": 0.0, "run": 1.0}

    (general,) = metrics["objectives"]["general_french"]
    assert general["dataset"] == "vox" and general["n"] == 30
    assert general["relative_wer_change"] > 0

    test = metrics["results"]["synth-medical-words/test"]["run"]
    assert test["role"] == "test voices"
    assert test["overall"]["term_recall"] == pytest.approx(0.5)
    assert set(test["by_speaker"]) == set(VOICES)
    assert set(test["by_source"]) == {"cim10", "orphanet"}
    assert "term_recall" not in metrics["results"]["vox"]["base"]["overall"]

    report = (out / "report.md").read_text(encoding="utf-8")
    for title in (
        "## Objective 1: learn new words",
        "## Objective 2: keep the learning with other voice timbres",
        "## Objective 3: keep general French",
        "## Most missed terms (`run`)",
    ):
        assert title in report
    assert "| test voices, male | 40 | 0.0 | 50.0 |" in report

    per_term = read_records(out / "per_term.csv")
    assert len(per_term) == 160
    assert {row["run_hit"] for row in per_term} == {"0", "1"}


def test_predictions_are_cached_per_model_path(ws):
    calls = []
    evaluate(ws, "e", _settings(), transcriber_factory=_factory(calls))
    assert sum(calls) == 2 * (160 + 30)

    calls.clear()
    evaluate(ws, "e", _settings(), transcriber_factory=_factory(calls))
    assert calls == []

    other = _settings(models={"base": BASE_MODEL, "run": str(ws.root / "runs")})
    evaluate(ws, "e", other, transcriber_factory=_factory(calls))
    assert sum(calls) == 160 + 30


def test_max_samples_keeps_the_splits_paired(ws):
    settings = _settings(models={"base": BASE_MODEL}, max_samples=10)
    out = evaluate(ws, "base", settings, transcriber_factory=_factory([]))

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["objectives"] == {"new_words": [], "voice_timbres": [], "general_french": []}
    assert metrics["results"]["synth-medical-words/train"]["base"]["overall"]["n"] == 10
    assert metrics["results"]["synth-medical-words/test"]["base"]["overall"]["n"] == 10
    per_term = read_records(out / "per_term.csv")
    ids = {
        split: {r["id"] for r in per_term if r["dataset"].endswith(split)}
        for split in ("train", "test")
    }
    assert ids["train"] == ids["test"]
    assert ids["train"] == {f"s{i}-male" for i in range(0, 40, 4)}
    assert "measured against a second model" in (out / "report.md").read_text(encoding="utf-8")


def test_general_speech_only(ws):
    out = evaluate(ws, "vox", _settings(synthetic=None), transcriber_factory=_factory([]))

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    objectives = metrics["objectives"]
    assert objectives["new_words"] == [] and objectives["voice_timbres"] == []
    assert len(objectives["general_french"]) == 1


def test_synthetic_dataset_needs_a_word_column(ws):
    write_csv(
        ws.root / "datasets" / "plain" / "train" / "metadata.csv",
        [{"file_name": "a.wav", "text": "Bonjour."}],
        ("file_name", "text"),
    )
    with pytest.raises(ValueError, match="word"):
        evaluate(ws, "x", _settings(synthetic="plain"), transcriber_factory=_factory([]))


def test_model_names_and_resolution(ws):
    assert model_name(BASE_MODEL) == "base"
    assert model_name("runs/cohere-medical/final") == "cohere-medical"
    assert parse_models(["base=" + BASE_MODEL, "runs/x/final"]) == {
        "base": BASE_MODEL,
        "x": "runs/x/final",
    }
    with pytest.raises(ValueError):
        parse_models(["a=x", "a=y"])
    final = str((ws.root / "runs" / "cohere-medical" / "final").resolve())
    assert resolve_model(ws, "cohere-medical") == final
    assert resolve_model(ws, "runs/cohere-medical") == final
    assert resolve_model(ws, BASE_MODEL) == BASE_MODEL
    with pytest.raises(FileNotFoundError):
        resolve_model(ws, "./missing")
    with pytest.raises(ValueError):
        EvalSettings(models={"run": "x"}, synthetic="d", reference="base")
    with pytest.raises(ValueError):
        EvalSettings(models={"run": "x"})


def test_cli_reads_the_evaluate_section(ws, monkeypatch):
    calls = []
    monkeypatch.setattr(
        evaluation, "_cohere_transcriber", lambda model, _s: FakeTranscriber(model, calls)
    )
    assert ws.section("evaluate")["synthetic"] == "synth-medical-words"

    result = CliRunner().invoke(
        app, ["evaluate", "--name", "cli", "--general", "vox", "-w", str(ws.root)]
    )

    assert result.exit_code == 0, result.output
    assert "Report written to" in result.output
    metrics = json.loads((ws.root / "evals" / "cli" / "metrics.json").read_text("utf-8"))
    assert list(metrics["models"]) == ["base", "run"]
    assert metrics["settings"]["synthetic"] == "synth-medical-words"
    assert metrics["settings"]["general"] == ["vox"]

    missing = CliRunner().invoke(app, ["evaluate", "--general", "absent", "-w", str(ws.root)])
    assert missing.exit_code == 1 and "metadata" in missing.output


def test_word_errors_add_up_to_the_corpus_wer():
    refs = ["Le patient va bien.", "Une acné varioliforme.", "Bonjour docteur"]
    hyps = ["le patient va mal", "une acné variole forme", "Bonjour, docteur !"]
    pairs = [word_errors(h, r) for h, r in zip(hyps, refs, strict=True)]
    wer = sum(e for e, _ in pairs) / sum(n for _, n in pairs)
    assert wer == pytest.approx(text_metrics(hyps, refs)["wer"])
    assert word_errors("rien", "") == (0, 0)


def test_term_hits_ignore_case_and_accents_but_need_the_whole_word():
    hits = term_hits(["ACNE VARIOLIFORME", "acné variole", "varioliformes"], ["varioliforme"] * 3)
    assert hits == [True, False, False]
