"""Evaluate Cohere Transcribe models on synthetic medical speech and general French.

The evaluation measures three objectives, comparing each model with a reference
model (the base model):

1. Learn new words: term recall and WER on ``<synthetic>/train``, the training
   sentences read by the training voices.
2. Keep that learning with other voice timbres: the gain over the reference on
   ``<synthetic>/test`` (the same sentences, test voices) compared with the gain
   on ``<synthetic>/train``.
3. Keep general French: WER on general speech datasets.

``<synthetic>`` is a paired dataset written by ``medstt tts synthesize``.
Predictions are cached in ``evals/<name>/predictions.jsonl``, so an interrupted
run resumes.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medstt.io import append_jsonl, iter_jsonl, read_records
from medstt.metrics import term_hits, text_metrics, word_errors
from medstt.workspace import Workspace

log = logging.getLogger(__name__)

BASE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"
PER_TERM_FILE = "per_term.csv"
REPORT_FILE = "report.md"
CACHE_CHUNK = 256
TRAIN_VOICES, TEST_VOICES, GENERAL = "train voices", "test voices", "general"
NEW_WORDS, VOICE_TIMBRES, GENERAL_FRENCH = "new_words", "voice_timbres", "general_french"
OBJECTIVES = (NEW_WORDS, VOICE_TIMBRES, GENERAL_FRENCH)


@dataclass(frozen=True)
class EvalSettings:
    """Models, datasets, batching, and the thresholds of the report."""

    models: dict[str, str]
    synthetic: str | None = None
    general: tuple[str, ...] = ()
    reference: str | None = None
    batch_size: int = 16
    max_new_tokens: int = 256
    max_samples: int | None = None

    def __post_init__(self) -> None:
        """Reject an empty model or dataset list and an unknown reference."""
        if not self.models:
            raise ValueError("No model to evaluate; set evaluate.models or pass --model.")
        if not self.synthetic and not self.general:
            raise ValueError(
                "No dataset to evaluate; set evaluate.synthetic or evaluate.general, "
                "or pass --synthetic or --general."
            )
        if self.reference is not None and self.reference not in self.models:
            raise ValueError(
                f"Reference model {self.reference!r} is not among {list(self.models)}."
            )

    @property
    def reference_name(self) -> str:
        """Return the model the others are compared with."""
        if self.reference:
            return self.reference
        return "base" if "base" in self.models else next(iter(self.models))


@dataclass
class EvalDataset:
    """Rows of one evaluation dataset and its role: train voices, test voices or general."""

    name: str
    role: str
    rows: list[dict[str, Any]]

    @property
    def medical(self) -> bool:
        """Whether the rows are synthetic medical sentences with a target word."""
        return self.role != GENERAL


TranscriberFactory = Callable[[str, EvalSettings], Any]


def model_name(spec: str) -> str:
    """Name a model given without a name: ``base``, the run name, or the last path part."""
    if spec == BASE_MODEL:
        return "base"
    path = Path(spec)
    return path.parent.name if path.name == "final" and path.parent.name else path.name


def parse_models(values: list[str] | dict[str, str]) -> dict[str, str]:
    """Read ``name=model`` entries (or bare models) into an ordered mapping."""
    if isinstance(values, dict):
        return {str(k): str(v) for k, v in values.items()}
    models: dict[str, str] = {}
    for value in values:
        name, sep, spec = value.partition("=")
        if not sep:
            name, spec = model_name(value), value
        if not name or not spec:
            raise ValueError(f"Invalid model {value!r}; expected name=model or a model.")
        if name in models:
            raise ValueError(f"Model name {name!r} given twice.")
        models[name] = spec
    return models


def resolve_model(ws: Workspace, spec: str) -> str:
    """Return a local model directory for ``spec``, or ``spec`` itself as a Hub ID.

    A run directory (``runs/<name>`` or just ``<name>``) resolves to its ``final``
    subdirectory.
    """
    raw = Path(spec).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [ws.root / raw, ws.root / "runs" / raw, ws.root / "models" / raw, Path.cwd() / raw]
    )
    for candidate in candidates:
        if candidate.is_dir():
            final = candidate / "final"
            return str((final if final.is_dir() else candidate).resolve())
    if raw.is_absolute() or spec.startswith((".", "~")):
        raise FileNotFoundError(f"Model directory {spec} not found.")
    return spec


def load_eval_dataset(ws: Workspace, name: str, role: str) -> EvalDataset:
    """Read an AudioFolder dataset (``metadata.csv`` or ``metadata.jsonl``)."""
    path = ws.resolve(name, "datasets")
    metadata = next(
        (path / f for f in ("metadata.csv", "metadata.jsonl") if (path / f).is_file()), None
    )
    if metadata is None:
        raise FileNotFoundError(
            f"{path} has no metadata.csv or metadata.jsonl; "
            "import it with `medstt dataset import` or synthesize it with `medstt tts synthesize`."
        )
    rows = []
    for record in read_records(metadata):
        file_name = str(record.get("file_name") or "")
        if not file_name:
            raise ValueError(f"{metadata}: a row has no file_name.")
        rows.append(
            {
                "id": str(record.get("id") or file_name),
                "audio": str(path / file_name),
                "text": str(record.get("text") or ""),
                "word": _clean(record.get("word")),
                "speaker": _clean(record.get("speaker")),
                "source": _clean(record.get("source")),
            }
        )
    rows = [row for row in rows if row["text"].strip()]
    if not rows:
        raise ValueError(f"{metadata} has no row with a transcript.")
    if role != GENERAL and not any(row["word"] for row in rows):
        raise ValueError(f"{metadata} has no `word` column; it is not a synthetic medical set.")
    return EvalDataset(name=name, role=role, rows=rows)


def load_eval_datasets(ws: Workspace, settings: EvalSettings) -> list[EvalDataset]:
    """Load the train and test splits of the synthetic dataset, then the general sets.

    With ``max_samples``, the same recordings are kept in both synthetic splits, so
    that they stay paired.
    """
    datasets = []
    if settings.synthetic:
        train = load_eval_dataset(ws, f"{settings.synthetic}/train", TRAIN_VOICES)
        test = load_eval_dataset(ws, f"{settings.synthetic}/test", TEST_VOICES)
        if settings.max_samples is not None:
            train.rows = _spread(train.rows, settings.max_samples)
            kept = {row["id"] for row in train.rows}
            test.rows = [row for row in test.rows if row["id"] in kept]
        datasets += [train, test]
    for name in settings.general:
        dataset = load_eval_dataset(ws, name, GENERAL)
        if settings.max_samples is not None:
            dataset.rows = _spread(dataset.rows, settings.max_samples)
        datasets.append(dataset)
    return datasets


def _spread(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Keep ``size`` rows evenly spaced over the dataset, so a rerun keeps the same ones."""
    if len(rows) <= size:
        return rows
    return [rows[i * len(rows) // size] for i in range(size)]


def _clean(value: Any) -> str:
    """Return a CSV cell as text, with empty and null markers as ''."""
    text = "" if value is None else str(value).strip()
    return "" if text in ("None", "nan", "null") else text


def evaluate(
    ws: Workspace,
    name: str,
    settings: EvalSettings,
    transcriber_factory: TranscriberFactory | None = None,
) -> Path:
    """Transcribe every dataset with every model, then write metrics and the report."""
    datasets = load_eval_datasets(ws, settings)
    models = {model: resolve_model(ws, spec) for model, spec in settings.models.items()}
    out_dir = ws.subdir("evals") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = transcriber_factory or _cohere_transcriber

    predictions = _predict_all(out_dir / PREDICTIONS_FILE, models, datasets, settings, factory)
    metrics = compute_metrics(datasets, predictions, settings)
    metrics["models"] = models
    metrics["settings"] = asdict(settings)
    (out_dir / METRICS_FILE).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_per_term(out_dir / PER_TERM_FILE, datasets, predictions, list(models))
    (out_dir / REPORT_FILE).write_text(
        render_report(name, metrics, datasets, predictions, settings), encoding="utf-8"
    )
    return out_dir


def _cohere_transcriber(model: str, settings: EvalSettings):
    """Load the real transcriber (imports torch and transformers)."""
    from medstt.inference import CohereTranscriber

    return CohereTranscriber(
        model, batch_size=settings.batch_size, max_new_tokens=settings.max_new_tokens
    )


Predictions = dict[tuple[str, str], list[str]]


def _predict_all(
    cache: Path,
    models: dict[str, str],
    datasets: list[EvalDataset],
    settings: EvalSettings,
    factory: TranscriberFactory,
) -> Predictions:
    """Return the predictions of each (model, dataset), transcribing what is not cached."""
    done: dict[tuple[str, str, str], str] = {}
    for record in iter_jsonl(cache):
        if models.get(record["model"]) == record.get("model_path"):
            done[(record["model"], record["dataset"], record["id"])] = record["prediction"]

    result: Predictions = {}
    for model, path in models.items():
        todo = [
            (dataset, row)
            for dataset in datasets
            for row in dataset.rows
            if (model, dataset.name, row["id"]) not in done
        ]
        if todo:
            log.info("%s: %d files to transcribe with %s", model, len(todo), path)
            transcriber = factory(path, settings)
            for chunk in _chunks(todo, CACHE_CHUNK):
                texts = transcriber.transcribe([row["audio"] for _, row in chunk])
                records = []
                for (dataset, row), text in zip(chunk, texts, strict=True):
                    done[(model, dataset.name, row["id"])] = text
                    records.append(
                        {
                            "model": model,
                            "model_path": path,
                            "dataset": dataset.name,
                            "id": row["id"],
                            "reference": row["text"],
                            "prediction": text,
                            "word": row["word"],
                            "speaker": row["speaker"],
                        }
                    )
                append_jsonl(cache, records)
            del transcriber
        for dataset in datasets:
            result[(model, dataset.name)] = [
                done[(model, dataset.name, row["id"])] for row in dataset.rows
            ]
    return result


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _wer(rows: list[dict[str, Any]], predictions: list[str]) -> float:
    """Corpus WER: all word errors over all reference words."""
    pairs = [word_errors(p, row["text"]) for p, row in zip(predictions, rows, strict=True)]
    words = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / words if words else float("nan")


def _recall(rows: list[dict[str, Any]], predictions: list[str]) -> float:
    """Share of recordings whose transcript contains the row's word."""
    if not rows:
        return float("nan")
    return sum(term_hits(predictions, [row["word"] for row in rows])) / len(rows)


def _select(rows, preds, keep: Callable[[dict[str, Any]], bool]):
    pairs = [(row, pred) for row, pred in zip(rows, preds, strict=True) if keep(row)]
    return [row for row, _ in pairs], [pred for _, pred in pairs]


def compute_metrics(
    datasets: list[EvalDataset], predictions: Predictions, settings: EvalSettings
) -> dict[str, Any]:
    """Return per-model results and the three objectives measured against the reference."""
    models = list(settings.models)
    reference = settings.reference_name
    results = {
        dataset.name: {m: _summary(dataset, predictions[(m, dataset.name)]) for m in models}
        for dataset in datasets
    }
    by_role = {dataset.role: dataset for dataset in datasets if dataset.medical}
    objectives: dict[str, list[dict[str, Any]]] = {name: [] for name in OBJECTIVES}
    for model in (m for m in models if m != reference):
        if TRAIN_VOICES in by_role:
            train = by_role[TRAIN_VOICES]
            objectives[NEW_WORDS].append(_new_words(train, predictions, model, reference))
            if TEST_VOICES in by_role:
                test = by_role[TEST_VOICES]
                objectives[VOICE_TIMBRES].append(
                    _voice_timbres(train, test, predictions, model, reference)
                )
        for dataset in (d for d in datasets if not d.medical):
            objectives[GENERAL_FRENCH].append(
                _general_french(dataset, predictions, model, reference)
            )
    return {"reference": reference, "results": results, "objectives": objectives}


def _summary(dataset: EvalDataset, predictions: list[str]) -> dict[str, Any]:
    """Overall WER/CER (and term recall), then per voice and per nomenclature source."""
    rows = dataset.rows
    overall = {"n": len(rows), **text_metrics(predictions, [row["text"] for row in rows])}
    if dataset.medical:
        overall["term_recall"] = _recall(rows, predictions)
    summary: dict[str, Any] = {"role": dataset.role, "overall": overall}
    if not dataset.medical:
        return summary
    for key in ("speaker", "source"):
        groups = sorted({row[key] for row in rows if row[key]})
        if not groups:
            continue
        summary[f"by_{key}"] = {}
        for group in groups:
            sub_rows, sub_preds = _select(rows, predictions, lambda r, k=key, g=group: r[k] == g)
            summary[f"by_{key}"][group] = {
                "n": len(sub_rows),
                "wer": _wer(sub_rows, sub_preds),
                "term_recall": _recall(sub_rows, sub_preds),
            }
    return summary


def _new_words(train: EvalDataset, predictions, model: str, reference: str) -> dict[str, Any]:
    """Term recall and WER of the training sentences read by the training voices."""
    rows = train.rows
    ref, run = predictions[(reference, train.name)], predictions[(model, train.name)]
    recall = {reference: _recall(rows, ref), model: _recall(rows, run)}
    wer = {reference: _wer(rows, ref), model: _wer(rows, run)}
    gain = recall[model] - recall[reference]
    return {
        "dataset": train.name,
        "model": model,
        "reference": reference,
        "n": len(rows),
        "term_recall": recall,
        "term_recall_gain": gain,
        "wer": wer,
        "wer_change": wer[model] - wer[reference],
    }


def _voice_timbres(train: EvalDataset, test: EvalDataset, predictions, model, reference):
    """Gain over the reference with training voices versus test voices, same recordings.

    The two splits hold the same sentence/voice slots under the same IDs; only the
    IDs present in both count.
    """
    shared = {row["id"] for row in train.rows} & {row["id"] for row in test.rows}
    gains, per_voice = {}, {}
    for dataset in (train, test):

        def keep(row):
            return row["id"] in shared

        rows, ref = _select(dataset.rows, predictions[(reference, dataset.name)], keep)
        _, run = _select(dataset.rows, predictions[(model, dataset.name)], keep)
        gains[dataset.role] = _recall(rows, run) - _recall(rows, ref)
        for voice in sorted({row["speaker"] for row in rows if row["speaker"]}):
            v_rows, v_ref = _select(rows, ref, lambda r, v=voice: r["speaker"] == v)
            _, v_run = _select(rows, run, lambda r, v=voice: r["speaker"] == v)
            per_voice[f"{dataset.role}, {voice}"] = {
                "n": len(v_rows),
                reference: _recall(v_rows, v_ref),
                model: _recall(v_rows, v_run),
            }
    train_gain, test_gain = gains[TRAIN_VOICES], gains[TEST_VOICES]
    ratio = test_gain / train_gain if train_gain > 0 else float("nan")
    return {
        "model": model,
        "reference": reference,
        "recordings": len(shared),
        "train_voice_gain": train_gain,
        "test_voice_gain": test_gain,
        "gain_ratio": ratio,
        "per_voice": per_voice,
    }


def _general_french(dataset, predictions, model: str, reference: str) -> dict[str, Any]:
    """Relative change of the WER on general speech."""
    rows = dataset.rows
    wer = {
        reference: _wer(rows, predictions[(reference, dataset.name)]),
        model: _wer(rows, predictions[(model, dataset.name)]),
    }
    base = wer[reference]
    change = (wer[model] - base) / base if base else float("nan")
    return {
        "dataset": dataset.name,
        "model": model,
        "reference": reference,
        "n": len(rows),
        "wer": wer,
        "relative_wer_change": change,
    }


def _write_per_term(
    path: Path, datasets: list[EvalDataset], predictions: Predictions, models: list[str]
) -> None:
    """Write one row per medical recording with each model's transcript and hit."""
    columns = ["dataset", "id", "word", "source", "speaker", "reference"]
    for model in models:
        columns += [f"{model}_prediction", f"{model}_hit"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for dataset in (d for d in datasets if d.medical):
            per_model = {m: predictions[(m, dataset.name)] for m in models}
            hits = {m: term_hits(per_model[m], [r["word"] for r in dataset.rows]) for m in models}
            for i, row in enumerate(dataset.rows):
                line = {"dataset": dataset.name, "reference": row["text"]}
                line.update({k: row[k] for k in ("id", "word", "source", "speaker")})
                for model in models:
                    line[f"{model}_prediction"] = per_model[model][i]
                    line[f"{model}_hit"] = int(hits[model][i])
                writer.writerow(line)


def _pct(value: float) -> str:
    return "n/a" if value != value else f"{100 * value:.1f}"


def render_report(
    name: str,
    metrics: dict[str, Any],
    datasets: list[EvalDataset],
    predictions: Predictions,
    settings: EvalSettings,
) -> str:
    """Write the Markdown report: results per model, then one section per objective."""
    reference = metrics["reference"]
    lines = [f"# Evaluation `{name}`", ""]
    lines += ["Models: " + ", ".join(f"`{m}` = `{p}`" for m, p in metrics["models"].items())]
    lines += [f"Reference model: `{reference}`. Rates are percentages.", ""]

    lines += ["## Results", "", "| Dataset | Role | Model | Files | WER | CER | Term recall |"]
    lines += ["| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset, per_model in metrics["results"].items():
        for model, summary in per_model.items():
            o = summary["overall"]
            recall = _pct(o["term_recall"]) if "term_recall" in o else ""
            lines.append(
                f"| {dataset} | {summary['role']} | {model} | {o['n']} | {_pct(o['wer'])} "
                f"| {_pct(o['cer'])} | {recall} |"
            )
    lines.append("")

    objectives = metrics["objectives"]
    if len(settings.models) < 2:
        lines += ["The three objectives are measured against a second model.", ""]
        return "\n".join(lines)

    lines += [
        "## Objective 1: learn new words",
        "",
        "Training sentences read by the training voices.",
        "",
        "| Dataset | Model | Recordings | Recall (ref) | Recall | Gain | WER (ref) | WER "
        "| Change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in objectives[NEW_WORDS]:
        r, m = c["reference"], c["model"]
        lines.append(
            f"| {c['dataset']} | {m} | {c['n']} | {_pct(c['term_recall'][r])} "
            f"| {_pct(c['term_recall'][m])} | {_pct(c['term_recall_gain'])} "
            f"| {_pct(c['wer'][r])} | {_pct(c['wer'][m])} | {_pct(c['wer_change'])} |"
        )
    if not objectives[NEW_WORDS]:
        lines.append("| no synthetic dataset | | | | | | | | |")
    lines.append("")

    lines += [
        "## Objective 2: keep the learning with other voice timbres",
        "",
        "The same recording slots in `train/` (training voices) and `test/` (test "
        "voices). Gain in term recall over the reference with each, and the ratio of "
        "the test-voice gain to the training-voice gain.",
        "",
        "| Model | Recordings | Gain, training voices | Gain, test voices | Ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for c in objectives[VOICE_TIMBRES]:
        ratio = "n/a" if c["gain_ratio"] != c["gain_ratio"] else f"{c['gain_ratio']:.2f}"
        lines.append(
            f"| {c['model']} | {c['recordings']} | {_pct(c['train_voice_gain'])} "
            f"| {_pct(c['test_voice_gain'])} | {ratio} |"
        )
    if not objectives[VOICE_TIMBRES]:
        lines.append("| no `train/` and `test/` synthetic splits | | | | |")
    lines.append("")
    for c in objectives[VOICE_TIMBRES]:
        r, m = c["reference"], c["model"]
        lines += [f"Term recall per voice, `{m}`:", ""]
        lines += [f"| Voice | Recordings | {r} | {m} |", "| --- | ---: | ---: | ---: |"]
        for voice, v in c["per_voice"].items():
            lines.append(f"| {voice} | {v['n']} | {_pct(v[r])} | {_pct(v[m])} |")
        lines.append("")

    lines += [
        "## Objective 3: keep general French",
        "",
        "WER on general speech and its relative change against the reference.",
        "",
        "| Dataset | Model | Files | WER (ref) | WER | Relative change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in objectives[GENERAL_FRENCH]:
        r, m = c["reference"], c["model"]
        lines.append(
            f"| {c['dataset']} | {m} | {c['n']} | {_pct(c['wer'][r])} | {_pct(c['wer'][m])} "
            f"| {_pct(c['relative_wer_change'])} |"
        )
    if not objectives[GENERAL_FRENCH]:
        lines.append("| no general dataset | | | | | |")
    lines.append("")

    lines += _missed_terms(datasets, predictions, list(settings.models)[-1])
    return "\n".join(lines)


def _missed_terms(datasets, predictions: Predictions, model: str, top: int = 30) -> list[str]:
    """List the words the given model misses most often, to look for patterns."""
    missed: Counter = Counter()
    total: Counter = Counter()
    for dataset in (d for d in datasets if d.medical):
        preds = predictions[(model, dataset.name)]
        for row, hit in zip(
            dataset.rows, term_hits(preds, [r["word"] for r in dataset.rows]), strict=True
        ):
            total[row["word"]] += 1
            missed[row["word"]] += not hit
    worst = [(w, n) for w, n in missed.most_common() if n][:top]
    if not worst:
        return []
    lines = [
        f"## Most missed terms (`{model}`)",
        "",
        "| Word | Missed | Recordings |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {word} | {n} | {total[word]} |" for word, n in worst]
    return [*lines, ""]
