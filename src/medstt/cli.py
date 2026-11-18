"""Define CLI commands for the workspace, data pipeline, and training.

Model and dataset modules load after workspace selection so ``--workdir`` can
set the Hugging Face cache before those libraries are imported.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from medstt.io import read_records
from medstt.nomenclatures.config import parse_nomenclatures
from medstt.nomenclatures.normalize import import_nomenclature
from medstt.sentences.generate import (
    DEFAULT_FORMS,
    DEFAULT_SCENES,
    DEFAULT_SUBJECTS,
    USER_PROMPT,
    GenerationSettings,
    generate_sentences,
)
from medstt.words.build import build_words
from medstt.words.rarity import RaritySettings
from medstt.workspace import Workspace, WorkspaceError, pick

app = typer.Typer(
    help="Medical speech-to-text: build synthetic datasets and fine-tune ASR models.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
nomenclature_app = typer.Typer(
    help="Phase 0: download and normalize source nomenclatures.", no_args_is_help=True
)
words_app = typer.Typer(help="Phase 1: select rare medical words.", no_args_is_help=True)
sentences_app = typer.Typer(help="Phase 1: generate sentences from words.", no_args_is_help=True)
tts_app = typer.Typer(help="Phase 2: synthesize audio datasets.", no_args_is_help=True)
dataset_app = typer.Typer(help="Import, finalize and inspect audio datasets.", no_args_is_help=True)
model_app = typer.Typer(help="Download models for offline use.", no_args_is_help=True)
app.add_typer(nomenclature_app, name="nomenclature")
app.add_typer(words_app, name="words")
app.add_typer(sentences_app, name="sentences")
app.add_typer(tts_app, name="tts")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")

log = logging.getLogger("medstt")

WorkdirOption = Annotated[
    Path | None,
    typer.Option(
        "--workdir",
        "-w",
        help="Workspace directory (default: $MEDSTT_WORKDIR, then the current directory).",
    ),
]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Configure application logging for the selected verbosity."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The OpenAI client logs every HTTP request at INFO level.
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _open(workdir: Path | None) -> Workspace:
    """Open the selected workspace and activate its Hugging Face cache."""
    try:
        ws = Workspace.open(workdir)
    except WorkspaceError as error:
        raise typer.BadParameter(str(error), param_hint="--workdir") from error
    ws.activate()
    return ws


def _api_key(section: dict, api_key: str | None, api_key_env: str | None) -> str:
    """Choose the LLM key from the CLI, environment, or workspace settings."""
    if api_key:
        return api_key
    env_name = pick(api_key_env, section, "api_key_env")
    from_env = os.environ.get(env_name) if env_name else None
    # Some local endpoints require a nonempty placeholder key.
    return from_env or section.get("api_key") or "EMPTY"


def _output_path(ws: Workspace, value: str, base: str, suffix: str = "") -> Path:
    """Place a relative output name under its workspace directory."""
    path = Path(value)
    if suffix and not path.suffix:
        path = path.with_suffix(suffix)
    return path if path.is_absolute() else ws.subdir(base) / path


# Workspace


@app.command()
def init(
    path: Annotated[Path | None, typer.Argument(help="Directory to initialize.")] = None,
) -> None:
    """Create workspace folders and default settings without replacing existing settings."""
    ws = Workspace.init(path)
    typer.echo(f"Workspace ready: {ws.root}\nEdit {ws.root / 'medstt.yaml'} to configure it.")


# Nomenclatures


def _nomenclature_specs(ws: Workspace, names: list[str] | None):
    """Return the configured nomenclatures, restricted to ``names`` when given."""
    try:
        specs = parse_nomenclatures(ws.section("nomenclatures"))
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="nomenclatures") from error
    if not specs:
        raise typer.BadParameter(
            "No nomenclature configured; add a `nomenclatures` section to medstt.yaml.",
            param_hint="nomenclatures",
        )
    if names:
        known = {spec.name: spec for spec in specs}
        missing = [name for name in names if name not in known]
        if missing:
            raise typer.BadParameter(
                f"Unknown nomenclature(s) {missing}; configured: {sorted(known)}.",
                param_hint="NAMES",
            )
        specs = [known[name] for name in names]
    return specs


@nomenclature_app.command("import")
def nomenclature_import(
    names: Annotated[
        list[str] | None,
        typer.Argument(help="Nomenclatures to import (default: all configured)."),
    ] = None,
    force: Annotated[bool, typer.Option(help="Download and normalize again.")] = False,
    workdir: WorkdirOption = None,
) -> None:
    """Download each configured nomenclature and write its normalized JSONL."""
    ws = _open(workdir)
    raw_root, normalized_root = ws.subdir("nomenclature/raw"), ws.subdir("nomenclature/normalized")
    for spec in _nomenclature_specs(ws, names):
        if spec.is_local:
            spec = replace(spec, path=str(ws.resolve(spec.path)))
        try:
            output, count = import_nomenclature(raw_root, normalized_root, spec, force=force)
        except (OSError, ValueError) as error:
            typer.echo(f"{spec.name}: {error}", err=True)
            raise typer.Exit(code=1) from error
        typer.echo(f"{spec.name}: {count} concepts written to {output}")


# Rare words


@words_app.command("build")
def words_build(
    sources: Annotated[
        list[str] | None,
        typer.Argument(help="Normalized nomenclatures to read (overrides words.sources)."),
    ] = None,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Word list name (overrides words.output).")
    ] = None,
    max_zipf: Annotated[
        float | None, typer.Option(help="Keep words with a French Zipf frequency <= this.")
    ] = None,
    min_length: Annotated[int | None, typer.Option(help="Minimum letters per word.")] = None,
    limit: Annotated[
        int | None, typer.Option(help="Only read the first N concepts per source.")
    ] = None,
    stats: Annotated[bool, typer.Option(help="Print counts and the Zipf histogram.")] = False,
    workdir: WorkdirOption = None,
) -> None:
    """Keep the rare words of each concept and write them to ``words/<output>.jsonl``."""
    ws = _open(workdir)
    conf = ws.section("words")
    configured = [spec.name for spec in parse_nomenclatures(ws.section("nomenclatures"))]
    names = list(sources or pick(None, conf, "sources", configured))
    if not names:
        raise typer.BadParameter(
            "No source given and words.sources is empty.", param_hint="SOURCES"
        )
    normalized_root = ws.subdir("nomenclature/normalized")
    paths = []
    for name in names:
        path = normalized_root / f"{name}.jsonl"
        if not path.is_file():
            raise typer.BadParameter(
                f"{path} not found; run `medstt nomenclature import {name}` first.",
                param_hint="SOURCES",
            )
        paths.append((name, path))
    settings = RaritySettings(
        max_zipf=float(pick(max_zipf, conf, "max_zipf", 3.0)),
        min_length=int(pick(min_length, conf, "min_length", 4)),
    )
    output_path = _output_path(ws, pick(output, conf, "output", "words"), "words", ".jsonl")
    count, build_stats = build_words(paths, output_path, settings, limit=limit)
    if stats:
        typer.echo(build_stats.report(settings.max_zipf))
    typer.echo(f"{count} words written to {output_path}")


# Sentences


@sentences_app.command("generate")
def sentences_generate(
    words: Annotated[
        Path | None,
        typer.Argument(
            help="Word list: txt/csv/tsv/jsonl/parquet (in words/); "
            "default: words/<words.output>.jsonl."
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output", "-o", help="Sentence set name (in sentences/); default: sentences.output."
        ),
    ] = None,
    column: Annotated[
        str, typer.Option(help="Word column (txt files: one word per line).")
    ] = "word",
    per_word: Annotated[
        int | None, typer.Option(help="Sentences per term (overrides sentences.per_word).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only process the first N words.")] = None,
    base_url: Annotated[str | None, typer.Option(help="OpenAI-compatible endpoint.")] = None,
    model: Annotated[str | None, typer.Option(help="LLM model name.")] = None,
    api_key: Annotated[str | None, typer.Option(help="LLM API key.")] = None,
    api_key_env: Annotated[str | None, typer.Option(help="Env var holding the API key.")] = None,
    concurrency: Annotated[int | None, typer.Option(help="Parallel requests.")] = None,
    min_words: Annotated[int | None, typer.Option()] = None,
    max_words: Annotated[int | None, typer.Option()] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(
            help="User prompt template; placeholders: {word}, {label}, {definition}, "
            "{context}, {domain}, {min_words}, {max_words} and any column of the word list."
        ),
    ] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Generate the configured number of French sentences per term; resume by ID."""
    ws = _open(workdir)
    llm, conf = ws.section("llm"), ws.section("sentences")
    if words is None:
        words = Path(ws.section("words").get("output", "words")).with_suffix(".jsonl")
    words_path = ws.resolve(words, "words")
    output = pick(output, conf, "output", "sentences")
    records = read_records(words_path, txt_column=column)
    if limit is not None:
        records = records[:limit]
    if records and column not in records[0]:
        raise typer.BadParameter(
            f"Column {column!r} not found in {words_path}.", param_hint="--column"
        )

    sentences_per_word = pick(per_word, conf, "per_word", 3)
    if (
        not isinstance(sentences_per_word, int)
        or isinstance(sentences_per_word, bool)
        or sentences_per_word < 1
    ):
        raise typer.BadParameter(
            "sentences.per_word must be a positive integer.", param_hint="--per-word"
        )

    settings = GenerationSettings(
        base_url=pick(base_url, llm, "base_url"),
        api_key=_api_key(llm, api_key, api_key_env),
        model=pick(model, llm, "model"),
        domain=conf.get("domain", "pratique clinique hospitalière"),
        min_words=pick(min_words, conf, "min_words", 8),
        max_words=pick(max_words, conf, "max_words", 20),
        temperature=llm.get("temperature", 0.8),
        max_tokens=llm.get("max_tokens", 200),
        concurrency=pick(concurrency, llm, "concurrency", 16),
        timeout=llm.get("timeout", 120),
        max_attempts=conf.get("max_attempts", 4),
        per_word=sentences_per_word,
        user_prompt=prompt_file.read_text(encoding="utf-8") if prompt_file else USER_PROMPT,
        extra_body=llm.get("extra_body") or None,
        scenes=tuple(conf.get("scenes", DEFAULT_SCENES) or ()),
        forms=tuple(conf.get("forms", DEFAULT_FORMS) or ()),
        subjects=tuple(conf.get("subjects", DEFAULT_SUBJECTS) or ()),
    )
    if not settings.base_url or not settings.model:
        raise typer.BadParameter("Set llm.base_url and llm.model in medstt.yaml or via options.")
    output_path = _output_path(ws, output, "sentences", ".jsonl")
    ok, failed = generate_sentences(records, column, output_path, settings)
    typer.echo(f"{ok} sentences written to {output_path} ({failed} failed, rerun to retry).")
    if failed:
        raise typer.Exit(code=1)


# Synthetic audio


@tts_app.command("synthesize")
def tts_synthesize(
    sentences: Annotated[Path, typer.Argument(help="Sentence set: jsonl/csv/txt (in sentences/).")],
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Override tts.output (in datasets/).")
    ] = None,
    base_url: Annotated[
        str | None, typer.Option(help="vLLM-Omni API base URL ending in /v1.")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Served Qwen VoiceDesign model name.")] = None,
    api_key: Annotated[
        str | None, typer.Option(help="TTS API key; prefer the configured env var.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option(min=1, help="Concurrent TTS requests per shard.")
    ] = None,
    timeout: Annotated[
        float | None, typer.Option(min=0.001, help="Deadline per TTS request in seconds.")
    ] = None,
    text_column: Annotated[str, typer.Option(help="Sentence column.")] = "text",
    min_duration: Annotated[float | None, typer.Option()] = None,
    max_duration: Annotated[float | None, typer.Option()] = None,
    shard_index: Annotated[
        int | None, typer.Option(help="Shard to process (default: $SLURM_ARRAY_TASK_ID or 0).")
    ] = None,
    num_shards: Annotated[
        int | None, typer.Option(help="Shard count (default: $SLURM_ARRAY_TASK_COUNT or 1).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only process the first N sentences.")] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Write matching train/ and test/ splits with different voice prompts."""
    ws = _open(workdir)
    backends = import_module("medstt.tts.backends")
    synthesis = import_module("medstt.tts.synthesize")
    conf = ws.section("tts")
    records = read_records(ws.resolve(sentences, "sentences"), txt_column=text_column)
    if limit is not None:
        records = records[:limit]
    if shard_index is None:
        shard_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if num_shards is None:
        num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    output_dir = _output_path(ws, pick(output, conf, "output", "synth-medical-words"), "datasets")
    train_dir, test_dir = output_dir / "train", output_dir / "test"
    settings = synthesis.SynthesisSettings(
        sampling_rate=conf.get("sampling_rate", 16000),
        min_duration=pick(min_duration, conf, "min_duration", 1.0),
        max_duration=pick(max_duration, conf, "max_duration", 20.0),
        shard_index=shard_index,
        num_shards=num_shards,
        concurrency=pick(concurrency, conf, "concurrency", 4),
    )

    async def run():
        """Keep both splits and the shared HTTP connection pool on one event loop."""
        tts = backends.QwenVoiceDesign(
            voice_prompts=conf.get("voice_prompts", backends.DEFAULT_VOICE_PROMPTS),
            base_url=pick(base_url, conf, "base_url", "http://localhost:8091/v1"),
            model=pick(model, conf, "model", backends.MODEL_ID),
            api_key=_api_key({"api_key_env": "MEDSTT_TTS_API_KEY", **conf}, api_key, None),
            timeout=pick(timeout, conf, "timeout", 300),
            max_attempts=conf.get("max_attempts", 3),
        )
        try:
            test_tts = tts.with_voice_prompts(
                conf.get("test_voice_prompts", backends.DEFAULT_TEST_VOICE_PROMPTS)
            )
            train_counts = await synthesis.synthesize_dataset_async(
                records, text_column, train_dir, tts, settings
            )
            typer.echo(
                f"Train shard {shard_index}: {train_counts[0]} written, "
                f"{train_counts[1]} rejected, {train_counts[2]} failed. "
                f"Metadata: {train_dir / 'metadata.csv'} "
                f"({'present' if (train_dir / 'metadata.csv').exists() else 'not published'})."
            )
            test_counts = await synthesis.synthesize_dataset_async(
                records, text_column, test_dir, test_tts, settings
            )
            return tuple(a + b for a, b in zip(train_counts, test_counts, strict=True))
        finally:
            await tts.close()

    written, rejected, failed = asyncio.run(run())
    typer.echo(
        f"Shard {shard_index}: {written} written, {rejected} rejected (duration), "
        f"{failed} failed (rerun to retry)."
    )
    if failed:
        raise typer.Exit(code=1)
    if num_shards == 1:
        total = synthesis.finalize_paired_datasets(train_dir, test_dir)
        typer.echo(f"Dataset {output_dir} finalized with {total} samples per split (train/test).")
    else:
        typer.echo(f"When all shards are done: medstt dataset finalize {output_dir}")


# Datasets


@dataset_app.command("finalize")
def dataset_finalize(
    name: Annotated[str, typer.Argument(help="Dataset name (in datasets/).")],
    workdir: WorkdirOption = None,
) -> None:
    """Merge successful audio records from all shards into ``metadata.csv``."""
    ws = _open(workdir)
    synthesis = import_module("medstt.tts.synthesize")
    path = ws.resolve(name, "datasets")
    if (path / "train").is_dir() or (path / "test").is_dir():
        total = synthesis.finalize_paired_datasets(path / "train", path / "test")
        typer.echo(f"{path}: {total} samples per split (train/test).")
    else:
        typer.echo(f"{path}: {synthesis.finalize_dataset(path)} samples.")


@dataset_app.command("import")
def dataset_import(
    source: Annotated[str, typer.Argument(help="Hugging Face hub id or local dataset path.")],
    output: Annotated[str, typer.Option("--output", "-o", help="Dataset name (in datasets/).")],
    config: Annotated[str | None, typer.Option(help="Dataset configuration (e.g. 'fr').")] = None,
    split: Annotated[
        list[str] | None,
        typer.Option(help="Source split to import; repeat for several. Default: train and test."),
    ] = None,
    audio_column: Annotated[str, typer.Option()] = "audio",
    text_column: Annotated[str, typer.Option(help="e.g. 'sentence', 'transcription'.")] = "text",
    max_samples: Annotated[int | None, typer.Option()] = None,
    streaming: Annotated[
        bool, typer.Option(help="Stream instead of downloading everything.")
    ] = True,
    workdir: WorkdirOption = None,
) -> None:
    """Import source splits into datasets/<output>/<split>/ AudioFolders."""
    ws = _open(workdir)
    hub = import_module("medstt.data.hub")
    output_dir = _output_path(ws, output, "datasets")
    splits = list(dict.fromkeys(split or ["train", "test"]))
    for name in splits:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise typer.BadParameter(
                "Each split must be a single directory name.", param_hint="--split"
            )
    if (output_dir / "metadata.csv").exists() or (output_dir / "metadata.jsonl").exists():
        raise typer.BadParameter(
            "Output contains an older flat dataset. Move it into the appropriate split "
            "subdirectory or choose a new output name.",
            param_hint="--output",
        )
    # Check every requested destination before starting a potentially long download.
    for name in splits:
        destination = output_dir / name
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise typer.BadParameter(
                f"{destination} is not empty. Import only missing splits with --split "
                "or choose a new output name."
            )
    for name in splits:
        destination = output_dir / name
        count = hub.import_dataset(
            source,
            destination,
            config=config,
            split=name,
            audio_column=audio_column,
            text_column=text_column,
            max_samples=max_samples,
            streaming=streaming,
        )
        typer.echo(f"{count} samples imported into {destination}")


@dataset_app.command("info")
def dataset_info(
    names: Annotated[list[str], typer.Argument(help="Dataset names or paths.")],
    workdir: WorkdirOption = None,
) -> None:
    """Show each dataset's sample count and up to three transcripts."""
    ws = _open(workdir)
    loading = import_module("medstt.data.loading")
    for name in names:
        ds = loading.load_dataset_spec(ws, loading.DatasetSpec(path=name))
        typer.echo(f"== {name}: {len(ds)} samples")
        for row in ds.select(range(min(3, len(ds)))):
            typer.echo(f"   {row['text']}")


# Model downloads


@model_app.command("download")
def model_download(
    repo_id: Annotated[str, typer.Argument(help="Hugging Face model id.")],
    revision: Annotated[str | None, typer.Option()] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Download model files into the Hugging Face cache selected by HF_HOME."""
    _open(workdir)
    hub = import_module("medstt.data.hub")
    typer.echo(f"Downloaded to {hub.download_model(repo_id, revision)}")


# Training and evaluation


@app.command()
def train(
    resume: Annotated[
        bool, typer.Option(help="Resume from the last checkpoint of the run.")
    ] = False,
    workdir: WorkdirOption = None,
) -> None:
    """Train Cohere decoder LoRA adapters using ``medstt.yaml``."""
    ws = _open(workdir)
    training = import_module("medstt.train.run")
    output = training.train(ws, resume)
    typer.echo(f"Trained model saved to {output}")


@app.command()
def evaluate(
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Evaluation name, written to evals/<name>/."),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="name=model, repeatable; a model is a run name, a local directory or a "
            "Hub ID (default: evaluate.models).",
        ),
    ] = None,
    synthetic: Annotated[
        str | None,
        typer.Option(
            help="Paired synthetic dataset in datasets/, with train/ and test/ splits "
            "(default: evaluate.synthetic)."
        ),
    ] = None,
    general: Annotated[
        list[str] | None,
        typer.Option(
            "--general",
            help="General speech dataset in datasets/, repeatable (default: evaluate.general).",
        ),
    ] = None,
    batch_size: Annotated[int | None, typer.Option(help="Files per generation batch.")] = None,
    max_samples: Annotated[
        int | None, typer.Option(help="Evaluate at most N files per dataset.")
    ] = None,
    workdir: WorkdirOption = None,
) -> None:
    """Compare models on the synthetic medical speech and on general French."""
    ws = _open(workdir)
    evaluation = import_module("medstt.eval.evaluate")
    conf = ws.section("evaluate")
    try:
        settings = evaluation.EvalSettings(
            models=evaluation.parse_models(model or conf.get("models") or {}),
            synthetic=pick(synthetic, conf, "synthetic"),
            general=tuple(general or conf.get("general") or ()),
            reference=conf.get("reference"),
            batch_size=int(pick(batch_size, conf, "batch_size", 16)),
            max_new_tokens=int(conf.get("max_new_tokens", 256)),
            max_samples=pick(max_samples, conf, "max_samples"),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    try:
        output = evaluation.evaluate(ws, pick(name, conf, "name", "eval"), settings)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Report written to {output / evaluation.REPORT_FILE}")
