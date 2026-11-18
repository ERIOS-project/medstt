"""Save two voice recordings per sentence as a Hugging Face AudioFolder.

Each successful sentence/voice pair gets one WAV and a progress record.
Progress files allow interrupted or sharded runs to resume without duplicates;
``finalize_dataset`` merges them into ``metadata.csv``.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from medstt.audio import peak_normalize, resample, to_mono
from medstt.io import append_jsonl, iter_jsonl, write_csv
from medstt.tts.base import TTSBackend

log = logging.getLogger(__name__)

PARTS_DIR = "_parts"
AUDIO_DIR = "audio"
METADATA_FILE = "metadata.csv"
VOICES = ("male", "female")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SynthesisSettings:
    """Output audio format, accepted duration, and current shard."""

    sampling_rate: int
    min_duration: float
    max_duration: float
    shard_index: int = 0
    num_shards: int = 1
    concurrency: int = 1


async def synthesize_dataset_async(
    sentences: list[dict[str, Any]],
    text_column: str,
    output_dir: Path,
    backend: TTSBackend,
    settings: SynthesisSettings,
) -> tuple[int, int, int]:
    """Generate this shard's missing sentence/voice pairs.

    Return counts of saved audio, duration rejections, and generation failures.
    Duration rejections are final; failures remain eligible for a later retry.
    """
    if not 0 <= settings.shard_index < settings.num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if (
        not isinstance(settings.concurrency, int)
        or isinstance(settings.concurrency, bool)
        or settings.concurrency < 1
    ):
        raise ValueError("tts.concurrency must be a positive integer.")
    audio_dir = output_dir / AUDIO_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    part_file = output_dir / PARTS_DIR / f"part-{settings.shard_index:05d}.jsonl"
    rejected_file = output_dir / PARTS_DIR / f"rejected-{settings.shard_index:05d}.jsonl"
    # Read all shards so a rerun can use a different shard count.
    with _progress_lock(output_dir):
        done = {
            record["id"]
            for progress in (output_dir / PARTS_DIR).glob("rejected-*.jsonl")
            for record in iter_jsonl(progress)
        }

        for progress in (output_dir / PARTS_DIR).glob("part-*.jsonl"):
            for record in iter_jsonl(progress):
                if _readable_audio(output_dir / record["file_name"]):
                    done.add(record["id"])

    items = []
    seen = set()
    for row, record in enumerate(sentences):
        text = str(record.get(text_column) or "").strip()
        if not text:
            continue
        sample_id = _UNSAFE_RE.sub("_", str(record.get("id") or f"{row:07d}"))
        if sample_id in seen:
            raise ValueError(f"Duplicate or colliding sentence ID: {sample_id}")
        seen.add(sample_id)
        for voice in VOICES:
            items.append((f"{sample_id}-{voice}", text, record, voice))
    expected = {item[0]: item[1] for item in items}
    items = items[settings.shard_index :: settings.num_shards]
    todo = [item for item in items if item[0] not in done]
    log.info(
        "Shard %d/%d: %d audio samples, %d left to synthesize.",
        settings.shard_index,
        settings.num_shards,
        len(items),
        len(todo),
    )

    async def work(item):
        """Generate and resample one sentence/voice pair."""
        sample_id, text, record, voice = item
        try:
            audio, sr = await backend.synthesize(text, voice)
            audio = peak_normalize(resample(to_mono(audio), sr, settings.sampling_rate))
        except Exception as error:  # noqa: BLE001 - left undone, retried on the next run
            log.warning("TTS failed for %s: %s", sample_id, error)
            return sample_id, text, record, voice, None
        return sample_id, text, record, voice, audio

    written = rejected = failed = 0

    def save(result):
        """Persist completions serially on the event loop, with no await during writes."""
        nonlocal written, rejected, failed
        sample_id, text, record, voice, audio = result
        if audio is None:
            failed += 1
            return
        duration = len(audio) / settings.sampling_rate
        if not settings.min_duration <= duration <= settings.max_duration:
            _append_progress(rejected_file, [{"id": sample_id, "text": text, "duration": duration}])
            rejected += 1
            return
        file_name = f"{AUDIO_DIR}/{sample_id}.wav"
        temporary = (output_dir / file_name).with_suffix(".wav.part")
        sf.write(temporary, audio, settings.sampling_rate, subtype="PCM_16", format="WAV")
        temporary.replace(output_dir / file_name)
        reserved = {
            "id",
            "file_name",
            "text",
            text_column,
            "duration",
            "speaker",
            "tts",
            "source_id",
        }
        extra = {k: v for k, v in record.items() if k not in reserved}
        _append_progress(
            part_file,
            [
                {
                    "id": sample_id,
                    "file_name": file_name,
                    "text": text,
                    "duration": round(duration, 3),
                    "speaker": voice,
                    "tts": backend.name,
                    "source_id": sample_id.removesuffix(f"-{voice}"),
                    **extra,
                }
            ],
        )
        written += 1

    pending = iter(todo)
    with tqdm(
        total=len(items),
        initial=len(items) - len(todo),
        desc=f"tts {output_dir.name} shard {settings.shard_index}/{settings.num_shards}",
        unit="utt",
    ) as progress:

        async def worker():
            """Pull one item at a time, bounding tasks and audio buffers by concurrency."""
            for item in pending:
                result = await work(item)
                save(result)
                progress.set_postfix(
                    written=written, rejected=rejected, failed=failed, refresh=False
                )
                progress.update(1)

        workers = [
            asyncio.create_task(worker()) for _ in range(min(settings.concurrency, len(todo)))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    _publish_split_metadata(output_dir, expected)
    return written, rejected, failed


@contextmanager
def _progress_lock(output_dir: Path):
    """Serialize progress appends and split publication across local/Slurm shards."""
    parts = output_dir / PARTS_DIR
    parts.mkdir(parents=True, exist_ok=True)
    with (parts / ".metadata.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _append_progress(path: Path, records: list[dict[str, Any]]) -> None:
    """Keep readers from observing a partially appended JSONL record."""
    with _progress_lock(path.parent.parent):
        append_jsonl(path, records)


def _publish_split_metadata(output_dir: Path, expected: dict[str, str]) -> None:
    """Publish all valid successes, even when other slots are missing or failed."""
    with _progress_lock(output_dir):
        records = {
            record["id"]: record
            for part in sorted((output_dir / PARTS_DIR).glob("part-*.jsonl"))
            for record in iter_jsonl(part)
        }
        rejected = {
            record["id"]
            for part in (output_dir / PARTS_DIR).glob("rejected-*.jsonl")
            for record in iter_jsonl(part)
            if expected.get(record["id"]) == record.get("text")
        }
        selected = []
        blocked = {"missing progress": [], "text mismatch": [], "unreadable audio": []}
        for key, text in sorted(expected.items()):
            if key in rejected:
                continue
            record = records.get(key)
            if record is None:
                blocked["missing progress"].append(key)
            elif record["text"] != text:
                blocked["text mismatch"].append(key)
            elif not _readable_audio(output_dir / record["file_name"]):
                blocked["unreadable audio"].append(key)
            else:
                selected.append(record)
        if any(blocked.values()):
            details = "; ".join(
                f"{reason}: {len(ids)} (e.g. {', '.join(ids[:3])})"
                for reason, ids in blocked.items()
                if ids
            )
            log.warning(
                "Split %s: publishing partial metadata.csv; %d/%d slots complete; %s. "
                "Check other shards and retry failed samples. Text mismatches require "
                "the original sentence input or a new output dataset.",
                output_dir,
                len(selected) + len(set(expected) & rejected),
                len(expected),
                details,
            )
        temporary = output_dir / "metadata.csv.part"
        write_csv(temporary, selected, first_columns=("file_name", "text"))
        temporary.replace(output_dir / METADATA_FILE)
        log.info("Split %s: metadata.csv published (%d valid samples).", output_dir, len(selected))


def synthesize_dataset(
    sentences, text_column, output_dir, backend, settings
) -> tuple[int, int, int]:
    """Run one split from synchronous callers; API clients should share one event loop."""
    return asyncio.run(
        synthesize_dataset_async(sentences, text_column, output_dir, backend, settings)
    )


def finalize_dataset(output_dir: Path) -> int:
    """Write ``metadata.csv`` from successful shard records and return its row count."""
    records: dict[str, dict[str, Any]] = {}
    for part in tqdm(
        sorted((output_dir / PARTS_DIR).glob("part-*.jsonl")),
        desc=f"finalize {output_dir.name}: read shards",
        unit="shard",
    ):
        for record in iter_jsonl(part):
            records[record["id"]] = record
    if not records:
        raise ValueError(f"No synthesized samples found in {output_dir / PARTS_DIR}.")
    invalid = [
        record["file_name"]
        for record in tqdm(
            records.values(), desc=f"finalize {output_dir.name}: check audio", unit="utt"
        )
        if not _readable_audio(output_dir / record["file_name"])
    ]
    if invalid:
        raise ValueError(
            f"Cannot finalize: {len(invalid)} missing or unreadable audio files "
            f"(e.g. {invalid[:3]}). Rerun synthesis to repair them."
        )
    ordered = [records[key] for key in sorted(records)]
    write_csv(output_dir / METADATA_FILE, ordered, first_columns=("file_name", "text"))
    return len(ordered)


def finalize_paired_datasets(train_dir: Path, test_dir: Path) -> int:
    """Finalize matching sentence/voice slots, excluding duration rejects on both sides.

    Missing counterparts without a rejection are unfinished work and block finalization.
    IDs and texts must match; neither metadata file is written until both sets validate.
    """
    if train_dir.resolve() == test_dir.resolve():
        raise ValueError("Train and test output directories must differ.")
    splits = []
    rejections = []
    for directory in (train_dir, test_dir):
        records = {}
        for part in tqdm(
            sorted((directory / PARTS_DIR).glob("part-*.jsonl")),
            desc=f"finalize {directory.name}: read shards",
            unit="shard",
        ):
            for record in iter_jsonl(part):
                records[record["id"]] = record
        rejected = {
            record["id"]
            for part in (directory / PARTS_DIR).glob("rejected-*.jsonl")
            for record in iter_jsonl(part)
        }
        splits.append(records)
        rejections.append(rejected)
    train, test = splits
    all_ids = set(train) | set(test) | rejections[0] | rejections[1]
    for records, rejected in zip(splits, rejections, strict=True):
        missing = all_ids - records.keys() - rejected
        if missing:
            raise ValueError(
                f"Cannot finalize: {len(missing)} unfinished train/test counterparts. "
                "Complete all shards and rerun synthesis to retry failures."
            )
    selected = sorted((train.keys() & test.keys()) - rejections[0] - rejections[1])
    if not selected:
        raise ValueError("No complete train/test pairs found.")
    for key in tqdm(selected, desc="finalize train/test: check pairs", unit="pair"):
        if train[key]["text"] != test[key]["text"]:
            raise ValueError(f"Train/test text mismatch for {key}.")
        for directory, records in zip((train_dir, test_dir), splits, strict=True):
            if not _readable_audio(directory / records[key]["file_name"]):
                raise ValueError("Cannot finalize: missing or unreadable audio. Rerun synthesis.")
    for directory, records in zip((train_dir, test_dir), splits, strict=True):
        write_csv(
            directory / METADATA_FILE,
            [records[key] for key in selected],
            first_columns=("file_name", "text"),
        )
    log.info(
        "Paired datasets: %d samples each; %d slots excluded.",
        len(selected),
        len(all_ids) - len(selected),
    )
    return len(selected)


def _readable_audio(path: Path) -> bool:
    """Check that a recorded success still points to a readable, nonempty file."""
    try:
        return path.is_file() and sf.info(path).frames > 0
    except (OSError, RuntimeError):
        return False
