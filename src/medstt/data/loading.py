"""Load AudioFolder or saved Hugging Face datasets for ASR training.

Every input is mapped to ``audio``, ``text``, ``word``, and ``source`` columns
before multiple datasets are combined. Audio stays undecoded until batching.
"""

from __future__ import annotations

import io
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm

from medstt.audio import decode_audio
from medstt.io import read_records
from medstt.workspace import Workspace

COLUMNS = ("audio", "text", "word", "source")


@dataclass(frozen=True)
class DatasetSpec:
    """Location and optional column, split, sampling, and weighting settings."""

    path: str
    name: str | None = None
    split: str | None = None
    audio_column: str = "audio"
    text_column: str = "text"
    word_column: str = "word"
    max_samples: int | None = None
    #: Reuse each row this many times when combining datasets.
    repeat: int = 1
    #: Target share of audio duration per epoch (all sources must specify it).
    weight: float | None = None

    @property
    def source_name(self) -> str:
        """Identify a source without collapsing distinct paths ending in train/."""
        return self.name or self.path

    @classmethod
    def parse(cls, value: str | dict[str, Any]) -> DatasetSpec:
        """Accept a path string or a mapping of dataset options."""
        if isinstance(value, str):
            return cls(path=value)
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown dataset options {sorted(unknown)} in {value}.")
        return cls(**value)


def load_dataset_spec(ws: Workspace, spec: DatasetSpec, seed: int = 42):
    """Load one dataset and map it to the common four-column schema."""
    path = ws.resolve(spec.path, base="datasets")
    name = spec.source_name
    metadata = _find_metadata(path)
    if metadata is not None:
        records = read_records(metadata)
        if records and spec.text_column not in records[0]:
            raise ValueError(f"{metadata}: missing text column {spec.text_column!r}.")
        ds = Dataset.from_dict(
            {
                "audio": [str(path / r["file_name"]) for r in records],
                "text": [str(r[spec.text_column] or "") for r in records],
                "word": [str(r.get(spec.word_column) or "") for r in records],
            }
        )
    elif (path / "dataset_dict.json").exists() or (path / "state.json").exists():
        ds = _load_from_disk(path, spec)
    else:
        raise FileNotFoundError(
            f"{path} is neither an AudioFolder (metadata.csv/jsonl) nor a save_to_disk dataset."
        )

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.filter(lambda text: bool(text.strip()), input_columns="text")
    if spec.max_samples is not None and len(ds) > spec.max_samples:
        ds = ds.shuffle(seed=seed).select(range(spec.max_samples))
    ds = ds.add_column("source", [name] * len(ds))
    if spec.repeat > 1:
        ds = concatenate_datasets([ds] * spec.repeat)
    return ds


def load_datasets(ws: Workspace, specs: list[DatasetSpec], seed: int = 42, *, mix: bool = True):
    """Concatenate datasets in the order listed in ``specs``."""
    if not specs:
        raise ValueError("At least one dataset is required.")
    weighted = any(spec.weight is not None for spec in specs)
    if weighted and any(
        spec.weight is None
        or not math.isfinite(spec.weight)
        or spec.weight <= 0
        or spec.repeat != 1
        for spec in specs
    ):
        raise ValueError("Set a positive weight on every source; do not combine weight and repeat.")
    parts = [load_dataset_spec(ws, spec, seed) for spec in specs]
    if weighted and mix:
        return mix_by_duration(parts, [spec.weight for spec in specs], seed)
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _find_metadata(path: Path) -> Path | None:
    """Find an AudioFolder metadata file, if the directory has one."""
    for filename in ("metadata.csv", "metadata.jsonl"):
        if (path / filename).is_file():
            return path / filename
    return None


def _load_from_disk(path: Path, spec: DatasetSpec):
    """Read a saved Hugging Face dataset and select its configured columns."""
    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        split = spec.split or ("train" if "train" in ds else None)
        if split is None:
            raise ValueError(f"{path} has splits {list(ds)}; set `split` in the dataset spec.")
        ds = ds[split]
    for column in (spec.audio_column, spec.text_column):
        if column not in ds.column_names:
            raise ValueError(f"{path}: missing column {column!r} (got {ds.column_names}).")
    ds = ds.rename_columns(
        {
            old: new
            for old, new in ((spec.audio_column, "audio"), (spec.text_column, "text"))
            if old != new
        }
    )
    if spec.word_column in ds.column_names and spec.word_column != "word":
        ds = ds.rename_column(spec.word_column, "word")
    if "word" not in ds.column_names:
        ds = ds.add_column("word", [""] * len(ds))
    return ds.select_columns(["audio", "text", "word"])


def audio_duration(audio: Any) -> float:
    """Read duration from an audio header, with a decoder fallback."""
    try:
        if isinstance(audio, dict):
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        else:
            source = audio
        return float(sf.info(source).duration)
    except (KeyError, TypeError, RuntimeError):
        return len(decode_audio(audio, 16000)) / 16000


def mix_by_duration(parts, weights: list[float], seed: int):
    """Build a seeded epoch with approximately weighted audio duration.

    Keep every usable example from the first source exactly once. Other sources
    target its duration multiplied by their weight relative to the first weight.
    They are shuffled in cycles, repeating only when their pool is exhausted.
    A target can overshoot by at most one clip. Unreadable or >=30s clips are excluded.
    """
    rng = random.Random(seed)
    pools = []
    durations = []
    for source_index, ds in enumerate(parts):
        indices, seconds = [], []
        for index, audio in enumerate(
            tqdm(
                ds["audio"],
                total=len(ds),
                desc=f"scan audio durations {source_index + 1}/{len(parts)}",
                unit="utt",
                disable=int(os.environ.get("RANK", "0")) != 0,
            )
        ):
            try:
                duration = audio_duration(audio)
            except (OSError, ValueError, RuntimeError):
                continue
            if math.isfinite(duration) and 0 < duration < 30:
                indices.append(index)
                seconds.append(duration)
        if not indices:
            raise ValueError("A weighted source contains no readable audio shorter than 30s.")
        logging.getLogger(__name__).info(
            "Duration scan: kept %d/%d clips, %.2f hours",
            len(indices),
            len(ds),
            sum(seconds) / 3600,
        )
        pools.append(ds.select(indices))
        durations.append(seconds)
    reference_duration = sum(durations[0])
    combined = [pools[0]]
    for pool, seconds, weight in zip(pools[1:], durations[1:], weights[1:], strict=True):
        target = reference_duration * weight / weights[0]
        selected, elapsed = [], 0.0
        while elapsed < target:
            order = list(range(len(pool)))
            rng.shuffle(order)
            for index in order:
                selected.append(index)
                elapsed += seconds[index]
                if elapsed >= target:
                    break
        logging.getLogger(__name__).info(
            "Mixed source: %d samples, %.2fh (target %.2fh)",
            len(selected),
            elapsed / 3600,
            target / 3600,
        )
        combined.append(pool.select(selected))
    return concatenate_datasets(combined).shuffle(seed=seed)
