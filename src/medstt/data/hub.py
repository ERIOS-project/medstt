"""Download model files and copy external datasets into the workspace."""

from __future__ import annotations

import logging
import shutil
from contextlib import suppress
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

from medstt.io import write_csv

log = logging.getLogger(__name__)


def import_dataset(
    source: str,
    output_dir: Path,
    *,
    config: str | None,
    split: str,
    audio_column: str,
    text_column: str,
    max_samples: int | None,
    streaming: bool,
) -> int:
    """Copy a Hub or local dataset to AudioFolder without decoding its audio.

    Return the number of rows whose audio could be written.
    """
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"Dataset destination is not empty: {output_dir}. Use a new output name."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        count = _import_dataset(
            source,
            staged,
            config=config,
            split=split,
            audio_column=audio_column,
            text_column=text_column,
            max_samples=max_samples,
            streaming=streaming,
        )
        # A same-filesystem rename publishes audio and metadata together. It fails
        # safely if another import has populated the destination in the meantime.
        staged.rename(output_dir)
        return count


def _import_dataset(
    source: str,
    output_dir: Path,
    *,
    config: str | None,
    split: str,
    audio_column: str,
    text_column: str,
    max_samples: int | None,
    streaming: bool,
) -> int:
    """Build the complete AudioFolder in a private staging directory."""
    ds = load_dataset(source, config, split=split, streaming=streaming)
    ds = ds.cast_column(audio_column, Audio(decode=False))
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    info = getattr(ds, "info", None)
    split_info = (getattr(info, "splits", None) or {}).get(split)
    total = getattr(split_info, "num_examples", None)
    if total is None:
        with suppress(TypeError):
            total = len(ds)
    if max_samples is not None:
        total = min(total, max_samples) if total is not None else None
        ds = islice(ds, max(0, max_samples))
    for index, row in enumerate(tqdm(ds, desc=f"import {source}/{split}", unit="utt", total=total)):
        audio = row[audio_column]
        suffix = Path(audio.get("path") or "").suffix or ".wav"
        file_name = f"audio/{index:08d}{suffix}"
        if audio.get("bytes"):
            (output_dir / file_name).write_bytes(audio["bytes"])
        elif audio.get("path") and Path(audio["path"]).is_file():
            shutil.copyfile(audio["path"], output_dir / file_name)
        else:
            log.warning("Row %d has no readable audio, skipped.", index)
            continue
        scalars = {
            key: value
            for key, value in row.items()
            if key not in (audio_column, text_column) and isinstance(value, (str, int, float, bool))
        }
        records.append({"file_name": file_name, "text": row[text_column], **scalars})

    write_csv(output_dir / "metadata.csv", records, first_columns=("file_name", "text"))
    return len(records)


def download_model(repo_id: str, revision: str | None) -> Path:
    """Download a model snapshot to the Hugging Face cache."""
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
    )
    return Path(path)
