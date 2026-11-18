"""Transcribe audio files with Cohere Transcribe, the base model or a trained run.

A trained run saves LoRA adapters, not a full model: a directory with an
``adapter_config.json`` is loaded on top of its base model, then merged.
Generation uses the same French decoder prompt as training. ``generate`` puts
the decoder start token before the prompt, as the training collator does, so
training and inference see the same decoder input.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import torch

from medstt.audio import decode_audio

log = logging.getLogger(__name__)

LANGUAGE = "fr"
SAMPLING_RATE = 16000
BASE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
ADAPTER_CONFIG = "adapter_config.json"


class Transcriber(Protocol):
    """Turn audio files into text, one transcript per file."""

    def transcribe(self, paths: Sequence[str | Path]) -> list[str]:
        """Return one transcript per audio path, in order."""


class CohereTranscriber:
    """Batch transcription with a Cohere Transcribe checkpoint."""

    def __init__(
        self,
        model: str,
        batch_size: int = 16,
        max_new_tokens: int = 256,
        device: str | None = None,
    ):
        """Load the processor and model from a Hub ID, a model directory or adapters."""
        from transformers import AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = load_model(model, dtype)
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        start = self.model.generation_config.decoder_start_token_id
        if start is None:
            start = (
                self.model.config.decoder_start_token_id or self.processor.tokenizer.bos_token_id
            )
        log.info("Loaded %s on %s; decoder start token: %s", model, self.device, start)

    def transcribe(self, paths: Sequence[str | Path]) -> list[str]:
        """Transcribe the files in batches; audio over 30 s is split and joined back."""
        texts: list[str] = []
        for batch in _batches(list(paths), self.batch_size):
            arrays = [decode_audio(path, SAMPLING_RATE) for path in batch]
            inputs = self.processor(
                audio=arrays, language=LANGUAGE, sampling_rate=SAMPLING_RATE, return_tensors="pt"
            )
            chunk_index = inputs.pop("audio_chunk_index", None)
            inputs = {
                key: value.to(self.device, dtype=self.model.dtype)
                if key == "input_features"
                else value.to(self.device)
                for key, value in inputs.items()
            }
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.processor.decode(
                generated,
                skip_special_tokens=True,
                audio_chunk_index=chunk_index,
                language=LANGUAGE,
            )
            texts += [text.strip() for text in decoded]
        return texts


def load_model(model: str, dtype: torch.dtype):
    """Load a full Cohere model, or a base model with the run's LoRA adapters merged in."""
    from transformers import CohereAsrForConditionalGeneration

    adapter_config = Path(model) / ADAPTER_CONFIG
    if not adapter_config.is_file():
        return CohereAsrForConditionalGeneration.from_pretrained(model, dtype=dtype)
    from peft import PeftModel

    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_id = config.get("base_model_name_or_path") or BASE_MODEL
    log.info("%s holds LoRA adapters; loading them on %s", model, base_id)
    base = CohereAsrForConditionalGeneration.from_pretrained(base_id, dtype=dtype)
    return PeftModel.from_pretrained(base, model).merge_and_unload()


def _batches(items: list, size: int) -> Iterator[list]:
    """Yield consecutive slices of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]
