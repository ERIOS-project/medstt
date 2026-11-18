"""Generate French VoiceDesign audio through the vLLM-Omni speech API."""

from __future__ import annotations

import asyncio
import io
import logging
import math
from copy import copy

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI

from medstt.tts.base import TTSBackend

log = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_VOICE_PROMPTS = {
    "male": (
        "Voix d'homme adulte francophone, naturelle, claire et posée, "
        "avec une diction médicale précise."
    ),
    "female": (
        "Voix de femme adulte francophone, naturelle, claire et posée, "
        "avec une diction médicale précise."
    ),
}


DEFAULT_TEST_VOICE_PROMPTS = {
    "male": (
        "Voix d'homme âgé francophone, grave et légèrement rauque, "
        "avec un débit lent et une articulation précise."
    ),
    "female": (
        "Voix de jeune femme francophone, timbre aigu et lumineux, "
        "avec un débit vif et une articulation précise."
    ),
}


class QwenVoiceDesign(TTSBackend):
    """Share an asynchronous API client across train and test voice descriptions."""

    def __init__(
        self,
        voice_prompts: dict[str, str],
        *,
        base_url: str = "http://localhost:8091/v1",
        api_key: str = "EMPTY",
        model: str = MODEL_ID,
        timeout: float = 300,
        max_attempts: int = 3,
    ):
        """Configure a remote vLLM-Omni model; no local model weights are loaded."""
        self._validate_prompts(voice_prompts)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("tts.timeout must be a finite positive number.")
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("tts.max_attempts must be a positive integer.")
        if not base_url or not model:
            raise ValueError("tts.base_url and tts.model must be nonempty.")
        self.voice_prompts = voice_prompts
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        # Only our retry loop retries; avoid multiplying SDK and application retries.
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    @staticmethod
    def _validate_prompts(prompts: dict[str, str]) -> None:
        """Require both voice descriptions before submitting requests."""
        if (
            not isinstance(prompts, dict)
            or set(prompts) != {"male", "female"}
            or not all(isinstance(v, str) and v.strip() for v in prompts.values())
        ):
            raise ValueError("Voice prompts must define nonempty male and female descriptions.")

    def with_voice_prompts(self, voice_prompts: dict[str, str]) -> QwenVoiceDesign:
        """Use other voice instructions with the same connection pool."""
        self._validate_prompts(voice_prompts)
        train_prompts = {v.strip().casefold() for v in self.voice_prompts.values()}
        if any(v.strip().casefold() in train_prompts for v in voice_prompts.values()):
            raise ValueError("Test voice prompts must differ from all training voice prompts.")
        backend = copy(self)
        backend.voice_prompts = voice_prompts
        return backend

    @property
    def name(self) -> str:
        """Return the served model identifier stored in metadata."""
        return self.model

    async def close(self) -> None:
        """Close the client shared by both split backends."""
        await self.client.close()

    async def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Request WAV audio with a wall-clock deadline and bounded retries."""
        for attempt in range(self.max_attempts):
            try:
                response = await asyncio.wait_for(
                    self.client.audio.speech.create(
                        model=self.model,
                        input=text,
                        voice="default",  # SDK requires this; VoiceDesign uses instructions.
                        response_format="wav",
                        extra_body={
                            "task_type": "VoiceDesign",
                            "language": "French",
                            "instructions": self.voice_prompts[voice],
                        },
                    ),
                    timeout=self.timeout,
                )
                audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("TTS returned empty or non-finite audio.")
                return audio, sr
            except Exception as error:
                status = getattr(error, "status_code", None)
                log.warning(
                    "TTS %s attempt %d/%d failed (%s, HTTP %s).",
                    voice,
                    attempt + 1,
                    self.max_attempts,
                    type(error).__name__,
                    status,
                )
                # Authentication, wrong model/endpoint and invalid payload need a fix, not retries.
                if (
                    status is not None
                    and 400 <= status < 500
                    and status not in {408, 409, 429}
                    or attempt + 1 == self.max_attempts
                ):
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        raise RuntimeError("TTS exhausted its attempts.")
