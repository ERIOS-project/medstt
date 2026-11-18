"""Exercise the actual OpenAI HTTP payload and concurrent pipeline without a GPU."""

import asyncio
import io
import json
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
import soundfile as sf
import yaml
from openai import APIStatusError, AsyncOpenAI
from typer.testing import CliRunner

from medstt.cli import app
from medstt.io import read_records
from medstt.tts import backends
from medstt.tts.backends import DEFAULT_TEST_VOICE_PROMPTS, DEFAULT_VOICE_PROMPTS, QwenVoiceDesign
from medstt.workspace import Workspace


def wav_bytes():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(2400), 24000, format="WAV")
    return buffer.getvalue()


def use_transport(monkeypatch, handler):
    def client(**kwargs):
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setattr(backends, "AsyncOpenAI", client)


def test_api_contract_shared_client_and_wav_decode(monkeypatch):
    payloads = []

    def handler(request):
        assert request.url.path == "/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer test-key"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, content=wav_bytes(), headers={"content-type": "audio/wav"})

    use_transport(monkeypatch, handler)

    async def run():
        train = QwenVoiceDesign(DEFAULT_VOICE_PROMPTS, api_key="test-key", model="served-alias")
        test = train.with_voice_prompts(DEFAULT_TEST_VOICE_PROMPTS)
        assert train.client is test.client
        try:
            audio, rate = await train.synthesize("Bonjour.", "male")
            assert rate == 24000 and audio.shape == (2400,)
            await test.synthesize("Bonjour.", "female")
            assert train.client.max_retries == 0
        finally:
            await train.close()
        assert train.client.is_closed()

    asyncio.run(run())
    assert payloads[0] == {
        "model": "served-alias",
        "input": "Bonjour.",
        "voice": "default",
        "response_format": "wav",
        "task_type": "VoiceDesign",
        "language": "French",
        "instructions": DEFAULT_VOICE_PROMPTS["male"],
    }
    assert payloads[1]["instructions"] == DEFAULT_TEST_VOICE_PROMPTS["female"]


@pytest.mark.parametrize("status,expected", [(400, 1), (401, 1), (429, 3), (503, 3), (200, 3)])
def test_api_retries_are_bounded_and_bad_wav_is_retried(monkeypatch, status, expected):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=b"not valid audio")

    use_transport(monkeypatch, handler)
    monkeypatch.setattr(backends.asyncio, "sleep", AsyncMock())

    async def run():
        backend = QwenVoiceDesign(DEFAULT_VOICE_PROMPTS, max_attempts=3)
        try:
            with pytest.raises((APIStatusError, RuntimeError)):
                await backend.synthesize("Bonjour.", "male")
        finally:
            await backend.close()

    asyncio.run(run())
    assert len(calls) == expected


def test_request_deadline_cancels_stalled_response(monkeypatch):
    cancelled = []

    async def handler(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    use_transport(monkeypatch, handler)

    async def run():
        backend = QwenVoiceDesign(DEFAULT_VOICE_PROMPTS, timeout=0.01, max_attempts=1)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await backend.synthesize("Bonjour.", "male")
        finally:
            await backend.close()

    asyncio.run(run())
    assert cancelled == [True]


@pytest.mark.parametrize("cli_override", [False, True])
@pytest.mark.parametrize("output_mode", ["default", "legacy", "yaml", "cli"])
def test_cli_concurrency_resume_and_paired_outputs(
    tmp_path, monkeypatch, cli_override, output_mode
):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "sentences/in.txt").write_text("Une première phrase.\nUne deuxième phrase.\n")
    config = ws.config
    config["tts"]["concurrency"] = 1 if cli_override else 2
    output_name = "synth-medical-words"
    if output_mode == "legacy":
        config["tts"].pop("output")
    elif output_mode in {"yaml", "cli"}:
        config["tts"]["output"] = "configured-audio"
        output_name = "paired" if output_mode == "cli" else "configured-audio"
    (ws.root / "medstt.yaml").write_text(yaml.safe_dump(config))
    active = peak = calls = 0

    async def handler(request):
        nonlocal active, peak, calls
        calls += 1
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(200, content=wav_bytes())
        finally:
            active -= 1

    use_transport(monkeypatch, handler)
    args = [
        "tts",
        "synthesize",
        "in.txt",
        "--workdir",
        str(ws.root),
        "--min-duration",
        "0.05",
    ]
    if output_mode == "cli":
        args += ["-o", "paired"]
    if cli_override:
        args += ["--concurrency", "2"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    assert peak == 2 and calls == 8
    for split in ("train", "test"):
        rows = read_records(ws.root / f"datasets/{output_name}/{split}/metadata.csv")
        assert len(rows) == len({row["id"] for row in rows}) == 4
        for row in rows:
            assert (
                sf.info(ws.root / f"datasets/{output_name}/{split}" / row["file_name"]).samplerate
                == 16000
            )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    assert calls == 8


def test_invalid_prompts_and_settings_fail_before_network():
    with pytest.raises(ValueError, match="male and female"):
        QwenVoiceDesign({"male": "male"})
    for options in ({"timeout": 0}, {"timeout": float("nan")}, {"max_attempts": 0}):
        with pytest.raises(ValueError):
            QwenVoiceDesign(DEFAULT_VOICE_PROMPTS, **options)

    async def run():
        backend = QwenVoiceDesign(DEFAULT_VOICE_PROMPTS)
        try:
            with pytest.raises(ValueError, match="must differ"):
                backend.with_voice_prompts(DEFAULT_VOICE_PROMPTS)
        finally:
            await backend.close()

    asyncio.run(run())


def test_api_failure_exits_nonzero_and_resume_requests_only_missing_audio(tmp_path, monkeypatch):
    ws = Workspace.init(tmp_path / "ws")
    (ws.root / "sentences/in.txt").write_text("Une phrase clinique.\n")
    calls = []
    fail = True

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload["instructions"])
        if fail and payload["instructions"] == DEFAULT_VOICE_PROMPTS["female"]:
            return httpx.Response(400, json={"error": {"message": "test failure"}})
        return httpx.Response(200, content=wav_bytes())

    use_transport(monkeypatch, handler)
    args = [
        "tts",
        "synthesize",
        "in.txt",
        "-o",
        "paired",
        "--workdir",
        str(ws.root),
        "--min-duration",
        "0.05",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "1 failed" in result.output
    assert len(calls) == 4
    assert len(read_records(ws.root / "datasets/paired/train/metadata.csv")) == 1
    assert (ws.root / "datasets/paired/test/metadata.csv").exists()
    fail = False
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.exception
    assert len(calls) == 5 and calls[-1] == DEFAULT_VOICE_PROMPTS["female"]
    for split in ("train", "test"):
        assert len(read_records(ws.root / f"datasets/paired/{split}/metadata.csv")) == 2
