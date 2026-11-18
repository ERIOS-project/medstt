"""Batching, prompt and long-audio handling of the Cohere transcriber, without weights."""

from unittest.mock import Mock

import numpy as np
import soundfile as sf
import torch

from medstt.inference import CohereTranscriber


def test_transcribe_batches_uses_french_prompt_and_joins_chunks(tmp_path):
    paths = []
    for i, seconds in enumerate([1.0, 2.0, 1.5]):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.zeros(int(16000 * seconds), dtype=np.float32), 16000)
        paths.append(path)

    transcriber = CohereTranscriber.__new__(CohereTranscriber)
    transcriber.device, transcriber.batch_size, transcriber.max_new_tokens = "cpu", 2, 64
    transcriber.model = Mock(dtype=torch.float32)
    transcriber.model.generate.side_effect = lambda **kw: torch.ones(
        (kw["input_features"].shape[0], 3), dtype=torch.long
    )
    calls = []

    def processor(audio, language, sampling_rate, return_tensors):
        calls.append((len(audio), language, sampling_rate))
        return {
            "input_features": torch.zeros((len(audio), 4, 8), dtype=torch.float64),
            "attention_mask": torch.ones((len(audio), 4), dtype=torch.long),
            "decoder_input_ids": torch.tensor([[7, 8]] * len(audio)),
            "audio_chunk_index": [(i, None) for i in range(len(audio))],
        }

    transcriber.processor = Mock(side_effect=processor)
    transcriber.processor.decode.side_effect = lambda ids, **kw: [
        f" texte {i} " for i, _ in kw["audio_chunk_index"]
    ]

    texts = transcriber.transcribe(paths)

    assert texts == ["texte 0", "texte 1", "texte 0"]
    assert calls == [(2, "fr", 16000), (1, "fr", 16000)]
    first = transcriber.model.generate.call_args_list[0].kwargs
    assert "audio_chunk_index" not in first
    assert first["input_features"].dtype == torch.float32
    assert first["decoder_input_ids"].tolist() == [[7, 8], [7, 8]]
    assert first["max_new_tokens"] == 64
    decode_kwargs = transcriber.processor.decode.call_args.kwargs
    assert decode_kwargs["skip_special_tokens"] and decode_kwargs["language"] == "fr"


def test_load_model_puts_run_adapters_on_their_base_model(tmp_path, monkeypatch):
    import json

    import peft
    import transformers

    from medstt.inference import load_model

    loaded = []
    merged = Mock(name="merged")
    monkeypatch.setattr(
        transformers.CohereAsrForConditionalGeneration,
        "from_pretrained",
        classmethod(lambda cls, name, dtype: loaded.append((name, dtype)) or f"model:{name}"),
    )
    adapters = Mock()
    adapters.merge_and_unload.return_value = merged
    from_pretrained = Mock(return_value=adapters)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", from_pretrained)

    assert load_model("CohereLabs/cohere-transcribe-03-2026", torch.float32) == (
        "model:CohereLabs/cohere-transcribe-03-2026"
    )

    run = tmp_path / "final"
    run.mkdir()
    (run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/checkpoint"}), encoding="utf-8"
    )
    assert load_model(str(run), torch.bfloat16) is merged
    assert loaded[-1] == ("base/checkpoint", torch.bfloat16)
    from_pretrained.assert_called_once_with("model:base/checkpoint", str(run))
