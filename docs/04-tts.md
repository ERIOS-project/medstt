# 04 — Generate synthetic speech

`medstt tts synthesize` sends concurrent requests to a **vLLM-Omni** server running
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. The server exposes
`POST /v1/audio/speech`; plain text-only vLLM is not sufficient.

Each sentence produces one male and one female recording in each of `train/`
and `test/`, using separate voice descriptions. The server holds the model;
the CLI decodes its WAV responses and builds the dataset locally.

## Client and server environments

The CLI uses the normal `.venv` (`uv sync --extra train` or `pip install -e '.[train]'`).
It no longer imports `qwen-tts` or loads Qwen weights locally. The `tts` extra is
retained as an empty compatibility alias; it does not install an inference server.

Install **vLLM-Omni and its matching vLLM/PyTorch versions** in a separate
`.venv-vllm-omni` environment following the upstream instructions for your GPU:

- [Installation](https://docs.vllm.ai/projects/vllm-omni/en/stable/getting_started/installation/)
- [Qwen3-TTS recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md)
- [Speech API](https://docs.vllm.ai/projects/vllm-omni/en/stable/serving/speech_api/)

Do not install the server in the Cohere environment. In particular, this project's
ROCm 6.4 client dependency configuration does not certify a vLLM-Omni server on
ROCm 6.4. Stage compatible server dependencies and model files on the offline
cluster before submitting jobs. No Docker is required by these scripts.

For a server already installed in its own environment:

```bash
export HF_HOME="$MEDSTT_WORKDIR/cache/hf"
.venv-vllm-omni/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --omni --host 127.0.0.1 --port 8091
```

The model's default deploy configuration is supplied by vLLM-Omni. Apply any
additional hardware-specific options from its recipe for your installed version.

## Configure in medstt.yaml

```yaml
tts:
  output: synth-medical-words
  base_url: http://localhost:8091/v1
  model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
  api_key_env: MEDSTT_TTS_API_KEY
  api_key: null
  concurrency: 4
  timeout: 300
  max_attempts: 3
  voice_prompts:
    male: >-
      Voix d'homme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
    female: >-
      Voix de femme adulte francophone, naturelle, claire et posée,
      avec une diction médicale précise.
  test_voice_prompts:
    male: >-
      Voix d'homme âgé francophone, grave et légèrement rauque,
      avec un débit lent et une articulation précise.
    female: >-
      Voix de jeune femme francophone, timbre aigu et lumineux,
      avec un débit vif et une articulation précise.
  sampling_rate: 16000
  min_duration: 1.0
  max_duration: 20.0
```

`concurrency` bounds simultaneous audio requests **per shard**, using a fixed pool
of async workers. Each shard completes train before starting test; the limit is
not doubled for the two splits. Four shards with concurrency four can submit up
to sixteen simultaneous requests in total. Completion order can differ from input
order, but IDs and final metadata ordering remain stable.

`timeout` is the wall-clock deadline in seconds for each HTTP attempt.
`max_attempts` bounds retries, with short backoff and a warning for each failure.
SDK retries are disabled to avoid multiplying attempts. Permanent HTTP errors
(such as 400/401/404) are not retried within that sample. Empty, malformed and
non-finite audio is treated as a failed generation, not a successful recording.

The request includes the text, `task_type: VoiceDesign`, `language: French`, the
selected `instructions`, and `response_format: wav`. Responses are resampled to
16 kHz, converted to mono and peak-normalized by the client. API credentials come
from `--api-key`, then the variable named by `tts.api_key_env`, then `tts.api_key`.
`HF_TOKEN`, if needed to download a gated model, belongs to the server environment;
it is not the speech API credential.

## Generate and resume

```bash
source .venv/bin/activate
medstt tts synthesize medical-sentences.jsonl
```

This reads the output name, concurrency and endpoint from YAML. `tts.output` defaults
to `synth-medical-words`, including for existing YAML files without this setting.
The dataset is written under `datasets/<output>/{train,test}/`. Use `-o` or
`--output` for a one-off override:

```bash
medstt tts synthesize medical-sentences.jsonl -o synth-medical-words \
  --base-url http://server:8091/v1 --concurrency 8 --timeout 300
```

The output is:

```text
synth-medical-words/
  train/
    audio/
    _parts/
    metadata.csv
  test/
    audio/
    _parts/
    metadata.csv
```

Completed WAV files are published by rename before appending progress records.
Writes within a shard are serialized on the event loop. Re-running skips readable
completed recordings and retries failures or missing/corrupt recordings. Duration
rejections are final. Duplicate or colliding sanitized sentence IDs are rejected.
Use distinct shard indices for simultaneous jobs; do not run the same shard twice.
Changing shard counts is supported only after stopping the old jobs.

The existing progress layout remains compatible with previous direct-Qwen runs:
valid audio is retained, so a resumed dataset can contain both old and new
recordings. Use a new output name to regenerate everything through vLLM-Omni.

## Slurm

Each array task starts one local vLLM-Omni server on its allocated GPU by default,
then runs the CLI client from `.venv`. Set `MEDSTT_TTS_VLLM_BIN` to override the
server executable (default: `<repo>/.venv-vllm-omni/bin/vllm`). The model comes from
`tts.model` in YAML. Startup is checked with `/health`, bounded by
`MEDSTT_TTS_STARTUP_TIMEOUT` (900 seconds by default). The server is stopped when
the client exits. Logs are written to `slurm-logs/vllm-tts-<job>-<shard>.log`.

```bash
job=$(sbatch --parsable --array=0-3 slurm/synthesize.sbatch \
  medical-sentences.jsonl synth-medical-words)
sbatch --dependency=afterok:${job%%;*} slurm/finalize_dataset.sbatch synth-medical-words
```

Add the account/partition options required by your cluster. Optional simple server
flags can be supplied as whitespace-separated `MEDSTT_TTS_SERVER_ARGS`; shell
quoting inside this variable is not interpreted. Use YAML `tts.model` rather than
a CLI `--model` override when the job launches its own server.

To use an **existing server**, set `MEDSTT_TTS_BASE_URL` before submitting. This
skips local server startup; the endpoint must be reachable from every task. For
example, on a CPU partition:

```bash
export MEDSTT_TTS_BASE_URL=http://tts-server:8091/v1
sbatch --gpus-per-node=0 --array=0-3 slurm/synthesize.sbatch \
  medical-sentences.jsonl synth-medical-words
```

The shell variable chooses between existing/local server modes; ordinary CLI use
reads `tts.base_url` from YAML. Configure `tts.concurrency` in YAML in either mode.

## Finalization and evaluation

Each split publishes its own `metadata.csv` at the end of each shard's generation
pass, including on resume and when some requests failed. The CSV contains all
successful, readable recordings found across shards for the current input;
missing, rejected, mismatched or unreadable entries are excluded. If no valid
recordings exist, a header-only CSV is written. Logs report incomplete slots.

Train metadata is therefore available before test finishes. Each completed shard
and subsequent retry refreshes the CSV with the available successes. Publication
is atomic, and progress reads/writes use a per-split file lock. A failed request
still gives the CLI a nonzero exit status, but does not prevent CSV publication.

A single shard then finalizes the paired splits automatically. For arrays, finalize the
parent dataset after all shards succeed. Failures cause a nonzero CLI exit, so
an `afterok` dependency cannot finalize failed jobs.

The early split CSVs contain that split's successful recordings. Paired finalization
can subsequently remove train rows whose test counterparts were rejected (and vice
versa). Use the paired final output when equal train/test example counts are required.

Final metadata retains only matching sentence/voice pairs. If either recording
is rejected for duration, its counterpart is excluded too. Thus train and test
have the same example counts and texts, but can differ in duration and disk size.
Unselected WAVs remain for resume; loaders use the metadata rows.

Use `synth-medical-words/train` for training and `synth-medical-words/test` for
evaluation. This is voice-style variation on known texts: VoiceDesign does not
guarantee fixed or disjoint speaker identities, and changing inference engines
does not guarantee identical waveforms. Keep the real-audio test separate.
