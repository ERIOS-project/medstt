# Run the steps on Slurm

Submit the scripts in [`slurm/`](../slurm) from the repository root. Set `MEDSTT_WORKDIR` and adapt the ROCm module setup in `slurm/env.sh` for your cluster. Jobs activate `.venv` for the CLI. Synthesis starts a separate vLLM-Omni server from `.venv-vllm-omni`, unless `MEDSTT_TTS_BASE_URL` selects an existing server. The shared setup enables Hugging Face offline mode and places MIOpen caches on node-local storage. Supply your account and partition to `sbatch`; logs go to `slurm-logs/`.

```bash
export MEDSTT_WORKDIR=$SCRATCH/medstt-ws
```

Prepare [models and datasets](datasets.md) on a connected machine before submitting these jobs. Install the client/training dependencies in `.venv` and provision a compatible vLLM-Omni server separately; see [TTS](04-tts.md).

## Sentences

With an LLM endpoint already reachable from the job:

```bash
sbatch -A <account> -p <partition> \
  slurm/generate_sentences.sbatch cardio.txt cardio
```

The script can also start vLLM in the job when `MEDSTT_VLLM_MODEL` is set. The model must already be available locally or in the cache. Set `MEDSTT_VLLM_BIN` if needed and request the required GPUs with `sbatch --gpus-per-node=<count>`.

## Audio

A job array splits sentence/voice pairs into shards. Each shard sends up to
`tts.concurrency` simultaneous requests (default 4). The default mode starts and
stops a local server per task; an existing endpoint can be selected with
`MEDSTT_TTS_BASE_URL`. The dependent job merges their metadata after all shards succeed. A generation error now makes its shard fail, so Slurm will not finalize incomplete output; duration rejections are recorded separately.

```bash
jid=$(sbatch -A <account> -p <partition> --parsable --array=0-7 \
  slurm/synthesize.sbatch cardio.jsonl synth-cardio)
sbatch -A <account> -p <partition> --dependency=afterok:$jid \
  slurm/finalize_dataset.sbatch synth-cardio
```

## Training

The script uses `torchrun` with one process per GPU and supports multiple nodes:

```bash
sbatch -A <account> -p <partition> --nodes=1 --gpus-per-node=4 \
  slurm/train.sbatch
```

Training reads the `train` section of the workspace's `medstt.yaml`. To resume interrupted training, pass `--resume` after `slurm/train.sbatch`. Sentence generation and audio synthesis skip items already recorded.

## Evaluation

One GPU is enough; predictions are cached, so a resubmitted job resumes:

```bash
sbatch -A <account> -p <partition> slurm/evaluate.sbatch --name first-run
```

Options after the script name go to [`medstt evaluate`](06-evaluate.md).

## Follow job progress

CLI commands display tqdm progress in the Slurm output files, including sentence
requests, train/test synthesis, shard finalization and the Trainer's optimization
steps. Preparation commands also report nomenclature downloads, normalization,
word selection, split imports and audio-duration scans. Hugging Face model
downloads retain their built-in progress display.

```bash
tail -f slurm-logs/medstt-tts-<job-id>_0.out
tail -f slurm-logs/medstt-finalize-<job-id>.out
tail -f slurm-logs/medstt-train-<job-id>.out
```

Slurm jobs default to `TQDM_MININTERVAL=10` seconds between intermediate refreshes;
set it before `sbatch` to change the interval. Completed bars still print their
final state. Counts and rates appear even when a streaming source has no known
total; percentages and ETA require a known total. TTS displays the split and shard
and includes previously completed/rejected items in its initial count on resume.
A completed bar counts processed items, including rejected or failed generations;
check the command's final summary and job exit status for success.

Training progress is enabled by default; `train.training.disable_tqdm: true` can
still disable the Trainer bar explicitly. No additional training run is needed
to enable this display.
