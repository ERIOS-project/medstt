# 06 — Evaluate

`medstt evaluate` compares a trained run with the base Cohere Transcribe model and measures three objectives:

| Objective | Recordings | Measure |
| --- | --- | --- |
| 1. Learn new words | `synth-medical-words/train`: training sentences, training voices | Term recall and WER, gain over the base model |
| 2. Keep that learning with other voice timbres | `synth-medical-words/test`: the same sentences, test voices | Term-recall gain with test voices versus training voices, and their ratio |
| 3. Keep general French | `voxpopuli-fr/test`, or the datasets listed in `general` | Relative WER change against the base model |

**Term recall** is the share of recordings whose transcript contains the target word, compared without case or accents (« varioliforme » matches « VARIOLIFORME », not « variole forme »). It measures the goal of the project: a missed medical word barely moves the WER of a 15-word sentence. **WER** (word error rate) is the number of substituted, deleted and inserted words over the number of reference words, after lowercasing and removing punctuation; it checks that the rest of the sentence does not get worse, and it is the only measure of general French. **CER** is the same per character, for information.

The evaluation runs on every recording. Objective 1 measures what the model learned, memorization included: `train/` is the training audio itself. For objective 2, [`medstt tts synthesize`](04-tts.md#paired-train-and-test-datasets) writes in `test/` the same sentence/voice slots, under the same IDs, with the test voice prompts (`tts.test_voice_prompts`); both gains are measured on the same slots, so a gap comes from the voice. Part of it can also come from memorizing the exact training audio, which `test/` does not have. With every recording evaluated, the numbers rest on tens of thousands of files and need no confidence interval.

## Datasets

The evaluation only reads datasets that already exist under `datasets/`:

- `synthetic`: the paired dataset written by [`medstt tts synthesize`](04-tts.md#paired-train-and-test-datasets), the one used for training. Its `train/` and `test/` splits must have the `word` and `speaker` columns that synthesis writes.
- `general`: any AudioFolder of general French speech (`metadata.csv` with `file_name` and `text`), for objective 3. By default `voxpopuli-fr/test`, the official VoxPopuli test split that [`medstt dataset import`](datasets.md) writes next to the `train` split used for training. Use speech that training never saw.

## Configure

```yaml
evaluate:
  name: eval
  models:
    base: CohereLabs/cohere-transcribe-03-2026   # without adapters
    run: cohere-medical                          # runs/cohere-medical/final: LoRA adapters
  synthetic: synth-medical-words                 # reads its train/ and test/ splits
  general: [voxpopuli-fr/test]     # general speech datasets under datasets/
  batch_size: 16
  max_samples: null                # at most N files per dataset (evenly spaced), for a quick check
```

The models run locally with Transformers, on a GPU (very slow on CPU), not through an API. A model is a Hub ID, a run name, or a local directory; a run name stands for `runs/<name>/final`. A directory with an `adapter_config.json` holds LoRA adapters: they are loaded on the base model named in that file and merged. The base model is gated on Hugging Face: download it beforehand with `medstt model download CohereLabs/cohere-transcribe-03-2026` on a machine with access. The other models are compared with `base`, or with `reference` when set.

## Run

```bash
# Before training: the base model alone (results only, no comparison)
medstt evaluate --name base -m CohereLabs/cohere-transcribe-03-2026

# After training: base model versus the run
medstt evaluate --name first-run \
  -m base=CohereLabs/cohere-transcribe-03-2026 -m run=cohere-medical
```

`--model`/`-m` (`name=model`, repeatable), `--synthetic`, `--general` (repeatable), `--batch-size` and `--max-samples` override the YAML. With `--max-samples N`, N files evenly spaced over each dataset are kept, always the same ones, and the same slots in `train/` and `test/`. On Slurm, see [evaluation](slurm.md#evaluation).

Predictions are appended to `predictions.jsonl` as they are produced. Rerunning the command transcribes only what is missing, and a model whose path changed is transcribed again; the base model is therefore transcribed once, by the first evaluation.

## Output

`<workdir>/evals/<name>/`:

| File | Content |
| --- | --- |
| `report.md` | Results per model and dataset, one table per objective, term recall per split and voice, the 30 most missed words |
| `metrics.json` | The same numbers, plus results per voice and per nomenclature source, models and settings |
| `per_term.csv` | One row per synthetic recording: dataset, word, source, voice, reference, and for each model its transcript and hit (1/0) |
| `predictions.jsonl` | Every transcript, also used as the cache |

The most missed words help decide whether the word list needs cleaning.
