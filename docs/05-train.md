# 05 — Train decoder LoRA adapters

Training adapts **only query/value attention projections in the decoder** of
`CohereLabs/cohere-transcribe-03-2026`, including cross-attention to the audio.
The encoder, embeddings, output head and all original weights remain frozen.
LoRA starts at rank 16, alpha 32 and dropout 0.05. These are initial experimental
settings, not a validated medical recipe.

## Prepare the model and real speech

On a connected machine, accept the model's Hugging Face access conditions and run:

```bash
medstt model download CohereLabs/cohere-transcribe-03-2026
medstt dataset import facebook/voxpopuli --config fr \
  --text-column normalized_text --output voxpopuli-fr
```

The official train and test splits are imported into `voxpopuli-fr/train` and
`voxpopuli-fr/test`. Only the train split is used for training. Use
`--max-samples N` to limit each imported split for a smaller pilot.
Import can download substantial audio. Copy the workspace (including its Hugging
Face cache) to the cluster for offline training. Generate and finalize
`synth-medical-words` using the earlier pipeline steps.

## Configure training without validation

New workspaces contain this configuration. Existing workspaces are not overwritten
by `medstt init`; update their `train` section explicitly and use a new run name
when switching from full training to adapters:

```yaml
train:
  run_name: cohere-medical-lora
  seed: 42
  lora:
    rank: 16
    alpha: 32
    dropout: 0.05
  data:
    train:
      - path: synth-medical-words/train
        weight: 0.8
      - path: voxpopuli-fr/train
        weight: 0.2
    eval: []
    eval_split: 0.0
    sampling_rate: 16000
    max_label_tokens: 256
  training:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 0.00001
    num_train_epochs: 3
    warmup_steps: 200
    logging_steps: 25
    save_steps: 500
    save_total_limit: 3
    dataloader_num_workers: 4
```

Weights specify approximate **audio duration shares**, not example counts. Every
source must have a positive weight, and `repeat` cannot be combined with weights.
After filtering transcripts by `max_label_tokens`, training measures audio duration, excludes unreadable/empty recordings and clips
of 30 seconds or longer, then constructs a seeded mixture. The **first source**
(synthetic audio in the default configuration) is retained in full, once per epoch.
Each additional source targets `first_source_duration * weight / first_weight`.
With 80% synthetic and 20% VoxPopuli, 100 hours of usable synthetic audio therefore
selects approximately 25 hours of VoxPopuli, not 20% of the entire VoxPopuli corpus.
VoxPopuli is shuffled using the configured seed and selected without replacement
until the duration target is reached. Only if its entire usable pool is exhausted
is it shuffled and repeated to fill the target. The target can be exceeded by at
most one clip. The selection is constructed once before training and reused across
epochs; changing the seed changes the selection. The combined dataset is shuffled
for training. Duration scanning reads audio headers and can take time on large
corpora. Long clips should be segmented with aligned transcripts before importing
if you want to retain them.

With `eval: []` and `eval_split: 0.0`, all retained examples are training data.
There is no evaluation run, held-out split, WER-based checkpoint selection or early
stopping. The final adapter is the last training step; training loss alone cannot
establish whether recognition on real medical speech improves.

## Run, resume and load

```bash
medstt train
medstt train --resume
```

The adapter, processor and adapter configuration are saved in
`runs/<run_name>/final/`. This directory contains **adapter weights**, not a full
standalone model. Keep the base checkpoint available. Intermediate checkpoints
support resuming adapter and optimizer state. Do not resume an old full-weight run.

```python
from peft import PeftModel
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

adapter = "/path/to/workspace/runs/cohere-medical-lora/final"
base = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026"
)
model = PeftModel.from_pretrained(base, adapter).eval()
processor = AutoProcessor.from_pretrained(adapter)
# Use processor(audio=..., sampling_rate=16000, language="fr") and model.generate(...).
# To export a standalone model when needed:
# model.merge_and_unload().save_pretrained("/path/to/merged-model")
# processor.save_pretrained("/path/to/merged-model")
```

Keep the original adapter/base copies if you want to disable adaptation later.
Full-scale training and recognition quality must be checked on the target GPU
cluster. CPU tests exercise adapter gradients, saving, reloading and resumption.

To compare the trained adapters with the base model, see [evaluation](06-evaluate.md).
