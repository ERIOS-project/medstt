# Prepare models and datasets

Run these commands on a machine with internet access before submitting jobs to offline compute nodes:

```bash
medstt model download CohereLabs/cohere-transcribe-03-2026
medstt model download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
medstt dataset import facebook/multilingual_librispeech \
  --config french --split train --text-column transcript \
  --max-samples 20000 -o mls-fr
```

`model download` saves both fixed model IDs in the workspace's Hugging Face cache. Keep the default cache location so training and TTS can load them by Hub ID. `dataset import` copies a Hugging Face dataset or a compatible local dataset into `datasets/<name>/<split>/` as one AudioFolder per split. By default it imports
`train` and `test` from the source. Specify `--split train` to import only train,
or repeat the option to select several splits. `--max-samples` applies separately
to each split. Adapt the import with `--audio-column`, `--text-column`, `--config`, `--split`, `--max-samples`, and `--streaming`. Import writes the audio files and `metadata.csv` without decoding audio during the copy.

For manually prepared data, the loader accepts an AudioFolder directory with `metadata.csv` or `metadata.jsonl` containing `file_name` and `text`, or a directory created by `Dataset.save_to_disk`. In `medstt.yaml`, each `train.data.train` or `train.data.eval` item can be a dataset name or an object with `path`, `split`, `max_samples`, `repeat`, `weight`, and column names. `medstt dataset info <name>/<split>` prints the sample count and a few transcripts.

If downloads were done on another machine, copy the workspace to the cluster. Compute nodes use its `cache/hf/` directory.

The default recipe uses `voxpopuli-fr/train` with 20% real audio and 80% synthetic audio
by duration. See [Train](05-train.md) for its import command and weighted configuration.
No validation dataset or automatic holdout is required.

Imports are built in a temporary directory beside the destination and published
only after both audio and metadata are complete. The destination must be absent or
empty; use a new output name to import another version. An interrupted import
does not replace an existing dataset or publish partial metadata.


For VoxPopuli, import both official splits with one output name:

```bash
medstt dataset import facebook/voxpopuli --config fr \
  --text-column normalized_text --output voxpopuli-fr
medstt dataset info voxpopuli-fr/train voxpopuli-fr/test
```

This creates `datasets/voxpopuli-fr/train/{audio/,metadata.csv}` and
`datasets/voxpopuli-fr/test/{audio/,metadata.csv}`. Source train/test membership is
preserved; unlike synthetic pairs, their texts and sizes need not match. The test
split must not be included in `train.data.train`.

Each split is published independently after a complete import. If train succeeds
but test fails, retry with `--split test`; the completed train remains unchanged.
An existing flat import is not moved automatically: move its contents into the
correct split subdirectory (usually `train/`), then import the missing `test` split.
