# CED audio tagging

CED (Consistent Ensemble Distillation) is a ViT-based multi-label AudioSet
classifier. It returns sound-category scores, not a speech transcript or
timestamped sound-event boundaries.

The packaged variants are CED Tiny, Mini, Small, and Base in F32. CED Tiny is
the default package.

- Upstream: [CED](https://github.com/RicherMans/CED).
- Checkpoints: [mispeech](https://huggingface.co/mispeech).
- Architecture: HTK log-mel frontend, frequency-wise batch normalization,
  non-overlapping patch convolution, separate frequency/time positions,
  pre-norm ViT encoder, mean pooling, LayerNorm and sigmoid classifier.
- The official Hugging Face checkpoints declare Apache-2.0. The original
  training repository separately uses GPL-3.0; its code is not bundled here.

## Usage

```bash
audiocpp_cli --family ced --task cls \
  --model /path/to/CED-GGUF/ced-tiny-f32.gguf \
  --backend cpu --threads 4 --audio input.wav \
  --request-option top_k=10 --log
```

`--family ced` should be supplied explicitly. CPU, CUDA, and Vulkan use the
same model graph. The output is `custom_schema_output` with schema
`ced.classification.v1` and a `scores` array containing `index`, `label`,
`score` (sigmoid probability), and `logit`. Categories are independent, so
their probabilities need not sum to one.

## Model

| Field | Value |
|---|---|
| Family | `ced` |
| Tasks | `classify` |
| Modes | `offline` |
| Languages | `language_agnostic` |
| Default package | `ced_tiny_f32` |
| Input | Audio recording |
| Output | Audio tag scores |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `top_k` | `int` | `10` | Number of highest-scoring AudioSet categories returned. Declared range: minimum 1, maximum 527. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `ced.attention` | `auto`, `flash`, `eager` | `auto` | Attention implementation; auto selects a supported fast path. |
| `ced.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_k` | `native` | Weight storage override; native preserves packaged tensor types. |

## Usage Notes


Input is mixed to mono and resampled to 16 kHz with the framework's torchaudio
sinc implementation. Audio must provide at least one 16-frame spectrogram
patch. This is offline classification, not streaming keyword spotting.

Long recordings are split after frontend normalization at the checkpoint's
`target_length`. The final partial window is zero-padded after batch
normalization. Encoder embeddings are averaged across windows before the
classifier, following the official Hugging Face implementation. The original
training repository instead averages per-window probabilities; those are not
equivalent longform reference paths. Complete final windows are retained,
including when the frame count is an exact multiple of `target_length`.

## Conversion

Download an official `mispeech/ced-*` checkpoint and the AudioSet
`class_labels_indices.csv` used by its Python configuration. Then run:

```bash
conda run -n qwen3-tts python tools/models/convert_ced_gguf.py \
  --source /path/to/ced-tiny \
  --labels-csv /path/to/class_labels_indices.csv \
  --output /path/to/audio.cpp-gguf/CED-GGUF/ced-tiny-f32.gguf
```

The conversion preserves the checkpoint tensors and embeds the spec,
configuration, processor settings, and complete AudioSet labels. It also stores
the fixed torchaudio Hann window and mel filterbank for the native frontend.
Use the converted GGUF for inference; an unprepared upstream safetensors
directory does not contain those frontend tensors. Label repair
is necessary because some names in the upstream serialized configuration are
truncated at commas; Python replaces them from the CSV when loading.
