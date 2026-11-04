# Underdog Bark 3.0

Conway Research's **Standard, Plus and Pro are English ASR variants** of the
`underdog_bark3` community family, sharing the implementation under
`parakeet_tdt`. These are ASR models, not Bark speech-generation models.
All share the 0.6B FastConformer/TDT architecture, publisher's 250 ms prefix,
original-sample timestamps and native runtime.

| Variant | Pinned Hugging Face source / revision | Source weights | Converter | Recommended GGUF |
|---|---|---|---|---|
| Standard | [ConwayResearch/Underdog-Bark-3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0) / `85db2ad18ab41e534a222ee253c29191d77ed919` | 163 MB `model.bark` | `convert_underdog_bark3_standard.py` | `underdog-bark-3.0-mixed-f16.gguf` |
| Plus | [ConwayResearch/Underdog-Bark-3.0-Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus) / `d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb` | 180 MB `model.bark` | `convert_underdog_bark3_plus.py` | `underdog-bark-3.0-plus-mixed-f16.gguf` |
| Pro | [ConwayResearch/Underdog-Bark-3.0-Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) / `4b5c51e04aa8d16dda9ecdff71bc29f7360149e4` | About 1.25 GB BF16 safetensors | `convert_underdog_bark3.py` | `underdog-bark-3.0-pro-q8_0.gguf` |

The compressed download sizes are not expanded GGUF/RAM/VRAM requirements.
All three packages select the `underdog_bark3` family. Checkpoint variant
labels describe the weights; they do not select runtime behavior. Use converted GGUF weights for this guide.

## Install

```bash
python tools/model_manager_v2.py install underdog_bark3_standard_f16 --models-root models
```

The Standard package is hosted in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Underdog-Bark-3.0-GGUF).
Plus and Pro use the same family; convert their checkpoints with the commands below.

## Quick Start

Use any recommended GGUF with the same flags:

```bash
audiocpp_cli --task asr --family underdog_bark3 \
  --backend cuda --threads 8 \
  --model models/Underdog-Bark-3.0-GGUF/underdog-bark-3.0-mixed-f16.gguf \
  --audio input-16k-mono.wav --text-out transcript.txt --words-out words.json --metrics

ffmpeg -i input.wav -ar 16000 -ac 1 -f f32le - | \
  audiocpp_cli --task asr --family underdog_bark3 \
    --backend cuda \
    --model models/Underdog-Bark-3.0-GGUF/underdog-bark-3.0-mixed-f16.gguf \
    --mode streaming --audio - --input-format f32le --input-rate 16000 --input-channels 1
```

Use `--backend cpu` or `vulkan` for other devices and
`--device` to choose the intended GPU when needed.

The single [server example](../../examples/server/underdog-bark3-asr.json) has
offline and buffered-streaming entries for all three variants. Replace the paths,
or keep only the entries you use. Lazy loading and `max_loaded_models: 1` avoid
keeping all three resident together. Pro's existing IDs remain available.

| Variant | Offline model ID | Streaming model ID |
|---|---|---|
| Standard | `bark3standard` | `bark3standard-stream` |
| Plus | `bark3plus` | `bark3plus-stream` |
| Pro | `bark3` | `bark3-stream` |

```bash
audiocpp_server --config examples/server/underdog-bark3-asr.json
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=bark3standard -F file=@input-16k-mono.wav
```

Streaming entries support the raw-PCM/SSE
[`/v1/audio/transcriptions/live`](../../app/server/api.md#post-v1audiotranscriptionslive)
route. These use the ordinary legacy server, without parallel-runtime flags.


## Model

| Field | Value |
|---|---|
| Family | `underdog_bark3` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Language | English |
| Input | Speech audio |
| Output | Transcript text and optional word timestamps |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model GGUF file or package directory. |
| `--backend` | Backend supported by the selected build. |
| `--audio` | Input recording; streaming raw PCM uses `--audio -`. |
| `--mode` | `offline` or `streaming`. |
| `--text-out` | Save the transcript. |
| `--words-out` | Save word timestamps. |
| `--log` | Enable runtime logging. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `return_timestamps` | `bool` | Model/session default | Accepted for CLI --words-out compatibility. Word timestamps are always returned, including when false or omitted. |
| `max_tokens` | `int` | `0` | Maximum TDT generated tokens; 0 or omitted uses the model-derived limit. Declared range: minimum 0. |
| `keep_language_tags` | `bool` | `false` | Keep language tag tokens in decoded text; default false. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline audio chunking mode. fixed selects overlapping long-form windows regardless of offline_mode; vad uses the configured VAD model (Silero by default) to skip silence; auto follows offline_mode. Offline input is downmixed and resampled as needed. |
| `audio_chunk_duration_sec` | `float` | Model/session default | VAD maximum chunk duration; omitted inherits the session duration. Fixed/long_form windows use the session setting, not this request override. Declared range: minimum 0.001. |
| `hotwords` | `string` | Model/session default | Optional native vocabulary bias: comma-separated terms or a JSON list; up to 25 terms. |
| `hotwords_score` | `float` | `2.0` | Phonon-2/Bark native matching-piece logit bonus; duration scores and blank are not biased. Declared range: minimum 0, maximum 100. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `underdog_bark3.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type; default native. |
| `underdog_bark3.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Encoder and decoder matmul weight storage type; defaults to weight_type, which defaults to native. Q8_0 measured 1.79x faster on the tested CPU and changed roughly 8 percent of transcripts without moving aggregate word error rate. |
| `underdog_bark3.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type; default native. |
| `underdog_bark3.perf_mode` | `off`, `flash_attention` | `off` | Encoder attention implementation. Default off uses the validated relative-attention path; flash_attention enables the fused implementation, which was numerically validated but slower on the tested hardware. |
| `underdog_bark3.weight_context_mb` | `int` | `3072` | Weight context arena size in MiB; default 3072. Declared range: minimum 1. |
| `underdog_bark3.encoder_graph_arena_mb` | `int` | `1024` | Encoder graph arena size in MiB; default 1024. Declared range: minimum 1. |
| `underdog_bark3.decoder_graph_arena_mb` | `int` | `256` | Decoder graph arena size in MiB; default 256. Declared range: minimum 1. |
| `underdog_bark3.audio_chunk_duration_sec` | `float` | `2` | Center-region duration for buffered streaming in seconds; default 2. Fixed context windows are re-encoded rather than cache-aware. Also sizes offline fixed/long_form windows. Declared range: minimum 0.001. |
| `underdog_bark3.left_context_sec` | `float` | `10` | Past context included when re-encoding each buffered-streaming window in seconds; default 10. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `underdog_bark3.right_context_sec` | `float` | `2` | Future lookahead included when re-encoding each buffered-streaming window in seconds; default 2 and adds equivalent partial-result latency. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `underdog_bark3.streaming_attention_mode` | `full_context` | `full_context` | Attention policy inside each buffered window. full_context preserves bidirectional attention over the bounded window. |
| `underdog_bark3.offline_mode` | `full_context`, `long_form`, `auto` | `full_context` | Offline encoder scheduling. full_context encodes the whole utterance, long_form uses bounded overlapping windows, and auto selects long_form beyond audio_chunk_threshold_sec. |
| `underdog_bark3.audio_chunk_threshold_sec` | `float` | `30.0` | Duration threshold used by offline_mode=auto before switching to bounded-window long-form execution; default 30 seconds. Declared range: minimum 0.001. |
| `underdog_bark3.vad_model_path` | `path` | `assets/framework/models/silero_vad` | VAD model path used by audio_chunk_mode=vad (any built-in VAD family); default assets/framework/models/silero_vad. |

## Input, context and timestamps

Supply original unpadded audio. The adapter adds **4000 zero samples at 16 kHz
(250 ms)** once per request/stream after resampling/downmixing. Word timestamps
subtract the prefix and stay within the recording; they follow native TDT emission
boundaries, not an independently validated forced aligner. For source parity use
identical mono 16 kHz inputs: the publisher uses a different resampler at other
rates. Streaming requires contiguous mono 16 kHz PCM chunks. Empty audio is
rejected; silence is valid.

Full context is the default. Bidirectional attention grows quadratically with
duration and retains the encoder frame limit. Explicit `offline_mode=auto`
switches to bounded windows after 30 seconds including the prefix; `long_form`
always uses windows. Windowing and offline VAD can change punctuation/case and
word spans. Enable offline-only VAD with:

```text
--request-option audio_chunk_mode=vad
--request-option audio_chunk_duration_sec=30
--session-option underdog_bark3.vad_model_path=assets/framework/models/silero_vad
```

The bundled default is Silero VAD. Each speech span receives its own prefix and
keeps the original sample clock. Optional `hotwords`/`hotwords_score` provide
native token-logit bias (also HTTP `prompt`), with comma-separated terms or JSON
word/spoken forms. Empty lists or zero score disable bias; this is an extension
to the original runner. Standard emits `Yeah.` on one-second silence; unfiltered
native inference preserves it, while VAD can skip non-speech.

Streaming is buffered, not native cache-aware: bidirectional attention and
whole-utterance normalization change earlier activations when future audio
arrives. The frontend only caches raw log-mel frames whose complete
FFT/preemphasis support fits both windows. Edges and F64 whole-window normalization
are recomputed and caches reset for every stream; encoder/context policy is unchanged.

## Validation

The [consolidated validation report](../reports/underdog_bark3_validation.md)
covers Standard, Plus, and Pro source reconstruction, original-reference
comparisons, full-recording performance, streaming, and server/lifecycle checks.
It separates historical profiles from the corrected results and records the
hardware, input availability, and limits of the evidence.

## Convert

Build shared binaries once. Disable both GPU flags for CPU-only builds; adjust
paths for your generator/platform (`.exe` on Windows).

```bash
cmake -S . -B build/bark3 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=underdog_bark3 \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DENGINE_BUILD_WARMBENCH=ON
cmake --build build/bark3 --parallel --target audiocpp_cli audiocpp_gguf
```

Conversion needs Python, `numpy`, `safetensors` and `huggingface_hub`; add
`constriction==0.5.0` for Standard/Plus or `torch` for Pro. Native inference
requires none of these Python dependencies.

### Standard or Plus

Use the Standard assignments below, or replace all five with the Plus values.
These assignments use Bash syntax; in PowerShell use `$NAME = '...'` syntax or
substitute the literal values.

```bash
NAME=Underdog-Bark-3.0
REVISION=85db2ad18ab41e534a222ee253c29191d77ed919
CONVERTER=convert_underdog_bark3_standard.py
OUTPUT=underdog-bark-3.0-mixed-f16.gguf
TYPE=mixed-f16
# Plus: NAME=Underdog-Bark-3.0-Plus
#       REVISION=d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb
#       CONVERTER=convert_underdog_bark3_plus.py
#       OUTPUT=underdog-bark-3.0-plus-mixed-f16.gguf
#       TYPE=mixed-f16

hf download "ConwayResearch/$NAME" --revision "$REVISION" \
  --include model.bark bark_loader.py mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir "models/$NAME-source"
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir "models/$NAME-source"
python "tools/community_models/$CONVERTER" \
  --source "models/$NAME-source" --staging "models/$NAME-prepared" \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type "$TYPE" --output "models/$OUTPUT"
```

The converters verify checkpoint SHA256, pinned NVIDIA sidecars and vocabulary,
decode ANS/FP16 levels without executing downloaded Python, and prepare 699
inference tensors after dropping 24 training counters. Immutable profiles reject
staging from the other checkpoint. No NVIDIA model weights are downloaded.

### Pro

```bash
hf download ConwayResearch/Underdog-Bark-3.0-Pro \
  --revision 4b5c51e04aa8d16dda9ecdff71bc29f7360149e4 \
  --include model.safetensors config.json processor_config.json tokenizer.json \
    tokenizer_config.json generation_config.json NOTICE README.md coreml/LICENSE \
    runtime/transcribe_bark_pro.py runtime/requirements.txt \
  --local-dir models/Underdog-Bark-3.0-Pro-source
python tools/community_models/convert_underdog_bark3.py \
  --source models/Underdog-Bark-3.0-Pro-source --staging models/Underdog-Bark-3.0-Pro-prepared \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type q8_0 \
  --output models/underdog-bark-3.0-pro-q8_0.gguf
```

All converters support `--prepare-only`, or reuse staging with `--pack-only`
and another `--type`/`--output`. `--overwrite` replaces generated GGUFs; sources
are never changed. Every GGUF embeds the English model spec v1, tokenizer/configs,
source/conversion provenance, NOTICE and full CC-BY-4.0 license. No unpublished
model-manager download is advertised. Projection scaling and BatchNorm variance
floor compensation match the existing loader; corrected gamma stays F32.

| Variant | Precision policy |
|---|---|
| Standard / Plus | `mixed-f16` preserves every staged FP32 value, keeping non-representable row-scale products, normalization and compensated projections in F32. Q8 keeps feed-forward/pointwise matrices in exact F16 and quantizes eligible attention matrices. Q8/BF16 are rounded derivatives. |
| Standard BF16 | Additionally retains predictor, encoder-to-joint projection and output head in F32 to protect case/duration decisions; Plus's BF16 default is unchanged. |
| Pro | BF16 is source precision; F32 is lossless expanded storage, not extra precision. Protected Q8 retains predictor/projector/output head in exact source BF16 and keeps convolutions floating point. |

Keep `matmul_weight_type`/`conv_weight_type` at `native`: explicit overrides can
round protected tensors again.
