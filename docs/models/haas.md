# Haas-ja-v1.0

[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) is a
Japanese ASR model. It reuses Parakeet-TDT's FastConformer encoder and TDT joint,
replacing the LSTM prediction network with two Kimi Delta Attention blocks.
Use `--family haas_asr` to select this model.

## Quick Start

```bash
./build/debug/bin/audiocpp_cli --task asr --family haas_asr \
  --model models/Haas-ja-v1.0-GGUF/haas-ja-v1.0-q8_0.gguf \
  --backend cuda --session-option haas_asr.matmul_weight_type=q8_0 \
  --audio japanese.wav --text-out transcript.txt --log
```

F16 and Q8_0 packages are available. The checkpoint is Japanese-only. The
encoder uses full-context attention; the family's buffered streaming interface
does not make it a native cache-aware streaming encoder. Long-form splitting
uses the existing Parakeet runtime policy rather than upstream's quiet-point
splitting policy.

The existing `haas_asr.matmul_weight_type=q8_0` session option also keeps
load-time packed and scaled projections in Q8_0. With `native`, projections
constructed from float32 values retain float32 storage. Omit the override when
using the F16 package if that precision is preferred.

Timestamp word boundaries follow the tokenizer’s word markers. Continuous
Japanese text can therefore occupy one phrase-sized span; this does not add
Japanese morphological word segmentation.

## Model

| Field | Value |
|---|---|
| Family | `haas_asr` |
| Tasks | `asr` |
| Modes | `offline`, `streaming` |
| Languages | `ja` |
| Default package | `haas_ja_v1_q8_0` |
| Input | Japanese speech audio |
| Output | Transcript text and optional timestamps |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

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
| `return_timestamps` | `bool` | Model/session default | Accepted for CLI --words-out compatibility. Word timestamps are always returned, including when false or omitted. |
| `max_tokens` | `int` | `0` | Maximum TDT generated tokens; 0 or omitted uses the model-derived limit. Declared range: minimum 0. |
| `keep_language_tags` | `bool` | `false` | Keep language tag tokens in decoded text; default false. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline audio chunking mode. fixed selects overlapping long-form windows regardless of offline_mode; vad uses the configured VAD model (Silero by default) to skip silence; auto follows offline_mode. Offline input is downmixed and resampled as needed. |
| `audio_chunk_duration_sec` | `float` | Model/session default | VAD maximum chunk duration; omitted inherits the session duration. Fixed/long_form windows use the session setting, not this request override. Declared range: minimum 0.001. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `haas_asr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type; default native. |
| `haas_asr.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Encoder and decoder matmul weight storage type; defaults to weight_type, which defaults to native. Q8_0 measured 1.79x faster on the tested CPU and changed roughly 8 percent of transcripts without moving aggregate word error rate. |
| `haas_asr.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type; default native. |
| `haas_asr.perf_mode` | `off`, `flash_attention` | `off` | Encoder attention implementation. Default off uses the validated relative-attention path; flash_attention enables the fused implementation, which was numerically validated but slower on the tested hardware. |
| `haas_asr.weight_context_mb` | `int` | `3072` | Weight context arena size in MiB; default 3072. Declared range: minimum 1. |
| `haas_asr.encoder_graph_arena_mb` | `int` | `1024` | Encoder graph arena size in MiB; default 1024. Declared range: minimum 1. |
| `haas_asr.decoder_graph_arena_mb` | `int` | `256` | Decoder graph arena size in MiB; default 256. Declared range: minimum 1. |
| `haas_asr.audio_chunk_duration_sec` | `float` | `2` | Center-region duration for buffered streaming in seconds; default 2. Fixed context windows are re-encoded rather than cache-aware. Also sizes offline fixed/long_form windows. Declared range: minimum 0.001. |
| `haas_asr.left_context_sec` | `float` | `10` | Past context included when re-encoding each buffered-streaming window in seconds; default 10. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `haas_asr.right_context_sec` | `float` | `2` | Future lookahead included when re-encoding each buffered-streaming window in seconds; default 2 and adds equivalent partial-result latency. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `haas_asr.streaming_attention_mode` | `full_context` | `full_context` | Attention policy inside each buffered window. full_context preserves bidirectional attention over the bounded window. |
| `haas_asr.offline_mode` | `full_context`, `long_form`, `auto` | `full_context` | Offline encoder scheduling. full_context encodes the whole utterance, long_form uses bounded overlapping windows, and auto selects long_form beyond audio_chunk_threshold_sec. |
| `haas_asr.audio_chunk_threshold_sec` | `float` | `30.0` | Duration threshold used by offline_mode=auto before switching to bounded-window long-form execution; default 30 seconds. Declared range: minimum 0.001. |
| `haas_asr.vad_model_path` | `path` | `assets/framework/models/silero_vad` | VAD model path used by audio_chunk_mode=vad (any built-in VAD family); default assets/framework/models/silero_vad. |

## Convert Weights

To convert the official checkpoint, download `Haas-ja-v1.0.nemo` from the
upstream repository, then run:

```bash
python tools/community_models/convert_haas.py \
  --source Haas-ja-v1.0.nemo --staging haas-converted \
  --converter build/debug/bin/audiocpp_gguf --type q8_0 \
  --output haas-ja-v1.0-q8_0.gguf
```

Conversion requires PyTorch, PyYAML, NumPy, safetensors, SentencePiece,
tokenizers, protobuf and Transformers (tested with Transformers 5.12.1 and
protobuf 5.29.5). NeMo and flash-linear-attention are needed only
to run the official Python implementation, not for conversion or C++ inference.
