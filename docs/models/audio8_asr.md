# Audio8-ASR-0.1B in audio.cpp

[Audio8-ASR-0.1B](https://huggingface.co/Audio8/Audio8-ASR-0.1B) is a compact
multilingual autoregressive ASR model (en / zh / yue / ja / ko / fr / de): a
Qwen3-ASR audio encoder adapted by an MLP tower into an 8-layer Qwen2-style
decoder with only ~103M language-model parameters (324M end-to-end). The
checkpoint ships under **CC-BY-NC-4.0**, so audio.cpp loads it from locally
converted weights only; the converted GGUF must not be redistributed.

## Usage

```bash
# Convert locally (requires the audiocpp_gguf tool and a self-downloaded
# checkpoint from the Audio8/Audio8-ASR-0.1B HF repository):
python tools/community_models/audio8_asr/convert_audio8_asr.py \
    --checkpoint models/Audio8-ASR-0.1B-hf \
    --converter build/debug/bin/audiocpp_gguf \
    --type q8_0 \
    --output models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf

# Transcribe
audiocpp_cli --task asr --family audio8_asr \
    --model models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf \
    --audio sample.wav

# The safetensors package loads directly, no conversion required:
audiocpp_cli --task asr --family audio8_asr \
    --model models/Audio8-ASR-0.1B-hf --audio sample.wav
```

## Model

| Field | Value |
|---|---|
| Family | `audio8_asr` |
| Tasks | `asr` |
| Modes | `offline` |
| Languages | `en`, `zh`, `yue`, `ja`, `ko`, `fr`, `de` |
| Default package | `audio8_asr_0_1b_q8_0` |
| Input | Speech audio |
| Output | Transcript text |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `audio8_asr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Decoder and adapter weight storage type. |
| `audio8_asr.audio_encoder_weight_type` | `native`, `f32`, `f16` | `native` | Audio encoder weight storage type (quantized storage is rejected for the encoder; already-quantized GGUF tensors load as native). |
| `audio8_asr.encoder_graph_arena_mb` | `int` | `128` | Audio encoder graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |
| `audio8_asr.projector_graph_arena_mb` | `int` | `256` | Adapter (MLP tower + pooling + projector) graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |
| `audio8_asr.prefill_graph_arena_mb` | `int` | `256` | Decoder prefill graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |
| `audio8_asr.decode_graph_arena_mb` | `int` | `256` | Decoder step decode graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |

## Architecture

- **Audio frontend**: 16 kHz Whisper log-mel, 128 bins, hop 160, n_fft 400
  (shared with the `qwen3_asr` family). The reference processor emits
  bfloat16 mel values; the audio8_asr frontend rounds to bfloat16 before
  encoding to match.
- **Audio encoder**: Qwen3-ASR audio tower (d_model 896, 18 layers, 14 heads,
  FFN 3584, 128 mel bins, output dim 1024) — bit-for-bit the same
  architecture as `Qwen/Qwen3-ASR-0.6B`, loaded through the shared
  `qwen3_asr` encoder implementation.
- **Adapter**: 4 pre-norm residual MLP blocks (1024 → 4096, erf GELU), a
  final LayerNorm, then an adaptive average pool (merge factor 4 against the
  mel-frame count) followed by LayerNorm + Linear(1024 → 512).
- **Decoder**: Qwen2-style causal LM, 8 layers, hidden 512, 8 heads,
  head_dim 64, SwiGLU FFN 1408, tied embeddings over a 151,936 Qwen BPE
  vocabulary, RoPE theta 1e6, RMSNorm eps 1e-6.
- **Prompt**: `<|user|><|begin_of_audio|><|audio|>×N<|end_of_audio|>Please
  transcribe this audio.<|assistant|>` where
  `N = max(floor(floor((floor(samples / hop) + 1) / 2) / merge_factor), 1)`
  (all divisions integer).

## Measured performance

Release build (`-DCMAKE_BUILD_TYPE=Release`), Apple M4, Metal backend,
Q8_0 GGUF (345 MB weights):

| Audio | Session wall (`session.wall_ms`) | Effective RTF | CLI wall (incl. ~2.2 s process + load) |
|---|---|---|---|
| 5.95 s | ~0.75 s | ~8x realtime | 2.2 s |
| 14.07 s | 757 ms | 18.6x realtime | 3.8 s |
| 61 s (3 windows) | 3.46 s | 17.6x realtime | 5.8 s |

The audio encoder dominates (~64% of session time; its Metal shaders are
insensitive to build type, so Debug and Release measure within 3%).
Peak RSS is ~1.0 GB for a single clip and ~1.3 GB for a three-window
transcription (Metal buffers + per-window-shape graph pools on top of the
345 MB of weights; CPU backend measures the same ~1.0 GB).

## Known limitations

- **Offline only**: no streaming mode, no word timestamps, no language-id
  output.
- **30-second windows**: audio longer than 30 s is transcribed in fixed
  windows sized at the input sample rate (0.5 s minimum tail folds into the
  previous window) and space-joined, without VAD segmentation. Unlike the
  single-pass reference, each window is peak-normalized independently, so
  relative loudness across a window boundary can shift.
- **CC-BY-NC-4.0**: non-commercial use only; convert locally, do not
  redistribute the converted GGUF. There is no release GGUF package and no
  WebUI catalog entry (the package manager cannot download
  unsupported-license packages).
- Hotword logit boosting from the reference implementation is not ported.
