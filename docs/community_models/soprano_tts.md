# Soprano TTS

Soprano is an ultra-lightweight (~80M parameter) English-only text-to-speech model
using a two-stage architecture: a Qwen3-style causal LM (17 layers, hidden 512,
vocab 8192) that autoregressively emits per-frame 512-dimensional features, and a
non-iterative Vocos-style decoder (ConvNeXt backbone + single ISTFT head, n_fft 2048 /
hop 512) that turns those features into 32 kHz audio. No diffusion refinement is
performed in the decoder.

| Field | Value |
|---|---|
| Family | `soprano_tts` |
| Task | `tts` |
| Mode | `offline`, `streaming` |
| Languages | `en` |
| Audio | WAV; 32 kHz mono |
| Streaming | Pull events (per-chunk audio) |

---

## Install

The model-spec manager installs the original safetensors package from the official
Hugging Face repository:

```bash
python3 tools/model_manager_v2.py install soprano_1_1_80m_original
```

Or download the checkpoint directly and convert the decoder manually:

```bash
# Download the official checkpoint
git lfs install
git clone https://huggingface.co/ekwek/Soprano-1.1-80M models/Soprano-1.1-80M

# Convert (folds weight-norm from decoder.pth, emits combined.safetensors)
pip install torch numpy safetensors
python3 tools/community_models/soprano_tts/convert_soprano.py \
  --input-dir models/Soprano-1.1-80M \
  --output-dir models/Soprano-1.1-80M-converted
```

---

## Build

Build audio.cpp with Soprano support:

```bash
# Soprano only (avoids OOM from 45-model parallel compilation)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts
cmake --build build --target audiocpp_cli --parallel

# With Vulkan backend
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soprano_tts \
  -DENGINE_ENABLE_VULKAN=ON
cmake --build build --target audiocpp_cli --parallel
```

---

## CLI

### Basic inference

```bash
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-converted \
  --text "Soprano is an extremely lightweight text to speech model." \
  --out soprano.wav
```

### With Vulkan backend

```bash
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-converted \
  --backend vulkan \
  --text "Soprano runs on CPU and Vulkan backends." \
  --out soprano_vulkan.wav
```

### Custom generation parameters

```bash
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-converted \
  --text "Warmer temperature and higher max tokens produce longer audio." \
  --request-option temperature=0.5 \
  --request-option max_tokens=256 \
  --seed 42 \
  --out custom.wav
```

### Long-form with custom chunk size

```bash
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-converted \
  --text "This is a longer text that will be split into sentence-aware chunks by the framework text chunker. Each chunk is generated and decoded separately, then concatenated into the final audio output." \
  --session-option text_chunk_size=320 \
  --out longform.wav
```

### Streaming mode

```bash
build/bin/audiocpp_cli --task tts --mode streaming --family soprano_tts \
  --model models/Soprano-1.1-80M-converted \
  --text "Streaming mode emits audio chunks as they are generated." \
  --out stream.wav \
  --out-dir stream_chunks
```

---

## Options

### Request options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option max_tokens=<n>` | integer | `512` | Maximum generated audio frames per chunk. |
| `--temperature` / `--request-option temperature=<f>` | float | `0.3` | AR sampling temperature. |
| `--top-p` / `--request-option top_p=<f>` | float | `0.95` | Nucleus sampling threshold. |
| `--repetition-penalty` / `--request-option repetition_penalty=<f>` | float | `1.2` | Repetition penalty. |
| `--request-option eos_bias=<f>` | float | `0.0` | Additive bias on EOS logit; positive stops sooner. |
| `--seed` / `--request-option seed=<n>` | integer | random | AR sampling seed. |

### Session options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--session-option text_chunk_size=<n>` | chars | `200` | Max codepoints per chunk. |

### Load options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--session-option backbone_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `f32` | LM weight storage. `f16`/`q8_0` are faster (see Performance). |
| `--session-option decoder_weight_type=<type>` | `native`, `f32`, `f16` | `f32` | Decoder weight storage. |

---

## Server

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "models": [
    {
      "id": "soprano",
      "family": "soprano_tts",
      "path": "models/Soprano-1.1-80M-converted",
      "task": "tts",
      "mode": "offline"
    }
  ]
}
```

```bash
audiocpp_server --config server.json

# OpenAI-compatible TTS endpoint
curl http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "soprano",
    "input": "Soprano is an extremely lightweight text to speech model.",
    "response_format": "wav"
  }' \
  -o server_output.wav
```

---

## GGUF package

Standalone GGUF packages are available on Hugging Face:

```bash
# Install with the model manager
python3 tools/model_manager_v2.py install soprano_1_1_80m_q8_0

# Or install the BF16 variant
python3 tools/model_manager_v2.py install soprano_1_1_80m_bf16
```

Inference with the GGUF package:

```bash
build/bin/audiocpp_cli --task tts --family soprano_tts \
  --model models/Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf \
  --text "GGUF packages are standalone and self-describing." \
  --out gguf_soprano.wav
```

To create a GGUF package from the converted safetensors yourself:

```bash
build/bin/audiocpp_gguf \
  --input models/Soprano-1.1-80M-converted/combined.safetensors \
  --output Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf \
  --type q8_0 \
  --root models/Soprano-1.1-80M-converted \
  --family soprano_tts \
  --overwrite
```
## Performance

Measured on an Intel i5-10400 (6C/12T) CPU and an AMD Radeon RX Vega (8 GB) GPU,
Release build, warm cache, short/medium sentences:

| Backend | Backbone storage | RTF | Details |
|---------|------------------|----:|--------|
| CPU | F32 (default) | ~0.23 | ~4.3x realtime; LM decode dominates (~12.3 ms/frame) |
| CPU | F16 | ~0.16 | ~6x realtime; output statistically identical to F32 |
| CPU | Q8_0 | ~0.12 | ~8x realtime; sampling diverges slightly from F32 |
| Vulkan | F32 (default) | ~0.15-0.21 | ~5-7x realtime after one-time shader warmup |
| Vulkan | F16 | ~0.11-0.13 | ~8x realtime |
| Vulkan | Q8_0 | ~0.11-0.13 | same as F16; long-form text amortizes to ~0.08 |

The LM decode step is memory-bandwidth bound: halving weight traffic (F16)
speeds it up ~1.6x on CPU. Storage types are selected per session (see below);
F32 remains the bit-exact reference, while F16 measured numerically identical
output for this checkpoint, and Q8_0 trades a small sampling drift for the
fastest inference.

### Tuning storage types

```
# CPU: F16 backbone (recommended)
build/bin/audiocpp_cli --task tts --family soprano_tts --model models/soprano-1.1-80m-converted \
  --text "..." --session-option backbone_weight_type=f16 --out out.wav

# CPU: Q8_0 backbone (fastest)
build/bin/audiocpp_cli --task tts --family soprano_tts --model models/soprano-1.1-80m-converted \
  --text "..." --session-option backbone_weight_type=q8_0 --out out.wav

# GPU: pre-quantized GGUF packages already run at the q8_0 rate
```

Timing logs are available through `--log`:
- `soprano_tts.lm.generate_ms` -- LM AR decode time
- `soprano_tts.lm.frames` -- generated frames
- `soprano_tts.decoder.decode_ms` -- Vocos decoder time
- `soprano_tts.lm.decode.plan_cached` -- plan caching status

---

## Memory

| Metric | Value | Conditions |
|--------|-------|------------|
| Model size (safetensors) | ~380 MB (backbone BF16) + ~18 MB (decoder F32) | Original HF checkpoint |
| Peak RSS (CPU) | ~1.2 GB | Graph arena (512 MB) + weight context (256 MB) + runtime overhead |
| Peak VRAM (Vulkan) | Not measured | Vega ~1.2 GB reported system RAM usage |

---

## Known limitations

- English-only (model limitation)
- No voice cloning
- EOS sampling unreliable at low temperature (C++ RNG != PyTorch RNG)
- Full composite build may OOM; use AUDIOCPP_MODEL_SET=custom with AUDIOCPP_MODELS=soprano_tts

---

## Architecture

Soprano uses a two-stage architecture:

1. **Qwen3 causal LM** (17 layers, hidden 512, 4 heads, 1 KV head, head_dim 128, vocab 8192,
   intermediate 2304, rope_theta 10000). Takes prompt `[STOP][TEXT]<text>[START]` and
   autoregressively generates tokens. Each step's last-layer hidden state (512-dim) equals
   one audio frame.

2. **Vocos decoder** (non-iterative): Interpolate x4 linear align_corners -> Conv1d(512->768,k=1)
   -> LN -> 8x ConvNeXt(dwconv k=3 groups, LN, Linear->2304, GELU, Linear->768, gamma) -> LN ->
   Linear(768->2050) -> split mag/phase -> exp*exp(i*phi) -> istft(center=True) with Hann window
   (n_fft=2048, hop=512).

Output: 32 kHz mono. Token ~ 2048 samples ~ 64 ms.

Reference: https://github.com/ekwek1/soprano
Weights: https://huggingface.co/ekwek/Soprano-1.1-80M
