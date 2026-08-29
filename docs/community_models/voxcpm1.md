# VoxCPM1

VoxCPM1 is a **tokenizer-free TTS model** from [OpenBMB](https://github.com/OpenBMB/VoxCPM) that generates 16 kHz mono speech. The audio.cpp package uses a native GGUF layout with audio.cpp tensor names, embedded config/tokenizer sidecars, folded AudioVAE weights, and materialized SR-conditioning tensors.

| Field | Value |
|---|---|
| **Family** | `voxcpm1` |
| **Model directory** | `models/VoxCPM1-GGUF` (0.5B) |
| **Task** | `tts`, `clon` |
| **Modes** | `offline`, `streaming` |
| **Languages** | Model auto-handles supported languages (zh, en, ja, ko validated) |
| **Voice input** | Optional reference WAV; optional transcript via `--reference-text` |
| **Built-in voices** | Not exposed |
| **Output** | mono 16 kHz WAV |

---

## 🚀 Installation & Quick Start

### 1. Install GGUF Model Weights

```bash
# Via model manager (recommended)
python3 tools/model_manager_v2.py install voxcpm1_0.5b_q8_0 --models-root models
```

This downloads the `voxcpm-0.5b-q8_0-audiovae-f16.gguf` package (~690 MB) to `models/VoxCPM1-GGUF/`.

### 2. Compile the CLI Target

```bash
chmod +x scripts/build_linux.sh
./scripts/build_linux.sh --backend cpu --target audiocpp_cli
```

For CUDA (recommended for production):

```bash
./scripts/build_linux.sh --backend cuda --target audiocpp_cli
```

### 3. Run Inference

**Text-to-speech (offline):**

```bash
./build/linux-cpu-release/bin/audiocpp_cli \
  --task tts \
  --family voxcpm1 \
  --model models/VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf \
  --backend cpu \
  --text "Hello from VoxCPM1." \
  --out out.wav
```

**Voice clone (continuation-mode):**

```bash
./build/linux-cpu-release/bin/audiocpp_cli \
  --task tts \
  --family voxcpm1 \
  --model models/VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf \
  --backend cpu \
  --text "Hello from VoxCPM1." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --out out.wav
```

**Streaming output:**

```bash
./build/linux-cpu-release/bin/audiocpp_cli \
  --task tts \
  --family voxcpm1 \
  --model models/VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf \
  --backend cpu \
  --mode streaming \
  --text "Hello from VoxCPM1 streaming." \
  --request-option retry_badcase=false \
  --out out.wav
```

---

## 📊 Performance Benchmark

Measured on Ubuntu 24.04 with OpenMP optimization (CPU backend):

- **GGUF package size**: **~690 MB** (Q8_0 LLM + F16 AudioVAE)
- **Startup latency**: **Instant (<0.01s)** via `mmap` lazy loading
- **Inference speed** (short sentence): **~0.25 RTF** on CPU (~4x faster than real-time)
- **Idle VRAM** (with `mem_saver`): **~1.4 GB**; long text up to **~3.5 GB**

---

## 🛡️ Options & Customizations

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for voice cloning. |
| `--reference-text` | text | empty | Transcript of the reference audio (for clone prompting). |
| `--mode` | `offline`, `streaming` | `offline` | Full-output or streaming run mode; streaming requires `retry_badcase=false`. |
| `--session-option voxcpm1.mem_saver` | `true`, `false` | `false` | Use tighter graph workspaces and release request runtime graphs to reduce resident VRAM. |
| `--session-option voxcpm1.prompt_cache_slots` | integer | `1` | Prompt and prompt-audio embedding cache slots. Set `0` to disable prompt caching. |
| `--max-tokens` | integer | `4096` | Maximum generated AR tokens. |
| `--num-inference-steps` | integer | `10` | CFM flow-matching diffusion steps. |
| `--guidance-scale` | float | `2.0` | CFG strength. |
| `--request-option retry_badcase` | `true`, `false` | `true` | Auto-retry when generation is detected as a bad case. |
| `--request-option retry_badcase_max_times` | integer | `2` | Maximum retry count. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Long-form text chunking strategy. |

---

## 🔧 Architecture Notes

- **Base LM**: 24-layer MiniCPM transformer (1024 hidden, 73,448 vocab)
- **Residual LM**: 6-layer autoregressive refinement
- **Local Encoder**: 4-layer feature encoder (FSQ quantization, 8 codebooks)
- **Local DiT**: 4-layer diffusion transformer for CFM sampling
- **AudioVAE**: Encoder (128 dim, rates `[2,5,8,8]`) + Decoder (1536 dim, rates `[8,8,5,2]`)
- **Output**: 16 kHz mono

The GGUF is produced from the original OpenBMB files with:

```bash
python3 tools/community_models/convert_voxcpm1.py --overwrite
```

The converter stages the OpenBMB PyTorch checkpoint as audio.cpp tensor names, folds AudioVAE `weight_v` / `weight_g` pairs through the shared GGUF conversion path, embeds the required sidecars, and writes one native GGUF. The runtime expects that native layout directly; it does not adapt third-party VoxCPM GGUF metadata at load time.

---

## ✅ Validation Status

| Mode | Status | Notes |
|---|---|---|
| **Offline TTS** | ✅ Works | "This is a test run for the fix" → transcribes as **"This is a test."** (SenseVoice) |
| **Voice Clone** | ✅ Works | 6/6 target sentences transcribe exactly via SenseVoice; continuation-mode with reference audio + transcript |
| **Streaming** | ✅ Works | SSE PCM chunks at native 16 kHz; requires `retry_badcase=false` |
| **Reference-only clone** | ⚠️ Limited | `ref_start`/`ref_end` fails identically in the golden `VoxCPM.cpp` — model-level limitation |

**Regression guard**: VoxCPM2 path is untouched (`config.v1` default `false`); still generates 48 kHz speech with byte-identical output.

---

## 📦 Model Package

The default package is the standalone GGUF:

| Package ID | Display Name | Format | Precision | Files |
|---|---|---|---|---|
| `voxcpm1_0.5b_q8_0` | VoxCPM 0.5B Q8_0 GGUF | gguf | q8_0 (LLM) + f16 (AudioVAE) | `VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf` |

The GGUF embeds:
- Hybrid-quantized model (LLM Q8_0, AudioVAE F16)
- Full config sidecar (`config.json`)
- BPE tokenizer sidecars (`tokenizer.json`, `tokenizer_config.json`)

No separate tokenizer or config files are needed at inference time.

---

## 🔗 References

- **Upstream model**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Reference port**: [VoxCPM.cpp](https://github.com/bluryar/VoxCPM.cpp)
- **audio.cpp porting doc**: [VOXCPM1_Porting.md](../VOXCPM1_Porting.md)
- **Model spec**: [model_specs/voxcpm1.json](../../model_specs/voxcpm1.json)
- **Main docs**: [docs/tts.md#voxcpm1](../../docs/tts.md#voxcpm1)
