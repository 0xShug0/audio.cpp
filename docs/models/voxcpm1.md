# VoxCPM1

VoxCPM1 is a **tokenizer-free TTS model** from [OpenBMB](https://github.com/OpenBMB/VoxCPM) that generates 16 kHz mono speech. The audio.cpp package uses a native GGUF layout with audio.cpp tensor names, embedded config/tokenizer sidecars, folded AudioVAE weights, and materialized SR-conditioning tensors.

## Model Package

The default package is the standalone GGUF:

| Package ID | Display Name | Format | Precision | Files |
|---|---|---|---|---|
| `voxcpm1_0_5b_q8_0` | VoxCPM 0.5B Q8_0 GGUF | gguf | q8_0 (LLM) + f16 (AudioVAE) | `VoxCPM1-GGUF/voxcpm-0.5b-q8_0-audiovae-f16.gguf` |

The GGUF embeds:
- Hybrid-quantized model (LLM Q8_0, AudioVAE F16)
- Full config sidecar (`config.json`)
- BPE tokenizer sidecars (`tokenizer.json`, `tokenizer_config.json`)

No separate tokenizer or config files are needed at inference time.

## Quick Start

### 1. Install GGUF Model Weights

```bash
# Via model manager (recommended)
python3 tools/model_manager_v2.py install voxcpm1_0_5b_q8_0 --models-root models
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

`clon` is an alias to the same synthesis path. Route selection does not change
the existing reference-audio or continuation-prompt behavior.

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

## Model

| Field | Value |
|---|---|
| **Family** | `voxcpm1` |
| **Model directory** | `models/VoxCPM1-GGUF` (0.5B) |
| **Tasks** | `tts`, `clon` |
| **Modes** | `offline`, `streaming` |
| **Languages** | Model auto-handles supported languages (zh, en, ja, ko validated) |
| **Voice input** | Optional reference WAV; optional transcript via `--reference-text` |
| **Built-in voices** | Not exposed |
| **Output** | mono 16 kHz WAV |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Text chunking mode; default tag_aware. |
| `seed` | `int` | Model/session default | Random seed for MiniCPM and diffusion sampling. |
| `max_tokens` | `int` | `4096` | Maximum MiniCPM output tokens; default 4096. Zero selects model max_length; negative counts are rejected. |
| `min_tokens` | `int` | `2` | Minimum MiniCPM output tokens before an EOS stop is honored; default 2. |
| `num_inference_steps` | `int` | `10` | CFM diffusion sampling steps; default 10. Must be positive. |
| `guidance_scale` | `float` | `2` | CFM classifier-free guidance rate. Must be nonnegative. |
| `retry_badcase` | `bool` | offline: `true`; streaming: `false` | Retry when generation is detected as a bad case; defaults to true offline and false in streaming. Streaming ignores an explicit true value with a warning. |
| `retry_badcase_max_times` | `int` | `3` | Maximum total generation attempts when retrying bad cases, including the initial attempt; default 3. Must be positive. |
| `retry_badcase_ratio_threshold` | `float` | Model/session default | Bad-case ratio threshold for retry decisions. Must be positive. |
| `prompt_text` | `string` | Model/session default | Text prompt for prompt-continuation voice cloning. |
| `reference_text` | `string` | Model/session default | Alias for prompt_text; text prompt for prompt-continuation voice cloning. |
| `text_chunk_size` | `int` | `2048` | Maximum characters per offline text chunk; must be positive. Does not control streaming chunk cadence. Declared range: minimum 1. |
| `language` | `string` | Model/session default | Language hint for Chinese-script normalization; Cantonese/yue preserves Traditional characters. Inherits structured transcript or voice language when provided. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `voxcpm1.lora` | `path` | not set | Native LoRA directory or checkpoint file with adjacent lora_config.json; merged into generator weights at session creation. |
| `voxcpm1.lora_scale` | `float` | `1.0` | Multiplier for the adapter's configured alpha/r scaling; default 1. |
| `voxcpm1.mem_saver` | `bool` | `false` | Use tighter graph workspaces and release request runtime graphs; default false. |
| `voxcpm1.prompt_cache_slots` | `int` | `1` | Prompt and prompt-audio embedding cache slots; default 1. |
| `voxcpm1.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Model weight storage type. |
| `voxcpm1.audiovae_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | AudioVAE weight storage type. |
| `voxcpm1.weight_context_mb` | `int` | Model/session default | Model weight graph context size in MB. |
| `voxcpm1.text_embedding_graph_arena_mb` | `int` | Model/session default | Text embedding host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.lm_step_graph_arena_mb` | `int` | Model/session default | LM step host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.projection_graph_arena_mb` | `int` | Model/session default | Projection host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.local_encoder_graph_arena_mb` | `int` | Model/session default | Local encoder host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.dit_graph_arena_mb` | `int` | Model/session default | DiT estimator host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.audiovae_weight_context_mb` | `int` | Model/session default | AudioVAE weight graph context size in MB. |
| `voxcpm1.audiovae_graph_arena_mb` | `int` | Model/session default | AudioVAE decoder host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.audiovae_encoder_graph_arena_mb` | `int` | Model/session default | AudioVAE encoder host graph-metadata arena size in MiB, not device workspace. |
| `voxcpm1.audiovae_latent_capacity` | `int` | Model/session default | AudioVAE decoder latent frame capacity. |
| `voxcpm1.audiovae_encoder_sample_capacity` | `int` | Model/session default | AudioVAE encoder sample capacity. |

## Session Graph Arenas

Use these names with `--session-option`. They size host graph metadata in MiB,
not device workspace. The old `*_graph_context_mb` aliases remain accepted with
the same `voxcpm1.` prefix. Weight-context controls are separate and unchanged.

## References

- **Upstream model**: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Reference port**: [VoxCPM.cpp](https://github.com/bluryar/VoxCPM.cpp)
- **audio.cpp porting doc**: [VOXCPM1_Porting.md](../VOXCPM1_Porting.md)
- **Model spec**: [model_specs/voxcpm1.json](../../model_specs/voxcpm1.json)
- **Main docs**: [docs/tts.md#voxcpm1](../../docs/tts.md#voxcpm1)
