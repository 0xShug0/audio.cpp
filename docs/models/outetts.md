# OuteTTS

OuteTTS 1.0 1B generates 24 kHz speech and supports voice cloning from reference
audio and its transcript. Standalone GGUF and the original Safetensors/DAC
layouts are supported.

## Install

```bash
python tools/model_manager_v2.py install outetts_1_0_1b_q8_0 --models-root models
```

The package is hosted by
[mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp/tree/main/Text%20to%20audio%20(TTS)).

## Quick Start

```bash
audiocpp_cli --task tts --family outetts \
  --model models/Llama-OuteTTS-1.0-1B_Q8 \
  --backend cuda --text "Hello from OuteTTS." \
  --max-tokens 1024 --out out.wav
```

For cloning, use a GGUF containing the DAC and forced aligner, such as the
package produced by the conversion command below:

```bash
audiocpp_cli --task clon --family outetts \
  --model models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --backend cuda --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --request-option reference_language=en \
  --text "This sentence uses the cloned voice." \
  --max-tokens 1024 --out cloned.wav
```

Around ten seconds of clean reference speech is recommended; the maximum is
twenty seconds. The transcript must match that recording. `--task tts` with
reference audio and its transcript also enables voice conditioning.

Safetensors and older GGUF packages without an embedded aligner need
`--session-option outetts.forced_aligner_path=<model>`. Cloning fails clearly
when the aligner is unavailable.

## Model

| Field | Value |
|---|---|
| Family | `outetts` |
| Model directory | `models/Llama-OuteTTS-1.0-1B` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `en`, `ar`, `zh`, `nl`, `fr`, `de`, `it`, `ja`, `ko`, `lt`, `ru`, `es`, plus moderate-data `pt`, `be`, `bn`, `ka`, `hu`, `lv`, `fa`, `pl`, `sw`, `ta`, `uk` |
| Voice input | Optional reference WAV plus its transcript |
| Output | mono 24 kHz WAV |

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
| `max_tokens` | `int` | Model/session default | Maximum generated audio tokens per chunk. When omitted, OuteTTS estimates a safe value from each chunk. Declared range: minimum 1. |
| `temperature` | `float` | Model/session default | Sampling temperature; default 0.4 for cloning, otherwise model config default. Declared range: minimum 0.0. |
| `top_k` | `int` | Model/session default | Top-k sampling; default 40 for cloning, otherwise model config default. Declared range: minimum 0. |
| `top_p` | `float` | Model/session default | Nucleus sampling in (0, 1]; default 0.9 for cloning, otherwise model config default. Declared range: minimum 0.0, maximum 1.0. |
| `min_p` | `float` | Model/session default | Minimum probability relative to the best token; default 0.05 for cloning, otherwise model config default. Declared range: minimum 0.0, maximum 1.0. |
| `repetition_penalty` | `float` | `1.1` | Windowed repetition penalty; must be > 0, default 1.1. Declared range: minimum 0.0. |
| `repetition_window` | `int` | `64` | Recent-token penalty window; default 64. Declared range: minimum 0. |
| `seed` | `int` | Model/session default | Sampling seed; cloning defaults to 4099 for native weights and 42 for quantized weights. Declared range: minimum 0, maximum 4294967295. |
| `reference_text` | `string` | Model/session default | Transcript matching the reference voice audio for voice cloning. |
| `reference_language` | `string` | Model/session default | Language code used to align the reference transcript; omitted inherits the request text language, then falls back to en. |
| `text_chunk_size` | `int` | `256` | Maximum UTF-8 codepoints per long-form text chunk; default 256. Chunks are split further when required by max_tokens or context budget. Declared range: minimum 1. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `word_budget` | Framework long-form text chunking mode; default word_budget. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `outetts.lora` | `path` | Model/session default | PEFT LoRA adapter directory or checkpoint file for the same OuteTTS checkpoint; merged into the language model at session creation. |
| `outetts.lora_scale` | `float` | `1.0` | Multiplier for the adapter's alpha/r scaling; default 1. |
| `outetts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `nvfp4` | `native` | Language-model weight storage type. Quantized CUDA voice cloning is expanded to F32 in memory for generation correctness. |
| `outetts.llama_weight_context_mb` | `int` | `4096` | Language-model weight context size in MiB; default 4096. Declared range: minimum 1. |
| `outetts.constant_context_mb` | `int` | `256` | Language-model constant tensor context size in MiB; default 256. Declared range: minimum 1. |
| `outetts.dac_weight_context_mb` | `int` | `1024` | DAC decoder weight context size in MiB; default 1024. Declared range: minimum 1. |
| `outetts.dac_graph_arena_mb` | `int` | `1536` | DAC decoder graph arena size in MiB; default 1536. Declared range: minimum 1. |
| `outetts.forced_aligner_path` | `path` | Model/session default | Optional Qwen3 Forced Aligner override. Cloning automatically uses the aligner embedded in a standalone OuteTTS GGUF when present. |
| `outetts.reference_cache_slots` | `int` | `1` | Prepared reference-profile cache slots; default 1, set 0 to disable. Declared range: minimum 0. |
| `outetts.mem_saver` | `bool` | `false` | Release cached-step and aligner runtime state after use; default false. |

## Compatibility

The released aliases `outetts.aligner_path`, `outetts.aligner_model_path`, and
`outetts.dac_graph_context_mb` remain accepted. Use the canonical names in the
tables above for new configurations.

## Convert Weights

The original source layout is available through the deprecated model manager:

```bash
python tools/model_manager_deprecated.py install outetts_1_0_1b --models-dir models
```

The installer places `DAC.speech.v1.0` and `Qwen3-ForcedAligner-0.6B` beside the OuteTTS directory. It converts the official DAC checkpoint to a safe tensor source. To do that conversion manually:

```bash
python tools/community_models/outetts/convert_outetts_dac.py \
  models/DAC.speech.v1.0/weights_24khz_1.5kbps_v1.0.pth \
  models/DAC.speech.v1.0/model.safetensors
```

Pack the language model, DAC, Qwen3 Forced Aligner, and sidecars into one standalone Q8 GGUF:

```bash
audiocpp_gguf \
  --input model_weights=models/Llama-OuteTTS-1.0-1B/model.safetensors \
  --input dac_weights=models/DAC.speech.v1.0/model.safetensors \
  --input aligner_weights=models/Qwen3-ForcedAligner-0.6B/model.safetensors \
  --root models/Llama-OuteTTS-1.0-1B \
  --sidecar models/DAC.speech.v1.0/config.json=dac/config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/config.json=aligner/config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/generation_config.json=aligner/generation_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/preprocessor_config.json=aligner/preprocessor_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/tokenizer_config.json=aligner/tokenizer_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/vocab.json=aligner/vocab.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/merges.txt=aligner/merges.txt \
  --output models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --type q8_0
```

The resulting GGUF contains all three tensor groups, the model specification, and all required sidecars. It does not need the original directories. Normal TTS does not run the embedded aligner; it is initialized only when reference cloning is requested:

```bash
audiocpp_cli --task tts --family outetts \
  --model models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --backend cuda --text "Hello from the standalone GGUF." \
  --max-tokens 1024 --out out.wav
```

## Long-form and Logging

With logging enabled, OuteTTS reports framework chunk count and token budget, per-chunk word/character counts, recommended and effective generation limits, the natural stop reason, reference-profile cache hits/evictions, Llama runtime and step-graph rebuild/reuse, released cache capacity, and timings for reference alignment, DAC encode/decode, prompt construction, generation, and the complete session request. See [OuteTTS validation](../reports/outetts_validation.md) for the reproducible long-lived session and memory test.
