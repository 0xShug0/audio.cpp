# F5-TTS (community model)

[F5-TTS](https://github.com/SWivid/F5-TTS) is an open-source zero-shot voice-cloning TTS built on a
flow-matching diffusion transformer (DiT) with a ConvNeXt text conditioner and a Vocos vocoder.
This community port also targets the [Habibi-TTS](https://github.com/SWivid/Habibi-TTS) finetune —
a multi-dialect Arabic checkpoint suite (MSA, SAU, UAE, ALG, IRQ, EGY, MAR, OMN, TUN, LEV, SDN, LBY)
from the same authors — which uses the identical architecture, giving Arabic support through the
same family (`habibi` / `habibi_tts` are registered as aliases).

## Quick Start

Everything needed is installable from this repository — no Python inference stack required:

```bash
# 1. Build (f5_tts is included in AUDIOCPP_MODEL_SET=full, or select it explicitly)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom \
      -DAUDIOCPP_MODELS=f5_tts -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_cli audiocpp_server

# 2. Download the model package (GGUF: DiT + Vocos vocoder in one file + vocab.txt)
python3 tools/model_manager_v2.py install habibi_unified
# Per-dialect specialized checkpoints (stronger accent): habibi_alg, habibi_egy,
# habibi_irq, habibi_mar, habibi_msa, habibi_sau, habibi_uae
# Standalone vocoder for the original safetensors checkpoints: vocos_mel_24khz

# 3. Synthesize (zero-shot: any short reference WAV + its transcript)
build/bin/audiocpp_cli --task tts --family habibi \
    --model models/Habibi-TTS/Unified \
    --voice-ref /path/to/reference.wav \
    --reference-text 'transcript of the reference audio' \
    --text 'أهلا، هذا نص عربي تجريبي.' \
    --request-option dialect=UNK --out out.wav
```

The GGUF package is self-contained: the DiT lives under the `transformer` namespace and the
Vocos vocoder under `vocos`, so no separate vocoder download or `f5_tts.vocos_path` option is
needed (the option and safetensors fallbacks still work for the original HF checkpoints).
Dialects: `UNK MSA SAU UAE ALG IRQ EGY MAR OMN TUN LEV SDN LBY`.
Reference audio longer than half the frame budget is truncated (with a warning): ~10.9s at the
default 2048-frame budget on CUDA, ~5.1s on CPU. There the DiT graph pool is allocated up front
(9 MiB per mel frame, hard-capped at 12 GiB for the batched-CFG graph and 6 GiB for the plain
one), so a pass above 1024 frames cannot build a graph at all and the budget is clamped to 960
frames. Keep refs under that and make sure the transcript matches, otherwise the transcript tail
leaks into the output.

## Model

| Field | Value |
|---|---|
| Family | `f5_tts` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `en`, `ar` |
| Default package | `habibi_unified` |
| Input | Text, reference voice audio, and the reference transcript |
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
| `reference_text` | `string` | Required | Transcript matching the reference voice audio; required by F5-TTS zero-shot cloning. |
| `dialect` | `string` | `UNK` | Habibi dialect token: UNK MSA SAU UAE ALG IRQ EGY MAR OMN TUN LEV SDN LBY; default UNK. |
| `sway_sampling_coef` | `float` | `-1` | Sway sampling coefficient for inference timesteps; default -1.0. Advertised bounds are recommended settings, not parser-enforced limits. Declared range: minimum -5, maximum 5. |
| `speed` | `float` | `1` | Speech speed multiplier for duration estimation; default 1.0. Generated text chunks shorter than 10 characters use 0.3 instead. Advertised bounds are recommended settings, not parser-enforced limits. Declared range: minimum 0.5, maximum 2. |
| `seed` | `int` | Model/session default | Non-negative noise seed; omitted selects a random seed. Explicit zero is deterministic. Declared range: minimum 0. |
| `num_inference_steps` | `int` | `32` | CFM Euler steps (NFE); default 32. The parser rejects nonpositive counts. Declared range: minimum 1. |
| `guidance_scale` | `float` | `2` | Classifier-free guidance strength; default 2.0. Advertised bounds are recommended settings, not parser-enforced limits. Declared range: minimum 0, maximum 10. |
| `strip_diacritics` | `bool` | `true` | Strip Arabic combining marks (harakat/tanwin/shadda/tatweel) before synthesis; Habibi is trained on undiacritized ASR transcripts and garbles diacritized input. Default true. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `f5_tts.lora` | `path` | Model/session default | PEFT DiT LoRA adapter directory or checkpoint file; merged into session-owned weights and graphs. Vocos is unchanged. |
| `f5_tts.lora_scale` | `float` | `1.0` | Multiplier for the adapter's alpha/r scaling; default 1. |
| `f5_tts.vocos_path` | `string` | Model/session default | Path to the Vocos vocoder checkpoint (vocos.safetensors); required unless placed next to the DiT checkpoint. |
| `f5_tts.dialect` | `string` | `UNK` | Default Habibi dialect token for requests that do not set it; default UNK. |
| `f5_tts.frame_budget` | `int` | `2048` | Total mel frames per CFM pass (ref + generated); larger budgets allow sentence-scale chunks at higher VRAM. Must be in [256, 8192]; omitted uses 2048. Explicit zero is not accepted. On CPU the allocated DiT graph pool caps a pass at 960 frames (10.2 s), so larger values are clamped there; CUDA keeps the full range. Declared range: minimum 256, maximum 8192. |

## Server usage

`audiocpp_server.json` entry: family `f5_tts`, model path = checkpoint directory (a `*.gguf`
package, or a safetensors checkpoint + `vocab.txt` for development). Session options:
`f5_tts.vocos_path` (only needed with safetensors checkpoints lacking a bundled vocoder),
`f5_tts.dialect` (default UNK), `f5_tts.frame_budget` (256-8192 mel frames per CFM pass;
omitted uses 2048, explicit zero is rejected).
Requests take `reference_text` (required), `dialect`, `speed`, `seed`, `num_inference_steps`,
`guidance_scale` (legacy alias `cfg_strength`), `sway_sampling_coef`, `strip_diacritics`.

Omitting `seed` selects fresh randomness; explicitly setting `seed=0` is deterministic.
`speed` defaults to 1.0 for duration estimation, but generated text chunks shorter
than 10 characters use 0.3 instead.

**Diacritics (harakat):** Habibi was trained on ASR transcripts, which are undiacritized —
the harakat/tanwin/shadda tokens exist in the vocab but are severely undertrained, and raw
diacritized input degrades to garbled speech with character repetitions (identical in the
Python reference; not a port issue). By default the frontend strips combining marks
(U+0640, U+064B–U+065F, U+0670) before synthesis, so `أَيْنَ اللَّوْنُ الأَحْمَر؟` reads as
`أين اللون الأحمر؟`. Disable per request with `strip_diacritics=false`.

## Convert Weights

For LoRA support, reconvert the matching original checkpoint with the current
converter; do not use the currently published F5-TTS GGUF packages as the base.
PEFT DiT adapters can then be loaded with `--session-option f5_tts.lora=<path>`.
Keep `adapter_config.json` beside the adapter weights. The optional
`f5_tts.lora_scale` multiplies alpha/r and defaults to `1`. Adapted weights and
DiT graph caches are session-owned; the Vocos vocoder is not adapted.
See the [LoRA routing table](../lora.md).

`tools/community_models/convert_f5_tts.py` wraps `audiocpp_gguf` and produces one self-contained GGUF per
checkpoint (`transformer` + `vocos` namespaces) plus the standalone vocoder package:

```bash
python3 tools/community_models/convert_f5_tts.py --checkpoint-root /models/Habibi-TTS \
    --vocos /models/vocos-mel-24khz/vocos.safetensors \
    --converter build/bin/audiocpp_gguf --output-dir gguf-packages
```

The safetensors source layout stays supported for development; the packaged/default format is GGUF.

## Checkpoints

| Model | Source | License |
|---|---|---|
| F5-TTS Base (en/zh) | `SWivid/F5-TTS` | cc-by-nc-4.0 |
| Habibi Unified (ar) | `SWivid/Habibi-TTS` | cc-by-nc-sa-4.0 |

Both are non-commercial licenses; keep that in mind before shipping anything built on them.
