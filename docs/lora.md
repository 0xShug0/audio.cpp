# LoRA

LoRA adapters must match the base checkpoint's architecture, size, and tensor
layout. An adapter trained for another release of the same family is not
automatically compatible.

## Supported Models

| Model | Family | Session options | Adapter format and target |
|---|---|---|---|
| Audio Flamingo | `audio_flamingo` | `audio_flamingo.lora`, `audio_flamingo.lora_scale` | Matching PEFT encoder/projector/decoder adapters. SpotSound uses the Audio Flamingo 3 base with `template_name=temporal_grounding`; see the [model guide](models/audio_flamingo.md#spotsound-temporal-grounding). |
| Qwen3-TTS | `qwen3_tts` | `qwen3_tts.lora`, `qwen3_tts.lora_scale` | PEFT Talker/Subtalker adapters. Custom-voice packages can also include `speaker_embedding.safetensors` and `tts_config.json`. |
| Qwen3-ASR | `qwen3_asr` | `qwen3_asr.lora`, `qwen3_asr.lora_scale` | PEFT encoder/decoder adapters; both original and Transformers weight layouts are supported. |
| VoxCPM2 | `voxcpm2` | `voxcpm2.lora`, `voxcpm2.lora_scale` | Native `lora_weights.safetensors` with adjacent `lora_config.json`; generator only, not AudioVAE. |
| VoxCPM-0.5B | `voxcpm1` | `voxcpm1.lora`, `voxcpm1.lora_scale` | Native VoxCPM adapter and `lora_config.json`; upstream tensor names are mapped to the GGUF layout. |
| Fish Audio S2 Pro | `fish_audio` | `fish_audio.lora`, `fish_audio.lora_scale` | Prepared Fish Speech adapters for slow/fast decoder projections and embeddings; codec weights are unchanged. |
| CosyVoice3 | `cosyvoice3` | `cosyvoice3.lora`, `cosyvoice3.lora_scale` | PEFT LLM adapters, or a native checkpoint with `lora_weights.pt`, `config.json`, `speech_embedding.pt`, and `llm_decoder.pt`. |
| F5-TTS | `f5_tts` | `f5_tts.lora`, `f5_tts.lora_scale` | PEFT DiT adapters. Reconvert the matching original checkpoint with the current converter; do not use the currently published GGUF packages as the base. Vocos is unchanged. |
| OuteTTS | `outetts` | `outetts.lora`, `outetts.lora_scale` | PEFT Llama adapters for the matching OuteTTS checkpoint. DAC is unchanged. |
| YuE2 | `yue2` | `yue2.ar_lora`, `yue2.ar_lora_scale`, `yue2.nar_lora`, `yue2.nar_lora_scale` | Separate AR/NAR adapters, including NAR projection replacements. |
| ACE-Step | `ace_step` | `ace_step.lora`, `ace_step.lora_scale` | DiT matrix LoRA/LoKr in PEFT or ComfyUI/LyCORIS format. |
| VibeVoice | `vibevoice` | `vibevoice.lora`, `vibevoice.lora_scale` | PEFT language-model adapters with optional diffusion-head and connector replacements. Also accepted as load options. |

## Session Loading

Container format and adapter layout are independent. Shared tensor I/O accepts
safetensors and PyTorch checkpoints (`.pt`, `.pth`, `.bin`, `.ckpt`); the suffix
does not select PEFT, native LoRA, or LoKr semantics. Runtime loading expects a
tensor state dictionary, not an arbitrary training checkpoint.

Directory discovery uses logical names such as `adapter_model`, `lora_weights`,
and `speaker_embedding`, with the supported extensions. It prefers safetensors,
then `.pt`, `.pth`, `.bin`, and `.ckpt`. A malformed selected file is an error;
the loader does not silently try another file. The filenames below show the
usual upstream packaging, not a restriction to that container.

CosyVoice3's native `lora_weights` package uses `config.json` and requires
speech-embedding/output-head sidecars. PEFT uses `adapter_config.json`.
When both layouts are present, directory loading and export select the native
package. An explicit file selects that file; `lora_weights` retains native
configuration and required sidecars, while `adapter_model` uses PEFT metadata.

Use `--session-option` on the CLI, or the same keys in a server model's
`session_options`. Adapters are applied once while loading session weights;
they do not add inference graph operations or modify the base files.
Pre-v1 GGUF packages still require a current external or compiled model spec.

In Studio, the **LoRA adapter** group provides a server-side adapter path and
strength for Qwen3-TTS/ASR, VoxCPM1/2, Fish Audio, CosyVoice3, F5-TTS, OuteTTS,
and VibeVoice. **Browse folder** selects an adapter directory on the server;
individual checkpoint paths can also be entered. Keep required configuration
and sidecar files beside the adapter.
Changing or clearing an adapter reloads the model before the next generation;
clearing it restores the base model. A blank strength uses the model's adapter
default. Models configured by the server show these settings read-only; edit
their `session_options` in the server config. ACE-Step and YuE2 use the same
group style; YuE2 retains separate AR/NAR adapters and upload controls.

```bash
audiocpp_cli --family qwen3_asr --task asr --model base.gguf \
  --backend cuda --audio speech.wav \
  --session-option qwen3_asr.lora=adapters/domain \
  --session-option qwen3_asr.lora_scale=1 --log
```

Keep PEFT's `adapter_config.json` next to `adapter_model.safetensors`. For native
VoxCPM, point to the directory containing `lora_weights.safetensors` and
`lora_config.json`. Fish Speech Lightning checkpoints need preparation:

```bash
python tools/lora/prepare_fish_audio_adapter.py \
  --input lora.ckpt --alpha 16 --output adapter.safetensors
```

Use the training alpha, not alpha/r. The exporter handles Fish embedding-factor
orientation and records scaling in the output.

For Qwen3-TTS custom voices, keep the trained speaker embedding and configuration
with the adapter and select the configured name with `voice_id`. A sidecar-backed
custom voice uses the TTS route, not the Base checkpoint's reference-clone prompt.

## Scaling

The new integrations use `lora_scale=1` as a multiplier for the adapter's
alpha/r coefficient; PEFT rsLoRA uses alpha/sqrt(rank). Full-weight replacements
are not scaled. To restore the unmodified model, omit the adapter rather than
setting its strength to zero when it includes replacements.

Existing controls retain their behavior: VibeVoice's optional `lora_scale`
overrides the effective coefficient, YuE2 native factors use their direct
AR/NAR strengths, and ACE-Step uses an adapter-strength multiplier with zero
disabling the adapter.

YuE2 export supports custom strengths for A/B adapters; LoKr export currently
requires unit strength.

## Merged GGUF

Use the shared model-aware merge tool for every family in the table:

The tool requires PyTorch, `safetensors`, `gguf`, and a built `audiocpp_gguf`.
Fish Lightning preparation also requires `omegaconf`.

```bash
python tools/lora/merge_lora.py --family qwen3_tts \
  --base models/qwen3-tts-base-bf16.gguf --adapter adapters/my-voice \
  --type q8_0 --output models/qwen3-tts-my-voice-q8_0.gguf
```

Original converter-compatible safetensors and shard indexes can be passed directly,
with the same component namespaces as `audiocpp_gguf`. For Qwen3-TTS:

```bash
python tools/lora/merge_lora.py --family qwen3_tts \
  --input model_weights=models/Qwen3-TTS/model.safetensors \
  --input speech_tokenizer_weights=models/Qwen3-TTS/speech_tokenizer/model.safetensors \
  --root models/Qwen3-TTS --adapter adapters/my-voice \
  --type bf16 --output models/qwen3-tts-my-voice-bf16.gguf
```

Checkpoints requiring model-specific preparation still use their existing
conversion tool first. The merge tool does not guess or rewrite an unsupported
base tensor layout.

| Model | Additional merge arguments |
|---|---|
| YuE2 | `--adapter <AR>` and/or `--nar-adapter <NAR>`; strengths are `--scale` and `--nar-scale`. |
| ACE-Step | Select `--component dit_turbo_weights` (or the matching DiT component) when the base bundles multiple variants. LoRA/LoKr parsing and merging use the shared framework. |
| Fish Audio S2 Pro | A prepared adapter, or a Lightning `.ckpt` plus its training `--alpha`. |
| Qwen3-TTS | Speaker embedding/config sidecars are applied and embedded automatically. |
| CosyVoice3 | Native speech-embedding and output-head sidecars are merged automatically. |
| VibeVoice | Diffusion-head and connector sidecars are merged when included. |
| Other listed models | Use `--adapter` and optional `--scale`; default strength is `1`. |

`--scale` is an adapter-strength multiplier, including for VibeVoice; it is not
VibeVoice's runtime coefficient override. `--keep-type '<pattern>=<type>'` can be
repeated. `--converter <path>` selects an existing `audiocpp_gguf` executable.
Use `--overwrite` to replace an existing output. The base files are never changed.

The script stages only adapter factors and any changed sidecars. It does not
write a full intermediate merged checkpoint, and removes its staging directory
after conversion. Run the resulting GGUF without passing the adapter again.

The GGUF converter can merge a normalized, fully bound adapter before applying
the requested weight type. Use a 16-bit or original-dtype base where available;
merging into quantized weights first dequantizes them, then requantizes the
result. Quantization can change output independently of LoRA.

The converter's `--weight-adapter` input uses complete GGUF tensor names, including
component namespaces. `--adapter-merge-mode` selects `delta_f32`, `rankwise_f32`
(VibeVoice), or `rounded_bf16_delta` (YuE2). Model-specific adapter names must be
bound before using this low-level interface. Quantization and `--keep-type`
follow the normal [GGUF conversion rules](gguf.md).
