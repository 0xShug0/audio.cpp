# M5-TTS

`m5_tts` supports the [M5-TTS](https://github.com/ASLP-lab/M5-TTS) 50M
VoiceClone and CustomVoice checkpoints. Both produce mono 24 kHz speech using
a temporal Llama backbone, a local codebook decoder, and the shared Qwen3
12.5-Hz speech codec. The published 50M size excludes embeddings, output heads
and the codec.

## Downloads

Packages are available in [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/M5-TTS-GGUF).
Choose VoiceClone or CustomVoice, each available with F16 or Q8 LM weights.
All four packages include the F32 speech codec.

## Usage

VoiceClone requires a reference recording and its accurate transcript:

```bash
audiocpp_cli --family m5_tts --task clon --backend cuda \
  --model /path/to/m5-tts-50m-voiceclone-f16.gguf \
  --voice-ref reference.wav --reference-text "The words in the reference recording." \
  --language en --text "We followed the river until we reached the old bridge." \
  --threads 8 --out speech.wav --log
```

CustomVoice is a separate checkpoint; selecting a speaker does not switch weights:

```bash
audiocpp_cli --family m5_tts --task tts --backend cuda \
  --model /path/to/m5-tts-50m-customvoice-f16.gguf \
  --request-option voice_id=1 --language en \
  --text "The morning train arrived beside the river." \
  --threads 8 --out speech.wav --log
```

Use `--mode streaming` to emit audio incrementally during generation. Streaming
resets codec state for each text chunk and retains the final 100 ms until EOS,
matching the model's tail trim. Use `--backend cpu` or `vulkan` to select another
compiled backend.

## Model

| Field | Value |
|---|---|
| Family | `m5_tts` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | Chinese (including Cantonese and Sichuanese), English, German, French, Japanese, Korean |
| Input | Text; VoiceClone also needs reference audio and its transcript |
| Output | Mono 24 kHz speech |
| Weights | Self-contained GGUF including text tokenizer and audio codec |
| License | Apache-2.0 checkpoint model cards; [attribution](../../tests/m5_tts/NOTICE) |

VoiceClone accepts reference audio through either task. CustomVoice accepts
`tts` with the published example speaker `1`. The native text frontend uses the
framework's Chinese and English normalization; it does not reproduce every
upstream WeText grammar. For other languages, supply already normalized text.
`--language yue` selects Cantonese; `--language zh-Sichuan` selects Sichuanese.
`zh-Sichuan` is a project-specific tag. The model maps these internally to Chinese
with dialect conditioning; no separate accent option is needed. Cantonese text
is not automatically converted to traditional Chinese.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Converted GGUF file. |
| `--backend` | `cuda`, `cpu`, or another compiled backend. |
| `--threads` | CPU thread count. |
| `--text`, `--language` | Spoken text and language tag, such as `en` or `zh`. |
| `--voice-ref`, `--reference-text` | VoiceClone reference recording and transcript. |
| `--mode` | `offline` or `streaming`. |
| `--log` | Enable component timings and `session.wall_ms`. |

## Session Options (`--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `m5_tts.weight_type` | `native` | LM weight storage: `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`. |
| `m5_tts.codec_weight_type` | `native` | Codec storage: `native`, `f32`, `f16`. |
| `m5_tts.attention` | `auto` | Shared `auto`, `flash`, or `eager` attention selection. |

The retained packages use F16 or Q8 LM weights and F32 codec weights.
`m5_tts.weight_type=f32` can upcast LM weights for numerical controls without
a separate F32 package. Quantization and backend arithmetic can change the
sampled token sequence and output audio.

## Request Options (`--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `voice_id` | `1` | CustomVoice speaker. |
| `reference_text` | empty | Exact transcript required for VoiceClone. |
| `temperature` | `0.9` | Audio-token sampling temperature. |
| `top_k` | `10` | Top-k sampling cutoff. |
| `max_tokens` | `500` | Maximum generated codec frames; 12.5 frames represent one second. |
| `seed` | `2026` | Sampling seed; `-1` selects a random seed. |
| `instruct` | empty | Checkpoint caption condition. |
| `text_chunk_size` | `200` | Maximum characters per text chunk. |
| `text_chunk_mode` | `default` | Framework text-chunking policy. |

## Conversion

Download the appropriate official checkpoint and
[ASLP-lab/M5-TTS-Audio-Tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer)
into local directories. Build `audiocpp_gguf`, then convert:

```bash
python tests/m5_tts/convert_gguf.py \
  --source /path/to/M5-TTS-50M-VoiceClone \
  --codec /path/to/M5-TTS-Audio-Tokenizer \
  --variant voice_clone --type f16 \
  --output /path/to/m5-tts-50m-voiceclone-f16.gguf
```

For CustomVoice, use its checkpoint directory and `--variant custom_voice`.
`--type q8_0` creates a Q8 LM package. The converter always retains the codec and RoPE factors in F32.
No Python environment is required for native inference after conversion.

## Validation

See the [validation report](../../ideas/m5_tts_validation.md) for ONNX comparisons,
teacher-forced AR checks, streaming latency and known GPU reference-encoder
differences.
