# Fish Audio S2 Pro

Fish Audio S2 Pro is wired as `--family fish_audio` for offline text to speech
and reference voice cloning. The integration supports standalone GGUF packages,
session-level reference caching, framework text chunking for long-form text, and
Fish-style multi-reference conditioning.

## Install

```bash
python3 tools/model_manager_v2.py install fish_audio_s2_pro_q8_0
```

The default package installs:

```text
models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf
```

## Quick Start

Text-to-speech:

```bash
audiocpp_cli --task tts --family fish_audio \
  --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf \
  --backend cuda \
  --text "Hello from Fish Audio." \
  --out out.wav
```

Reference voice clone:

```bash
audiocpp_cli --task tts --family fish_audio \
  --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf \
  --backend cuda \
  --text "The final render is ready for review." \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature." \
  --out out_ref.wav
```

## Model

| Field | Value |
|---|---|
| Family | `fish_audio` |
| Task | `tts` |
| Modes | `offline` |
| Model path | `models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf` |
| Languages | Model auto-handles language; tested paths cover English and Chinese-style prompts |
| Voice input | Optional reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

## Multi-Reference Conditioning

`multi_reference_cond` is a Fish Audio request option. It takes an ordered JSON
array of reference pairs:

```json
[
  {
    "audio": "assets/resources/a.wav",
    "text": "First reference transcript."
  },
  {
    "audio": "assets/resources/b.wav",
    "text": "Second reference transcript."
  }
]
```

Each entry must include:

| Field | Meaning |
|---|---|
| `audio` | Reference WAV path. |
| `text` | Transcript for that reference audio. |

The option has two useful cases.

### Case 1: One Generated Voice

Use multiple references without speaker tags when you want them to condition one
generated voice/style.

```bash
audiocpp_cli --task tts --family fish_audio \
  --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf \
  --backend cuda \
  --text "The review is ready, and I will check the final numbers." \
  --request-option 'multi_reference_cond=[{"audio":"assets/resources/a.wav","text":"First reference transcript."},{"audio":"assets/resources/b.wav","text":"Second reference transcript."}]' \
  --out out_multi_ref.wav
```

In this mode the references are packed into the prompt context. The target text
does not assign individual lines to reference speakers.

### Case 2: Speaker-Tagged Turns

Use Fish speaker tags in the target text when you want the generated output to
follow a speaker order. Reference array index `0` maps to `<|speaker:0|>`, index
`1` maps to `<|speaker:1|>`, and so on. Speaker ids should stay within the
provided reference indexes.

```bash
audiocpp_cli --task tts --family fish_audio \
  --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf \
  --backend cuda \
  --text "<|speaker:0|>The review is ready. <|speaker:1|>I will check the final numbers. <|speaker:0|>Please send me the final summary." \
  --request-option 'multi_reference_cond=[{"audio":"assets/resources/a.wav","text":"First reference transcript."},{"audio":"assets/resources/b.wav","text":"Second reference transcript."}]' \
  --request-option max_new_tokens=220 \
  --out out_speakers.wav
```

The reference transcripts may already contain speaker tags. If they do not, the
runtime tags them by reference order before packing the prompt.

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for voice cloning. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--max-new-tokens` | integer | `1024` | Maximum generated semantic tokens per chunk. `0` uses the default. |
| `--text-chunk-size` | integer chars | `200` | Long-form chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |
| `--temperature` | float | `0.8` | Sampling temperature. |
| `--top-k` | integer | `30` | Top-k sampling limit. |
| `--top-p` | float | `0.8` | Nucleus sampling limit. |
| `--seed` | integer | random when omitted | Sampling seed for reproducible output. |
| `--request-option multi_reference_cond=<json>` | JSON array | not set | Ordered Fish Audio reference conditioning pairs. Each entry requires `audio` and `text`. |
| `--session-option fish_audio.mem_saver=true\|false` | bool | `false` | Release cached AR runtime graphs after each request. |
| `--session-option fish_audio.reference_cache_slots=<n>` | integer | `1` | Prepared reference-audio cache slots. |
| `--session-option fish_audio.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | AR matmul weight storage type. |
| `--session-option fish_audio.codec_weight_type=<type>` | `native`, `f32`, `f16`, `q8_0` | `native` | Codec conv/matmul weight storage type. |

For GGUF packages, leave weight options at `native` unless you are deliberately
testing conversion or storage behavior.
