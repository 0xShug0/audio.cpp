# FireRedTTS3

FireRedTTS3 is a TTS family with separate Base and Instruct GGUF packages.
Base supports zero-shot voice cloning. Instruct supports reference cloning,
voice design, semantic editing, and acoustic editing.

## Quick Start

Base voice cloning:

```bash
audiocpp_cli \
  --task clon \
  --family fireredtts3 \
  --model models/FireRedTTS3-Base-GGUF/fireredtts3-base-orig.gguf \
  --backend cuda \
  --language Chinese \
  --text "今天天气很好，我们一起去公园散步吧。" \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years. Twenty two thousand five hundred times longer than you." \
  --request-option num_inference_steps=4 \
  --request-option guidance_scale=2.0 \
  --request-option stop_threshold=0.5 \
  --request-option seed=1234 \
  --out fireredtts3_base.wav
```

Instruct voice cloning:

```bash
audiocpp_cli \
  --task tts \
  --family fireredtts3 \
  --model models/FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-orig.gguf \
  --backend cuda \
  --language Chinese \
  --text "今天天气很好，我们一起去公园散步吧。" \
  --voice-ref assets/resources/b.wav \
  --reference-text "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years. Twenty two thousand five hundred times longer than you." \
  --request-option template_name=instruct_tts \
  --request-option num_inference_steps=4 \
  --request-option guidance_scale=2.0 \
  --out fireredtts3_instruct.wav
```

Voice design:

```bash
audiocpp_cli \
  --task vdes \
  --family fireredtts3 \
  --model models/FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-orig.gguf \
  --backend cuda \
  --language Chinese \
  --text "今天天气很好，我们一起去公园散步吧。" \
  --request-option template_name=voice_design \
  --request-option instruction="一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。" \
  --out fireredtts3_design.wav
```

## Model

| Field | Base | Instruct |
|---|---|---|
| Family | `fireredtts3` | `fireredtts3` |
| Tasks | `clon` | `tts`, `clon`, `vdes` |
| Mode | `offline` | `offline` |
| GGUF | `models/FireRedTTS3-Base-GGUF/fireredtts3-base-orig.gguf` | `models/FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-orig.gguf` |
| Primary use | Zero-shot voice clone | Clone, voice design, semantic edit, acoustic edit |
| Languages | Official 24 language tags plus 21 Chinese dialect tags | Same request language tags for text frontend handling |

## Templates

| `template_name` | Task | Inputs |
|---|---|---|
| `instruct_tts` | `tts` or `clon` | `--text`, `--voice-ref`, `--reference-text` |
| `voice_design` | `vdes` | `--text`, `--request-option instruction=<text>` |
| `semantic_edit` | `tts` | `--audio`, `--request-option instruction=<text>` |
| `acoustic_edit` | `tts` | `--audio`, `--request-option instruction=<text>` |

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language` / `--request-option language=<name>` | official FireRedTTS3 language or dialect tag | `Chinese` | Generation language tag. |
| `--voice-ref` | WAV path | required for clone | Prompt/reference voice. |
| `--reference-text` / `--request-option reference_text=<text>` | text | empty | Transcript for the prompt audio. |
| `--request-option template_name=<name>` | `instruct_tts`, `voice_design`, `semantic_edit`, `acoustic_edit` | path-dependent | Instruct request template. |
| `--request-option instruction=<text>` | text | required for design/edit | Voice design or edit instruction. |
| `--request-option text_chunk_size=<n>` | integer > 0 | `600` | Long-form text chunk size. |
| `--request-option text_chunk_mode=<mode>` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunk mode. |
| `--request-option num_inference_steps=<n>` | integer > 0 | `10` | FireRed DiT flow steps per latent patch. |
| `--request-option guidance_scale=<f>` | float >= 0 | `2.0` | FireRed DiT CFG scale. |
| `--request-option stop_threshold=<f>` | `0..1` | `0.5` | AR stop probability threshold. |
| `--request-option seed=<n>` | integer >= 0 | `1234` | Generation seed. |
| `--session-option fireredtts3.reference_cache_slots=<n>` | integer >= 0 | `4` | Prepared reference-audio cache slots. |
| `--session-option fireredtts3.mem_saver=true\|false` | bool | `false` | Release runtime graphs after request phases. |
| `--session-option fireredtts3.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage override for experiments. |

Language tags accepted by FireRedTTS3 Base match the official tokenizer:
`Arabic`, `Cantonese`, `Chinese`, `Czech`, `Dutch`, `English`, `Finnish`,
`French`, `German`, `Greek`, `Hindi`, `Indonesian`, `Italian`, `Japanese`,
`Korean`, `Polish`, `Portuguese`, `Romanian`, `Russian`, `Spanish`, `Thai`,
`Turkish`, `Ukrainian`, `Vietnamese`, plus Chinese dialect tags such as
`ZH_Sichuan`, `ZH_Shanghai`, and `ZH_Wu`.

Text normalization note: the native frontend applies local normalization for
Chinese, English, Cantonese, and `ZH_*` dialect tags. Other official language
tags are accepted but currently use basic whitespace cleanup, matching the
official Python path when LLM-based TN is not enabled.

Quality note: based on local checks against the original Python mode, non-English
and non-Chinese generation can be weaker even though the language tags are
accepted. Validate the specific language, task, and voice-reference setup before
depending on those outputs.
