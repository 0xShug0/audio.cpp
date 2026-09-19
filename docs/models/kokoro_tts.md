# Kokoro 82M

Kokoro 82M is a compact multilingual text-to-speech model exposed as
`--family kokoro_tts`. audio.cpp packages the model as standalone GGUF files
with all 54 upstream voice packs.

## Install

```bash
python3 tools/model_manager_v2.py install kokoro_82m_q8_0
```

The default package installs:

```text
models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf
```

BF16 is also available:

```bash
python3 tools/model_manager_v2.py install kokoro_82m_bf16
```

## Quick Start

```bash
audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --backend cpu \
  --language en-us \
  --voice-id af_heart \
  --text "Hello from Kokoro." \
  --out out.wav
```

Chinese example:

```bash
audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --backend cpu \
  --language zh \
  --voice-id zf_xiaobei \
  --text "你好，这是中文语音测试。" \
  --out out_zh.wav
```

## Model

| Field | Value |
|---|---|
| Family | `kokoro_tts` |
| Task | `tts` |
| Modes | `offline` |
| Default package | `kokoro_82m_q8_0` |
| Other package | `kokoro_82m_bf16` |
| Model file | `models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf` |

## Voices and Languages

Use `--voice-id <id>` to select one of the packaged voices. The voice prefix
selects the language family:

| Prefix | Language | Voice IDs |
|---|---|---|
| `af`, `am` | American English | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| `bf`, `bm` | British English | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` |
| `ef`, `em` | Spanish | `ef_dora`, `em_alex`, `em_santa` |
| `ff` | French | `ff_siwis` |
| `hf`, `hm` | Hindi | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |
| `if`, `im` | Italian | `if_sara`, `im_nicola` |
| `jf`, `jm` | Japanese | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo` |
| `pf`, `pm` | Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa` |
| `zf`, `zm` | Mandarin Chinese | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` |

The request language must match the selected voice. For example, use
`--language zh` with `zf_*` or `zm_*` voices.

## Runtime Resources

The release GGUF includes model weights, config, vocabulary, all voice packs,
and the generated `g2p/ja.json` and `g2p/zh.json` tables.

English, Spanish, French, Hindi, Italian, and Portuguese use the shared eSpeak
runtime. Install or package eSpeak data as described in
[`docs/espeak_phonemizer.md`](../espeak_phonemizer.md).

Chinese works from the release GGUF because `g2p/zh.json` is bundled. Avoid
mixed Latin words inside Chinese text unless they are known to map to Kokoro's
vocabulary.

Japanese also needs MeCab and UniDic. The small release GGUF does not include
UniDic because it is large. Export a local full multilingual GGUF with
`--embed-multilingual-resources` if you need Japanese to work from a bundled
package.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language` | language code | voice prefix | Text frontend language. |
| `--voice-id` | voice ID listed above | `af_heart` | Built-in voice pack. |
| `--seed` | integer | random | Decoder noise seed. |
| `--text-chunk-size` | integer chars | `240` | Long-form chunk size. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `language` | language code | voice prefix | Text frontend language. |
| `seed` | integer | random | Decoder noise seed. |
| `text_chunk_size` | integer chars | `240` | Long-form chunk size. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `kokoro_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Matmul weight storage type. |
| `kokoro_tts.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type. |

## Conversion

Convert from the official `hexgrad/Kokoro-82M` source checkout:

```bash
python tools/prepare_kokoro_gguf.py \
  --source /path/to/Kokoro-82M \
  --output-dir /path/to/Kokoro-82M-GGUF \
  --type both \
  --overwrite
```

For a fully bundled local multilingual package with eSpeak data and UniDic:

```bash
python tools/prepare_kokoro_gguf.py \
  --source /path/to/Kokoro-82M \
  --output-dir /path/to/Kokoro-82M-GGUF \
  --type both \
  --embed-multilingual-resources \
  --overwrite
```

The detailed validation notes live in
[`tests/kokoro_tts/MULTILINGUAL_GGUF.md`](../../tests/kokoro_tts/MULTILINGUAL_GGUF.md).
