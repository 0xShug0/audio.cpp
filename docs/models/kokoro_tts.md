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

| Prefix | Language |
|---|---|
| `af`, `am` | American English |
| `bf`, `bm` | British English |
| `ef`, `em` | Spanish |
| `ff` | French |
| `hf`, `hm` | Hindi |
| `if`, `im` | Italian |
| `jf`, `jm` | Japanese |
| `pf`, `pm` | Brazilian Portuguese |
| `zf`, `zm` | Mandarin Chinese |

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

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language <code>` / `--request-option language=<code>` | language code | voice prefix | Text frontend language. |
| `--voice-id <id>` | packaged voice id | `af_heart` | Built-in voice pack. |
| `--seed <n>` / `--request-option seed=<n>` | integer | random | Decoder noise seed. |
| `--text-chunk-size <n>` / `--request-option text_chunk_size=<n>` | integer chars | `240` | Long-form chunk size. |
| `--session-option kokoro_tts.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Matmul weight storage type. |
| `--session-option kokoro_tts.conv_weight_type=<type>` | `native`, `f32`, `f16` | `native` | Convolution weight storage type. |

For GGUF packages, leave weight options at `native` unless you are testing a
conversion or storage policy.

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
