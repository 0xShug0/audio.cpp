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
| `--speaking-rate` | positive float | `1.0` | Speech speed multiplier. |
| `--text-chunk-size` | integer chars | `240` | Long-form chunk size. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `language` | language code | voice prefix | Text frontend language. |
| `seed` | integer | random | Decoder noise seed. |
| `speed` | positive float | `1.0` | Speech speed multiplier; `speaking_rate` is also accepted, but conflicting values are rejected. |
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

## Phoneme-group timings

Kokoro predicts a per-token frame count before the decoder runs, and the decoder upsamples by
exactly those counts, so where each unit lands in the output is known rather than estimated. Ask
for it with the `return_timestamps` request option, which is off by default:

```bash
audiocpp_cli --task tts --family kokoro_tts --model /path/to/Kokoro-82M-GGUF --backend cpu \
  --language en --text "The button was forgotten on the cotton coat." --voice-id af_heart \
  --out out.wav --words-out words.json
```

`--words-out` sets the option for you. The entries arrive in `word_timestamps`
(`audiocpp_result_word()` on the C API).

⚠ **These are phoneme groups, not written words, and they do not map one-to-one onto the input
text.** Read the next section before joining them to anything.

### What a group is, and what it is not

A group is a run of tokens between the space tokens Kokoro's vocabulary carries, labelled with its
own phonemes. Nothing in this family maps tokens back to the input text: the built-in G2P keeps no
span, and on the supplied-phoneme path there is no text being spoken at all. So the boundaries are
the model's, and whose they are depends on which path produced them:

- **Supplied phonemes.** The caller's own G2P chose the spacing, so the caller already knows which
  of its words became which group and can join the two in order.
- **Built-in G2P (the text path).** eSpeak-ng chose the spacing, and *it merges function words*.
  `on the` becomes the single group `ɔnðə`, `at a` becomes `æTə`, `in the` becomes `ɪnðə`. Three of
  five ordinary English sentences tested this way produced fewer groups than words:

  | text | words | groups |
  |---|---|---|
  | The button was forgotten **on the** cotton coat. | 8 | 7 |
  | She read the schedule aloud **at a** quarter past three. | 10 | 9 |
  | He said it was **in the** box under the table. | 10 | 9 |
  | Uranium and aluminium are both elements. | 6 | 6 |
  | I went to the store and bought a loaf of bread. | 11 | 11 |

  **Zipping these onto whitespace-split words is therefore wrong**, and wrong silently — the counts
  differ only sometimes, and where they do every later word is off by one. A caller that needs a
  written-word timeline from the text path needs its own G2P and the supplied-phoneme path.

What the groups always give, on both paths, is a correct division of the audio: the spans are
contiguous in output order and cover the buffer, so following the speech is exact even when
labelling it is not.

### Details

- **Punctuation the G2P spaced off is not reported.** An opening quote or a standalone dash gets no
  entry, because reporting one would shift a caller joining in order. Its duration is still
  consumed, so the following group starts after it. A mark attached to a word (`lˈɛft.`) stays in
  that word's span, which means a sentence-final pause falls inside the last word rather than after
  it — matching the reference implementation.
- **Spans are in output samples** at the result's own sample rate, offset across chunks so a
  multi-chunk render is one continuous timeline.
- **`confidence` is always 0.** The model does not score its own duration prediction and any number
  there would be read as one.
- **`audiocpp_model_supports_timestamps()` reports 0** for this family, and that is deliberate: the
  capability describes a written-word timeline, which this is not. Pass the option and read the
  result.
