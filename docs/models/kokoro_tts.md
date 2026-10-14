# Kokoro 82M

Kokoro 82M is a compact multilingual text-to-speech model exposed as
`--family kokoro_tts`. audio.cpp packages the model as standalone GGUF files
with all 54 upstream voice packs.

## Install

For server speed control, use the Kokoro GGUF version published on
[Hugging Face on September 22, 2026 (UTC)](https://huggingface.co/audio-cpp/audio.cpp-gguf/commit/406756ee8e3b16e902ce40112986c1010775f888)
or a later published version. Its embedded spec declares
`speed`. Older GGUFs missing that declaration must be updated to use non-default
speed through `/v1/audio/speech`; `speed: 1` remains accepted when the declaration is absent.

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

## Streaming

Use `--mode streaming` to emit each synthesized text chunk immediately. Text
chunking uses the same `text_chunk_size` control as offline
generation; it does not split an already generated waveform into fake streaming
events. A short input that fits in one chunk produces one audio event. Longer
inputs produce incremental, non-cumulative audio chunks and a final merged WAV.

## Model
| Field | Value |
|---|---|
| Family | `kokoro_tts` |
| Task | `tts` |
| Modes | `offline`, `streaming` |
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
| `--language` | language code | voice prefix | Language selection precedence: voice-style language, transcript language (CLI --language), request-option language, then voice prefix. Must match the selected voice. |
| `--voice-id` | voice ID listed above | `af_heart` | Built-in voice pack. |
| `--seed` | integer | random | Decoder noise seed for the current request; omitted requests choose a fresh random seed. |
| `--speaking-rate` | positive float | `1.0` | Speech speed multiplier. |
| `--text-chunk-size` | integer chars | `240` | Long-form chunk size. |

## Request Options (use with `--request-option`)

`--voice-id` takes precedence over `--request-option voice_id=...`.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `voice_id` | packaged voice ID | `af_heart` | Built-in voice pack, also selectable with `--voice-id`. |
| `language` | language code | voice prefix | Language selection precedence: voice-style language, transcript language (CLI --language), request-option language, then voice prefix. Must match the selected voice. |
| `seed` | integer | random | Decoder noise seed for the current request; omitted requests choose a fresh random seed. |
| `speed` | positive float | `1.0` | Speech speed multiplier; default 1.0. Must be finite and positive. |
| `text_chunk_size` | integer chars | `240` | Long-form chunk size. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `kokoro_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Matmul weight storage type. |
| `kokoro_tts.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type. |
| `kokoro_tts.predictor_duration_graph_arena_mb` | positive integer MiB | `384` | Duration predictor host graph-metadata arena. |
| `kokoro_tts.predictor_text_graph_arena_mb` | positive integer MiB | `256` | Text predictor host graph-metadata arena. |
| `kokoro_tts.predictor_tail_graph_arena_mb` | positive integer MiB | `640` | Tail predictor host graph-metadata arena. |
| `kokoro_tts.rknn_generator` | path to a `.rknn` | unset | Pre-compiled RKNN engine for the iSTFTNet generator, voiced on a Rockchip NPU. Needs a build with `ENGINE_ENABLE_RKNN=ON`; see [docs/build/rockchip-npu.md](../build/rockchip-npu.md). |
| `kokoro_tts.rknn_generator_overlap` | latent frames, `0` .. window-1 | `0` | Advance the window walk by `emitted - overlap` frames and blend the shared spectrogram columns with a raised-cosine ramp instead of hard-writing them. Measured on RK3588: 1.54x wall for no change in the per-window loudness spread (7.23 → 7.14 dB), so it is a seam-smoothing knob, not a loudness one. |
| `kokoro_tts.rknn_generator_gain` | `0` / `1` | `0` | Correct each NPU window's loudness toward a full-length CPU decode of the same latent (additive gain on the log-magnitude rows). Measured on RK3588: per-window loudness spread 7.23 → 3.27 dB for 1.9x wall and +544 MB RSS, because the ggml generator graph stays loaded and runs once per utterance. |

The graph-arena controls do not size device workspace. Their former
`predictor_duration_graph_mb`, `predictor_text_graph_mb`, and
`predictor_tail_graph_mb` names remain accepted with the `kokoro_tts.` prefix,
including with older embedded specs.

On RK3588 the generator is 83% of the runtime, so the NPU path is worth the extra file: measured
RTF 1.415 → 0.867 and peak RSS 2777 → 2334 MB on `kokoro-82m-q8_0` (185 Russian characters,
`--threads 4`, clipped-sine engine; the range-reduced engine the producer now exports by default
costs +21..36% generator time depending on the utterance, and buys 63 dB of sine accuracy, see the
RKNN document). The engine is compiled for a 128-latent-frame window and the runtime walks it over
the latent; see the RKNN document for the window limit and the per-window normalization caveat. The
NPU render also carries a few content-locked near-Nyquist clicks per utterance (2 to 3 on a 33 s
render, at the same samples for every engine and for every window grid, 2.5-5x the CPU render's
9-12 kHz energy at those columns): the RK3588 DPU's fp16 rounding at the columns where the graph's
activations peak, absent from the CPU path.

A `--global-stats` variant of the engine — the per-channel AdaIN mean/variance of a full-length
decode fed into the engine as graph I/O, so every window normalizes with the whole utterance's
statistics — is not in the build. In fp32 it is exact (the stitched windowed decode moves from
9.7 dB to 135.4 dB against the full-length decode), and that is exactly why it was dropped: on the
RK3588 the fed statistics scale the window activations past the engine's fp16 range and the graph
collapses — every window after the first renders as a constant-energy tone (rms flat at 51.5 dB,
spectral flatness 1.7e-17, energy at DC, 4800 Hz and 9600 Hz), for 1.8x the wall time and +1.5 GB
RSS. The producer, the runtime and the engine contract carry no code for it.

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

The spans follow output order but need not be contiguous or cover the full buffer.
Space tokens, standalone punctuation, and padding can leave gaps between or around groups.

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
