# tiny-voca

`tiny_voca` runs [tiny-voca](https://huggingface.co/n4ze3m/tiny-voca), an
English text-to-speech model with 4.5 million parameters and two voices. It
makes 24 kHz mono audio and is meant for the CPU. Nothing else has to be
installed: the model's own phonemizer,
[dizione](https://github.com/n4ze3m/dizione), is part of the port, and its data
is inside the GGUF file.

## Install

```bash
python tools/model_manager_v2.py install tiny_voca_orig --models-root models
```

The package is one 24 MB file, `models/tiny-voca-GGUF/tiny-voca-orig.gguf`.

## Run

```bash
audiocpp_cli --task tts --family tiny_voca \
  --model models/tiny-voca-GGUF/tiny-voca-orig.gguf \
  --backend cpu --text "Hello! This is tiny voca." \
  --voice-id M2 --out hello.wav
```

The voices are `M2` (male, the default) and `F2` (female).

### Request options (use with `--request-option`)

`--voice-id` takes precedence over `--request-option voice_id=...`.

| Option | Range | Default | Description |
|---|---:|---:|---|
| `voice_id` | `M2`, `F2` | `M2` | Built-in voice, also set with `--voice-id`. |
| `speed` | 0.25 to 4 | `1.0` | Speech speed; larger is faster. The common `--speaking-rate` option sets it too. |
| `semitone_shift` | whole number, -24 to 24 | `0` | Pitch shift in semitones. |
| `pitch_shift` | -1.4 to 1.4 | `0.0` | Fine pitch shift, added to `semitone_shift`. The unit is the natural log of the pitch: 0.0578 is one semitone, 0.693 one octave. |
| `pitch_range` | 0 to 3 | `1.0` | Scales the movement of the pitch around its mean: below 1 is flatter, above 1 livelier. |
| `seed` | whole number, 0 or more | `0` | Seed of the noise in the voice. The same seed gives the same audio. |
| `language` | `en`, `en-us`, `English` | `en` | The model reads English only. Any other language is refused: the command-line tool fails with an error and the server answers HTTP 400. The same holds for `--language` and for the `language` field of a server request. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `text_chunk_size` | positive integer | `400` | Maximum Unicode codepoints per chunk. |

There are no session options.

## How text is read

The framework splits long text into chunks. Each chunk is then spoken one
sentence at a time, because the model was trained on single sentences. A
sentence that is too long for one pass (500 phoneme tokens, or 3,000 frames,
which is 32 seconds) is cut after commas, then between words. Text that ends
without a mark gets a full stop, so the voice falls at the end. A part with
nothing to pronounce (another script, signs only) is skipped; when nothing at
all can be read, the request fails and says so.

The phonemizer writes out numbers, dates, money, units and the like by rule,
looks words up in a dictionary built from CMUdict, and uses a small model for
words the dictionary does not have. A pronunciation can be given in the text
in CMUdict notation: `{HH AH0 L OW1}`.

## How it runs

- The three transformer stacks of the model (text encoder, prosody, acoustic)
  run as ggml graphs built from the framework's modules.
- The small steps between them run on the host: the duration predictor,
  repeating each phoneme for its frames, and the pitch rules.
- There is no neural vocoder. The waveform head shapes a pulse train (made
  from the pitch) and noise with a filter that changes every frame. This is
  signal processing on the host and uses the framework's real FFT.
- The phonemizer is plain C++ with no dependencies, in
  `src/community_models/tiny_voca/dizione`. That folder does not use the
  framework and can be built by itself.

Memory is kept low on purpose. Attention scores are worked out for blocks of
rows, so they never take more than 4 MB however long the sentence is. The
waveform head works through the audio in blocks.

A graph is made for one stack and one length. The session keeps the graphs it
used last in a bounded cache (the framework's `CacheSlots`) and runs a kept
graph again when the same stack meets the same length. It keeps 6 graphs at
most (three for a sentence) and 4 MB of buffers at most, least recently used
first out. A graph larger than 4 MB, which is a sentence of more than about
four seconds, is freed after its sentence, because a kept graph of the longest
sentence would hold 25 MB. So the cache serves short sentences that come
back; a long sentence builds its graphs each time, which takes under a
millisecond. With `--log`, the lines
`tiny_voca.<stack>_graph.cache_hit` show each reuse and
`tiny_voca.graph_cache.kept_mb` what is kept.

## Convert the released model

Download `tiny-voca.pt`, `config.json` and the `phonemizer` folder from the
[model repository](https://huggingface.co/n4ze3m/tiny-voca), then run:

```bash
python tools/community_models/convert_tiny_voca.py \
  --source /path/to/tiny-voca \
  --output models/tiny-voca-GGUF/tiny-voca-orig.gguf --check
```

This needs `torch`, `numpy`, `safetensors` and a built `audiocpp_gguf`
(`--check` also needs the `gguf` package). Nothing is changed on the way: the
weights stay float32, bit for bit, and `--check` reads the file back and
compares every tensor.

## Checked against PyTorch

The phonemizer gives the same phoneme ids as the Python package on every one
of the model's 8,749 training sentences and on 300,000 generated texts. On
two million other sentences, 4 differ: each has a word the dictionary lacks
for which the small model is at a near-tie between two phonemes.

The audio was compared with the PyTorch model for the same phoneme ids and
the same noise: 200 texts, both voices, some at other speeds and pitches (700
utterances, 42 minutes of audio).

| Measure | Result |
|---|---:|
| Durations equal to PyTorch | 700 of 700 |
| Largest waveform difference, as a fraction of the peak: median | 0.12% |
| Largest waveform difference, as a fraction of the peak: worst utterance | 1.5% |
| Difference below the signal (RMS): median | 64 dB |
| Pitch of a frame, relative difference: median of the largest in each utterance | 8e-7 |
| Waveform difference when PyTorch is given the same pitch values: worst of 55 | 0.003% |

The whole waveform difference comes from the last digits of the predicted
pitch: the pulse train adds the pitch up along the sentence, so a difference
of one part in a million moves the pulses a little by the end.

`tests/community_models/tiny_voca/reference_audio.py` and the
`tiny_voca_synth_probe` target repeat the audio comparison;
`reference.py` and `tiny_voca_dizione_probe` repeat the phonemizer one.
`language_test.py` checks with the real command-line tool and server that
English is taken and other languages are refused.

## Speed and memory

AMD EPYC 4585PX, Linux, CPU backend, FP32, one session, release build.

| Threads | Real-time factor | Peak memory |
|---:|---:|---:|
| 1 | 0.043 | 54 MB |
| 2 | 0.026 | 54 MB |
| 4 | 0.019 | 54 MB |

120 utterances, 518 seconds of audio, best of five runs on a shared machine
(8 virtual cores). Memory is the resident set of the
process: 32 MB once the model is loaded, of which the weights are 17 MB, and
it does not grow with the number of requests (2,100 utterances in one session
peak at 54 MB, of which the kept graphs are up to 4 MB). A single sentence of
14 seconds peaks at 43 MB, and one at the model's limit of 32 seconds at
60 MB. The command-line tool peaks at 39 MB for a short sentence and at
62 to 66 MB for three 42-second requests in one session.

## Limits

- English only. A request that names another language is refused.
- Offline only: the audio of a request is returned when all of it is made.
- Tested on the CPU backend. FP32 weights only.
- A seed gives different noise here than in the Python package, because the
  framework's noise generator is used. The comparison above gives both the
  same noise.
- `semitone_shift` takes whole numbers, as it does in other models; use
  `pitch_shift` for less than a semitone.

## Licence

The model and its code are Apache-2.0. The dictionary inside the package is
built from CMUdict (BSD-style licence; its notice is kept in the package) and
from Google's Wikipedia Homograph Data (Apache-2.0).
