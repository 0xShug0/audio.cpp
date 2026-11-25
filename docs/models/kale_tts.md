# kale-tts-1

`kale_tts` runs [kale-tts-1-tiny](https://huggingface.co/KaleComputer/kale-tts-1-tiny) (45.7M parameters) and
[kale-tts-1-nano](https://huggingface.co/KaleComputer/kale-tts-1-nano) (7.5M), small English text-to-speech models with
ten designed voices (eight US English, two British). They make 24 kHz mono audio. Both are distilled from
kale-tts-1-base, a Kokoro-82M fine-tune, which runs on the existing `kokoro_tts` family (see below).

How they differ from Kokoro: nano has a small ALBERT and prosody predictor (192 wide); tiny reads the words with
Ettin-encoder-17m (a ModernBERT encoder) and keeps Kokoro's prosody predictor. Neither uses iSTFTNet: a few ConvNeXt
layers at the frame rate shape the spectrum of a 64-harmonic sine source built from the pitch, plus plain, pitch-pulsing
and short-burst noise. Each voice carries its own end-of-sentence pitch floor.

## Convert

Package downloads target `audio-cpp/audio.cpp-gguf/kale-tts-1-GGUF`.
Until the converted files are published there, use the local conversion commands below.

The weights are the release's float32 graph (`kale-tts-1-nano.fp32.onnx` / `kale-tts-1-tiny.fp32.onnx`); the converter stores them as float32 under their
PyTorch names, with the voices, the word lists and (tiny) the Ettin tokenizer:

```bash
pip install numpy onnx safetensors
python tools/gguf/kale_tts_convert.py --source /path/to/kale-tts-1-nano \
  --output models/kale-tts-1-GGUF/kale-tts-1-nano-f32.gguf
python tools/gguf/kale_tts_convert.py --source /path/to/kale-tts-1-tiny \
  --output models/kale-tts-1-GGUF/kale-tts-1-tiny-f32.gguf
```

`--source` is a copy of the Hugging Face repository. nano is a 39 MB file, tiny 194 MB.

## Run

```bash
audiocpp_cli --task tts --family kale_tts \
  --model models/kale-tts-1-GGUF/kale-tts-1-nano-f32.gguf \
  --backend cpu --text "The museum opens at nine o'clock and tickets cost twelve pounds." \
  --voice-id amy --out hello.wav
```

Voices: `kenzie`, `simone`, `amy` (British), `frankie`, `darlene`, `owen`, `toby`, `jack` (British), `earl`, `magnus`.

### Request options (use with `--request-option`)

| Option | Range | Default | Description |
|---|---:|---:|---|
| `voice_id` | a voice above | `kenzie` | Built-in voice, also set with `--voice-id`. |
| `speed` | 0.25 to 4 | `1.0` | Speech speed; 1.0 is each voice's own pace. |
| `seed` | whole number, 0 or more | `0` | Seed of the decoder noise; the same seed gives the same audio. |
| `phonemes` | list of strings | | Kokoro-symbol phonemes to speak instead of reading the text, one entry per sentence (native C++ / C API list option). For tiny, the text should hold the same sentences: its word reader reads them. |
| `text_chunk_mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `text_chunk_size` | positive integer | `400` | Maximum Unicode codepoints per chunk. |

### Session options (use with `--session-option`)

| Option | Description |
|---|---|
| `kale_tts.espeak_library_path` | Path to the eSpeak-ng shared library. |
| `kale_tts.espeak_data_path` | Path to espeak-ng-data. |
| `kale_tts.parity_noise_file` | Tests only: raw float32 samples used as the decoder noise, to compare with a reference run. |

## How text is read

Text is split into sentences and spoken one sentence at a time, with 0.12 s between sentences, as the models were
trained. Words come from the model's own checked word list (built from CMUdict, with hand-checked overrides and spoken
forms for technical terms such as `AWS` or `Node.js`); the British voices use the British list. Numbers and words that
are not in the list go through eSpeak-ng, which is loaded only when such a word comes up. eSpeak-ng is GPL-3.0 and is
loaded at run time, not linked: with the `phonemes` option, or a text whose words are all in the list, it is never
loaded and need not be installed. The IPA is written in Kokoro's
own symbols, as the models were trained. The release's full text front end (dates, money and units by rule, a g2p
model for unknown words) is JavaScript; to use it, pass its output with the `phonemes` option.

Each sentence accepts at most 510 phoneme symbols and 4096 prosody frames. Oversized sentences return an error; reduce `text_chunk_size`
or provide shorter entries in the native `phonemes` list. Input is never silently truncated.
The runtime retains one prepared graph per phase, reusing it while its shape is unchanged.

## Backends

The text side, durations and the pitch and loudness heads (small, sequential LSTMs) run on the CPU, where the duration
rounding matches the release graphs exactly; the decoder runs on the session's backend. CPU and Metal are tested.

## kale-tts-1-base

kale-tts-1-base is Kokoro-82M with its text side and prosody predictor fine-tuned on the same ten voices; its decoder is
stock Kokoro, so it runs on the `kokoro_tts` family. Convert it with the Kokoro script; `--checkpoint` names its
checkpoint, and a `kokoro_id` per voice in its `voices.json` (`af_kenzie`, `bf_amy`, `bm_jack`, ...) gives the
Kokoro-style voice ids, whose first letter selects the language:

```bash
python tools/gguf/kokoro_tts_convert.py --source /path/to/kale-tts-1-base --output-dir models/kale-tts-1-GGUF \
  --checkpoint kale-tts-1-base.pth --name kale-tts-1-base --type bf16
audiocpp_cli --task tts --family kokoro_tts --model models/kale-tts-1-GGUF/kale-tts-1-base-bf16.gguf \
  --backend cpu --text "Hello there!" --voice-id af_kenzie --out base.wav
```

## Validation

Against the release ONNX graphs (float32), on the release's 20 check lines x 2 voices (Kenzie, Jack), with the same
phonemes and the same decoder noise (`parity_noise_file`): see the pull request for the numbers and the scripts.

## License

Weights and word lists: Apache-2.0 (the word lists are built from CMUdict, BSD-style, its notice kept). Ettin-encoder-17m
and its tokenizer: MIT. The decoder head follows paradee (Apache-2.0). Kokoro-82M: Apache-2.0.
