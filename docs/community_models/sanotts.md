# sanoTTS heart-nano

`sanotts` provides native GGML inference for
[sanoTTS](https://github.com/Ampixa/sanoTTS) **heart-nano**, a
294,279-parameter English text-to-speech model that also runs on
microcontrollers. The graph is a duration student, a contextual acoustic
student producing a mel-100 spectrogram, and a noise-fed ConvNeXt-1D decoder
whose [log-magnitude | phase] head feeds an inverse STFT. Output is 24 kHz
mono. Offline FP32 inference only.

The GGUF is published on Hugging Face at
[ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS) under `gguf/`: the
int8 on-device rows are dequantised to FP32 tensors in PyTorch shapes, with
the audio.cpp exact-shape metadata and an embedded model spec, so the package
is standalone.

## Install

Install eSpeak-ng and its English voice data first. On Debian or Ubuntu:

```bash
sudo apt install espeak-ng libespeak-ng1
```

On macOS:

```bash
brew install espeak-ng
```

Then install the GGUF package:

```bash
python tools/model_manager_v2.py install sanotts_heart_nano_orig --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family sanotts \
  --model models/sanoTTS-heart-nano-GGUF --backend cpu \
  --text "Hello from sano T T S, a very small neural text to speech model." \
  --out sanotts.wav
```

eSpeak-ng is loaded dynamically at runtime, never linked. If it is not on the
default library path:

```bash
audiocpp_cli --task tts --family sanotts \
  --model models/sanoTTS-heart-nano-GGUF --backend cpu \
  --session-option sanotts.espeak_library_path=/path/to/libespeak-ng.so \
  --session-option sanotts.espeak_data_path=/path/to/espeak-ng-data \
  --text "A configured eSpeak installation." --out sanotts.wav
```

## Options

- `speaking_rate` (request, 0.5..2.0, default 1.0) — duration multiplier
  applied before rounding; larger is slower.
- `seed` (request, default 0) — decoder noise seed. The decoder is noise-fed,
  so a given seed picks one of many valid renderings. `0` derives the seed
  from each text chunk as `sha256(text)[:8]`, which is what the reference
  implementations do; an explicit seed advances by one per long-form chunk.
- `text_chunk_size` (request, default 280) — maximum codepoints per long-form
  chunk; chunks are split on sentence punctuation first.

## Determinism and parity

The runtime reproduces the reference implementations' exact semantics:

- ATen-compatible MT19937 noise (24-bit uniform, Box–Muller in blocks of 16),
  so a seed renders the same waveform as the PyTorch and MCU runtimes up to a
  few ulp of libm difference.
- The phonemizer punctuation-preservation pipeline and misaki E2M rewrite,
  byte-identical token streams against the Python front end across a
  punctuation corpus.
- torch.istft window normalisation and centre trim, and the reference's
  DC-blocking filter `H(z) = (1 - z^-1)/(1 - 0.9973 z^-1)`.

Measured against the project's numpy reference (same text, same seed, same
eSpeak-ng build): correlation **0.999999985**, max sample delta 1.7e-05
(the WAV's own int16 quantisation), identical sample count. The numpy
reference is itself gated at 0.987–1.000 against the float PyTorch model.

## Performance

CPU-only, 12-thread x86 (default 4 backend threads), FP32:

- 38.7 s of audio synthesized in 0.22 s wall including model load
  (about 175x faster than real time); peak RSS 76 MB.
- Per stage on a 5.7 s utterance (`--log`): duration 0.4 ms, acoustic
  0.5 ms, decoder 15.1 ms, host iSTFT 5.3 ms.

Graphs are cached per token count (duration and token stages) and per frame
count (decoder), so repeated lengths skip graph construction; `--log` prints
the cache hits and stage timings.

## Licensing

The sanoTTS runtime and weights are MIT-licensed. eSpeak-ng is GPL-3.0 and is
therefore opened with `dlopen` at runtime and never linked, matching how
`inflect_v2` treats it.
