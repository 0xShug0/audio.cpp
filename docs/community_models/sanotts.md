# sanoTTS voice family

`sanotts` provides native GGML inference for the
[sanoTTS](https://github.com/Ampixa/sanoTTS) voice family — very small
text-to-speech models, the smallest of which also runs on microcontrollers.
All packages download from Hugging Face
([ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS) `gguf/`) as
standalone FP32 GGUFs with embedded model specs. Offline FP32 inference only.

Two graphs share one family:

- **nano** — duration student → contextual acoustic student → mel-100 →
  noise-fed ConvNeXt-1D decoder → [log-magnitude | phase] head → inverse
  STFT. 24 kHz. A seed picks one of many valid renderings.
- **piperlite** — duration student → contextual acoustic student → 192-channel
  latent → 3-stage ConvTranspose1d decoder with dilated residual banks →
  tanh waveform. 22.05 kHz. Fully deterministic (no seed).

| Package | Voice | Graph | Params | Language | Notes |
|---|---|---|---:|---|---|
| `sanotts_heart_orig` | heart | nano | 2,272,145 | en | best quality of the nano pair |
| `sanotts_heart_nano_orig` | heart-nano | nano | 294,279 | en | microcontroller-class |
| `sanotts_amy_orig` | amy | piperlite | 1,454,284 | en | Piper-distilled |
| `sanotts_hfc_orig` | hfc | piperlite | 1,834,380 | en | largest piperlite voice |
| `sanotts_kristin_orig` | kristin | piperlite | 1,396,151 | en | carries a learned post filter |
| `sanotts_vi_orig` | vi | piperlite | 1,565,484 | vi | Vietnamese |
| `sanotts_id_orig` | id | piperlite | 1,562,124 | id | Indonesian |

## Install

Install eSpeak-ng and its voice data first. On Debian or Ubuntu:

```bash
sudo apt install espeak-ng libespeak-ng1
```

On macOS:

```bash
brew install espeak-ng
```

Then install any package, e.g.:

```bash
python3 tools/model_manager_v2.py install sanotts_heart_orig --models-root models
python3 tools/model_manager_v2.py install sanotts_amy_orig --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family sanotts \
  --model models/sanoTTS-heart-GGUF --backend cpu \
  --text "Hello from sano T T S, a very small neural text to speech model." \
  --out sanotts.wav
```

Swap `--model` for any installed package directory
(`models/sanoTTS-amy-GGUF`, `models/sanoTTS-vi-GGUF`, ...). The Vietnamese
and Indonesian voices accept `--language vi` / `--language id`; a session
rejects text tagged with a language the voice was not trained on.

eSpeak-ng is loaded dynamically at runtime, never linked. If it is not on the
default library path:

```bash
audiocpp_cli --task tts --family sanotts \
  --model models/sanoTTS-heart-GGUF --backend cpu \
  --session-option sanotts.espeak_library_path=/path/to/libespeak-ng.so \
  --session-option sanotts.espeak_data_path=/path/to/espeak-ng-data \
  --text "A configured eSpeak installation." --out sanotts.wav
```

## Options

- `speaking_rate` (request, 0.5..2.0, default 1.0) — duration multiplier on
  the voice's tuned length scale; larger is slower.
- `seed` (request, default 0) — nano voices only: the decoder is noise-fed,
  so a given seed picks one of many valid renderings. `0` derives the seed
  from each text chunk as `sha256(text)[:8]`, which is what the reference
  implementations do; an explicit seed advances by one per long-form chunk.
  Piperlite voices are deterministic and ignore the seed.
- `text_chunk_size` (request, default 280) — maximum codepoints per long-form
  chunk; chunks split on sentence punctuation first, and a chunk that
  phonemizes past the voice's token limit is bisected at whitespace.

## Determinism and parity

The runtimes reproduce the reference implementations' exact semantics:

- Front ends: the phonemizer punctuation-preservation pipeline through the
  same eSpeak-ng library. The nano voices add the misaki E2M rewrite with
  tie characters; the piperlite voices use Piper's NFD-decompose-to-
  codepoints convention, per-voice `phoneme_id_map`, `[BOS, PAD, (id, PAD)…,
  EOS]` framing, and the schwa fallback for ids outside a component's
  trained vocabulary.
- nano: ATen-compatible MT19937 noise (24-bit uniform, Box–Muller in blocks
  of 16), torch.istft window normalisation and centre trim, and the
  reference's DC blocker `H(z) = (1 - z^-1)/(1 - 0.9973 z^-1)`.
- Shared: torch.linspace / expand_features float behaviour, LayerNorm eps
  1e-6 (nano), ties-to-even duration rounding.

Measured against the project's numpy references (same text, same
eSpeak-ng build), every voice: **correlation ≥ 0.99999996 with identical
sample counts**; max sample delta ~1.7e-05 is the WAV's own int16
quantisation. The numpy references are themselves gated ≥ 0.987 against the
float PyTorch models.

## Performance

CPU-only, 12-thread x86 (default 4 backend threads), FP32, the shared 6 kB
long-form text:

| Voice | Audio | Wall | vs real time | Peak RSS |
|---|---:|---:|---:|---:|
| heart-nano | 373 s | 1.3 s | ~283× | 220 MB |
| amy | 394 s | 18.5 s | ~21× | 497 MB |

The nano decoder runs at frame rate with a host iSTFT; the piperlite decoder
runs convolutions at audio rate, which is why it is heavier. Graphs are
cached per token count (duration and token stages) and per frame count
(decoder); `--log` prints cache hits and per-stage timings.

## Licensing

The sanoTTS runtimes and weights are MIT-licensed. eSpeak-ng is GPL-3.0 and
is therefore opened with `dlopen` at runtime and never linked, matching how
`inflect_v2` treats it.
