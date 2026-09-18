# sanoTTS voice family

`sanotts` provides native GGML inference for the
[sanoTTS](https://github.com/Ampixa/sanoTTS) voice family — very small
text-to-speech models across fourteen languages, the smallest of which also
runs on microcontrollers.
All packages download from Hugging Face
([ampixa/sanoTTS](https://huggingface.co/ampixa/sanoTTS) `gguf/`) as
standalone FP32 GGUFs with embedded model specs. Offline FP32 inference only.

Two graphs share one family:

- **nano** — duration student → contextual acoustic student → mel-100 →
  noise-fed ConvNeXt-1D decoder → [log-magnitude | phase] head → inverse
  STFT. 24 kHz. A seed picks one of many valid renderings.
- **piperlite** — duration student → contextual acoustic student → 192-channel
  latent → optional calibration adapter → 3-stage ConvTranspose1d decoder with
  dilated residual banks → tanh waveform. 22.05 kHz. Fully deterministic
  (no seed).

  The adapter is a small layer trained onto an already-finished acoustic
  student to re-calibrate the latent it hands the decoder: a bias-free
  depthwise convolution, a low-rank residual `x + up(tanh(down(x)))`, then a
  per-channel scale and bias. Voices that carry one declare it in
  `acoustic.adapter`; the rest omit the key and the latent goes straight to
  the decoder. Of the packages below only `hi` has one.

| Package | Voice | Graph | Params | Language | Notes |
|---|---|---|---:|---|---|
| `sanotts_heart_orig` | heart | nano | 2,272,145 | en | best quality of the nano pair |
| `sanotts_heart_nano_orig` | heart-nano | nano | 294,279 | en | microcontroller-class |
| `sanotts_amy_orig` | amy | piperlite | 1,454,284 | en | Piper-distilled |
| `sanotts_hfc_orig` | hfc | piperlite | 1,834,380 | en | largest piperlite voice |
| `sanotts_kristin_orig` | kristin | piperlite | 1,396,151 | en | carries a learned post filter |
| `sanotts_vi_orig` | vi | piperlite | 1,565,484 | vi | Vietnamese |
| `sanotts_id_orig` | id | piperlite | 1,562,124 | id | Indonesian |
| `sanotts_cs_orig` | cs | piperlite | 1,568,044 | cs | Czech, from `cs_CZ-jirka-medium` |
| `sanotts_de_orig` | de | piperlite | 1,565,324 | de | German, from `de_DE-thorsten-medium` |
| `sanotts_es_orig` | es | piperlite | 1,562,924 | es | Spanish, from `es_ES-davefx-medium` |
| `sanotts_fr_orig` | fr | piperlite | 1,565,324 | fr | French, from `fr_FR-siwis-medium` |
| `sanotts_it_orig` | it | piperlite | 1,565,484 | it | Italian, from `it_IT-serena-medium` |
| `sanotts_pt_orig` | pt | piperlite | 1,565,324 | pt | Brazilian Portuguese, from `pt_BR-cadu-medium` |
| `sanotts_ro_orig` | ro | piperlite | 1,565,164 | ro | Romanian, from `ro_RO-mihai-medium` |
| `sanotts_ru_orig` | ru | piperlite | 1,566,764 | ru | Russian, from `ru_RU-irina-medium` |
| `sanotts_tr_orig` | tr | piperlite | 1,562,284 | tr | Turkish, from `tr_TR-dfki-medium` |
| `sanotts_ne_orig` | ne | piperlite | 1,474,771 | ne | Nepali, from `ne_NP-chitwan-medium` |
| `sanotts_hi_orig` | hi | piperlite | 1,499,407 | hi | Hindi, from `hi_IN-pratham-medium`; carries a calibration adapter |

## Install

SanoTTS uses the [shared eSpeak-ng phonemizer](../espeak_phonemizer.md),
including shared synchronization with other model frontends.

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
(`models/sanoTTS-amy-GGUF`, `models/sanoTTS-de-GGUF`, ...). Each non-English
voice accepts its own `--language` code (`vi`, `id`, `cs`, `de`, `es`, `fr`,
`it`, `pt`, `ro`, `ru`, `tr`, `ne`, `hi`); a session rejects text tagged with a
language the
voice was not trained on. Every voice drives the eSpeak-ng voice its Piper
teacher was trained against — `pt` uses `pt-br`, the rest use the bare
language code — so eSpeak-ng must have that language's data installed.

By default eSpeak-ng is loaded dynamically. Static builds with
`AUDIOCPP_STATIC_ESPEAK=ON` instead include its code and use executable-local data.
For a dynamic build, if eSpeak-ng is not on the
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
eSpeak-ng build), the seven original voices: **correlation ≥ 0.99999996 with
identical sample counts**; max sample delta ~1.7e-05 is the WAV's own int16
quantisation. The numpy references are themselves gated ≥ 0.987 against the
float PyTorch models.

The nine language voices were measured the same way, on out-of-domain Tatoeba
text, 3–4 sentences each, comparing the reference's float audio against
audio.cpp's int16 WAV:

| Voice | Sample counts | Min correlation | Samples that quantise identically | Max int16 gap |
|---|---|---:|---:|---:|
| cs | identical | 0.9999999961 | 99.847% | 1 LSB |
| de | identical | 0.9999999865 | 99.929% | 1 LSB |
| es | identical | 0.9999999970 | 99.645% | 1 LSB |
| fr | identical | 0.9999999851 | 99.897% | 1 LSB |
| it | identical | 0.9999999962 | 99.832% | 1 LSB |
| pt | identical | 0.9999999762 | 99.897% | 1 LSB |
| ro | identical | 0.9999999859 | 99.893% | 1 LSB |
| ru | identical | 0.9999999678 | 99.944% | 1 LSB |
| tr | identical | 0.9999999435 | 99.958% | 1 LSB |
| amy (control, re-measured) | identical | 0.9999999920 | 99.818% | 1 LSB |

Turkish is the one voice that does not reach 0.99999996, and the reason is the
metric rather than the runtime: correlation against a 16-bit WAV is bounded by
the quantisation floor, which scales with the voice's amplitude. Turkish is the
quietest voice in the set (RMS 0.027, peak 0.271), and its measured correlation
sits at that voice's ceiling — as every other voice's does, including the amy
control. The scale-free statement holds everywhere: audio.cpp's int16 sample is
the correctly rounded quantisation of the reference float sample for 99.6–99.96%
of samples, never differs by more than one LSB, and the implied divergence
between the two runtimes is at most 5.1e-06 (Turkish: 4.5e-07, the tightest in
the set).

Both stacks ran the same eSpeak-ng 1.52.0 build. The front end is the part that
is sensitive to the eSpeak-ng version: rendering the same probe sentences
against Debian's eSpeak-ng 1.51 gives byte-identical audio for eight of the nine
voices, but two of the four Russian sentences phonemize differently there and
come out 3.5-8% shorter. Any voice is only as reproducible as the eSpeak-ng
build under it.

Nepali and Hindi were measured on the same harness, four out-of-domain Tatoeba
sentences each:

| Voice | Sample counts | Min correlation | Samples that quantise identically | Max int16 gap |
|---|---|---:|---:|---:|
| ne | identical | 0.9999999865 | 99.916% | 1 LSB |
| hi | identical | 0.9999999943 | 99.859% | 1 LSB |

Both clear the 0.99999996 of PR #449, and the implied divergence between the
two runtimes is at most 5.3e-07 (ne) and 1.9e-06 (hi). Hindi is the first
package to exercise the acoustic calibration adapter, so its number is also the
evidence that the new adapter path matches the reference: an adapter
implemented wrongly would not land within half an int16 LSB of a reference that
computes it. The ten voices that predate the adapter re-measure to correlations
identical to the ones above, digit for digit, which is what says the new code
path is inert when a voice does not declare an adapter.

## Performance

CPU-only, 12-thread x86 (default 4 backend threads), FP32, the shared 6 kB
long-form text:

| Voice | Audio | Wall | vs real time | Peak RSS |
|---|---:|---:|---:|---:|
| heart-nano | 373 s | 1.3 s | ~283× | 220 MB |
| amy | 394 s | 18.5 s | ~21× | 497 MB |

A 806-character German paragraph through `sanotts_de_orig` on the same host:
37.7 s of audio in 2.7 s wall, ~14× real time.

The nano decoder runs at frame rate with a host iSTFT; the piperlite decoder
runs convolutions at audio rate, which is why it is heavier. Graphs are
cached per token count (duration and token stages) and per frame count
(decoder); `--log` prints cache hits and per-stage timings.

## Intelligibility of the language voices

The only quality evidence for the language voices is automatic transcription.
16 out-of-domain Tatoeba sentences per language, rendered by the numpy
reference and transcribed by Whisper small, scored against the source text:

| Voice | CER | WER |
|---|---:|---:|
| pt | 0.014 | 0.038 |
| it | 0.026 | 0.080 |
| ru | 0.031 | 0.120 |
| de | 0.032 | 0.099 |
| es | 0.050 | 0.147 |
| tr | 0.059 | 0.303 |
| fr | 0.088 | 0.221 |
| ro | 0.104 | 0.392 |
| cs | 0.111 | 0.351 |
| hi | 0.289 | 0.662 |
| ne | 0.576 | 1.116 |

This measures whether the words survive, nothing else. 16 sentences per language
is directional, not precise. No listening study has been run on any of these
voices and there is no SCOREQ or MOS estimate for any of them, so nothing here
says how natural they sound.

CER is also not comparable across languages, because Whisper is far better at
some than at others, and Nepali and Hindi are two it is weak at. The comparable
number is each student against its own Piper teacher through the same ASR,
since the teacher is the ceiling on what any distillation of it can reach:

| Voice | Student CER | Teacher CER | Retained |
|---|---:|---:|---:|
| ne | 0.576 | 0.515 | 0.89× |
| hi | 0.289 | 0.154 | 0.53× |

Read the gap rather than the absolute for these two. Whisper small puts the
Nepali *teacher* — a published Piper voice, not ours — at 0.515, so the Nepali
absolute is mostly a statement about the transcriber, and only the small gap to
its teacher is evidence about the voice. Hindi keeps about half its teacher's
margin, the same retention German shipped at (0.475). Both are materially
weaker than the nine European voices above, and no human has listened to
either.

## Not included

- **Arabic.** The weights exist and run, but Piper's Arabic front end diacritizes
  the text with a separate 1.16M-parameter model before eSpeak-ng sees it —
  Arabic script omits the short vowels eSpeak-ng needs. Without that step 55% of
  the phoneme ids are wrong, which produces different words rather than a
  degraded voice, and this front end has no diacritizer. Arabic ships once one
  exists.
- **Chinese.** The weights exist, run, and phonemize correctly — the question of
  the front end is settled, not open. Piper ships Chinese voices under two
  different front ends: `zh_CN-chaowen-medium` and `zh_CN-xiao_ya-medium`
  declare `phoneme_type: "pinyin"` over an 85-symbol pinyin inventory, which
  eSpeak-ng cannot drive, while this voice's teacher `zh_CN-huayan-medium`
  declares `phoneme_type: "espeak"` with `espeak.voice: "cmn"` over the same
  152-symbol IPA inventory every other eSpeak Piper voice uses. It is the
  eSpeak one, and eSpeak-ng's `cmn` reads it correctly: the teacher transcribes
  at 0.090 CER through exactly this path. What fails is the distilled student,
  which comes in at 0.468 CER on the same sentences — it keeps 0.19× of its
  teacher, against 0.475× for the weakest voice that has shipped. Nearly half
  the characters are wrong, so it is held back for being unintelligible rather
  than for being mis-phonemized. Chinese ships when a student worth shipping
  exists. (Both numbers fold traditional and simplified to simplified before
  scoring; Tatoeba mixes the two and Whisper answers in either, which charges a
  voice for orthography it never pronounced. Unfolded the same rows read 0.144
  teacher and 0.502 student — the gap is the same.)
- **The ~511k "tiny" variants** of eight of these languages. They are a separate
  size tier and need their own evidence.

## Licensing

The sanoTTS runtimes and weights are MIT-licensed. eSpeak-ng is GPL-3.0-or-later.
The default build loads an external library; `AUDIOCPP_STATIC_ESPEAK=ON`
statically links it. Distributors must comply with the applicable license terms
for their build, including corresponding-source requirements for combined static
builds. Dynamic loading does not itself waive license obligations. See the
[shared component documentation](../espeak_phonemizer.md).
