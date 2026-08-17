# MOSS-VoiceGenerator

Voice design: the speaker comes from a written description instead of a reference
recording, and the model speaks the supplied text in that voice.

- Family: `moss_voicegen`
- Task: `vdes` (voice design), offline
- Languages: English, Chinese
- Weights: [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator), Apache-2.0
- Architecture: `moss_tts_delay` — a Qwen3-1.7B backbone with 16 audio codebook
  embeddings and `1 + 16` output heads, decoded on a delay pattern
- Codec: MOSS-Audio-Tokenizer v1, 24 kHz mono, hop 1920

## Usage

```bash
audiocpp_cli --family moss_voicegen --model <model-dir> --task vdes \
  --instruct "A warm male radio voice in his fifties, calm, never shrill." \
  --text "Good evening, and welcome back to the late show." \
  --language English --output out.wav
```

`--instruct` describes the speaker: timbre, age, pace, emotion. `--language` must be the
full language name the model was trained on — "English", not "en".

### Decoding defaults

From the model card, and applied as the family's defaults rather than left to the caller:

| Option | Default |
|---|---:|
| `audio_temperature` | 1.5 |
| `audio_top_p` | 0.6 |
| `audio_top_k` | 50 |
| `audio_repetition_penalty` | 1.1 |

This family is genuinely sensitive to these. At a generic TTS preset it degenerates into
an immediate end of speech.

## Behaviour Worth Knowing

**Duration is guarded, not steered.** Upstream has no text-derived length control — only a
flat 1000-step cap and a 16-step floor on the turn-end token — and it does not need one:
measured over the same 168-character line, the reference produced 68%, 87% and 100% of a
natural reading, and this port produced 62% to 99%. The same spread, no systematic
truncation.

What the session adds is a guard rail, not a correction. It derives a minimum and maximum
frame count from the text length (0.95 frames per character, measured; floor at 0.45 of
that, ceiling at 1.6 plus slack) and gates the two decisions the model would otherwise make
freely: opening the flush window, and ending the turn. Across every take measured the
bounds never bind — the floor sits at 71 frames where the shortest natural take was 99 —
but they do catch the pathological outlier, a take that stopped at 39 frames, a quarter of
the text. Override them per request with `min_frames` and `max_frames`, or set them to 1
and a large number to get upstream's behaviour exactly.

**An instruction constrains the speaker; it does not fix one.** The same instruction with
a different seed is audibly a different person. `generate()` upstream takes no seed at all
and draws from the global RNG, so this is how the model works rather than a property of
this port.

**A seed reproduces a take, not a voice.** Generation here is deterministic: the same
seed, instruction and text give a bit-identical WAV. But the seed does not carry the
speaker to *other* text — holding seed and instruction fixed while changing the sentence
moved the median F0 by 2.2 and 7.6 semitones across three lines, which is a different
person, not a different mood. Anything that needs one voice across several utterances has
to keep the generated audio and clone from it, rather than re-generating from the same
seed.

**A run can start in text mode.** At the documented `text_temperature` of 1.5 the model
puts about 0.76 on the audio-start token at the first step, so roughly one run in four
samples an ordinary text token instead and produces no audio. This is upstream behaviour,
measured against the reference implementation, not a quirk of this port.

**bf16 or f32 only, for the backbone.** It carries attention-sink activations far beyond
f16's range and produces NaN from position zero when stored as f16.

**The codec wants the other 16-bit format.** Storing the audio tokenizer as bf16 costs a
lot of accuracy — worst probe deviation 1.1e-2 against the reference decode, against 5e-7
from the f32 source, which fails this port's 1e-3 gate. f16 brings it back to 7.1e-4 and
is the same size, since both formats are two bytes. So a package converts as
`--type bf16 --keep-type "audio_tokenizer_weights*=f16"`: the two halves of the file want
different 16-bit formats, for opposite reasons — the backbone needs the exponent range, the
codec needs the mantissa.

**Ship decode only.** VoiceGenerator never encodes audio, so the codec's encoder is dead
weight in a package: `--exclude-prefix "audio_tokenizer_weights/encoder"` takes the bf16
build from 7.3 GB to 5.7 GB.

## Validation

Parity is measured against the checkpoint's own PyTorch implementation, which ships with
the weights. The scripts under `tools/community_models/` regenerate every reference used
here.

| Check | Result |
|---|---|
| Prompt tokens vs `MossTTSDelayProcessor` | identical, 4 cases (trailing punctuation, none, Chinese, no instruction) |
| Backbone hidden state vs `MossTTSDelayModel`, bf16 | max relative 1.9e-5 (prefill), 1.6e-5 (cached step) |
| Backbone hidden state, f32 | max relative 3.5e-7 |
| Backbone hidden state, f16 | 2048/2048 non-finite — the f16 failure, reproducible |
| Greedy generation vs `generate()`, f32 | 40/40 rows identical, text tokens and all 16 codes |
| Greedy generation, bf16 | first 16 rows identical, then diverges on a 0.003 logit gap |
| Codec decode vs `MossAudioTokenizerModel`, f32 source | max abs 3e-5, correlation 1.000000; worst probe 5.0e-7 |
| Codec decode from the shipped package, codec f16 | worst probe 7.1e-4 |
| Codec decode with the codec stored bf16 | worst probe 1.1e-2 — rejected, hence the f16 override |

Greedy row-for-row parity only holds at f32: greedy decoding is a step function, and bf16
rounding is coarser than the gaps between near-tied candidates, so the trajectories part
company. That is a property of greedy decoding, not of the port.

## Framework Changes This Needed

MOSS-Audio-Tokenizer v1 is a different generation from the v2 and Nano codecs already in
the tree, so `src/models/moss/shared/` needed five additions. None of them change v2 or
Nano behaviour:

| Change | Why |
|---|---|
| `samples_per_frame` taken from the config rather than a constant | v1's hop is 1920, not 3840 |
| `channels` config field; the stereo de-interleave is skipped when it is 1 | v1 is mono |
| The stage output projection is optional | upstream only creates one when a stage changes width; v1 omits it on three of four decoder stages |
| Attention projections also accept `in_projs.0` / `out_projs.0` | v1 keeps them in an indexed `ModuleList` |
| Feed-forward also accepts `linear1` / `linear2` | v1 names them directly where v2 uses an `nn.Sequential` |

A local copy of the codec would have meant duplicating roughly 1500 lines to change five
names and two numbers.
