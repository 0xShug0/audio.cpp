# MOSS-VoiceGenerator

Voice design: the speaker comes from a written description instead of a reference
recording, and the model speaks the supplied text in that voice.

- Family: `moss_voicegen`
- Task: `vdes` (voice design), offline
- Languages: English, Chinese
- Weights: [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator), Apache-2.0
- Codec weights: [OpenMOSS-Team/MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer), Apache-2.0
- Architecture: `moss_tts_delay` — a Qwen3-1.7B backbone with 16 audio codebook
  embeddings and `1 + 16` output heads, decoded on a delay pattern
- Codec: MOSS-Audio-Tokenizer v1, 24 kHz mono, hop 1920

## Usage

The recommended package is the self-contained GGUF:

```text
MOSS-VoiceGenerator-GGUF/
  moss_voicegen_bf16_codec_f16_decode.gguf
```

```bash
audiocpp_cli --family moss_voicegen --model <model-dir> --task vdes \
  --instruct "A warm male radio voice in his fifties, calm, never shrill." \
  --text "Good evening, and welcome back to the late show." \
  --language English --out out.wav
```

`--instruct` describes the speaker: timbre, age, pace, emotion. `--language` takes the full
language name the model was trained on — "English", not "en".

For safetensors, the upstream voice generator and audio tokenizer are separate Hugging Face
repositories. Place the tokenizer snapshot under `audio_tokenizer/` inside the voice
generator root:

```text
MOSS-VoiceGenerator/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  merges.txt
  audio_tokenizer/
    config.json
    model.safetensors.index.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
```

The standalone `OpenMOSS-Team/MOSS-VoiceGenerator` snapshot does not contain
`audio_tokenizer/`, so it is not enough for safetensors inference by itself.

### Decoding defaults

From the model card, applied as the family's defaults rather than left to the caller:

| Option | Default |
|---|---:|
| `audio_temperature` | 1.5 |
| `audio_top_p` | 0.6 |
| `audio_top_k` | 50 |
| `audio_repetition_penalty` | 1.1 |

The family is sensitive to these. At a generic TTS preset it ends speech on the first frame.

## Behaviour Worth Knowing

**The instruction sets a class, the seed picks a member.** A different seed under the same
description is audibly a different person; upstream `generate()` takes no seed at all and
draws from the global RNG. The class itself holds: running the same 6026-character text
with only "male"/"his" changed to "female"/"her" moved median F0 from 117 Hz to 190 Hz over
47 independent designs, and by ear every chunk of both runs landed on the requested side.
Pitch alone does not decide that — the lowest chunk of the female run sits at 120 Hz and
still reads as female, so timbre moves with the instruction too.

**A seed reproduces a take, not a voice.** The same seed, instruction and text give a
bit-identical WAV. The same seed on *different* text does not: median F0 moved 2.2 and 7.6
semitones across three lines. Anything needing one voice across several utterances has to
keep the generated audio and clone from it.

**Duration is guarded, not steered.** Upstream has no text-derived length control — a
1000-step cap and a 16-step floor on the turn-end token — and does not need one: over the
same 168-character line the reference produced 68%, 87% and 100% of a natural reading, and
this port 62% to 99%. The session adds a guard rail on top: frame bounds derived from text
length (0.95 frames per character, floor at 0.45, ceiling at 1.6) gate the two decisions the
model would otherwise make freely, opening the flush window and ending the turn. They caught
one take that stopped at 39 frames and never bound otherwise — the floor sat at 71 where the
shortest natural take was 99. `min_frames` and `max_frames` override them per request.

**A run can start in text mode, about 2% of the time.** Nothing forces the first step to
open an audio segment; when it samples an ordinary text token instead, the request yields no
audio and the session returns an error naming the retry rather than silence. Measured: 2 in
100 requests, 0 across the 47 long-form chunks.

**bf16 or f32 for the backbone.** It carries attention-sink activations far beyond f16's
range and produces NaN from position zero when stored as f16.

**f16 for the codec.** Stored as bf16 the audio tokenizer drifts 1.1e-2 from the reference
decode, against 5e-7 from the f32 source, failing this port's 1e-3 gate; f16 brings it back
to 7.1e-4 at the same size. Hence `--type bf16 --keep-type "audio_tokenizer_weights*=f16"` —
the backbone needs the exponent range, the codec the mantissa.

**Ship decode only.** VoiceGenerator never encodes audio, so
`--exclude-prefix "audio_tokenizer_weights/encoder"` takes the package from 7.3 GB to 5.7 GB.

## Validation

Parity is measured against the checkpoint's own PyTorch implementation, which ships with the
weights. The scripts under `tools/community_models/` regenerate every reference used here.

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

Row-for-row greedy parity holds at f32 only: bf16 rounding is coarser than the gaps between
near-tied candidates, so the trajectories separate after 16 rows.

## Long-form

The shared long-form case (`tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json`,
6026 characters) runs through the framework text chunker without special handling:

| | |
|---|---|
| Chunks | 47, none silent |
| Audio | 359 s from 77 s of compute — **RTF 0.22** |
| Speech ratio | 78.7% |
| Peak | 1.000, one sample in 8.6 million — not clipping |

The chunker, the session and the per-chunk failure rate hold over 47 consecutive
generations. Two limitations remain, both inherent to voice design:

**The speaker does not survive chunking.** 47 chunks are 47 independent designs: median F0
spans 13 semitones, and neighbouring chunks jump 2.3 semitones on median, 9.9 at worst. For
one voice over a long text, generate once here and clone from the result with a
reference-driven family.

**Chunks are not levelled against each other.** Per-segment peaks span 0.20 to 1.00, roughly
14 dB, since each chunk sets its own level. An assembler should normalise per chunk.

## Performance

Measured through `audiocpp_server` over one long-lived session, 100 requests of varying
length, on a Radeon AI PRO R9700 (ROCm/HIP, bf16 backbone, f16 codec):

| | |
|---|---|
| Real-time factor | **0.20 to 0.35, median 0.23** end to end over HTTP (523 s of audio in 124 s) |
| First request | RTF 1.53 — it carries the one-time runtime build |
| VRAM | 7783 MiB after the first request, 7811 after the hundredth: **+28 MiB total** |
| Completed | 98 of 100; both failures were the text-mode start described above |

The server calls `prepare()` on every request against the same session, so it is idempotent
and the runtimes are built once. Rebuilding them per request cost about four seconds of
re-upload on top of roughly one second of work.

Other backends, same model and prompt:

| Backend | Device | Works | Note |
|---|---|---|---|
| HIP | R9700 (gfx1201) | yes | the numbers above |
| Vulkan | R9700 | yes | slower than HIP |
| Vulkan | Radeon 780M (gfx1103) | yes | iGPU, RTF above 1; useful as a correctness target |
| HIP | Radeon 780M (gfx1103) | **no** | segfaults inside ggml's `get_rows` |
| CPU | Ryzen 7 255 | yes | RTF around 9 |

The gfx1103 HIP crash is not specific to this model: the same workload runs on that card
under Vulkan, and a hand-written HIP gather kernel runs on it too, so it is ggml's HIP
`get_rows` path on this ROCm build (7.1, distribution packages). Reported separately.

## Framework Changes This Needed

MOSS-Audio-Tokenizer v1 is a different generation from the v2 and Nano codecs already in the
tree, so `src/models/moss/shared/` needed five additions. None of them change v2 or Nano
behaviour:

| Change | Why |
|---|---|
| `samples_per_frame` taken from the config rather than a constant | v1's hop is 1920, not 3840 |
| `channels` config field; the stereo de-interleave is skipped when it is 1 | v1 is mono |
| The stage output projection is optional | upstream only creates one when a stage changes width; v1 omits it on three of four decoder stages |
| Attention projections also accept `in_projs.0` / `out_projs.0` | v1 keeps them in an indexed `ModuleList` |
| Feed-forward also accepts `linear1` / `linear2` | v1 names them directly where v2 uses an `nn.Sequential` |

A local copy of the codec would have meant duplicating roughly 1500 lines to change five
names and two numbers.
