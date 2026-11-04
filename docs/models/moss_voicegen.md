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
  --request-option "instruct=A warm male radio voice in his fifties, calm, never shrill." \
  --text "Good evening, and welcome back to the late show." \
  --language English --out out.wav
```

`instruct` describes the speaker: timbre, age, pace, emotion. Legacy `instruction`
remains accepted. `text_temperature` controls text-stream sampling (default 1.5),
with `moss_voicegen.text_temperature` retained as an alias. `--language` takes the full
language name the model was trained on — "English", not "en".

Use `--request-option min_frames=<int>` and `--request-option max_frames=<int>`
to override generated audio-frame bounds per text chunk. Both default to 0;
nonpositive values derive bounds from text length. The runtime raises the
ceiling to at least the effective minimum plus the number of codebooks.

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

Use the model-specific sampling defaults in the tables below. Generic TTS
settings can cause speech to end on the first frame.

## Model

| Field | Value |
|---|---|
| Family | `moss_voicegen` |
| Tasks | `design` |
| Modes | `offline` |
| Languages | `en`, `zh` |
| Default package | `moss_voicegen_bf16_codec_f16_decode` |
| Input | Text and a voice description |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `min_frames` | `int` | `0` | Minimum generated audio frames per text chunk; nonpositive values derive a text-length-based floor. |
| `max_frames` | `int` | `0` | Generated audio-frame ceiling per text chunk; nonpositive values derive a text-length-based ceiling. Runtime raises it to at least the effective min_frames plus the number of codebooks. |
| `instruct` | `string` | Model/session default | Voice description: timbre, age, pace, emotion. |
| `text_temperature` | `float` | `1.5` | Text-token sampling temperature. Must be finite and positive when sampling. |
| `language` | `string` | Model/session default | Optional full language name, such as English or Chinese; not an ISO language code. |
| `seed` | `int` | `0` | Generation seed; later text chunks add their chunk index. Parsed as signed 32-bit integer then converted to uint32; negative values wrap. |
| `temperature` | `float` | `1.5` | Audio-token sampling temperature. Must be finite and positive when sampling. |
| `top_p` | `float` | `0.6` | Audio-token nucleus sampling cutoff. |
| `top_k` | `int` | `50` | Audio-token top-k sampling limit. |
| `repetition_penalty` | `float` | `1.1` | Audio-token repetition penalty. Must be finite and positive when history is nonempty. |
| `text_chunk_size` | `int` | `200` | Maximum characters per generated text chunk; must be positive. Declared range: minimum 1. |
| `text_chunk_mode` | `word_budget`, `tag_aware`, `japanese`, `endline` | `word_budget` | Text chunking policy. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `moss_voicegen.weight_type` | `native`, `f32`, `bf16`, `q8_0` | `bf16` | Backbone weight storage. f16 is not supported. |

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
