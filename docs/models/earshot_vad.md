# Earshot VAD

Native C++ port of [Earshot](https://github.com/pykeio/earshot) 1.2.2, a small
streaming voice activity detector by pyke.io (MIT OR Apache-2.0). It scores
16 ms frames of 16 kHz audio with a 40 KB network: a mel front end, three
depthwise-separable convolutions and two int16 minGRU layers.

The weights ship in the repository, so there is nothing to download. The port is
plain C++ and does not use a ggml graph or a GPU. It runs on the CPU with SSE2,
AVX2 or NEON, whatever backend the session asks for.

## Quick Start

```bash
audiocpp_cli --task vad --model assets/framework/models/earshot_vad \
  --audio speech.wav --segments-out segments.json
```

Streaming emits speech-start and speech-end events as frames arrive. It accepts
16 kHz chunks of any size:

```bash
audiocpp_cli --task vad --mode streaming --model assets/framework/models/earshot_vad \
  --audio speech_16k.wav --segments-out segments.json
```

| Field | Value |
|---|---|
| Family | `earshot_vad` (alias `earshot`) |
| Model directory | `assets/framework/models/earshot_vad` |
| Task | `vad` |
| Modes | `offline`, `streaming` |
| Input | Offline: any sample rate and channel count, converted to 16 kHz mono and mapped back to input samples. Streaming: 16 kHz mono, any chunk size. |
| Output | Speech segments (`--segments-out`), with the mean frame score as `confidence` |
| Threads | `--threads` parallelises the offline front end and convolutions; the recurrent layers run in order |
| Build | Part of the default `full` model set; with `AUDIOCPP_MODEL_SET=custom`, add `earshot_vad` to `AUDIOCPP_MODELS` |

On the server, configure a `"task": "vad"` model entry and call `POST /v1/tasks/run`. The response lists `segments`.

## Options

Pass options with `--request-option name=value`, or as session options with an
`earshot_vad.` prefix. Durations are rounded up to whole 16 ms frames. The
defaults reproduce upstream `SegmenterOptions::default()`.

| Option | Default | Meaning |
|---|---:|---|
| `threshold` | 0.6 | Smoothed score needed to start a segment (upstream `attack_threshold`). |
| `neg_threshold` | 0.5 | Smoothed score below which a segment starts to end (`decay_threshold`). |
| `smoothing_factor` | 0.4 | Exponential smoothing of raw frame scores, `[0, 1)`. |
| `start_delay_ms` | 32 | Consecutive voiced time needed to open a segment. |
| `min_silence_duration_ms` | 64 | Consecutive silence needed to close a segment (`end_delay`). |
| `speech_pad_ms` | 80 | Upstream `preroll_frames`. A segment starts `preroll - 2 * start_delay + 1` frames before its first voiced frame (2 frames, 32 ms, with the defaults). |
| `min_speech_duration_ms` | 96 | Shorter segments are dropped (`min_duration`). |
| `keep_trailing_segment` | true | Keep a segment that is still open at the end of the audio. |

Other Silero option names, such as `max_speech_duration_s`, are accepted and ignored.

## VAD chunking for ASR

`gigaam_asr` loads its chunking VAD through the model registry, so it can use
Earshot directly:

```bash
audiocpp_cli --task asr --family gigaam_asr --model <gigaam.gguf>   --audio long.wav --request-option audio_chunk_mode=vad   --session-option gigaam_asr.vad_model_path=assets/framework/models/earshot_vad
```

The other ASR families that support `audio_chunk_mode=vad` load Silero VAD directly
for now.

## Differences from upstream

Per-frame scores match the Rust crate to within 1e-4. The FFT is a different
implementation, so the last float bits differ. Segments match upstream
`segments()` except for two deliberate fixes:

- **Trailing speech is kept.** Upstream drops a segment that is still open when the
  audio ends. That would lose the last words of an ASR input, so audio.cpp closes
  it at the end of the audio. Set `keep_trailing_segment=false` for the upstream
  behaviour.
- **No timestamp drift.** Upstream's segment iterator skips its frame counter on
  every frame that returns a segment. Each later segment is therefore reported one
  frame (16 ms) earlier per earlier segment, so the error grows over a long file.
  audio.cpp counts every frame.

A segment start is also clamped so it never lies after the current frame. That only
matters when `speech_pad_ms` is close to 0, where upstream's formula would place the
start in the future (or underflow).

## Validation

`tests/earshot_vad/reference_rs` is a small Cargo program pinned to upstream. It
dumps per-frame scores and segments. `tests/earshot_vad/check_parity.py` compares
it with `earshot_vad_probe` and the CLI. It checks offline and streaming, and
allows for upstream's frame-counter lag:

```bash
cargo build --release --manifest-path tests/earshot_vad/reference_rs/Cargo.toml
python tests/earshot_vad/check_parity.py \
  --probe build/<preset>/bin/earshot_vad_probe \
  --reference tests/earshot_vad/reference_rs/target/release/earshot_reference \
  --cli build/<preset>/bin/audiocpp_cli --bench 5 <16 kHz mono wavs...>
```

`earshot_vad_test` (ctest) checks the committed upstream golden scores. It also
checks:
- streaming equals offline for any chunk size;
- silence and noise produce no segments;
- 48 kHz stereo input maps back to input samples;
- every streamed segment is announced by a SpeechStart event;
- the options, loader detection, and VAD chunk planning through the registry.

`earshot_vad_baseline_test` repeats it with the SSE2 kernel forced
(`AUDIOCPP_EARSHOT_BASELINE=1`).

### Speed

The table shows wall time to score 10 minutes of 16 kHz audio (37,583 frames)
with `earshot_vad_probe --bench 5`. It was measured on Windows 11 with MSVC 19.44
on a 32-thread x86-64 CPU with AVX2.

| Path | Threads | ms | x real time |
|---|---:|---:|---:|
| Upstream Rust crate, frame loop | 1 | 147 | 4,100 |
| audio.cpp offline | 1 | 111 | 5,400 |
| audio.cpp offline | 8 | 32 | 18,800 |
| audio.cpp streaming (frame by frame) | 1 | 119 | 5,000 |

The second table covers the 30 clips of the TEN VAD test set, repeated to 43.7 min.
It was run on Google Colab with a 12-vCPU Xeon @ 2.20 GHz, gcc 13 and a Release
build. Each figure is `session.wall_ms` from `audiocpp_cli --task vad`.

| Model | Threads | ms | x real time |
|---|---:|---:|---:|
| `earshot_vad` | 1 | 1,201 | 2,185 |
| `earshot_vad` | 4 | 484 | 5,421 |
| Upstream Rust crate, frame loop | 1 | 2,311 | 1,135 |
| `marblenet_vad` | 4 | 4,586 | 572 |
| `pulsevad` | 4 | 5,327 | 492 |
| `firered_vad` | 4 | 6,694 | 392 |
| `silero_vad` | 4 | 19,526 | 134 |

### Accuracy

This is frame accuracy on the [TEN VAD test set](https://github.com/TEN-framework/ten-vad/tree/main/testset)
(30 labelled clips, 262 s), at 10 ms resolution, with every model at its default
settings through `audiocpp_cli`.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| `silero_vad` | 0.900 | 0.974 | 0.935 |
| `earshot_vad` | 0.893 | 0.962 | 0.926 |
| `marblenet_vad` | 0.892 | 0.948 | 0.919 |
| `firered_vad` | 0.863 | 0.978 | 0.917 |
| `pulsevad` | 0.922 | 0.905 | 0.914 |

Earshot's raw frame scores reach a PR-AUC of 0.977 on the same set.

## Weights

`assets/framework/models/earshot_vad/earshot_vad.safetensors` contains upstream
`src/weights.bin` and the `src/filters.rs` Hann window and mel tables, unchanged.
Regenerate it with:

```bash
python tests/earshot_vad/convert_earshot.py \
  --output assets/framework/models/earshot_vad/earshot_vad.safetensors
```
