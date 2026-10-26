# Earshot VAD

Native C++ port of [Earshot](https://github.com/pykeio/earshot) 1.2.2, a small
streaming voice activity detector by pyke.io (MIT OR Apache-2.0). It scores
16 ms frames of 16 kHz audio with a 40 KB network: a mel front end, three
depthwise-separable convolutions and two int16 minGRU layers.

The weights ship in the repository, so there is nothing to download. The port is
plain C++ and does not use a ggml graph or a GPU. It runs on the CPU whatever
backend the session asks for, with the widest kernel set the CPU supports:
AVX-512 (F, BW and VNNI), AVX2, or the SSE2 / NEON baseline. Every kernel set,
compiler, thread count and the streaming path give the same scores, bit for bit.

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
| Threads | `--threads` parallelises the offline front end, convolutions and the minGRU matrix products; only the elementwise recurrence runs in order |
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

Per-frame scores match the Rust crate to within 1e-4. The last float bits differ
because the FFT is a different implementation, `ln` and `exp` are Cephes
polynomials (relative error below 2e-7) instead of the C library's, and the mel
sums use eight interleaved partial sums. Those choices are what make the scores
identical on every platform. The integer minGRU layers are exact. Segments match
upstream `segments()` except for two deliberate fixes:

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
- the frame-by-frame detector and 4-thread scoring equal 1-thread scoring bit for bit;
- streaming equals offline for any chunk size;
- silence and noise produce no segments;
- 48 kHz stereo input maps back to input samples;
- every streamed segment is announced by a SpeechStart event;
- the options, loader detection, and VAD chunk planning through the registry.

`earshot_vad_baseline_test` and `earshot_vad_avx2_test` repeat it with the kernel
set capped by `AUDIOCPP_EARSHOT_ISA=baseline` or `avx2` (`avx512` is the default
cap; `AUDIOCPP_EARSHOT_BASELINE=1` still means baseline).

### Speed

The table shows wall time to score 10 minutes of 16 kHz audio (37,583 frames)
with `earshot_vad_probe --bench`, best of five runs, on a Ryzen 9 9950X3D. The
upstream Rust crate takes 147 ms for the same file on Windows and 133 ms on
Linux (frame loop, one thread; 127 ms with `-C target-cpu=native`).

| Build | Kernels | Offline, 1 thread | Offline, 8 threads | Streaming |
|---|---|---:|---:|---:|
| Windows 11, MSVC 19.44 | AVX-512 | 25 ms | 5 ms | 29 ms |
| Windows 11, MSVC 19.44 | AVX2 | 38 ms | 6 ms | 39 ms |
| Ubuntu 24.04 (WSL), gcc 13 | AVX-512 | 20 ms | 3.1 ms | 25 ms |
| Ubuntu 24.04 (WSL), gcc 13 | AVX2 | 28 ms | 4.5 ms | 31 ms |
| Ubuntu 24.04 (WSL), clang 18 | AVX-512 | 17 ms | 2.6 ms | 25 ms |

The second table covers the 30 clips of the TEN VAD test set, repeated to 43.7 min.
It was run on Google Colab with a 12-vCPU Xeon @ 2.20 GHz (AVX-512 with VNNI),
gcc 13 and a Release build. Each figure is `session.wall_ms` from
`audiocpp_cli --task vad`, best of three.

| Model | Threads | ms | x real time |
|---|---:|---:|---:|
| `earshot_vad` | 1 | 345 | 7,603 |
| `earshot_vad` | 4 | 99 | 26,557 |
| Upstream Rust crate, frame loop | 1 | 2,200 | 1,192 |
| `marblenet_vad` | 4 | 4,518 | 581 |
| `pulsevad` | 4 | 5,405 | 485 |
| `firered_vad` | 4 | 6,862 | 382 |
| `silero_vad` | 4 | 19,942 | 132 |

On the same machine, `earshot_vad_probe --bench` gives these times for the
Earshot model alone on that 43.7 min file:

| Kernels | Offline, 1 thread | Offline, 4 threads | Offline, 12 threads | Streaming |
|---|---:|---:|---:|---:|
| AVX-512 | 351 ms | 97 ms | 60 ms | 443 ms |
| AVX2 | 486 ms | 134 ms | 82 ms | 511 ms |
| Baseline (SSE2) | 892 ms | 335 ms | 298 ms | 980 ms |

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
