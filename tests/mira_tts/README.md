# MiraTTS validation

Historical harness documentation: the removed runner and original commands
are preserved in the [pre-cleanup backup](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/mira_tts/README.md).
Reported measurements, hashes and limitations are unchanged. These recipes
are not instructions for the current tree.

The historical harness followed the long-lived validation pattern used for the OuteTTS
port in PR #63. The model is loaded once and exercised by a sequence containing
cold, repeat, changed-prompt, long-form, and post-long-form repeat requests.
Streaming has a separate long-lived sequence so that offline and streaming
sessions do not duplicate model weights in VRAM.

The benchmark reported one JSON object per request with wall time, generated
duration, real-time factor, sample/frame counts, a deterministic audio hash,
and output path. Streaming runs additionally reported first-event latency and the
number of progressively emitted audio events.

## Historical workflow

The archived guide above contains the original build, native offline/streaming,
trusted Python comparison, memory-hold and artifact validation commands.
The historical comparison gates were waveform cosine >= 0.95 and log-mel
cosine >= 0.95, with exact frame counts required for the strongest parity run.
No replacement timing or parity results are claimed after harness removal.

## Acceptance matrix

| Check | Required evidence |
|---|---|
| Cold/repeat determinism | `clone_cold`, `clone_repeat`, and `clone_repeat_after_longform` have identical hashes |
| Long-lived lifecycle | All offline requests finish in one process without reloading the model |
| Long form | `longform` produces non-empty audio within its token budget |
| Streaming | More than one event, finite first-event latency, merged non-empty WAV |
| Streaming determinism | `stream_cold` and `stream_repeat` have identical hashes and event counts |
| Python/native parity | Per-case WAV cosine, log-mel cosine, and frame counts are recorded |
| Audio validity | Every WAV is readable by FFmpeg and has the reported sample rate/channels |
| Resource use | Peak RSS, peak GPU memory, wall time, duration, and RTF are recorded |
| Backends | CUDA is required; CPU/Vulkan results or explicit limitations are documented |

Run deterministic checks before experimenting with unseeded sampling. If a
repeat hash changes, inspect generated speech-token boundaries, prompt/context
tokens, and component traces before comparing subjective audio quality.

## Historical exact-output comparisons

The pinned guide above preserves the optimization comparison commands and
original artifact conventions. They compared request sets, audio hashes,
WAV bytes, sample metadata and streaming event counts, and reported timing
changes without treating a single noisy measurement as a performance failure.
