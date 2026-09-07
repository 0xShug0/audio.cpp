# MiraTTS local validation record

Date: 2026-09-02

Model: `mira-tts-q8.gguf`

Reference: `00-reference-voice.wav`

Host: Windows, NVIDIA GeForce RTX 3090 24 GB

This record is intentionally local and is not a claim that untested devices or
backends have parity. Generated WAVs and machine-readable summaries are under
`build/logs/warmbench/mira_tts/` and are not source-controlled.

## CUDA offline — long-lived session

One model and one session handled all five requests.

| Request | Wall ms | Audio s | RTF | FNV-1a audio hash |
|---|---:|---:|---:|---|
| clone_cold | 1707.41 | 5.12 | 0.3335 | `426924e4695337c6` |
| clone_repeat | 1560.66 | 5.12 | 0.3048 | `426924e4695337c6` |
| short_second_prompt | 1566.03 | 5.12 | 0.3059 | `8255f4342afeb014` |
| longform | 4771.24 | 15.36 | 0.3106 | `5708fcccf7c25080` |
| clone_repeat_after_longform | 1583.74 | 5.12 | 0.3093 | `426924e4695337c6` |

The three identical requests remain bit-deterministic before and after the
different-prompt and long-form requests. Internal traces show the decode graph
reused after its first build for matching capacity; larger/different prompt
capacities can build a separate prefill graph.

## Upstream Python performance comparison

The revision-pinned upstream MiraTTS implementation was run on the same RTX
3090 with the same prompts, reference voice, sampling parameters, token budgets,
and request order. Model initialization is excluded from both request loops.

| Test | Original MiraTTS | audio.cpp Q8 | Difference |
|---|---:|---:|---|
| Cold request, 5.12 s audio | 2.227 s | 1.707 s | audio.cpp 1.30× faster* |
| Repeated request, 5.12 s | 0.911 s | 1.561 s | Original 1.71× faster |
| Second short prompt, 5.12 s | 0.895 s | 1.566 s | Original 1.75× faster |
| Long-form, 15.36 s | 2.976 s | 4.771 s | Original 1.60× faster |
| Repeat after long-form, 5.12 s | 0.918 s | 1.584 s | Original 1.73× faster |

\* The original cold request paid a one-time ONNX Runtime CUDA fallback cost;
the warm measurements are the representative steady-state comparison.

The three warm short requests average 908.02 ms upstream and 1570.14 ms in
native Q8, making upstream about 1.73x faster in this measurement. For the
15.36-second long-form case, upstream achieved RTF 0.1938 (5.16x real time)
and native Q8 achieved RTF 0.3106 (3.22x real time).

This is not a precision-matched comparison: upstream uses BF16 while the native
run uses GGUF Q8. Upstream also encodes and retains the reference context before
the timed request sequence, whereas the native request currently processes its
reference input on each run.

## CUDA streaming

| Request | Wall ms | First event ms | Events | Audio s | RTF | FNV-1a audio hash |
|---|---:|---:|---:|---:|---:|---|
| stream_cold | 6639.50 | 2489.93 | 4 | 21.14 | 0.3141 | `26eb3542a073a8db` |
| stream_repeat | 6498.52 | 2368.89 | 4 | 21.14 | 0.3074 | `26eb3542a073a8db` |

The stream emits multiple independently consumable audio events and the merged
repeat is bit-deterministic.

## Resident memory — Q8 CUDA

With the Q8 model held after the five-request sequence:

| Measurement | Resident usage |
|---|---:|
| Process working set | 1429.54 MiB |
| Process private bytes | 5958.28 MiB |
| Windows GPU Process Memory dedicated usage | 2551.55 MiB |
| `nvidia-smi` total device memory in use | 2552 MiB |

These are held-resident samples, not an instrumented peak across model loading.
The README therefore still requires peak sampling for a formal performance
submission.

## Python/native decoder parity

The previously saved exact-token comparison uses identical upstream speech and
context tokens:

| Metric | Python | Native | Result |
|---|---:|---:|---:|
| Sample rate | 48000 Hz | 48000 Hz | Match |
| Frame count | 122880 | 122880 | Exact match |
| Waveform cosine | — | — | 0.99996735 |
| Log-mel cosine | — | — | 0.99943239 |

This isolates the native processor/decoder path from autoregressive sampling.

## Backend coverage

| Backend | Build | Runtime result | Status |
|---|---|---|---|
| CUDA, RTX 3090 | Passed | Offline, streaming, determinism, and WAV validation passed | Validated |
| Vulkan, RTX 3090 | Passed | Deterministic, but under-generated (3.44 s vs 5.12 s) and diverged on other prompts | Not parity-clean |
| Vulkan, AMD integrated GPU | Passed | Deterministic, but generated only 0.10 s for the 5.12 s CUDA case | Not parity-clean |
| CPU | Not run | No runtime measurement recorded | Untested |

`mira_tts_warm_bench` builds successfully with the Vulkan configuration. It is
not parity-clean in this test:

- AMD integrated Vulkan device: deterministic repeats, but only 0.10 seconds
  for the 5.12-second CUDA case.
- RTX 3090 Vulkan device: deterministic repeats, but 3.44 seconds for the same
  case and severe under-generation on the other prompts.

Vulkan is therefore recorded as compile-tested but unsupported for MiraTTS
output parity until the backend divergence is diagnosed. CUDA is the validated
runtime for this model.

## Automated result

`tests/mira_tts/validate_bench.py` passed all offline/streaming lifecycle,
determinism, duration, event-count, and WAV-readability checks for the CUDA
summaries.
