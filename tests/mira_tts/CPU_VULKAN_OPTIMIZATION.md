# MiraTTS CPU/Vulkan optimization validation

Baseline: `f3bd8255` (upstream main merged into the MiraTTS branch).
Tested on Windows, September 8, 2026: Ryzen 9 7950X3D, RTX 3090
(Vulkan device 1), Release/MSVC builds, eight CPU threads.

## Changes

- CPU FlashSR: accumulate convolution taps across contiguous output positions
  rather than calculating one complete output sample at a time. This exposes
  independent samples to SIMD while retaining the input-channel/tap accumulation
  order for each sample, including omission of out-of-range taps. Work is still
  partitioned into disjoint 256-position blocks. No weight or precision changes.
- Vulkan MiraTTS: use the existing model-local speech/EOS output-head view on
  Vulkan as well as CPU. The view shares the original embedding buffer and keeps
  its existing lifetime. It avoids projecting unused vocabulary rows, not
  changing the sampling alphabet, weights, or token IDs. The shared Qwen runtime
  is untouched. The first pass retains the existing Vulkan attention/precision
  settings; the later decode-only KV-view pass is documented below.
- `AUDIOCPP_MIRA_TTS_SPARSE_HEAD=0` disables the local head on CPU or Vulkan for
  diagnostic comparisons. CUDA retains the existing full-head behavior.

## Method

Reference: `assets/resources/b.wav`. Models: local `mira-tts-q8.gguf` and
`mira-tts-bf16.gguf`. Request parameters/seeds come from
`offline_requests.json` and `streaming_requests.json`. Offline cases cover cold,
repeat, changed prompt, long request, and repeat after the long request in one
session. The offline token budgets cap outputs at 5.12/15.36 seconds; these are
lifecycle/parity tests, not proof that the complete prompt was spoken.

Timings are per-request wall times, excluding initial model loading. Cold means
the first request in a prepared session, not total process startup. Runs were
serial with no competing inference. One before/after sequence per precision is
reported; these are local measurements, not statistically bounded guarantees.

CPU baseline uses the original FlashSR implementation, candidate uses the new
loop. Vulkan compares the full head (`AUDIOCPP_MIRA_TTS_SPARSE_HEAD=0`) with the
local head (`1`). Both Vulkan runs use the same candidate binary, so the only
runtime difference is the head switch.

Reproduction (change backend, device, precision, and mode as appropriate):

```powershell
.\build\windows-vulkan-release\bin\mira_tts_warm_bench.exe --model ..\models_v3_test\MiraTTS-GGUF\mira-tts-q8.gguf --backend vulkan --device 1 --threads 8 --voice-ref assets/resources/b.wav --request-file tests/mira_tts/offline_requests.json --run-mode offline --model-spec-override model_specs/mira_tts.json --audio-out-dir ../outputs/candidate --summary-file ../outputs/candidate.json --log-file ../outputs/candidate.log
python tests/mira_tts/compare_optimization.py --before ../outputs/baseline.json --after ../outputs/candidate.json
```

## Results

| Backend / weights | Request | Before (s) | After (s) | Time reduction |
|---|---|---:|---:|---:|
| CPU Q8 | clone_cold | 8.245 | 7.446 | 9.7% |
| CPU Q8 | clone_repeat | 8.210 | 7.445 | 9.3% |
| CPU Q8 | short_second_prompt | 8.211 | 7.407 | 9.8% |
| CPU Q8 | longform | 26.923 | 24.106 | 10.5% |
| CPU Q8 | clone_repeat_after_longform | 8.180 | 7.457 | 8.8% |
| CPU Q8 | stream_cold | 32.852 | 28.893 | 12.0% |
| CPU Q8 | stream_repeat | 33.243 | 28.826 | 13.3% |
| CPU BF16 | clone_cold | 8.933 | 8.121 | 9.1% |
| CPU BF16 | clone_repeat | 8.851 | 8.170 | 7.7% |
| CPU BF16 | short_second_prompt | 8.832 | 8.488 | 3.9% |
| CPU BF16 | longform | 28.677 | 26.416 | 7.9% |
| CPU BF16 | clone_repeat_after_longform | 8.767 | 8.383 | 4.4% |
| CPU BF16 | stream_cold | 36.138 | 33.739 | 6.6% |
| CPU BF16 | stream_repeat | 36.528 | 32.801 | 10.2% |
| Vulkan Q8 | clone_cold | 0.994 | 0.940 | 5.5% |
| Vulkan Q8 | clone_repeat | 0.881 | 0.815 | 7.5% |
| Vulkan Q8 | short_second_prompt | 0.930 | 0.829 | 10.8% |
| Vulkan Q8 | longform | 2.942 | 2.743 | 6.8% |
| Vulkan Q8 | clone_repeat_after_longform | 0.937 | 0.881 | 6.0% |
| Vulkan Q8 | stream_cold | 3.747 | 3.444 | 8.1% |
| Vulkan Q8 | stream_repeat | 3.739 | 3.432 | 8.2% |
| Vulkan BF16 | clone_cold | 1.015 | 1.065 | -4.9% |
| Vulkan BF16 | clone_repeat | 0.928 | 0.857 | 7.7% |
| Vulkan BF16 | short_second_prompt | 0.951 | 0.863 | 9.3% |
| Vulkan BF16 | longform | 3.173 | 2.898 | 8.7% |
| Vulkan BF16 | clone_repeat_after_longform | 1.017 | 0.999 | 1.8% |
| Vulkan BF16 | stream_cold | 3.973 | 3.687 | 7.2% |
| Vulkan BF16 | stream_repeat | 3.969 | 3.681 | 7.3% |

All 28 paired requests have identical complete WAV bytes (SHA-256), reported
audio hashes, frame counts, sample formats, durations, and event counts.
`validate_bench.py` passes for all four backend/precision combinations, including
repeat-after-long-form and streaming repeat determinism. Every streaming request
produced four audio events. Cross-backend/precision outputs are not expected to
be identical; each candidate is compared only to its matching baseline.

The BF16 Vulkan cold-request regression in this sequence illustrates startup
variability; warm requests and streaming improved. No universal cold-start
speedup is claimed.

An additional short-prompt Q8 CPU check ("Hello, this is a native Mira
TTS test.", seed 1234, max_tokens 1024, same reference and eight threads) produced
6.18 seconds of identical audio: repeated-request wall time 10.222 -> 8.994 s,
and FlashSR stage time 2.085 -> 1.110 s. This is separate from the capped fixtures
in the table.

`flashsr_utility_test` passes its two Python-reference fixtures plus eight input
length cases (1, 7, 85, 86, 255, 256, 257, 1025) checking finite output, 3x length,
and exact one/eight-thread agreement around convolution block boundaries.
The six no-model tests for `compare_optimization.py` also pass, including
rejection of changed WAV bytes despite an unchanged reported hash.

Local artifacts: `../outputs/perf-cpu-{q8,bf16}-{before,after}-{offline,streaming}`
and `../outputs/perf-vk-{q8,bf16}-{0,1}-{offline,streaming}`, with sibling `.json`
summaries and `.log` traces. Large WAV/model files are not added to the repository.

## Experiments not adopted

- Raising CPU thread count from 4 to 16 improved elapsed time, but changed the
  generated hash and duration under otherwise identical sampling parameters.
  It is not an exact-output optimization; the default thread count is unchanged.
- CPU graph rebuilding costs only a few milliseconds versus seconds in
  generation/decoding. Its repeat-determinism safeguard is retained.
- No lower precision, changed sampling defaults, shortened token budgets, or
  reduced audio quality settings were introduced.

## Second CPU pass: local accumulator and pointer-based convolution

The local comparison tree (`TEST_22_audio.cpp_qwen_3.8`,
`src/community_models/miratts/dsp_internal.h`) uses a small stack accumulator
and pointer-based convolution loops to help MSVC vectorize. This pattern is
adapted to FlashSR's residual convolutions and final output convolution, rather
than replacing MiraTTS's generator/decoder wholesale. Each tile is written to
the output once. The input-channel/tap summation order and final `tanhf` are
unchanged. Empty tap intersections are rejected before forming input pointers.

The short-prompt Q8 comparison was rerun against a saved first-pass binary on
the same machine: repeated request 9.005 -> 8.671 s (3.7% less time), FlashSR
1.115 -> 0.637 s (42.8% less time), identical complete WAV. The generator's
runtime varied slightly between runs, so stage and total improvements differ.

Additional experiments were rejected and removed:

- Packing only speech/EOS weight rows: same audio, but CPU slowed slightly and
  Vulkan's warm improvement was only about 1.5%, insufficient to justify an
  extra copied weight allocation in this pass.
- Skipping odd/noncontributing upsampling taps with a variable-bound loop:
  preserved audio but did not improve measured performance.
- Restricting the eight-thread benchmark process to either 16-logical-CPU
  group: slower than the unrestricted process. No affinity settings persist,
  and no affinity/default-thread-count changes are included.

| CPU weights | Request | First pass (s) | Second pass (s) | Time reduction |
|---|---|---:|---:|---:|
| Q8 | clone_cold | 7.446 | 7.018 | 5.7% |
| Q8 | clone_repeat | 7.445 | 7.035 | 5.5% |
| Q8 | short_second_prompt | 7.407 | 7.082 | 4.4% |
| Q8 | longform | 24.106 | 22.381 | 7.2% |
| Q8 | clone_repeat_after_longform | 7.457 | 7.016 | 5.9% |
| Q8 | stream_cold | 28.893 | 28.053 | 2.9% |
| Q8 | stream_repeat | 28.826 | 27.965 | 3.0% |
| BF16 | clone_cold | 8.121 | 7.550 | 7.0% |
| BF16 | clone_repeat | 8.170 | 7.553 | 7.6% |
| BF16 | short_second_prompt | 8.488 | 7.644 | 9.9% |
| BF16 | longform | 26.416 | 23.891 | 9.6% |
| BF16 | clone_repeat_after_longform | 8.383 | 7.504 | 10.5% |
| BF16 | stream_cold | 33.739 | 31.376 | 7.0% |
| BF16 | stream_repeat | 32.801 | 31.435 | 4.2% |

All 14 second-pass sequence outputs are byte-identical to their first-pass
counterparts; both precisions pass the offline/streaming lifecycle validator.
The FlashSR reference/boundary tests and comparison-tool unit tests also pass.
The first-pass sequence timings were collected earlier on the same machine;
the fresh short-prompt comparison above corroborates the stage-level benefit.
These are local observations, not multi-trial confidence intervals.

Artifacts: `../outputs/round2-cpu-{q8,bf16}-{offline,streaming}` and their sibling
JSON/log files. CPU/Vulkan CLI and server targets were rebuilt. There is no
additional Vulkan algorithm change in this pass; its earlier optimization is
retained. The alternate source tree was inspected read-only.

## Third pass: Vulkan decode-only KV views

Vulkan now uses `FlashGroupedViewKV` for single-token decoding, avoiding
materialization of grouped cached heads on each token. Prefill remains
`FlashGrouped`: the previous strided-prefill precision issue is not bypassed.
F32 projection precision, sampling, token budgets and CPU execution are unchanged.
Set `AUDIOCPP_MIRA_TTS_VULKAN_VIEW_DECODE=0` to compare the previous decode path.

RTX 3090, Vulkan device 1, eight host threads, same fixtures as above:

| Weights | Request | Before (s) | After (s) | Time reduction |
|---|---|---:|---:|---:|
| Q8 | clone_repeat | 0.816 | 0.792 | 2.9% |
| Q8 | longform | 2.787 | 2.674 | 4.1% |
| Q8 | stream_repeat | 3.395 | 3.424 | -0.9% |
| BF16 | clone_repeat | 0.853 | 0.824 | 3.3% |
| BF16 | longform | 2.977 | 2.731 | 8.3% |
| BF16 | stream_repeat | 3.735 | 3.551 | 4.9% |

All 14 paired offline/streaming WAV files are byte-identical, and both lifecycle
validators pass. Q8 streaming did not improve in this measurement; the change
does not claim universal speedups. Other Vulkan devices are not validated here.

The initial BF16 offline runs had abnormal first-request times (61–65 seconds)
in **both** modes. Their cause is unconfirmed. These timings are excluded from
the table; a fresh paired rerun returned to normal timings. Artifacts are retained
for inspection rather than treating the anomaly as an optimization gain.

Artifacts: `../outputs/round3-vk-{q8,bf16}-{0,1}-{offline,streaming}`; the BF16
offline table uses `../outputs/round3-rerun-bf16-{0,1}.json`. The final default
build is additionally exercised in `../outputs/round3-final-{q8,bf16}.json`.
Use `compare_optimization.py` with the corresponding before/after JSON paths.

### CPU pool experiment rejected

An opt-in persistent ggml thread pool was tested in an off/on/on/off sequence.
Repeated short-request timings were 8.747 / 8.551 / 8.748 / 8.632 seconds, with
identical WAVs. This does not establish a repeatable improvement, so the pool
code was removed. The current build uses OpenMP, which already reuses OS worker
threads; allocating a ggml pool per graph is not equivalent to creating new OS
threads per token. The accepted CPU convolution optimizations remain intact.

## Fourth pass: no additional optimization retained

The following experiments were removed rather than claiming noise as a gain:

- CPU direct-convolution tiles enlarged from 256 to 1024: repeated short Q8
  request 8.684 -> 8.660 s, cold request 8.699 -> 8.722 s. Inconclusive.
- CPU activation downsampling tiled across positions with the same tap order:
  repeated request 8.739 s, slower than the 8.684 s baseline.
- Channel-major direct interpolation was tried, but initially benchmarked on
  Vulkan, which uses the GGML upsampler rather than this CPU DSP function.
  Those measurements do not validate the CPU optimization; it was removed.

The CPU tile and downsampling WAVs match the baseline byte-for-byte; FlashSR
utility tests pass. Artifacts: `../outputs/round4-cpu-tile-*` and
`../outputs/round4-cpu-downsample.*`. The Vulkan `round4-interp-*` and
`round4-tile-*` measurements are retained only as diagnostic artifacts, not
evidence for a speedup. Previous accepted optimizations remain unchanged.
