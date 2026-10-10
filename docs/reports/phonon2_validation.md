# Phonon-2: minimal variant validation

Revision responding to [maintainer feedback](https://github.com/0xShug0/audio.cpp/pull/863#issuecomment-6100453087).
The existing Parakeet family/loader/backend math remains in use. Model-local
hotwords, rolling streaming and `word_timestamp_mode` remain. The F32 precision
API, CPU weight-expansion policy and runtime BN/input-scale overrides are removed.
Vulkan optimizations are reviewed independently. No backend or GGML files change.

## Controlled mathematics

All 699 F32 inference tensors match independently decoded original records after
the documented conversion transforms, and match the GGUF exactly. The 1/32
projection transform cancels the normal loader's scale fold. BN gamma compensation
preserves the reference formula in two channels affected by its variance floor.
No runtime arithmetic customization is required.

CPU F32 encoder relative L2 error on the 3 s speech fixture is
`1.04e-06` (limit `1e-4`); maximum absolute error is
`3.5e-07`. Text and actual token-duration word timestamps
match the independently pinned reference exactly. No randomness was introduced;
all backend/storage requests repeat their own outputs exactly.

## Ordinary requests

Twenty inputs: speech, silence, short/longer video excerpts and music as arithmetic
stress coverage. Counts below use the published dense F32 reference. Exact word
timing uses its 1 ms JSON print tolerance; whole 80 ms shifts are recorded as
differences. Word edits use case/punctuation-normalized Levenshtein distance over
450 reference words. They are not a human-labelled WER benchmark.

| Storage | Backend | Exact text | Exact timing | Reference word edits | Exact repeat |
|---|---|---:|---:|---:|---:|
| f32 | cpu | 20/20 | 20/20 | 0/450 | 20/20 |
| f32 | cuda | 20/20 | 19/20 | 0/450 | 20/20 |
| f32 | vulkan | 20/20 | 20/20 | 0/450 | 20/20 |
| bf16 | cpu | 20/20 | 19/20 | 0/450 | 20/20 |
| bf16 | cuda | 19/20 | 18/20 | 2/450 | 20/20 |
| bf16 | vulkan | 20/20 | 19/20 | 0/450 | 20/20 |
| q8_0 | cpu | 19/20 | 17/20 | 1/450 | 20/20 |
| q8_0 | cuda | 18/20 | 17/20 | 3/450 | 20/20 |
| q8_0 | vulkan | 19/20 | 17/20 | 1/450 | 20/20 |

F32 text matches on every input/backend. Reduced-precision differences and CUDA
duration/timing differences are explicitly retained in the evidence; this is not
a claim of universal precision equivalence or no corpus-level quality change.
All nine combinations complete and pass their upload endpoint text check.

## Features and regressions

- Original hotword policy: 9 lists, 329 transition/bonus vectors exactly match.
- Original decoder/hotword/live final text: 54/54; final stream segments: 18/18.
  Original offline timings match in 27/36 cases and provisional sequences in
  5/18.
  These diagnostic differences are included rather than hidden by forcing GPU precision.
- CPU/CUDA/Vulkan actual CLI and legacy server: uploaded streaming, unaligned
  live PCM, per-request isolation and repeated requests pass.
- CPU/CUDA/Vulkan lifecycle: 13/2049/whole-input chunking, invalid order/retry,
  repeat finalize/post-final rejection, nonfinite input and callback failure
  recovery, idle/reuse and multiple phrases pass. Six named-term F32 cases match
  original final text. Committed deltas equal final text; timestamps stay in range.
- Stock Parakeet Q8 offline/buffered streaming: six before/after text/timing checks
  pass. Bad strength/JSON recovery and biased silence pass.
- CPU CTest: 54/54; combined-build targeted CTest: 4/4; converter/timestamp Python:
  14/14. Windows source encoding and embedded catalog are enabled; no assertions
  or unrelated framework tests were weakened.

Dedicated HTTP hotword fields were removed after those measurements. Hotwords
use generic `options` on JSON and batch multipart requests; single-file uploads
and live PCM use the existing `prompt` interface. A subsequent CUDA check covered
18 requests across standard and parallel servers, including upload, upload
streaming and live PCM: prompt bias took effect, removing it restored the
baseline, and committed deltas concatenated to the final text.
After separating variant selection from timestamp formatting, eight further
CUDA live requests using a newly packaged F32 GGUF matched the previous final
text and unprompted snapshot sequences exactly. Both server modes preserved
prompt bias and repeated-request isolation, with `session.wall_ms` emitted for
each request. These checks do not establish new performance results.

## Functional timing and memory

Windows Ryzen 7950X3D (8 threads), RTX 3090, MSVC 14.43, CUDA 12.4, Vulkan SDK
1.4.350. Median of five warm HTTP requests for an 11.9925 s clip, excluding loading.
CPU peak is process RSS; GPU peak is device-wide NVML sampled every 20 ms, after
subtracting the desktop baseline, over the complete 20-input sequence. Both
include weights/work buffers. Samples may miss short peaks or include unrelated
driver/desktop allocations; do not interpret this as cache alone or a controlled
before/after speedup experiment. Some CPU functional runs overlap the reference
process; the separate optimization report uses alternating timing trials.

| Storage | Backend | Warm request (ms) | Peak RSS (GiB) | Additional peak VRAM (GiB) |
|---|---|---:|---:|---:|
| f32 | cpu | 549.3 | 4.63 | n/a |
| f32 | cuda | 42.5 | 2.95 | 3.24 |
| f32 | vulkan | 68.4 | 2.88 | 3.38 |
| bf16 | cpu | 518.0 | 3.51 | n/a |
| bf16 | cuda | 45.0 | 2.15 | 2.62 |
| bf16 | vulkan | 62.1 | 2.11 | 2.72 |
| q8_0 | cpu | 530.4 | 3.46 | n/a |
| q8_0 | cuda | 41.5 | 2.19 | 2.37 |
| q8_0 | vulkan | 67.5 | 2.12 | 3.90 |

## Reproduce

Use [conversion and CLI/server examples](../community_models/phonon2.md).
Actual build commands use Ninja Release, `AUDIOCPP_MODEL_SET=custom`,
`ENGINE_BUILD_TESTS=ON`, `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF` and
`AUDIOCPP_BUILD_SERVER_FRONTENDS=OFF`. The combined build enables CUDA/Vulkan,
`CMAKE_CUDA_ARCHITECTURES=86`, `AUDIOCPP_MODELS=parakeet_tdt` and
`ENGINE_BUILD_MODEL_TESTS=ON`; the common CPU suite disables GPU/model tests,
sets `AUDIOCPP_DEPLOYMENT_BUILD=ON` and links
`parakeet_tdt;sortformer_diar_v2;supertonic`. MSVC uses `/EHsc /utf-8`.
Build CLI/server, `phonon_features_probe`, `phonon_hotwords_test`,
`parakeet_variant_config_test`, `partial_text_render_test`, the common CTest
targets, and the combined-build `tdt_decoder_duration_loop_test`/`parakeet_parity_dump`.

In an MSVC 14.43 developer environment with CUDA/Vulkan SDKs on PATH,
the combined runtime build used:

```bash
cmake -S . -B ../build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8' \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=parakeet_tdt \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DENGINE_BUILD_WARMBENCH=ON \
  -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF -DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF \
  -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DGGML_CCACHE=OFF \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build ../build-gpu -j 6 --target audiocpp_cli audiocpp_server audiocpp_gguf \
  phonon_features_probe phonon_hotwords_test parakeet_variant_config_test \
  partial_text_render_test tdt_decoder_duration_loop_test parakeet_parity_dump
ctest --test-dir ../build-gpu --output-on-failure \
  -R 'phonon_hotwords_test|parakeet_variant_config_test|tdt_decoder_duration_loop_test|partial_text_render_test'
```

```bash
ctest --test-dir build-cpu --output-on-failure
python tests/parakeet_tdt/test_phonon2_conversion.py
python tests/parakeet_tdt/test_phonon2_timestamps.py
python tests/parakeet_tdt/validate_phonon2_hotwords.py \
  --probe build/bin/phonon_features_probe \
  --original-hotwords ORIGINAL_WHEEL/fermion/_speech/hotwords.py \
  --source-config models/Phonon-2-source/extracted/config.json
```

`validate_phonon2.py` audits weights and numerical bounds; exact text/timing is
diagnostic unless `--require-exact-reference` is supplied for the controlled CPU
F32 run. The [portable JSON](phonon2_validation.json) includes source/model hashes,
all matrix summaries and differing fixture IDs. Raw commands/configs/outputs are
under `outputs/phonon2-review-20261010/`; earlier pre-compensation evidence is
preserved in its `pre-bn-compensation/` directory. No user audio is uploaded.
RTX 5090, Metal/HIP/MUSA runtime coverage is absent; removing the failing API path
does not constitute a verified RTX 5090 fix. Original WebSocket protocol and
packed kernels remain outside this PR.

## Mixed GGUF storage follow-up

Opt-in `mixed-f16` preserves all 699 staged tensor values exactly (327 F16,
372 F32 tensors). `mixed-q8` keeps sensitive feed-forward/pointwise weights in
exact F16 and non-F16-representable, normalization and compensated projection
values in F32, with ordinary rounded Q8_0 for the remaining matrices (120 Q8_0,
144 F16, 435 F32 tensors). It is not lossless whole-model storage. Both use
the normal Parakeet runtime; existing `orig`, `bf16`, `q8_0` defaults and shared
backends are unchanged. Mixed-FP16 is not a new BF16 package.

The 21-input follow-up adds the maintainer's 28 s WAV to the existing ten original
and ten holdout clips. Reference is the published dense Python FP32 route.
Timing counts below require every word in a clip to match within 1 ms.
Native FP32 GPU results also have reference differences; this does not claim
that either mixed package improves every timestamp or surpasses FP32 accuracy.

| Package | GB (decimal) | Exact transcript | Exact word-timing clip | Exact repeat |
|---|---:|---:|---:|---:|
| FP32 | 2.510 | 63/63 | 61/63 | 63/63 |
| Old BF16 | 1.264 | 62/63 | 56/63 | 63/63 |
| Old Q8 | 0.944 | 59/63 | 51/63 | 63/63 |
| Mixed-FP16 | 1.302 | 63/63 | 59/63 | 63/63 |
| Mixed-Q8 | 1.184 | 63/63 | 60/63 | 63/63 |

Mixed-FP16 is 48.1% smaller than FP32. Mixed-Q8 is 52.8% smaller than FP32,
25.5% larger than old Q8 and 9.1% smaller than mixed-FP16. All mixed-Q8 final
tensor payloads/types match the tested candidate; all 63 final request pairs
repeat exactly. Converter/timestamp Python: **16/16**; targeted CTest: **4/4**.
SHA256 mixed-FP16: `3ca25f950f46ab66f5b09ad806610ad0ff826c0d0cf0360fc87d4d704eb104c7`;
mixed-Q8: `238f4d03628ae7d594fe6345535cadba4f8576331ff5d7572c838add583ca25d`. These are file/storage comparisons, not VRAM
measurements. Existing optional F32 weight-loading session settings reproduce
the corresponding native FP32 package for mixed-FP16, using its working memory.

Recorded loaded-model request times for the same 28 s input, in seconds:

| Backend | FP32 | Old BF16 | Old Q8 | Mixed-FP16 | Mixed-Q8 |
|---|---:|---:|---:|---:|---:|
| CPU | 2.831 | 2.823 | 2.416 | 3.002 | 2.506 |
| CUDA | 0.120 | 0.109 | 0.111 | 0.116 | 0.104 |
| VULKAN | 0.211 | 0.187 | 0.191 | 0.199 | 0.185 |

These are single HTTP request measurements from separate validation passes,
excluding model loading, on Ryzen 7950X3D / RTX 3090 with 4 threads. They are
not alternating controlled speed trials and establish no general speedup.
Earlier timing tables use a different clip/thread setting; do not combine them.

Selected word-end offsets from Python FP32 on the maintainer WAV (milliseconds):

| Backend | Word | Native FP32 | Old BF16 | Old Q8 | Mixed-FP16 | Mixed-Q8 |
|---|---|---:|---:|---:|---:|---:|
| CPU | solar | 0 | 0 | +160 | 0 | 0 |
| CPU | sunrise. | 0 | +320 | 0 | 0 | 0 |
| CPU | spare | 0 | +80 | 0 | +80 | 0 |
| CUDA | solar | 0 | 0 | +160 | 0 | 0 |
| CUDA | sunrise. | 0 | +320 | 0 | 0 | 0 |
| CUDA | spare | 0 | +80 | 0 | +80 | +80 |
| VULKAN | solar | 0 | 0 | 0 | 0 | 0 |
| VULKAN | sunrise. | +320 | +320 | +320 | +320 | +320 |
| VULKAN | spare | +80 | +80 | +80 | +80 | +80 |

Reference ends: solar 3.84 s, sunrise. 14.56 s, spare 18.80 s. Other words on
this input match for these configurations. On `movie_offset_240_12s`, mixed-Q8
CPU introduces `to` start +80/end +240 ms and `improve` start/end +80 ms, where
old Q8 matches. Mixed-FP16 has other CPU/CUDA timing differences as recorded in
the portable JSON. Corpus WER and RTX 5090/Metal/HIP/MUSA remain untested.

## Duration diagnostics

One encoder frame is 80 ms (160-sample hop, 8x subsampling, 16 kHz). Scores near
ties can change blank/token choice or duration while preserving final text.
On the maintainer WAV, Python's period beats blank by 0.003579 score units;
duration 1 beats 2 by 0.001392. The Vulkan +320 ms `sunrise.` shift combines
later punctuation emission with a longer duration; it is not a clock error.

The same Python encoder output replayed through native FP32 decoders gives
identical IDs/frames/durations for all 166 retained tokens on CPU/CUDA/Vulkan.
FP32 encoder relative L2 on 350 valid rows is 8.77e-7 / 5.39e-4 / 5.03e-3;
the unused 351st capacity row is excluded. Encoder and decoder rounding can
reinforce or cancel differences. The original Fermion CPU wrapper also marks
2801 feature frames valid versus the dense route's 2800. A mask-only crossover
removes its word-time differences on this WAV, not a claim about every input.
No timestamp offsets, fitted duration choices or mandatory backend precision
controls were introduced. Raw captures remain in the follow-up artifact folders
listed in the existing portable JSON; user audio is not uploaded.
