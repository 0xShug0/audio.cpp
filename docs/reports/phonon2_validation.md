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
- CPU/CUDA/Vulkan actual CLI and legacy server: JSON/multipart hotwords, uploaded
  streaming, unaligned live PCM, per-request isolation and repeated requests pass.
- CPU/CUDA/Vulkan lifecycle: 13/2049/whole-input chunking, invalid order/retry,
  repeat finalize/post-final rejection, nonfinite input and callback failure
  recovery, idle/reuse and multiple phrases pass. Six named-term F32 cases match
  original final text. Committed deltas equal final text; timestamps stay in range.
- Stock Parakeet Q8 offline/buffered streaming: six before/after text/timing checks
  pass. Repeated multipart fields, bad strength/JSON recovery and biased silence pass.
- CPU CTest: 54/54; combined-build targeted CTest: 4/4; converter/timestamp Python:
  14/14. Windows source encoding and embedded catalog are enabled; no assertions
  or unrelated framework tests were weakened.

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
