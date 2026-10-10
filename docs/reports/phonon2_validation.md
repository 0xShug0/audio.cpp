# Phonon-2 native validation

Windows, 2026-10-10; Ryzen 7950X3D, 8 inference threads, RTX 3090.
Final pre-PR matrix rebuilt on upstream `497c7f02` plus the Phonon model
changes and CUDA precision dependency [PR #861](https://github.com/0xShug0/audio.cpp/pull/861).
Earlier before-fix measurements used upstream `6c3ab9ad`. Standard serial
server; full-context inference.

## Cause and changes

The earlier ten-clip pass had three failing combinations: F32 CUDA and Q8
CPU/CUDA. These were duration decisions, after fixing word formatting.
Replaying identical published-reference encoder activations through the real
F32/Q8 decoder on CPU, CUDA and Vulkan matched every emitted token ID, frame,
and duration in all 12 replays on the two failing clips. This isolates those
differences to the encoder arithmetic rather than the word formatter.

CUDA's default cuBLAS math mode permits TF32, including with F32 weights and
accumulation. A process-local `NVIDIA_TF32_OVERRIDE=0` probe removed both
original CUDA failures. The production fix uses no environment override:
Phonon's backend instance selects full-F32 matmul inputs through
`ggml_backend_cuda_set_f32_matmul`. It disables TF32 in that instance's existing
and future cuBLAS handles and bypasses matmul/fusion routes that round or
quantize activations. Existing F32 kernels/cuBLAS perform the computation.
Stricter routing also removed three additional Q8 CUDA holdout failures that
disabling TF32 alone did not fix.

CPU Q8 dot products additionally quantize activations. Generated Q8 packages
now opt into `audiocpp_cpu_matmul_weight_type=f32`, expanding matmul weights at
load time to avoid that extra activation quantization. The on-disk Q8 weights
are unchanged. Explicit session weight-storage overrides remain available.

The shared CUDA backend gained an **instance-scoped opt-in policy**, not new
kernel implementations. The default policy and other instances' dispatch are
unchanged. Configuration occurs before graph preparation/capture; changing
the policy after capture is rejected. HIP/MUSA retain their old math-mode
branch, but those platforms were not built/tested in this Windows pass.
Vulkan received no further kernel changes.

## Reference checks

Reference: pinned Phonon Transformers F32 model, actual token frames/durations,
and independently pinned `fermion-research` 0.2.11 word formatting. Timing
passes require matching words and starts/ends within 1 ms for JSON rounding;
an 80 ms frame difference fails. This is equality to the published dense
reference, not to every original packed runtime or a corpus-level WER result.

The original ten inputs total 95.4275 s. Ten additional inputs include short
speech, 18/30 s video excerpts, and excerpts from `test.mp3`. The latter includes
music/non-English singing and serves as arithmetic/output stress coverage,
not a claim of ASR accuracy outside Phonon's English speech scope.

| Storage | CPU text / timings | CUDA text / timings | Vulkan text / timings |
|---|---|---|---|
| F32 | 20/20 / 20/20 | 20/20 / 20/20 | 20/20 / 20/20 |
| BF16 | 20/20 / 19/20 | 20/20 / 19/20 | 20/20 / 19/20 |
| Q8_0 | 20/20 / 20/20 | 20/20 / 20/20 | 20/20 / 20/20 |

All nine combinations pass the original ten clips. Every combination repeats
its own text/word timings exactly on all 20 clips and passes the upload endpoint
text check. These are 180 original-reference comparisons plus repeated requests.

**BF16 remains non-equivalent on `video_30s`: one word timing differs by
80 ms on all three backends.** Keeping its head in original F32 or expanding
its CPU matmul weights to F32 did not remove the difference. The experimental
head package was not installed and that converter change was reverted. BF16
weights are rounded; exact timing for this input requires the F32 package
(Q8 also matches on this sampled input). No timestamp tolerance was relaxed.

The independent exact-weight/encoder/CLI validator passes on CPU and CUDA,
including the formerly failing video excerpt starting at 240 s. Encoder relative L2 error
on the 3 s speech input is `1.04e-6` on CPU and `8.85e-7` on CUDA.

## Latency and memory tradeoff

Five warm HTTP requests of the same 11.9925 s clip, excluding model loading.
Before and after are separate passes, not a controlled speedup experiment.
Final after-pass timings were collected after compilation completed.
Additional GPU memory subtracts the
desktop baseline and **includes model weights and buffers**; sampling is
device-wide NVML every 20 ms and can miss brief peaks.

| Storage | Backend | Earlier warm time | Current warm time | Earlier additional VRAM | Current additional VRAM |
|---|---|---:|---:|---:|---:|
| F32 | CPU | 547 ms | 524 ms | CPU | CPU |
| F32 | CUDA | 44 ms | 46 ms | 2.80 GiB | 2.78 GiB |
| F32 | VULKAN | 77 ms | 70 ms | 2.64 GiB | 2.61 GiB |
| BF16 | CPU | 512 ms | 499 ms | CPU | CPU |
| BF16 | CUDA | 39 ms | 59 ms | 2.18 GiB | 2.17 GiB |
| BF16 | VULKAN | 70 ms | 66 ms | 1.99 GiB | 1.97 GiB |
| Q8_0 | CPU | 484 ms | 535 ms | CPU | CPU |
| Q8_0 | CUDA | 40 ms | 50 ms | 2.12 GiB | 2.18 GiB |
| Q8_0 | VULKAN | 69 ms | 69 ms | 1.91 GiB | 1.94 GiB |

Precision has a real CUDA latency cost, especially for BF16/Q8. Q8 still
processes this approximately 12 s input in about 50 ms once loaded. CPU Q8
also uses more host RAM: sampled process RSS peaks at 3.21 GiB over the original
ten inputs and 4.17 GiB over the expanded suite. Original-suite CPU F32/BF16
peaks are 4.64/2.83 GiB. These RSS values include model loading and temporary
buffers; they are not cache-only figures. Expanded-suite additional VRAM:

| Storage | CUDA peak | Vulkan peak |
|---|---:|---:|
| F32 | 3.03 GiB | 2.90 GiB |
| BF16 | 2.60 GiB | 2.82 GiB |
| Q8_0 | 2.35 GiB | 3.34 GiB |

## Regression checks and boundaries

- Shared CPU CTest suite: 53/53 after using MSVC UTF-8 source encoding and
  enabling the embedded catalog. No test assertions were relaxed.
- Converter/timestamp Python tests: 12/12. Variant/duration CTests: CPU 2/2;
  combined build model tests 2/2 plus the dependency PR CUDA precision-isolation test.
- The CUDA isolation test checks F32, BF16 and Q8 weights at two matrix shapes
  against a double-accumulation reference. Another backend instance's outputs
  remain bit-identical before/after strict inference.
- Stock Parakeet Q8 full-context and long-form golden transcription/word
  checks pass. This is targeted compatibility coverage, not all-model testing.
- Q8 CLI automatic family selection and full-context reference equality pass
  on all three backends for the previously failing 20 s excerpt.
- On that 12 s input, native long-form Q8 matches native long-form F32 exactly
  on all three backends, but **both differ in timing from the full-context
  reference**. The bounded/padded encoder window is a separate schedule;
  this is not proof of original Phonon long-audio segmentation parity.
The complete 299.9935 s video also runs through native Q8 long-form mode on all
three backends: 817 words, identical transcript SHA256, monotonic word starts
and sample bounds within the recording. Cold process wall times, including
loading and output capture, were CPU 116.82 s, CUDA 7.11 s, Vulkan 7.64 s.
The CPU run overlapped compilation, so this is a correctness/RTF smoke check,
not a controlled long-form benchmark or original-reference alignment check.

Twelve identical server requests per storage/GPU backend verify repeatability
and bounded warm memory. After two warm-up requests, sampled GPU memory did
not grow on any of the six combinations; RSS varied by at most 1.82 MiB.
Results are in the committed summary and local `warm-memory-validation.json`;
device-wide NVML cannot attribute unrelated desktop activity to this server.

The model files F32/BF16 are unchanged in this pass. The installed Q8 file
contains only the new CPU precision metadata/provenance: all 699 tensor
payloads and types match the preceding Q8 file. New Q8 size: 931,186,432 bytes;
SHA256 `93ba6be8e949acca6d9e13ac11909bceb365416e8e5a75b3c6f436693dd3b68f`.
Its previous file is retained as a backup.

Final PR artifacts: `outputs/phonon2-pr-20261010/`; earlier diagnostic artifacts:
`outputs/phonon2-timing-fix-20261010/`. A portable result summary is committed as
[phonon2_validation.json](phonon2_validation.json). The local
`matrix-cpu-cuda-vulkan.json` contains every case, latency sample and memory scope;
`decoder-replay.json`, independent validators, CLI checks, package audit and
`manifest.json` retain the evidence and source/binary/model/input hashes.
Earlier artifacts remain in `outputs/phonon2-timestamp-investigation-20261009/`.
The CUDA precision API is [PR #861](https://github.com/0xShug0/audio.cpp/pull/861), submitted separately as required by CONTRIBUTING.md.
The Phonon model PR depends on that support for CUDA; CPU/Vulkan need no such API.

## Reproduce

Use the source conversion commands in [the model guide](../community_models/phonon2.md).
The CUDA precision dependency must be present before running CUDA reference checks.

```bash
cmake -S . -B build/phonon2 -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=parakeet_tdt \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON \
  -DENGINE_BUILD_WARMBENCH=ON
cmake --build build/phonon2 --parallel 6 --target audiocpp_cli audiocpp_server \
  audiocpp_gguf parakeet_parity_dump parakeet_warm_bench \
  parakeet_variant_config_test parakeet_golden_transcription_test \
  tdt_decoder_duration_loop_test cuda_matmul_precision_test
ctest --test-dir build/phonon2 --output-on-failure \
  -R 'parakeet_variant_config|tdt_decoder_duration_loop|cuda_matmul_precision'
python tools/check_loader_catalog_sync.py --self-test
python tools/check_loader_catalog_sync.py
python -m unittest discover -s tests/parakeet_tdt -p 'test_phonon2_*.py'
```

For a CPU-only build set both backend options OFF. On this Windows machine we
used Ninja, MSVC 14.43 and CUDA 12.4 with `-DCMAKE_CUDA_ARCHITECTURES=86`.
Windows executables below have the `.exe` extension. The full shared CPU test
run used `ENGINE_BUILD_MODEL_TESTS=OFF`, `ENGINE_BUILD_TESTS=ON`,
`ENGINE_BUILD_WARMBENCH=OFF`, `AUDIOCPP_DEPLOYMENT_BUILD=ON`, and
`AUDIOCPP_MODELS=parakeet_tdt;sortformer_diar_v2;supertonic` (the extra model
objects are needed by existing shared tests). MSVC flags retained `/EHsc /GR`
and added `/utf-8`; replacing all default flags with `/utf-8` alone disables
exception unwinding and is not a valid test configuration. All 53 registered
CTest executable targets and the eSpeak test DLLs were built, then run with
`ctest --test-dir build/phonon2-cpu --output-on-failure --parallel 4`.
Optional unrelated warmbench/probe targets are excluded from that reduced-model
unit build; the GPU model targets above were built separately.

```bash
build/phonon2/bin/parakeet_golden_transcription_test \
  --model models/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf
build/phonon2/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model models/Phonon-2-GGUF/phonon-2-q8_0.gguf --backend cuda \
  --audio speech.wav --threads 8
build/phonon2/bin/parakeet_warm_bench \
  --model models/Phonon-2-GGUF/phonon-2-q8_0.gguf --backend cuda \
  --audio speech.wav --iterations 5 --timing-file warm-timing.log
```

Exact-F32 tensor/encoder/text/word-time validation is documented in the model
guide; supply the pinned original word formatter. Replay `enc_out.npy` through
the real decoder with `parakeet_parity_dump --decoder-input encoder.npy`,
alongside `--model`, `--audio`, `--backend` and `--output-dir`.
