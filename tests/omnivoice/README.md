# OmniVoice tests

Voice-clone benchmark scripts and a Python-to-C++ integration test for the
`omnivoice` family. All scripts resolve the repository root themselves, so they
can be run from any directory.

They expect `models/OmniVoice` to be installed:

```bash
python3 tools/model_manager.py install omnivoice
```

## Build

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
cmake --build build --parallel --target audiocpp_cli audiocpp_server
```

## Weight-type benchmarks

`run_clone_benchmark.sh` sweeps single voice-clone requests and
`run_clone_batch_benchmark.sh` sweeps `--batch-text-file` runs, both across
every combination of `--num-inference-steps` (8 to 64),
`omnivoice.generator_weight_type`, and `omnivoice.audio_tokenizer_weight_type`.

```bash
./tests/omnivoice/run_clone_benchmark.sh
./tests/omnivoice/run_clone_batch_benchmark.sh
```

Each script runs 375 combinations, so a full sweep takes hours. Edit
`STEPS_LIST`, `WEIGHT_TYPES`, and `AUDIO_TOKENIZER_WEIGHT_TYPES` at the top to
narrow it. WAVs and a full log are written under `tests/omnivoice/outputs/`,
which is ignored by Git.

Measured results are in
[OmniVoice runtime weight-type benchmark](../../docs/reports/omnivoice_weight_type_benchmark.md).
The headline finding is that runtime `omnivoice.audio_tokenizer_weight_type=q8_0`
produces unusable audio without being any faster, while
`omnivoice.generator_weight_type=f16` is 1.26x faster than the native path.

## Assets

`assets/` holds the shared Bengali fixtures used by every script here:

```text
ref_audio_01.wav  ref_text_01.txt   longer reference voice
ref_audio_02.wav  ref_text_02.txt   reference voice used by default
clone_batch_prompts.txt             10 prompts, one request per line
```

## Python-to-C++ integration

[`python_cpp_simple/`](python_cpp_simple/) drives OmniVoice from Python while
keeping inference in C++, by sending HTTP requests to `audiocpp_server` using
only the Python standard library.

## Warm benchmark

`omnivoice_warm_bench.cpp` and `omnivoice_python_warm_bench.py` are the
framework warmbench helpers driven by `tests/warmbench.py`, and are unrelated to
the sweep scripts above.

## Native batch test

`omnivoice_batch_test.cpp` (target `omnivoice_batch_test`, built with
`-DENGINE_BUILD_WARMBENCH=ON`) runs the same requests through `run()` one at a
time and through one `run_batch()` call for each `--batch-sizes` entry, and
reports both wall times and per-request parity. Request *i* uses line
*i* of `--text-file` (cycled) and seed `--seed + i`.

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON -DENGINE_BUILD_WARMBENCH=ON
cmake --build build --parallel --target omnivoice_batch_test
./build/bin/omnivoice_batch_test --model models/OmniVoice \
  --clone-audio tests/omnivoice/assets/ref_audio_02.wav \
  --reference-text-file tests/omnivoice/assets/ref_text_02.txt \
  --text-file tests/omnivoice/assets/clone_batch_prompts.txt \
  --num-step 16 --batch-sizes 1,2,4,8 \
  --session-option omnivoice.generator_weight_type=f16
```

- `--out-dir` keeps every output as WAV and raw float32 (`.f32`).
- `--compare-dir <dir>` compares the `run()` outputs bit for bit with the
  `.f32` files of an earlier build, e.g. to check that a change left the
  single-request path untouched. A mismatch or a missing baseline makes it
  exit with status 1.
- The `run()` vs `run_batch()` lines are informational: with f16 weights the
  batched kernels can round differently. The parity test below asserts the
  exact guarantees.
- `--timing-log <file>` writes the engine `[TIMING]` lines (generator
  compute, readback, scoring, decode) of every call.
- `--session-option` and `--request-option` pass options through. The
  earlier code path is `--session-option omnivoice.packed_cfg=false
  --session-option omnivoice.device_scoring=false --session-option
  omnivoice.packed_projections=false`.

## Batch parity test

`batch_parity_test.cpp` (target `omnivoice_batch_parity_test`, built with
`-DENGINE_BUILD_MODEL_TESTS=ON`) asserts the batching and fast-path guarantees
with f32 generator weights and exits with status 1 if any check fails. Five
requests of different lengths, seeds, step counts and guidance scales, one of
them sampled (`class_temperature=0.8`), are checked:

- `batch_of_one`: `run_batch({r})` equals `run(r)` bit for bit, for every
  request.
- `batched_vs_run`: `run_batch()` of all five equals `run()` of each greedy
  request bit for bit on CUDA. Other backends can pick different kernels for
  the batched shapes, so there the check requires the same length and finite
  audio and prints the difference.
- `batched_repeat`: a second `run_batch()` gives the same audio for every
  request, the sampled one included.
- `device_scoring_off`: `omnivoice.device_scoring=false` equals the default,
  for every request.
- `packed_projections_off`: `omnivoice.packed_projections=false` equals the
  default, for every greedy request.

Sampled requests are not compared across layouts: batching and packed
projections change GEMM shapes, so f32 logits can differ in the last bits and
sampling can draw a different token.

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON -DENGINE_BUILD_MODEL_TESTS=ON \
  -DAUDIOCPP_OMNIVOICE_TEST_MODEL=$PWD/models/OmniVoice
cmake --build build --parallel --target omnivoice_batch_parity_test
ctest --test-dir build -R omnivoice_batch_parity --output-on-failure
```

Without `AUDIOCPP_OMNIVOICE_TEST_MODEL` the target is built but not
registered with ctest; run it directly with `--model`, `--clone-audio`,
`--reference-text-file` and optionally `--backend cuda|vulkan|cpu`,
`--device` and `--num-step` (default 8).
