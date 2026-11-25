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
BATCH_FILE=/path/to/prompts.txt ./tests/omnivoice/run_clone_batch_benchmark.sh
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

General clone and batch tests reuse the shared English LibriSpeech pair:

- `assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.wav`
- its matching `.txt` transcript

The pair has source/license metadata in `assets/asr_validation/`. Prompts are
English; supply your own batch requests with `BATCH_FILE=/path/to/prompts.txt`
for the sweep script or `--text-file /path/to/prompts.txt` for the native test.
The prompt file contains one request per non-empty line.
Rate variants should be created temporarily from this shared input rather than
stored as another recording. The removed local Bengali examples were not used
by a Bengali-specific assertion.

## Python-to-C++ integration

[`python_cpp_simple/`](python_cpp_simple/) drives OmniVoice from Python while
keeping inference in C++, by sending HTTP requests to `audiocpp_server` using
only the Python standard library.

## Historical session and batch benchmarks

The removed session runners and informational native batch benchmark are
preserved in the [pre-cleanup guide](https://github.com/0xShug0/audio.cpp/blob/checkpoint/before-tests-cleanup/tests/omnivoice/README.md).
Use the batch parity test below for its documented assertions; it does not
replace the historical timing measurements.

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
