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
