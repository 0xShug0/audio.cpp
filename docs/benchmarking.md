# Benchmarking & Testing Guide

This document outlines the procedures for running benchmarks and integration tests across the `audio.cpp` framework.

## 1. Managing Models

The test suites require specific models to be present in the `models/` directory.

You can list all available models and their package IDs using the model manager:

```bash
python3 tools/model_manager.py list
```

To install a specific model (e.g., `qwen3_tts_1_7b_base`):

```bash
python3 tools/model_manager.py install qwen3_tts_1_7b_base --models-root models
```

_Note: Downloading all 38+ test models requires ~200 GB of free space._

## 2. Running Path Tests (`audiocpp_cli_path_tests.py`)

The `run_audiocpp_cli_path_tests.py` script executes end-to-end integration tests using the compiled `audiocpp_cli` binary. It validates that entire pipelines (TTS, ASR, VAD, separation) execute correctly and do not crash or produce malformed audio.

**Command:**

```bash
python tools/audiocpp_cli/run_audiocpp_cli_path_tests.py \
  --models-root models \
  --backend vulkan \
  --out out \
  --log \
  --audiocpp-cli-bin ./result/bin/audiocpp_cli
```

**Filtering by Family:**
If you only want to test a specific model architecture (e.g., `qwen3_tts` or `ace_step`), use the `--family` flag to save time:

```bash
python tools/audiocpp_cli/run_audiocpp_cli_path_tests.py --family qwen3_tts ...
```

## 3. Running Throughput Benchmarks (`warmbench.py`)

The `warmbench.py` script performs rigorous throughput, memory usage, and latency testing. Unlike the CLI tests, this script targets dedicated testing binaries (`*_warm_bench`) rather than the monolithic CLI, allowing it to bypass system overhead and measure raw inference loop speeds.

We provide a convenient wrapper script, `tools/bench.py`, to simplify the setup and execution of these benchmarks.

### Setup for Warmbench

Because `warmbench.py` targets testing binaries compiled alongside `audiocpp_cli`, you must explicitly compile them.

1. **Compile the Warmbench Binaries:**
   Use the benchmarking CLI to automatically trigger the build script for your platform:

   ```bash
   ./tools/bench.py setup --platform linux --backend vulkan
   ```

### Running Warmbench

Once compiled, you can run the benchmarking tool using the wrapper:

```bash
./tools/bench.py warmbench --family qwen3_tts --backend vulkan
```

### A/B Performance Testing (Baseline Comparison)

If you are optimizing the C++ backend and want to measure your improvements against a known baseline, you can pass a path to a pre-compiled baseline binary. The tool will run the benchmark for both builds and output a Markdown-ready performance report!

```bash
./tools/bench.py warmbench \
  --family qwen3_tts \
  --backend vulkan \
  --baseline-bin /path/to/baseline/build/bin/audiocpp_cli
```

### Warmbench Output & Reports

`warmbench` will generate extensive JSON outputs and telemetry logs (`.timing.log`) inside the `out/` directory, detailing:

- `peak_rss_kb` and `peak_vms_kb` (Peak memory usage)
- `total_ms` (End-to-end latency)
- `compute_ms` (Raw graph execution time)
- Highly granular step timings for autoregressive loops and graph building phases

When running an A/B benchmark, a detailed Markdown report will automatically be generated in your terminal at the end of the run. You can also manually re-generate this Markdown report (perfect for pasting into GitHub Pull Requests) from any `summary.json` file:

```bash
./tools/bench.py report --summary-json out/summary.json
```