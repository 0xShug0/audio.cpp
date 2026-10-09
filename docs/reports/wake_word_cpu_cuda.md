# Wake-word CPU and CUDA measurements

F32 GGUFs only. Ryzen 9 7900X, RTX 5090, four CPU threads, Debug build,
`--log` enabled. Each measurement used one warmup followed by five requests
in the same session; the table reports their median. Runs were sequential.
Timing includes the frontend and inference, but excludes model loading and
WAV decoding. RTF is inference seconds divided by input seconds.

| Model / runtime | Before (ms) | After (ms) | After RTF | Speedup |
| --- | ---: | ---: | ---: | ---: |
| microWakeWord C++ CPU | 5.325 | 5.145 | 0.001715 | 1.03x |
| microWakeWord C++ CUDA | 21.954 | 15.409 | 0.005136 | 1.42x |
| microWakeWord Python CPU | --- | 3.037 | 0.001012 | --- |
| Sherpa KWS C++ CPU | 93.623 | 65.217 | 0.009844 | 1.44x |
| Sherpa KWS C++ CUDA | 87.526 | 64.577 | 0.009747 | 1.36x |
| Sherpa KWS Python CPU | --- | 59.430 | 0.008971 | --- |
| Sherpa KWS Python CUDA | --- | 210.121 | 0.031716 | --- |

microWakeWord used the 3-second `speech.wav` from `pymicro-features`.
The Python reference used LiteRT with the original INT8 Okay Nabu TFLite
model and `pymicro_features`, following upstream input quantization and
output scaling. The C++ package contains dequantized F32 weights and preserves
the activation quantization boundaries. This is not an F32-versus-F32
comparison. There is no Python CUDA result for this TFLite reference.

Sherpa used `en_0.wav` (6.625 seconds) from
`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`, the original F32 ONNX weights,
four active hypotheses, keyword score 1, threshold 0.25, one trailing blank,
the supplied keyword file, and 0.66 seconds of final padding. Python used
the official sherpa-onnx 1.13.8 API. The CUDA reference used its official
CUDA 12 / cuDNN 9 / ONNX Runtime 1.28.2 wheel, not the CPU wheel's fallback.

The CPU performance gap is reduced, not eliminated: microWakeWord C++ takes
about 1.69x the Python time, and Sherpa C++ about 1.10x. Sherpa C++ CUDA is
3.25x faster than Python CUDA. CPU remains preferable for microWakeWord on
this machine; moving its small sequential graph to a GPU does not make it faster.

## Peak CUDA memory

Measured separately with the existing path-test NVML resource monitor at a
5 ms sampling interval. Each process handled one warmup and five repeated
requests using the same inputs and settings above. Peak covers process startup,
model loading, and inference, includes CUDA runtime/context allocations, and
excludes other GPU processes. These are sampled process peaks, not exact
allocator high-water marks. No pre-optimization VRAM baseline was measured.

| Runtime | Warm inference in monitored run | RTF | Observed peak VRAM |
| --- | ---: | ---: | ---: |
| microWakeWord C++ F32 CUDA | 15.067 ms | 0.005022 | 544 MiB |
| Sherpa C++ F32 CUDA | 64.673 ms | 0.009762 | 672 MiB |
| Sherpa Python F32 CUDA | 208.899 ms | 0.031532 | 636 MiB |

Sherpa C++ uses 36 MiB more peak VRAM than Python in this test, while running
3.23x faster. microWakeWord has no official Python CUDA reference in this test.

## Peak CPU memory

| CPU runtime | Observed peak RAM (RSS) |
| --- | ---: |
| microWakeWord C++ F32 | 367.66 MiB |
| microWakeWord Python LiteRT INT8 | 41.27 MiB |
| Sherpa C++ F32 | 378.79 MiB |
| Sherpa Python F32 | 87.42 MiB |

Measured sequentially at a 5 ms sampling interval, including process startup,
model loading, one warmup, and five requests. C++ was launched directly so
the monitoring Python process is not included. The Debug C++ binary includes
CPU, CUDA, and Vulkan support but selects `--backend cpu`; this is not a
CPU-only-build comparison. All four runs showed zero process VRAM usage.
These are whole-process RSS peaks, not model tensor allocation sizes.

## Changes and correctness

- Recurrent state stays in session-owned cached graphs instead of crossing
  the host/backend boundary on every step. Reset explicitly zeros the state.
- Sherpa's two pointwise convolutions use framework linear modules.
- Sherpa skips zero filterbank coefficients while preserving F32 accumulation
  order, and reuses predictor results until the token context changes.
- No framework, ggml, or kernel changes; no graph optimizer.

CPU and CUDA detection JSON matched the original code byte-for-byte across
three recorded Okay Nabu fixtures, speech/music/silence inputs, nine Sherpa
English/Chinese fixtures, option changes, and repeated requests. Streaming
and offline detections matched for the checked clips. microWakeWord's
logged CPU/CUDA probability sequences also matched the original code exactly.
Vulkan correctness was checked separately; its performance is not the focus
of this report.

Against Python, microWakeWord matched all four wake events and their sample
positions in the three recorded fixtures. Sherpa matched keyword labels and
counts on all nine fixtures. This is not a blanket claim of exact timestamp
parity: on `zh_3.wav`, the first keyword starts at 0.48 seconds in C++ versus
0.64 seconds in Python. The same difference was reproduced with the original
C++ implementation. Python's timestamps after a stream reset are relative to
that reset; C++ reports absolute sample positions.

## Reproducing the C++ checks

The existing CLI path-test catalog contains `micro_wake_word_requests`,
`micro_wake_word_streaming`, `sherpa_kws_requests`, and `sherpa_kws_streaming`.
Place the upstream fixtures in `resources/wakeword/micro` and
`resources/wakeword/sherpa`:

- [Okay Nabu recordings](https://github.com/OHF-Voice/pymicro-wakeword/tree/main/tests/okay_nabu): rename `1.wav` through `3.wav` to `okay_nabu_1.wav` through `okay_nabu_3.wav`.
- [Micro frontend fixtures](https://github.com/OHF-Voice/pymicro-features/tree/main/tests): `speech.wav`, `music.wav`, `silence.wav`.
- The Sherpa checkpoint's `test_wavs` directory: `en_0.wav`, `en_1.wav`, and `zh_0.wav` through `zh_6.wav`.

```bash
conda run -n qwen3-tts python tools/audiocpp_cli/run_audiocpp_cli_path_tests.py \
  --audiocpp-cli-bin "$PWD/build/debug/bin/audiocpp_cli" \
  --models-root /path/to/audio.cpp-gguf --backend cpu --threads 4 --log \
  --only micro_wake_word_requests,micro_wake_word_streaming,sherpa_kws_requests,sherpa_kws_streaming
```

Run again with `--backend cuda`. For warmed timing, use a CLI request sequence
containing the same clip six times and take the median of `session.wall_ms`
after discarding the first request.
