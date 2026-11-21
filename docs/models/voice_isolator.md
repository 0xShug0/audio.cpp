# Voice Isolator

`voice_isolator` ports Inworld's Voice Isolator v1, a 48 kHz speech isolation model: it keeps the foreground voice and removes background sound. The release has two checkpoints that share one network. The offline checkpoint is two-sided and renders a recording in overlapping windows; the realtime checkpoint is causal and streams with a trained look-ahead.

Upstream: [inworld/voice-isolator, pinned revision](https://huggingface.co/inworld/voice-isolator/tree/9b0285b3fe8403aa8e2f1dbb2165a80a4294c970). The weights are Apache-2.0; see [model licenses](../model_licenses.md).

| Checkpoint | Package | Mode | Look-ahead |
|---|---|---|---|
| Realtime | `voice_isolator_realtime_f32` (default) | Streaming, or offline over a whole file | 2 frames (20 ms) by default; 0, 2 or 4 |
| Offline | `voice_isolator_offline_f32` | Offline, two-sided | None |

Both GGUF packages install into `models/voice-isolator-GGUF` from [Etherll/voice-isolator-GGUF](https://huggingface.co/Etherll/voice-isolator-GGUF). `voice_isolator_safetensors` installs the two upstream files, `denoiser.safetensors` and `denoiser_realtime.safetensors`, into `models/voice-isolator`.

## Run

Select the `s2s` task. Input may be any sample rate and channel count; it is mixed to mono and resampled to 48 kHz, and the output is 48 kHz mono.

```bash
python tools/model_manager_v2.py install voice_isolator_realtime_f32 --models-root models
python tools/model_manager_v2.py install voice_isolator_offline_f32 --models-root models

build/debug/bin/audiocpp_cli --task s2s --family voice_isolator \
  --model models/voice-isolator-GGUF/voice-isolator-offline-f32.gguf \
  --backend cpu --threads 8 --audio noisy.wav --out isolated.wav --log
```

Add `--mode streaming` for incremental processing:

```bash
build/debug/bin/audiocpp_cli --task s2s --mode streaming --family voice_isolator \
  --model models/voice-isolator-GGUF/voice-isolator-realtime-f32.gguf \
  --backend cpu --threads 8 --audio noisy.wav --out isolated.wav --log
```

Live PCM input uses the standard CLI interface:

```bash
cat microphone.s16le | build/debug/bin/audiocpp_cli \
  --task s2s --mode streaming --family voice_isolator \
  --model models/voice-isolator-GGUF/voice-isolator-realtime-f32.gguf \
  --backend cpu --audio - --input-format s16le --input-rate 48000 \
  --input-channels 1 --out isolated.wav --log
```

A GGUF holds one checkpoint, so pass the file as `--model`. The realtime GGUF without `--mode streaming` runs the causal model over a whole file. The offline checkpoint cannot stream.

With the safetensors directory `models/voice-isolator` as `--model`, streaming loads the realtime checkpoint and offline loads the offline one (or the realtime one when it is the only file present). Passing one of its files as `--model` selects that checkpoint.

## Session options

Pass each as `--session-option voice_isolator.<name>=<value>`.

| Option | Default | Effect |
|---|---|---|
| `lookahead_frames` | -1 (checkpoint default, 2) | Realtime checkpoint: look-ahead of 0, 2 or 4 frames of 10 ms. |
| `block_frames` | 4 | Realtime checkpoint: frames computed together when a chunk brings several. Adds no latency. |
| `attention_cache_type` | `native`: Q8_0 on CPU, F16 elsewhere | Streaming attention keys and values: `f32`, `f16`, `q8_0`, `q5_0` or `q4_0`. Every frame reads the whole 12 s cache, so smaller storage streams faster. |
| `weight_type` | `native`: BF16 on CPUs with AVX512-BF16, F32 elsewhere | Projection weights: `f32` or `bf16`. Without AVX512-BF16, BF16 is slower than F32 on CPU, and it does not speed up CUDA. |
| `audio_chunk_duration_sec` | 0 (checkpoint's own, 12 s) | Offline checkpoint: render window. |
| `audio_chunk_overlap_sec` | 0 (checkpoint's own, 2 s) | Offline checkpoint: overlap between windows. |

Streaming latency is 2048 + 480 × look-ahead samples at 48 kHz: the STFT window plus one 480-sample hop per look-ahead frame.

| Look-ahead | Latency |
|---|---:|
| 0 frames | 42.7 ms |
| 2 frames (default) | 62.7 ms |
| 4 frames | 82.7 ms |

## Accuracy

Largest absolute sample difference against the Python reference (`inworld.voice_isolation` 1.0.0) on CPU:

| Setting | Fixture cases | 20 s stream |
|---|---:|---:|
| Offline, F32 weights | 1.5e-7 | |
| Stream, F32 weights and cache | 2.1e-7 | 1.8e-7 |
| Stream, Q8_0 cache (CPU default) | 6.5e-7 | 1.5e-5 |
| Stream, Q5_0 cache | | 1.2e-4 |
| Stream, F16 cache | | 3.5e-4 |
| Stream, Q4_0 cache | | 6.0e-4 |
| BF16 weights (AVX512-BF16 CPU default) | 1.5e-4 | 1.4e-4 |

ggml's CUDA backend runs F32 matmuls on TF32 tensor cores, so CUDA output matches the reference to about 1.5e-4. Vulkan (RTX 5090) matches it to about 3.5e-4.

Both checkpoints also run in the browser with WASM64 and WebGPU. Use the
[WebGPU build guide](../build/webgpu.md) with `AUDIOCPP_MODELS=voice_isolator`,
the corresponding F32 GGUF, `--backend webgpu` and `--log`. WebGPU uses the
existing decomposed Mamba operations and materializes Q/K/V through the
framework's flash-attention lowering to avoid overlapping storage bindings.
See the [browser validation report](../reports/webgpu_browser_validation.md)
for measured results.

## Performance

Real-time factor (RTF) is processing time divided by audio duration, so values below 1.0 run faster than real time. On CPU (AMD Ryzen 9 9950X3D with AVX512-BF16, 16 threads, default Q8_0 cache and 4-frame blocks, best of three, `voice_isolator_warm_bench --backend cpu --threads 16`):

| Case | F32 weights | BF16 weights | Push p50 / p95 (BF16) |
|---|---:|---:|---:|
| Offline, 8 s clip | 0.63 | 0.60 | |
| Offline, 30 s clip | 0.83 | 0.77 | |
| Stream, 100 ms pushes | 0.82 | 0.76 | 78 / 87 ms |
| Stream, 40 ms pushes | 0.81 | 0.76 | 31 / 35 ms |
| Stream, 10 ms pushes | 1.21 | 1.02 | 10.7 / 11.9 ms |
| Stream, 100 ms pushes, 10-frame blocks | 0.69 | 0.64 | 66 / 73 ms |

Offline audio is rendered in 12 s windows that overlap by 2 s, so a clip a little longer than one window costs two: a 14 s clip runs at about 1.0 with BF16 weights, and a 418 s recording at 0.78. A 10 ms push runs one frame per graph and does not keep up with real time on this CPU; pushes of 40 ms or more do. The CLI feeds a stream in chunks of `block_frames` frames, 40 ms by default.

On CUDA (NVIDIA RTX 5090, F32 weights, default F16 cache, best of three, `voice_isolator_warm_bench --backend cuda`):

| Case | RTF | Push p50 / p95 |
|---|---:|---:|
| Offline, 8 s clip | 0.017 | |
| Offline, 30 s clip | 0.021 | |
| Stream, 100 ms pushes | 0.050 | 4.9 / 6.0 ms |
| Stream, 40 ms pushes | 0.045 | 1.7 / 2.4 ms |
| Stream, 10 ms pushes | 0.14 | 1.4 / 1.5 ms |
| Stream, 100 ms pushes, 10-frame blocks | 0.026 | 2.4 / 3.7 ms |

On Vulkan with the same GPU and the same settings: offline 30 s at 0.047, 40 ms pushes at 0.22 and 10 ms pushes at 0.43.

Streams run 20 s, past the 12 s attention window. The attention cache is a ring, so a push never moves cached keys. Every 4096 frames (41 s) the cached keys are rotated back to keep their rotary positions small; that push takes longer than the others.

Peak memory, rendering 300 s offline or streaming with the default cache:

| Case | CPU RAM | CUDA VRAM |
|---|---:|---:|
| Offline, F32 weights | 1.1 GB | 1.5 GB |
| Offline, BF16 weights | 1.0 GB | 1.4 GB |
| Stream | 540 MB | 1.4 GB |
| Stream, Q4_0 cache | 430 MB | 1.2 GB |
| Stream, BF16 weights and Q4_0 cache | 310 MB | 1.1 GB |

Offline memory does not grow with the recording beyond its samples: each window runs every block over slices of about 4096 frame-band positions. The stream's 12 s attention cache is 240 MB at Q8_0 (875 MB at F32).

## GGUF

The hosted GGUFs are made with `tests/voice_isolator/convert_gguf.py`, which packages either checkpoint as a self-contained F32 GGUF with the model spec and the checkpoint's architecture embedded:

```bash
python tests/voice_isolator/convert_gguf.py \
  --source models/voice-isolator/denoiser_realtime.safetensors \
  --output models/voice-isolator-gguf/voice-isolator-realtime-f32.gguf
```

## Limits

- The realtime checkpoint attends to the last 1200 frames (12 s of audio); older context is not used.
- Output is mono 48 kHz.
- On CPU, Vulkan and Metal the Mamba layers use ggml's fused Mamba-1 operations. CUDA, HIP and WebGPU use the older scan and convolution operations.

Validation commands are in [tests/voice_isolator/README.md](../../tests/voice_isolator/README.md).
