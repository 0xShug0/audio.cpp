# Kroko Community ASR validation

This report covers the native `kroko_asr` community-model implementation,
public 64-L and 128-L Kroko packages, stateful streaming, word timestamps,
multilingual behavior, and standalone Q8 GGUF loading. The original
sherpa-onnx execution path is used as the reference.

## Implementation scope

- Native Kaldi-compatible filterbank, Conv2dSubsampling/ConvNeXt frontend,
  19-layer Zipformer2, stateless RNN-T predictor, joiner, and greedy decoder.
- Offline and stateful streaming sessions with partial transcripts.
- Word timestamps derived from RNN-T emission frames.
- Public free German, English, Spanish, French, Italian, Hebrew, Dutch,
  Portuguese, Swedish, and Turkish packages.
- Dynamic support for both public chunk layouts: 64-L (`141/128`) and
  128-L (`269/256`).
- Converted safetensors and movable standalone Q8 GGUF packages.
- CLI, generic server task routes, and OpenAI-compatible transcription routes.
- Model-manager installation with a language/size-specific output directory.

## Test environment

Tests were run on 2026-07-27 with:

- Windows 11 Pro Insider Preview 10.0.26220
- AMD Ryzen 9 7950X3D, 16 cores / 32 logical processors
- 63.1 GiB system RAM
- NVIDIA RTX 3090 24,576 MiB, driver 591.86
- CUDA 12.4 and CMake 3.29.2
- `sherpa-onnx` 1.13.4 with CPUExecutionProvider

The native package paths were:

```text
..\models\Kroko-ASR\Kroko-{DE,EN,ES,FR,IT,IW,NL,PT,SV,TR}-Community-64-L-Native
..\models\Kroko-ASR\Kroko-EN-Community-128-L-Native
```

The source packages were downloaded from
[Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR). Only packages
whose header declares `free=true` were used.

## Exact build commands

Configure the CUDA build with the validation probes enabled:

```powershell
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $cuda
$env:CUDAToolkit_ROOT = $cuda
$env:CUDACXX = Join-Path $cuda "bin\nvcc.exe"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"

.\scripts\build_windows.ps1 `
  -Preset windows-cuda-release -ConfigureOnly

cmake -S . -B build\windows-cuda-release `
  -DENGINE_BUILD_WARMBENCH=ON `
  -DCMAKE_CUDA_COMPILER="$env:CUDACXX"
```

The final build command was:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul && cmake --build build\windows-cuda-release --config Release --target audiocpp_cli audiocpp_server audiocpp_gguf kroko_asr_streaming_probe kroko_asr_encoder_probe -j 8'
```

The CLI, server, GGUF converter, and both Kroko probes built successfully.

Loader/package-catalog validation:

```powershell
.\venv\Scripts\python.exe tools\check_loader_catalog_sync.py --self-test
.\venv\Scripts\python.exe tools\check_loader_catalog_sync.py
```

Result:

```text
Ran 2 tests ... OK
active_loaders=37 commented_loaders=0 catalog_packages=48
ok: installable catalog families match registered loaders
```

## Package conversion

Install the conversion and reference dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install `
  numpy onnx onnxruntime safetensors sherpa-onnx `
  soundfile huggingface_hub edge-tts psutil
```

Convert one free package through the model manager:

```powershell
.\venv\Scripts\python.exe tools\model_manager.py install `
  kroko_asr_community_converted `
  --source-file ..\models\Kroko-ASR\Kroko-DE-Community-64-L-Streaming-001.data `
  --models-root ..\models\Kroko-ASR --overwrite
```

The manager infers `Kroko-DE-Community-64-L-Native`; the language packages
therefore do not overwrite each other. The direct equivalent is:

```powershell
.\venv\Scripts\python.exe tools\community_models\convert_kroko_onnx.py `
  ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Streaming-001.data `
  ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native --overwrite
```

Each converted directory contains `config.json`, `model.safetensors`, and
`tokens.txt`.

## Original-model comparison

The test samples and manifest were generated with:

```powershell
.\venv\Scripts\python.exe tests\kroko_asr\generate_multilingual_samples.py `
  --output ..\outputs\kroko_multilingual_samples
```

The matrix invokes the original sherpa-onnx greedy decoder and audio.cpp
separately for every language. It reports exact transcript parity, normalized
WER, ground-truth WER, word-timestamp delta, wall time, RTF, and process RSS.

```powershell
.\venv\Scripts\python.exe tests\kroko_asr\kroko_multilingual_matrix.py `
  --cli build\windows-cuda-release\bin\audiocpp_cli.exe `
  --models-root ..\models\Kroko-ASR `
  --samples ..\outputs\kroko_multilingual_samples `
  --output ..\outputs\kroko_multilingual_results `
  --backend cpu
```

Per-request results:

| Lang | Audio s | Reference ms / RTF | audio.cpp ms / RTF | Exact | Reference WER | Ground-truth WER | Timestamp max delta | Peak RSS MiB |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| de | 5.544 | 103.878 / 0.0187 | 491.524 / 0.0887 | Yes | 0 | 0 | 0.200 us | 1344.1 |
| en | 4.992 | 105.852 / 0.0212 | 463.043 / 0.0928 | Yes | 0 | 0 | 0.162 us | 1343.8 |
| es | 4.800 | 106.780 / 0.0222 | 455.371 / 0.0949 | Yes | 0 | 0 | 0.172 us | 1343.7 |
| fr | 5.688 | 103.348 / 0.0182 | 497.482 / 0.0875 | Yes | 0 | 0 | 0.229 us | 1344.4 |
| it | 4.680 | 103.907 / 0.0222 | 455.115 / 0.0972 | Yes | 0 | 0 | 0.191 us | 1343.7 |
| he | 5.040 | 111.102 / 0.0220 | 469.215 / 0.0931 | Yes | 0 | 0 | 0.162 us | 1343.9 |
| nl | 5.400 | 102.360 / 0.0190 | 477.863 / 0.0885 | Yes | 0 | 0 | 0.153 us | 1343.9 |
| pt | 5.136 | 112.117 / 0.0218 | 480.211 / 0.0935 | Yes | 0 | 0 | 0.305 us | 1343.7 |
| sv | 4.968 | 107.267 / 0.0216 | 455.093 / 0.0916 | Yes | 0 | 0.1667 | 0.181 us | 1336.8 |
| tr | 5.472 | 108.245 / 0.0198 | 488.966 / 0.0894 | Yes | 0 | 0 | 0.200 us | 1344.0 |

`Reference WER` compares audio.cpp with the original decoder. All ten requests
are exact string matches and have zero reference WER. Swedish ground-truth WER
is one token because both paths write `Idag` while the source prompt writes
`I dag`; this is not a port mismatch. Word counts match for every request.
The worst word-start difference is 0.305 microseconds, which is floating-point
representation noise around the same 40 ms frame grid.

The complete machine-readable result and per-language artifacts are:

```text
..\outputs\kroko_multilingual_results\multilingual_parity.json
..\outputs\kroko_multilingual_results\<language>_reference.json
..\outputs\kroko_multilingual_results\<language>_audiocpp.txt
..\outputs\kroko_multilingual_results\<language>_audiocpp_words.json
```

The optimized ONNX Runtime CPU reference is roughly 4.3-4.7 times faster in
this short-request matrix. The native port is still about 10-11 times faster
than real time on CPU.

## Encoder boundary parity

The probes feed two deterministic feature chunks through the native model and
compare the frontend boundary and final encoder boundary with the source ONNX.

128-L English:

```powershell
build\windows-cuda-release\bin\kroko_asr_streaming_probe.exe `
  ..\models\Kroko-ASR\Kroko-EN-Community-128-L-Native `
  ..\outputs\kroko_en128_streaming_final.f32 `
  ..\outputs\kroko_en128_embedding_final.f32

.\venv\Scripts\python.exe tests\kroko_asr\kroko_onnx_streaming_parity.py `
  ..\models\Kroko-ASR\extracted-en-128\encoder.onnx `
  ..\models\Kroko-ASR\Kroko-EN-Community-128-L-Native\config.json `
  ..\outputs\kroko_en128_streaming_final.f32 `
  --native-embedding ..\outputs\kroko_en128_embedding_final.f32
```

64-L Swedish uses the same commands with `extracted-sv-64`,
`Kroko-SV-Community-64-L-Native`, and `kroko_sv64_*`.

| Variant / boundary | Chunk 1 cosine | Chunk 2 cosine |
|---|---:|---:|
| 128-L frontend embedding | 0.999959055 | 0.999957586 |
| 128-L final encoder | 0.999573775 | 0.999731910 |
| 128-L Zipformer with native embedding | 0.999580690 | 0.999710517 |
| 64-L frontend embedding | 0.999954853 | 0.999955603 |
| 64-L final encoder | 0.999739554 | 0.999864404 |
| 64-L Zipformer with native embedding | 0.999724176 | 0.999878200 |

The remaining small numerical drift is expected from executing converted ONNX
operators through ggml rather than ONNX Runtime. End-to-end greedy transcripts
and timestamp frames nevertheless match exactly in the multilingual matrix.

## Streaming and long-audio path

A 49.680-second 16 kHz Swedish sample was run through both modes:

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode streaming --family kroko_asr `
  --model ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --backend cpu --audio ..\outputs\kroko_sv_49s.wav --language sv `
  --text-out ..\outputs\kroko_sv_49s_streaming.txt `
  --words-out ..\outputs\kroko_sv_49s_streaming_words.json --log

build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode offline --family kroko_asr `
  --model ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --backend cpu --audio ..\outputs\kroko_sv_49s.wav --language sv `
  --text-out ..\outputs\kroko_sv_49s_offline.txt `
  --words-out ..\outputs\kroko_sv_49s_offline_words.json --log
```

| Mode | Session ms | RTF | Encoder chunks | Peak retained audio values |
|---|---:|---:|---:|---:|
| Streaming | 3981.768 | 0.0801 | 40 | 38,560 |
| Offline | 3936.170 | 0.0792 | 40 | n/a |

The transcript files and word-timestamp JSON files are byte-identical. The
stream accepted 794,880 samples but retained only 38,560 values at peak,
demonstrating bounded 16 kHz waveform buffering.

## Standalone Q8 GGUF and CUDA

Build:

```powershell
build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --input ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\model.safetensors `
  --root ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native `
  --family kroko_asr --type q8_0 --overwrite `
  --output ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\Kroko-SV-Community-64-L-Q8.gguf
```

Run:

```powershell
build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode streaming --family kroko_asr `
  --model ..\models\Kroko-ASR\Kroko-SV-Community-64-L-Native\Kroko-SV-Community-64-L-Q8.gguf `
  --backend cuda --audio ..\outputs\kroko_multilingual_samples\sv.wav `
  --language sv --text-out ..\outputs\kroko_sv_q8_cuda.txt `
  --words-out ..\outputs\kroko_sv_q8_cuda_words.json --log
```

The 167,754,176-byte GGUF has SHA-256
`2CBC4B2C69217D44FEDCD3B890AD563BA0E4FD9DDF78C105AFCC003D245EBB4F`.
It embeds the config, tokens, and package spec and runs from the file path with
no external sidecars. On CUDA it completed the 4.968-second sample in
462.556 ms (RTF 0.0931). Its transcript and word JSON are byte-identical to
the native safetensors CPU result.

## Server validation

Server configuration:

```json
{
  "host": "127.0.0.1",
  "port": 18080,
  "backend": "cuda",
  "models": [{
    "id": "kroko-sv-stream",
    "family": "kroko_asr",
    "path": "../models/Kroko-ASR/Kroko-SV-Community-64-L-Native/Kroko-SV-Community-64-L-Q8.gguf",
    "task": "asr",
    "mode": "streaming"
  }]
}
```

```powershell
build\windows-cuda-release\bin\audiocpp_server.exe `
  --config ..\outputs\kroko_server_test.json --log

curl.exe -N http://127.0.0.1:18080/v1/audio/transcriptions `
  -F "file=@../outputs/kroko_multilingual_samples/sv.wav" `
  -F "model=kroko-sv-stream" -F "language=sv" `
  -F "stream=true" -F "response_format=json"
```

`GET /health` returned `status=ok`, `backend=cuda`, and one model. Two
long-lived-server requests both returned the exact reference transcript:

| Request | Wall ms | TTFT ms |
|---|---:|---:|
| 1 | 579.880 | 189.824 |
| 2 | 491.622 | 111.753 |

After request 2 the server used 619.5 MiB RSS and 1478.1 MiB private memory.
Reliable per-process CUDA VRAM is not available through Windows WDDM, so only
the physical device capacity is reported. Artifacts are:

```text
..\outputs\kroko_server_request_1.json
..\outputs\kroko_server_request_2.json
..\outputs\kroko_server_stdout_final.log
..\outputs\kroko_server_stderr_final.log
```

The generic `/v1/tasks/run` and `/v1/tasks/stream` result schemas preserve
`word_timestamps`. The OpenAI-compatible transcription route returns its
standard text/delta events and currently does not serialize the model's word
array.

## Backend coverage

| Backend | Coverage |
|---|---|
| CPU | All ten languages; original parity; timestamps; 64-L/128-L probes; offline/streaming long-audio equality |
| CUDA | Standalone Q8 CLI streaming and two-request server session |
| Vulkan | Not tested |
| Metal | Not tested |

## Known limitations

- Conversion accepts only public packages whose header declares `free=true`.
  Commercial/encrypted models require Kroko's licensed runtime.
- Each package recognizes one language. Automatic routing across packages is
  not implemented.
- The decoder implements greedy RNN-T search. Modified beam search, hotwords,
  blank penalty, and automatic endpoint segmentation are not exposed.
- The 16 kHz streaming path has bounded waveform buffering. Resampled input is
  correct but retains the accumulated source waveform until finalization.
- Native CPU execution currently prioritizes parity and is slower than the
  optimized ONNX Runtime reference.
- Vulkan and Metal still require contributor testing.

Source references:

- [kroko-ai/kroko-onnx](https://github.com/kroko-ai/kroko-onnx)
- [Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR)
- [Kroko API documentation](https://docs.kroko.ai/api/)
