# GLM-TTS validation

This report covers the community GLM-TTS implementation, its prepared
safetensors package, and a standalone mixed Q8/F16 GGUF. Measurements were
captured per request, with deterministic Python parity completed before the
normal sampled path was measured.

## Implementation scope

- Adds the community `glm_tts` loader and offline `tts` / `clon` session.
- Runs the ChatGLM tokenizer, Llama speech-token generator, Whisper-VQ
  reference encoder, CAMPPlus speaker encoder, Flow/DiT mel generator, and
  HiFT vocoder natively through the shared framework.
- Adds the `glm_tts` model-manager package, package spec, CLI/server registry
  entry, user documentation, converter, and validation probes.
- Supports prepared safetensors packages and standalone multi-component GGUF
  files with embedded configs, tokenizer files, and package metadata.
- Keeps reusable component runtimes resident, caches prepared references,
  reuses shape-compatible graphs, and restricts the Llama output projection to
  GLM-TTS audio-token rows.
- Extends the shared OuteTTS Llama runtime and Whisper embedding surface.
  GLM-TTS's CAMPPlus partial-segment normalization is additive and explicitly
  enabled only by the GLM-TTS session; the established shared default remains
  unchanged for Chatterbox, IndexTTS2, and Seed-VC.

## Test environment

Tests were run on 2026-07-24 with:

- Windows 11 and MSVC 2022 14.43
- NVIDIA RTX 3090 24 GB
- CUDA toolkit 12.4.131
- PyTorch 2.5.1+cu121 and Transformers 5.13.0
- GLM-TTS branch based on audio.cpp main `f8fb0c1`

The official Python reference was checked out at:

```text
..\_reference\GLM-TTS
```

The prepared package id and paths were:

```text
model-manager package: glm_tts
safetensors package:   ..\models\GLM-TTS
standalone GGUF:       ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf
reference WAV:         ..\SAMPLES\EN_2.wav
reference transcript:  If you actually care about security
```

Install the package with:

```powershell
python tools\model_manager_deprecated.py install glm_tts --models-dir ..\models
```

The installer downloads `zai-org/GLM-TTS`, converts Flow and HiFT from the
official PyTorch checkpoints to safetensors, exports the tokenizer resources,
and installs matching CAMPPlus safetensors weights.

## Exact build commands

CPU:

```powershell
.\scripts\build_windows.ps1 `
  -Preset windows-cpu-release -ConfigureOnly

cmake -S . -B build\windows-cpu-release `
  -DENGINE_BUILD_WARMBENCH=ON

cmake --build build\windows-cpu-release --parallel 8 --target `
  audiocpp_cli audiocpp_server `
  campplus_shared_default_probe `
  glm_tts_warm_bench glm_tts_llama_probe `
  glm_tts_frontend_probe glm_tts_campplus_probe `
  glm_tts_flow_conditioned_probe
```

CUDA:

```powershell
$cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:CUDA_PATH = $cuda
$env:CUDAToolkit_ROOT = $cuda
$env:CUDA_TOOLKIT_ROOT_DIR = $cuda
$env:CUDACXX = Join-Path $cuda "bin\nvcc.exe"
$env:CMAKE_CUDA_COMPILER = $env:CUDACXX
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"

.\scripts\build_windows.ps1 `
  -Preset windows-cuda-release -ConfigureOnly

cmake -S . -B build\windows-cuda-release `
  -DENGINE_BUILD_WARMBENCH=ON `
  -DCMAKE_CUDA_COMPILER="$env:CUDACXX"

cmake --build build\windows-cuda-release --parallel 8 --target `
  audiocpp_cli audiocpp_server audiocpp_gguf `
  campplus_shared_default_probe `
  glm_tts_warm_bench glm_tts_llama_probe `
  glm_tts_frontend_probe glm_tts_campplus_probe `
  glm_tts_flow_conditioned_probe
```

Both clean builds completed. The loader catalog advertises `glm_tts` with
offline `tts` and `clon` tasks and the `/v1/audio/speech` endpoint.

## Standalone GGUF

The quality-tested package keeps the autoregressive Llama group in F16 and
stores the speech tokenizer, Flow, HiFT, and CAMPPlus groups as Q8_0:

```powershell
.\build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --input llama_weights=..\models\GLM-TTS\llm\model.safetensors.index.json `
  --input speech_tokenizer_weights=..\models\GLM-TTS\speech_tokenizer\model.safetensors `
  --input flow_weights=..\models\GLM-TTS\flow\model.safetensors `
  --input hift_weights=..\models\GLM-TTS\hift\model.safetensors `
  --input campplus_weights=..\models\GLM-TTS\frontend\campplus.safetensors `
  --root ..\models\GLM-TTS `
  --family glm_tts `
  --model-spec .\model_specs\glm_tts.json `
  --type q8_0 `
  --keep-type "llama_weights/*=f16" `
  --overwrite `
  --output ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf
```

Inspection:

```powershell
.\build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --inspect ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf
```

```text
tensors=2076
rank0_scalars=122
embedded_sidecars=true
embedded_model_spec=true
model_spec_family=glm_tts
namespace=campplus_weights
namespace=flow_weights
namespace=hift_weights
namespace=llama_weights
namespace=speech_tokenizer_weights
```

The file is 5,143,813,728 bytes and has SHA-256
`28E2B66333FD5F051466520F6A8A5F26CBA7F3B768891006168A4560972107CB`.
It needs no external model spec, tokenizer, config, or weight file.

## Exact run commands

Standalone CLI:

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task clon --family glm_tts `
  --model ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf `
  --backend cuda `
  --voice-ref ..\SAMPLES\EN_2.wav `
  --reference-text "If you actually care about security" `
  --text "Hello from G L M T T S. This voice was generated locally." `
  --top-k 25 --top-p 0.8 --temperature 1.0 `
  --seed 0 --max-tokens 256 `
  --out ..\outputs\glm_tts_pr_validation\standalone_q8_cli.wav `
  --log
```

The one-file path was materialized to audio.cpp's temporary GGUF resource
directory, produced 131 speech tokens, 262 mel frames, and 5.24 seconds of
valid 24 kHz mono PCM. `ffmpeg -v error -i ... -f null -` completed without an
error. The output SHA-256 is
`DF8ACFDCD308EA10FDBEF37D03581DAA4E2222922E2DC54A3DD417EEF0263450`.

Server configuration:

```json
{
  "host": "127.0.0.1",
  "port": 18083,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "lazy_load": false,
  "models": [
    {
      "id": "glm-tts-q8",
      "family": "glm_tts",
      "path": "E:/models/GLM-TTS-Q8/GLM-TTS_Q8.gguf",
      "task": "clon",
      "mode": "offline",
      "default_voice_preset": {
        "voice_ref": "E:/samples/EN_2.wav",
        "reference_text": "If you actually care about security"
      }
    }
  ]
}
```

```powershell
.\build\windows-cuda-release\bin\audiocpp_server.exe `
  --config ..\outputs\glm_tts_pr_validation\server.cuda.json `
  --log

curl.exe http://127.0.0.1:18083/v1/audio/speech `
  -H "Content-Type: application/json" `
  -o ..\outputs\glm_tts_pr_validation\server_q8.wav `
  -d '{\"model\":\"glm-tts-q8\",\"input\":\"Hello from the GLM TTS server.\",\"max_tokens\":256,\"seed\":0,\"temperature\":1.0,\"top_k\":25,\"top_p\":0.8}'
```

`GET /health` returned
`{"status":"ok","backend":"cuda","models":1}`. The speech request produced a
valid WAV with SHA-256
`9C818AB9804FF92A04F0EC5AB22B06B45F6E9DA60704F1A851C1367787F5169C`.

## Deterministic Python parity

The official model uses RAS sampling with temperature 1.0, top-k 25, and
top-p 0.8. Those defaults remain unchanged for normal synthesis. For logic
parity only, the upstream RAS selection and audio.cpp sampling were changed to
argmax. Native FP32 weights were used on both sides. This isolates prompt,
token, duration, Flow, and vocoder logic from stochastic sampling drift.

Python:

```powershell
python tests\glm_tts\reference_warm_bench.py `
  ..\_reference\GLM-TTS `
  --request-file tests\glm_tts\parity_greedy_requests.json `
  --voice-ref ..\SAMPLES\EN_2.wav `
  --reference-text "If you actually care about security" `
  --out-dir ..\outputs\glm_tts_pr_validation\python_f32_greedy `
  --greedy
```

audio.cpp:

```powershell
.\build\windows-cuda-release\bin\glm_tts_warm_bench.exe `
  --model ..\models\GLM-TTS `
  --backend cuda --threads 8 `
  --voice-ref ..\SAMPLES\EN_2.wav `
  --reference-text "If you actually care about security" `
  --request-file tests\glm_tts\parity_greedy_requests.json `
  --iterations 1 `
  --session-option glm_tts.weight_type=f32 `
  --audio-out-dir ..\outputs\glm_tts_pr_validation\cpp_f32_greedy
```

Per-request results:

| Request | Speech tokens C++ / Python | WAV frames C++ / Python | WAV cosine | Log-mel cosine | C++ wall | Python wall |
|---|---:|---:|---:|---:|---:|---:|
| `short_cold` | 75 / 75, exact | 72,000 / 72,000 | 0.150483 | 0.992423 | 2,684.2 ms | 2,687.7 ms |
| `short_repeat` | 75 / 75, exact | 72,000 / 72,000 | 0.150483 | 0.992423 | 887.8 ms | 1,929.9 ms |
| `technical` | 139 / 139, exact | 133,440 / 133,440 | 0.144861 | 0.991100 | 1,442.5 ms | 3,208.2 ms |
| `longform` | 305 / 305, exact | 292,800 / 292,800 | -0.012583 | 0.991946 | 3,161.7 ms | 6,505.0 ms |

The low waveform cosine is phase-sensitive and reflects independent HiFT
excitation. Frame counts and token ids are exact, while log-mel similarity is
0.9911 to 0.9924. No parity-only trace dumping is enabled during normal
requests.

### Component boundaries

The component probes use fixed inputs and fixed Flow noise:

| Component | Compared boundary | Result |
|---|---|---|
| Text tokenizer and prompt | normalized strings and full prompt ids | exact |
| Whisper-VQ | reference speech-token ids | 63 / 63 exact |
| Llama greedy | generated speech-token ids | exact for 75, 139, and 305-token requests |
| CAMPPlus | 192-value speaker embedding | cosine `0.999999999977`; max abs `2.52e-05`; RMSE `8.86e-06` |
| CAMPPlus shared default | deterministic 250 x 80 input against unmodified main | byte-for-byte exact; max abs `0` |
| Flow, 10 steps | 114 x 80 mel | cosine `0.999999999231`; max abs `0.001515`; RMSE `0.000279` |

`CampplusEncoderConfig::normalize_partial_segment_by_full_length` defaults to
`false`. Only GLM-TTS sets it to `true` to match the published GLM-TTS
CAMPPlus ONNX graph. Chatterbox, IndexTTS2, Seed-VC, and other shared callers
continue through the previous pooling graph. The regression probe
`campplus_shared_default_probe` was compiled once against unmodified main and
once against this branch; both emitted the same 192 serialized values.

The request options `flow_noise_path`, `hift_source_random_path`, and
`hift_prior_noise_count` expose stochastic boundaries for targeted parity
tests without adding normal-request log spam.

## Normal sampled-path parity

The normal official settings were then restored: RAS, temperature 1.0, top-k
25, top-p 0.8, and seed 0. Native FP32 and the mixed Q8/F16 GGUF were compared
against the official FP32 output per request:

| Request | Native FP32 log-mel / frames | Mixed GGUF log-mel / frames |
|---|---:|---:|
| `short_cold` | 0.902897; 77,760 / 89,280 | 0.902589; 77,760 / 89,280 |
| `short_repeat` | 0.902897; 77,760 / 89,280 | 0.902589; 77,760 / 89,280 |
| `technical` | 0.929266; 150,720 / 147,840 | 0.857288; 135,840 / 147,840 |
| `longform` | 0.862186; 313,920 / 306,240 | 0.861948; 313,920 / 306,240 |

For the short sampled request, token ids matched Python through token 37 and
then diverged stochastically. The deterministic tests above demonstrate that
the model logic and frame accounting match when sampling variation is removed.

## Per-request performance

The normal mixed Q8/F16 matrix was run in three fresh processes. Each request
was executed twice inside one long-lived session. Iteration 1 can build a new
shape-specific graph; iteration 2 measures reuse of that request shape.

```powershell
.\build\windows-cuda-release\bin\glm_tts_warm_bench.exe `
  --model ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf `
  --backend cuda --threads 8 `
  --voice-ref ..\SAMPLES\EN_2.wav `
  --reference-text "If you actually care about security" `
  --request-file tests\glm_tts\warm_bench_requests.json `
  --audio-out-dir ..\outputs\glm_tts_pr_validation\q8_run1\audio `
  --log-file ..\outputs\glm_tts_pr_validation\q8_run1\trace.log
```

Mean and sample standard deviation across the three processes:

| Request | Iteration | Audio | Wall mean ± SD | RTF mean ± SD |
|---|---:|---:|---:|---:|
| `short_cold` | 1, reference miss | 3.24 s | 2,469.5 ± 637.1 ms | 0.7622 ± 0.1966 |
| `short_cold` | 2 | 3.24 s | 626.2 ± 2.4 ms | 0.1933 ± 0.0008 |
| `short_repeat` | 1 | 3.24 s | 617.1 ± 2.4 ms | 0.1905 ± 0.0007 |
| `short_repeat` | 2 | 3.24 s | 618.1 ± 2.2 ms | 0.1908 ± 0.0007 |
| `technical` | 1 | 5.66 s | 1,023.2 ± 2.6 ms | 0.1808 ± 0.0005 |
| `technical` | 2 | 5.66 s | 1,002.8 ± 2.6 ms | 0.1772 ± 0.0005 |
| `longform` | 1 | 13.08 s | 2,231.0 ± 3.6 ms | 0.1706 ± 0.0003 |
| `longform` | 2 | 13.08 s | 2,189.1 ± 6.1 ms | 0.1674 ± 0.0005 |

The four Python sampled requests took 3,314.2, 2,429.6, 3,808.6, and
7,151.9 ms respectively. Using the comparable reference-cached audio.cpp
requests, the mixed GGUF path was approximately 3.2x to 3.9x faster than the
official Python reference.

All 24 WAVs from the three mixed-GGUF runs reduced to three hashes: one shared
by the two identical short cases, one for `technical`, and one for `longform`.
The repeated outputs are byte-identical across processes.

### Memory and mem-saver tradeoff

Memory was sampled every 200 ms with `nvidia-smi`:

| Path | Peak GPU memory | Warm repeated short request |
|---|---:|---:|
| Native FP32 audio.cpp | 8,846 MiB | 887.8 ms |
| Official Python FP32 | 8,628 to 8,787 MiB | 1,929.9 ms |
| Mixed GGUF, resident | 5,320 MiB | 608.6 ms |
| Mixed GGUF, balanced `glm_tts.mem_saver=true` | 3,878 MiB | 622.3 ms |
| Mixed GGUF, `glm_tts.aggressive_mem_saver=true` | 3,372 MiB | 3,761.7 ms |

Balanced mem-saver releases the approximately 1.58 GiB reference-only
Whisper-VQ and CAMPPlus path after its output is cached, but keeps Llama, Flow,
and HiFT warm. Across three warm repeats it saved 1,442 MiB (27.1%) versus
resident mode while adding 13.7 ms (2.3%) mean latency. A reference-cache miss
releases the generation path before reconstructing the reference encoders so
the groups do not overlap in VRAM.

Aggressive mem-saver preserves the former unload-every-stage policy. It saved
another 506 MiB, but rebuilding Llama, Flow, and HiFT made warm requests about
6.0x slower than balanced mode. All balanced, aggressive, and resident
outputs reduced to the same SHA-256 hash:
`72672D49755B1B3987318EC264FBEC0DBB0AA0175461818AAC56BD1404F56E88`.

Windows reported peak working set around 25.0 GiB and private bytes between
33.8 and 44.5 GiB while loading this model. These counters include mapped GGUF
pages, converted host tensors, and backend virtual reservations; private bytes
must not be interpreted as committed physical RAM.

## CPU path

A full short request was executed with the CPU build:

```powershell
.\build\windows-cpu-release\bin\glm_tts_warm_bench.exe `
  --model ..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf `
  --backend cpu --threads 16 `
  --voice-ref ..\SAMPLES\EN_2.wav `
  --reference-text "If you actually care about security" `
  --text "Hello." --iterations 1 `
  --audio-out-dir ..\outputs\glm_tts_pr_validation\cpu_short\audio
```

It produced 0.64 seconds of valid audio in 4,963.3 ms, RTF 7.755. CPU is
functional, but CUDA is the practical tested backend for interactive use.

## Generated artifacts

```text
..\outputs\glm_tts_pr_validation\standalone_q8_cli.wav
..\outputs\glm_tts_pr_validation\server_q8.wav
..\outputs\glm_tts_pr_validation\python_f32
..\outputs\glm_tts_pr_validation\python_f32_greedy
..\outputs\glm_tts_pr_validation\cpp_f32
..\outputs\glm_tts_pr_validation\cpp_f32_greedy
..\outputs\glm_tts_pr_validation\q8_run1
..\outputs\glm_tts_pr_validation\q8_run2
..\outputs\glm_tts_pr_validation\q8_run3
..\outputs\glm_tts_pr_validation\components
..\outputs\glm_tts_pr_validation\cpu_short
..\models\GLM-TTS-Q8\GLM-TTS_Q8.gguf
```

## Known limitations

- GLM-TTS supports Chinese and English.
- Both advertised task names use the same zero-shot reference-conditioned
  path. A clean reference WAV and its exact transcript are required.
- Normal sampled runs can diverge from Python because small numeric
  differences alter autoregressive multinomial choices. Greedy FP32 token and
  frame parity is exact.
- Independent HiFT excitation makes waveform cosine phase-sensitive even when
  token ids and frame counts match; log-mel similarity is the useful metric.
- The mixed GGUF keeps Llama in F16 because a fully Q8 autoregressive head had
  a larger quality and token-selection penalty.
- CUDA and CPU were tested. Vulkan and Metal were not tested.
- The upstream WeText normalizer was unavailable in the Windows Python
  environment. Parity uses English text that the identity fallback does not
  alter.
