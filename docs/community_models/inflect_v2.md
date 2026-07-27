# Inflect Micro v2 and Nano v2

`inflect_v2` provides native GGML inference for the English
[Inflect Micro v2](https://huggingface.co/owensong/Inflect-Micro-v2) and
[Inflect Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2) TTS models.
Both variants use the same runtime and produce 24 kHz mono audio. The initial
integration supports offline FP32 inference only.

The runtime does not use ONNX Runtime. The model manager downloads pinned
official ONNX exports, validates their complete tensor inventories and
dimensions, and writes one `model.safetensors` file locally. PyTorch is not
required.

## Install

Install eSpeak-ng and its English voice data first. On Debian or Ubuntu:

```bash
sudo apt install espeak-ng libespeak-ng1
```

On macOS:

```bash
brew install espeak-ng
```

Then install either model:

```bash
uv run --with onnx --with safetensors python tools/model_manager.py install inflect_micro_v2 --models-root models
uv run --with onnx --with safetensors python tools/model_manager.py install inflect_nano_v2 --models-root models
```

## Run

```bash
audiocpp_cli --task tts --family inflect_v2 \
  --model models/Inflect-Micro-v2 --backend cpu \
  --text "Hello from Inflect Micro version two." \
  --seed 0 --out inflect.wav
```

Use `models/Inflect-Nano-v2` to run Nano. CUDA uses the same model layout and
request surface. CUDA sessions keep duration alignment on the native CPU GGML
path so small TF32 differences cannot change the monotone expansion; the four
flows and waveform decoder execute on CUDA.

If eSpeak-ng is installed outside the dynamic loader's search path, provide
the library and data paths explicitly:

```bash
audiocpp_cli --task tts --family inflect_v2 \
  --model models/Inflect-Micro-v2 --backend cpu \
  --session-option inflect_v2.espeak_library_path=/path/to/libespeak-ng.so \
  --session-option inflect_v2.espeak_data_path=/path/to/espeak-ng-data \
  --text "A configured eSpeak installation." --out inflect.wav
```

On Windows, `espeak-ng.dll` and `libespeak-ng.dll` are searched automatically.
If eSpeak-ng is not installed system-wide, `espeakng-loader` can provide the
external DLL and data directory:

```powershell
$espeakPaths = uv run --with espeakng-loader==0.2.4 python -c `
  "import espeakng_loader; print(espeakng_loader.get_library_path()); print(espeakng_loader.get_data_path())"
$espeakLibrary = $espeakPaths[0]
$espeakData = $espeakPaths[1]

.\audiocpp_cli.exe --task tts --family inflect_v2 `
  --model ../../models/Inflect-Micro-v2 --backend cpu `
  --session-option "inflect_v2.espeak_library_path=$espeakLibrary" `
  --session-option "inflect_v2.espeak_data_path=$espeakData" `
  --text "Hello from Inflect Micro version two." `
  --seed 0 --out inflect.wav
```

Python is used only to locate the external files. Synthesis remains entirely
inside the native audio.cpp process.

## Options

| Option | Range | Default | Meaning |
|---|---:|---:|---|
| `speaking_rate` | `0.5`–`2.0` | `1.0` | Inflect speed multiplier; larger values shorten the output. |
| `variation` | `0.0`–`1.0` | `0.667` | Scale of the seed-based Gaussian latent variation. |
| `seed` | non-negative integer | `0` | Latent noise seed. Long-form chunks use `seed + chunk_index`. |
| `text_chunk_mode` | `word_budget` | `word_budget` | Long-form splitting mode. |
| `text_chunk_size` | positive integer | `280` | Maximum Unicode codepoints per chunk. |

Long-form text is split near punctuation. Adjacent chunks receive 5 ms edge
fades and punctuation-dependent pauses. A request is rejected if its expanded
latent would exceed 4000 frames. The model has one fixed voice and exposes no
language, speaker, cloning, or streaming controls.

## Standalone GGUF

The `inflect_v2` model specification supports standalone GGUF packaging. Pack
the FP32 `model.safetensors` together with `config.json`:

```bash
audiocpp_gguf \
  --input weights=models/Inflect-Nano-v2/model.safetensors \
  --root models/Inflect-Nano-v2 \
  --output models/Inflect-Nano-v2-FP32/model.gguf \
  --type orig --family inflect_v2 \
  --model-spec model_specs/inflect_v2.json
```

eSpeak-ng remains an external runtime dependency and is never embedded in the
GGUF.

FP16 and quantized packages are intentionally unsupported until separate
parity and listening validation is available.

## Validation

The frontend goldens use `phonemizer==3.3.0` and
`espeakng-loader==0.2.4`. Fixed-latent CPU output was checked against the
official FP32 ONNX graphs:

| Variant | Mean absolute error | Maximum error | Correlation |
|---|---:|---:|---:|
| Micro v2 | `1.54e-5` | `9.16e-5` | `0.999999976` |
| Nano v2 | `1.53e-5` | `9.16e-5` | `0.999999964` |

Fixed-input CUDA comparisons produced mean absolute errors of `2.81e-4` and
`3.39e-4`, with correlations of `0.999892` and `0.999832`, respectively.
The component run also covered deterministic seeds, rate control, long-form
output without non-finite samples, the 4000-frame rejection, and standalone
FP32 GGUF loading.

The committed `inflect_v2_tts_longform` path case runs two 783-codepoint
requests through one loaded session. Each request produces 12 chunks and a
48.544-second WAV. On a Ryzen 7 7800X3D and RTX 4090 with CUDA 13.3:

| Backend | Cold request | Repeated request | RTF cold / repeat | Memory |
|---|---:|---:|---:|---:|
| CPU | `11952.7 ms` | `11485.5 ms` | `0.246 / 0.237` | `370.17 MiB` observed peak RSS |
| CUDA | `672.274 ms` | `563.121 ms` | `0.0138 / 0.0116` | Per-process VRAM unavailable from NVML under Windows WDDM |

Within each backend, the two generated WAVs have identical frame counts and
SHA-256 hashes:
`2ff1deaf7abc70b34ebf830d1afac9fd87d70261962b60ed56d8d93b334f2a27`
on CPU and
`b5ef595317439a75a17927e45561ac36120e56f0350347eaf6e068af5d4de4fa`
on CUDA. Trace output records duration- and decoder-graph cache hits; the
retained caches are bounded to four duration shapes and two decoder shapes.

### Reproduce the validation

Run the build commands from an MSVC x64 developer shell:

```powershell
cmake -S . -B build-inflect -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DENGINE_BUILD_TESTS=ON
cmake --build build-inflect --parallel 8 `
  --target audiocpp_cli inflect_v2_frontend_test

cmake -S . -B build-inflect-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DENGINE_BUILD_TESTS=OFF -DGGML_CUDA=ON
cmake --build build-inflect-cuda --parallel 8 --target audiocpp_cli
```

Install the exact package used by the path test and resolve the external
eSpeak-ng paths:

```powershell
uv run --with onnx --with safetensors python tools/model_manager.py `
  install inflect_micro_v2 --models-root build-inflect/models

$espeakLibrary = uv run --with espeakng-loader python -c `
  "import espeakng_loader; print(espeakng_loader.get_library_path())"
$espeakData = uv run --with espeakng-loader python -c `
  "import espeakng_loader; print(espeakng_loader.get_data_path())"
```

Run the catalog, frontend, CPU long-form, and CUDA long-form checks:

```powershell
uv run python tools/check_loader_catalog_sync.py --self-test
uv run python tools/check_loader_catalog_sync.py
ctest --test-dir build-inflect -R "inflect_v2_frontend_test|model_spec_system_test" `
  --output-on-failure

uv run --with psutil --with nvidia-ml-py python `
  tools/audiocpp_cli/run_audiocpp_cli_path_tests.py `
  --cases tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json `
  --only inflect_v2_tts_longform `
  --audiocpp-cli-bin build-inflect/bin/audiocpp_cli.exe `
  --model-path build-inflect/models/Inflect-Micro-v2 `
  --backend cpu --threads 8 --measure-resources `
  --out-root build-inflect/validation/inflect-v2-cpu `
  --session-option "inflect_v2.espeak_library_path=$espeakLibrary" `
  --session-option "inflect_v2.espeak_data_path=$espeakData" --log

uv run python tools/audiocpp_cli/run_audiocpp_cli_path_tests.py `
  --cases tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json `
  --only inflect_v2_tts_longform `
  --audiocpp-cli-bin build-inflect-cuda/bin/audiocpp_cli.exe `
  --model-path build-inflect/models/Inflect-Micro-v2 `
  --backend cuda --threads 8 `
  --out-root build-inflect-cuda/validation/inflect-v2-cuda `
  --session-option "inflect_v2.espeak_library_path=$espeakLibrary" `
  --session-option "inflect_v2.espeak_data_path=$espeakData" --log
```

The generated WAVs, request files, commands, logs, and summaries are under:

- `build-inflect/validation/inflect-v2-cpu/inflect_v2_tts_longform/`
- `build-inflect-cuda/validation/inflect-v2-cuda/inflect_v2_tts_longform/`

## Known limitations

- English, one fixed voice, and offline TTS only.
- eSpeak-ng and its data remain external runtime dependencies.
- Only FP32 packages are supported.
- CPU and CUDA are runtime-validated. Vulkan and Metal are not practically
  validated for this release.
- CUDA deliberately performs duration alignment on CPU. CPU and CUDA are
  deterministic within a backend but are not bit-identical to each other.
- Windows WDDM did not expose per-process VRAM through NVML, so no CUDA peak
  VRAM figure is claimed.
