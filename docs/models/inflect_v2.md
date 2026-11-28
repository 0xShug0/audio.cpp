# Inflect Micro v2 and Nano v2

Default model-manager downloads use the published GGUF package when available;
the original source/conversion instructions below remain valid for manual use.

`inflect_v2` provides native GGML inference for the English
[Inflect Micro v2](https://huggingface.co/owensong/Inflect-Micro-v2) and
[Inflect Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2) TTS models.
Both variants use the same runtime and produce 24 kHz mono audio. Inference is offline; the published packages use FP32 weights.

The runtime does not use ONNX Runtime. The default download is the published
GGUF package. The deprecated converter can still rebuild local safetensors from
the official ONNX exports for manual testing.

## Install

Inflect v2 uses the [shared eSpeak-ng phonemizer](../espeak_phonemizer.md),
including shared synchronization with other model frontends.

With `AUDIOCPP_STATIC_ESPEAK=ON`, the build includes eSpeak code and stages its
data beside the CLI/server; no separate installation is needed.
Otherwise install eSpeak-ng and its English voice data first. On Debian or Ubuntu:

```bash
sudo apt install espeak-ng libespeak-ng1
```

On macOS:

```bash
brew install espeak-ng
```

Then install the default GGUF package:

```bash
python tools/model_manager_v2.py install inflect_micro_v2_orig --models-root models
```

The deprecated source-conversion workflow remains available through
`tools/model_manager_deprecated.py` for users who want to rebuild from the
upstream ONNX assets themselves.

## Run

```bash
audiocpp_cli --task tts --family inflect_v2 \
  --model models/Inflect-Micro-v2-GGUF/inflect-micro-v2-orig.gguf --backend cpu \
  --text "Hello from Inflect Micro version two." \
  --seed 0 --out inflect.wav
```

Use a converted Nano checkpoint to run Nano. CUDA uses the same model layout and
request surface. CUDA sessions keep duration alignment on the native CPU GGML
path so small TF32 differences cannot change the monotone expansion; the four
flows and waveform decoder execute on CUDA.

If eSpeak-ng is installed outside the dynamic loader's search path, provide
the library and data paths explicitly:

```bash
audiocpp_cli --task tts --family inflect_v2 \
  --model models/Inflect-Micro-v2-GGUF/inflect-micro-v2-orig.gguf --backend cpu \
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

## Model

| Field | Value |
|---|---|
| Family | `inflect_v2` |
| Tasks | `tts` |
| Modes | `offline` |
| Languages | `en` |
| Default package | `inflect_micro_v2_orig` |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `speed` | `float` | `1.0` | Speech speed multiplier mapped directly to Inflect speed; default 1.0. Declared range: minimum 0.5, maximum 2.0. |
| `variation` | `float` | `0.667` | Latent Gaussian variation; default 0.667. Declared range: minimum 0.0, maximum 1.0. |
| `seed` | `int` | `0` | Non-negative latent noise seed; default 0. Declared range: minimum 0. |
| `text_chunk_mode` | `word_budget` | `word_budget` | Long-form text chunking mode. |
| `text_chunk_size` | `int` | `280` | Maximum Unicode codepoints per long-form text chunk; default 280. Declared range: minimum 1. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `inflect_v2.espeak_library_path` | `path` | Model/session default | Optional explicit path to the eSpeak-ng shared library. |
| `inflect_v2.espeak_data_path` | `path` | Model/session default | Optional explicit path to the directory containing espeak-ng-data. |
| `inflect_v2.exec_plan` | `string` | `{}` | JSON object of component execution plans. Component: decoder. Each plan accepts impl, backend, artifacts, and params. impl is default/ggml/rknn. GGML keeps the session backend. RKNN requires artifacts.main and ENGINE_ENABLE_RKNN; omit backend. Neither implementation takes params. |

## Usage Notes

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

## Known limitations

- English, one fixed voice, and offline TTS only.
- eSpeak-ng and its data remain external runtime dependencies.
- Only FP32 packages are supported.
- Windows CPU/CUDA and Linux CPU under WSL2 are runtime-validated. Native
  Linux CUDA, Vulkan, and Metal are not practically validated for this
  release.
- CUDA deliberately performs duration alignment on CPU. CPU and CUDA are
  deterministic within a backend but are not bit-identical to each other.
- Windows WDDM did not expose per-process VRAM through NVML, so no CUDA peak
  VRAM figure is claimed.
