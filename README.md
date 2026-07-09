# audio.cpp

`audio.cpp` is a high-performance C++ audio inference framework built on top of `ggml`, designed to make modern local audio models practical, portable, and fast.

Tired of juggling a dozen Conda environments, hundreds of Python packages, and dependency conflicts just to try a few audio models? audio.cpp gives those paths a shared native runtime instead.

> [!IMPORTANT]
> **CUDA performance headline:** multiple TTS paths already run **1.8x-5.0x faster than their Python reference paths** while cutting end-to-end latency by **45%-80%**.
> **VibeVoice 1.5B:** generates a **93.9-minute podcast in 18.2 minutes** with **10 diffusion steps** and without quantization, running about **5.15x faster than real time**.

It is built for real end-to-end execution rather than one-off model demos: the same runtime powers TTS, voice cloning, voice conversion, ASR, diarization, VAD, source separation, alignment, codec-style models, and higher-level workflows through a common framework surface.

Highlights:

- **Parity.** Strong parity tooling against Python reference paths.
- **Performance.** Performance-focused execution, reusable sessions, and batch-style offline inference. **Optimized for CUDA**.
- **Portability.** A portable native stack centered on `ggml`, with CLI and server entry points instead of Python-only deployment paths.
- **Pipelines.** Experimental JSON pipeline support for higher-level multi-step workflows.
- **Audio Utilities.** Built-in denoise, enhancement, resampling, and STFT/ISTFT utilities for real production-style task paths.

<p><strong><span style="font-size:1.1em;">The goal of the framework is to provide highly optimized, reusable building blocks for audio-related models, so new model integrations can be brought up faster, shared components can be improved once and benefit many families, and real end-to-end inference paths can stay efficient, maintainable, and portable.</span></strong></p>

> [!TIP]
> **Contribution focus:** the most helpful contributions right now are improvements to the UI, API server, and pipeline/workflow subsystems. These areas make the existing model surface easier to use, serve, compose, and validate. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
>
> **New model PRs:** before starting a new model port, **please check the supported model table because several families are already implemented or under testing**. If you do add a model, follow the validation style in [PR #19](https://github.com/0xShug0/audio.cpp/pull/19): include exact build/run commands, model paths or package ids, generated outputs, parity or path-test results, and relevant performance or memory notes.

## News

> [!IMPORTANT]
> **2026-07-08:** Four new ASR families are now released in the framework: Higgs Audio STT, Hviske ASR, Nemotron ASR, and VibeVoice ASR. Initial model-specific streaming support also lands for VoxCPM2 TTS, Nemotron ASR, and Higgs Audio STT, with server SSE configuration and request examples for streaming speech generation and transcription.

> [!IMPORTANT]
> **2026-07-03:** Conv1DTransp module CUDA optimization: VibeVoice reaches **5.15x realtime** on **93.9-minute long-form generation**. Overall, VibeVoice inference time was reduced by **73.17%**, PocketTTS by **35.32%**, Chatterbox by **33.56%**, Qwen3-TTS by **30.60%**, HeartMuLa by **17.03%**, and VoxCPM2 by **14.7%** compared with the previous release.

> [!IMPORTANT]
> **2026-07-02:** Music generation and source separation expanded in the released framework surface: ACE-Step 1.5 Turbo/Base, HeartMuLa, Stable Audio 3 Small Music/SFX and Medium, Mel-Band RoFormer, and HTDemucs are now available through the normal audio.cpp CLI/framework paths.

- **2026-07-02:** VibeVoice 7B joins the 1.5B model, and full fine-tune adapters — language-model LoRA plus fine-tuned diffusion head and acoustic/semantic connectors — can now be merged at load time through `--load-option vibevoice.lora`.
- **2026-06-30:** VibeVoice 1.5B is now released in the framework, bringing long-form, multi-speaker dialogue TTS into the normal audio.cpp model surface.
- **2026-06-30:** More detailed usage documentation is now available in [docs/usage.md](docs/usage.md), covering model setup, CLI usage, server usage, and common workflows.
- **2026-06-26:** The speech intelligence side grew with released Citrinet ASR, MarbleNet VAD, and Sortformer diarization paths.
- **2026-06-25:** The first release wave landed with TTS, voice cloning, voice conversion, alignment, VAD, codec, and multilingual generation support across Chatterbox, MioCodec, MioTTS, OmniVoice, PocketTTS, Qwen3, SeedVC, Silero VAD, Vevo2, and VoxCPM2.

## Supported Models

Current model status in the framework:

- `released`: The model is fully wired into the broader framework surface and ready for normal use.
- `testing`: The model is implemented and working in this repo, and is still being validated, polished, or promoted into the broader released surface.
- `optimization`: The model is end-to-end working, but still needs more optimization work before it should be treated like a released or testing-level path.

| Family | Task | Supported language(s) | Supported variant(s) in this repo | Release status |
|---|---|---|---|---|
| **ace_step** | music generation, music editing | 50+ langs | ACE-Step 1.5 Turbo and Base with acestep-5Hz-lm-1.7B | **released** |
| **chatterbox** | TTS, voice cloning | ar, da, de, el, en, es, fi, fr, hi, it, ko, ms, nl, no, pl, pt, sv, sw, tr | Chatterbox with 0.5B backbone | **released** |
| **citrinet_asr** | ASR | en | Citrinet-256 | **released** |
| **heartmula** | music generation | zh, en, ja, ko, es | HeartMuLa-oss-3B with HeartCodec-oss | **released** |
| **higgs_audio_stt** | ASR | en | Higgs Audio v3 STT | **released** |
| **htdemucs** | source separation | lang agnostic | HTDemucs, HTDemucs_ft | **released** |
| **hviske_asr** | ASR | da | Hviske v5.3 | **released** |
| **marblenet_vad** | VAD | lang agnostic | MarbleNet VAD | **released** |
| **mel_band_roformer** | vocal separation | lang agnostic | Mel-Band RoFormer MLX vocal separation variants | **released** |
| **miocodec** | audio codec, voice conversion backend | lang agnostic | MioCodec v2, 25 Hz, 44.1 kHz | **released** |
| **miotts** | TTS, voice cloning | en, ja | MioTTS-1.7B | **released** |
| **omnivoice** | TTS, voice cloning, voice design | 646+ langs | OmniVoice, Qwen3-0.6B based | **released** |
| **pocket_tts** | TTS, voice cloning | en, de, it, pt, es | PocketTTS-100M | **released** |
| **nemotron_asr** | ASR | 100+ ASR prompt codes incl. auto | Nemotron 3.5 ASR Streaming 0.6B | **released** |
| **qwen3_asr** | ASR | zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, ro, hu, mk | Qwen3-ASR-0.6B | **released** |
| **qwen3_forced_aligner** | forced alignment | zh, yue, en, de, es, fr, it, pt, ru, ko, ja | Qwen3-ForcedAligner-0.6B | **released** |
| **qwen3_tts** | TTS, voice cloning, voice design | zh, en, fr, de, it, ja, ko, pt, ru, es | Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-1.7B-Base, Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen3-TTS-12Hz-1.7B-VoiceDesign | **released** |
| **seed_vc** | voice conversion | lang agnostic | SeedVC XLS-R + HiFT, SeedVC Whisper-small + BigVGAN | **released** |
| **silero_vad** | VAD | lang agnostic | Silero VAD | **released** |
| **sortformer_diar** | diarization | en | Sortformer-4spk-v1 | **released** |
| **stable_audio** | music generation, sound generation, audio editing | en | Stable Audio 3 Small Music, Stable Audio 3 Small SFX, Stable Audio 3 Medium | **released** |
| **vevo2** | TTS, singing generation, voice conversion, singing conversion, editing | en, zh | Vevo2 with Qwen2.5-0.5B AR model | **released** |
| **vibevoice** | TTS, multi-speaker dialogue TTS | en, zh | VibeVoice-1.5B, VibeVoice-7B | **released** |
| **vibevoice_asr** | ASR | auto | VibeVoice ASR | **released** |
| **voxcpm2** | TTS, voice cloning, voice design | ar, da, de, el, en, es, fi, fr, he, hi, id, it, ja, km, ko, lo, ms, my, nl, no, pl, pt, ru, sv, sw, th, tl, tr, vi, zh | VoxCPM2-2B, 48 kHz | **released** |
| higgs_tts | TTS, voice cloning, expressive speech | 100+ languages | Higgs Audio v3 TTS 4B | testing |
| index_tts2 | TTS, voice cloning, expressive speech | zh, en | IndexTTS-2 | testing |
| irodori_tts | TTS, voice cloning, voice design | ja | Irodori-TTS-500M-v3, Irodori-TTS-600M-v3-VoiceDesign | testing |
| kokoro_tts | TTS | en-us, en-gb | Kokoro-82M | testing |
| moss_tts | TTS, voice cloning | zh, yue, en, ar, cs, da, nl, fi, fr, de, el, he, hi, hu, it, ja, ko, mk, ms, fa, pl, pt, ro, ru, es, sw, sv, tl, th, tr, vi | MOSS-TTS-Local | testing |
| supertonic | TTS | en | Supertonic 3 | testing |

PocketTTS language selection is a model-load option. When the model path points at the PocketTTS root, the loader uses `english` unless you pass `--load-option language=<name>`. Kyutai's normal non-English PocketTTS releases are smaller distilled language models intended for the fast PocketTTS path. The `_24l` variants are larger 24-layer, undistilled preview models that can sound better but are slower. Kyutai currently publishes French only as `french_24l`, not as a normal distilled `french` language directory, so French is not listed as a normal PocketTTS language here.

## Docker

Docker CPU and CUDA images are available for both CLI and server use. See [Docker.md](Docker.md) for build commands and working Docker examples.

## Build

### Linux Build

On Linux, use a normal CMake build directory such as `build/`.

For single-config generators, the default build type is `RelWithDebInfo`.

That default configure is a CPU build unless you enable an accelerator backend explicitly.

Use GCC 13 or newer for Linux builds.

Native ggml CPU optimization is enabled by default for local performance. If your compiler or assembler rejects a generated CPU instruction such as `vpdpbusd`, reconfigure with `-DENGINE_ENABLE_NATIVE_CPU=OFF` to build portable CPU kernels.

Common Linux configure examples:

CPU-only:

```bash
cmake -S . -B build
```

CUDA:

```bash
cmake -S . -B build -DENGINE_ENABLE_CUDA=ON
```

Vulkan:

```bash
cmake -S . -B build -DENGINE_ENABLE_VULKAN=ON
```

Portable CPU-kernel fallback:

```bash
cmake -S . -B build -DENGINE_ENABLE_NATIVE_CPU=OFF
```

Build the CLI and server from the configured tree:

```bash
cmake --build build --parallel --target audiocpp_cli --target audiocpp_server
```

If you use an environment manager or custom toolchain, activate it before running the commands above.

The optional Linux helper script wraps the same CMake flow and uses aligned build directory names:

- `build/linux-cuda-release`
- `build/linux-vulkan-release`
- `build/linux-cpu-release`

Examples:

```bash
scripts/build_linux.sh --backend cuda --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend vulkan --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend cpu --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend cuda --native-cpu OFF --target audiocpp_cli --target audiocpp_server
```

Use `--build-dir <dir>` only when you intentionally want a custom output directory.

### Windows Build

The recommended native Windows build is command-line only:

- Visual Studio Build Tools 2022 or newer with the C++ desktop workload
- MSVC x64 compiler, Windows SDK, CMake, Ninja, and MSVC OpenMP components
- Official NVIDIA CUDA Toolkit installed under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\...` for CUDA builds

Use MSVC `cl.exe` as the compiler. For CUDA builds, `cl.exe` is also used as the CUDA host compiler. Native Windows `nvcc` does not support `clang-cl` as its host compiler, and the Visual Studio IDE is not required.

From PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

CPU-only:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli
```

From `cmd.exe`:

```bat
scripts\build_windows.cmd
```

If GNU Make is available on Windows:

```bash
make -f Makefile.windows cpu JOBS=16
make -f Makefile.windows cuda JOBS=16
make -f Makefile.windows cuda NATIVE_CPU=OFF JOBS=16
```

The Windows script configures `build/windows-cuda-release` by default and builds `audiocpp_cli`. CUDA presets enable CUDA, CUDA graphs, OpenMP, Ninja, `/utf-8`, `/EHsc`, MSVC OpenMP SIMD support with `/openmp:experimental`, and native CPU optimization by default. The CPU preset uses the same MSVC/Ninja/OpenMP setup without requiring CUDA. CUDA presets auto-detect the local GPU CUDA architecture when `nvidia-smi` is available. Pass `-NativeCpu OFF` or `NATIVE_CPU=OFF` to use portable CPU kernels.

For Windows prebuilt release zips and CPU compatibility profiles, see [docs/windows_build.md](docs/windows_build.md).

Useful variants:

```powershell
.\scripts\build_windows.ps1 -Target audiocpp_server -Jobs 16
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli
.\scripts\build_windows.ps1 -Preset windows-cuda-debug -Target audiocpp_cli
.\scripts\build_windows.ps1 -NativeCpu OFF -Target audiocpp_cli
.\scripts\build_windows.ps1 -ConfigureOnly
.\scripts\build_windows.ps1 -CudaArchitectures 120a-real
```

If multiple Build Tools installations are present, pass the one you want explicitly:

```powershell
.\scripts\build_windows.ps1 -VsInstall "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
```

The built CLI is written to the selected preset directory:

```powershell
build\windows-cpu-release\bin\audiocpp_cli.exe
build\windows-cuda-release\bin\audiocpp_cli.exe
```

### Metal Build

On macOS, use the Metal helper script to build against ggml's Metal backend. It requires Xcode or the Xcode Command Line Tools with the Metal compiler available through `xcrun`.

```bash
scripts/build_metal.sh --target audiocpp_cli
```

The script configures `build/macos-metal-release` by default, enables `ENGINE_ENABLE_METAL=ON`, disables CUDA and Vulkan, embeds the Metal shader library, and builds static libraries plus the requested target.

Useful variants:

```bash
scripts/build_metal.sh --target audiocpp_server
scripts/build_metal.sh --build-type Release --archs arm64 --target audiocpp_cli
scripts/build_metal.sh --with-tests --target audio_dsp_test
scripts/build_metal.sh --openmp auto --target audiocpp_cli
scripts/build_metal.sh --native-cpu OFF --target audiocpp_cli
```

The built CLI is written to:

```bash
build/macos-metal-release/bin/audiocpp_cli
```

Build options:

| Option | Meaning | Default |
|---|---|---|
| `ENGINE_ENABLE_CUDA` | Enable the ggml CUDA backend. Required for `--backend cuda`. | `OFF` |
| `ENGINE_ENABLE_VULKAN` | Enable the ggml Vulkan backend. Required for `--backend vulkan`. | `OFF` |
| `ENGINE_ENABLE_METAL` | Enable the ggml Metal backend. Required for `--backend metal`. | `OFF` on most platforms, `ON` on Apple |
| `ENGINE_ENABLE_LLAMAFILE` | Enable llamafile SGEMM support in ggml CPU builds. | `ON` |
| `ENGINE_ENABLE_CUDA_GRAPHS` | Enable ggml CUDA graphs support when CUDA is enabled. | `ON` |
| `ENGINE_ENABLE_NATIVE_CPU` | Build ggml CPU kernels with native host ISA flags such as `-march=native`. Disable this for portable CPU kernels or toolchains that reject generated CPU instructions. | `ON` |
| `ENGINE_ENABLE_OPENMP` | Enable OpenMP for host-side parallel work. | `ON` |
| `ENGINE_BUILD_EXAMPLES` | Build example binaries. | `OFF` |
| `ENGINE_BUILD_TESTS` | Build framework unit tests. | `OFF` |
| `ENGINE_BUILD_WARMBENCH` | Build warmbench helper binaries. | `OFF` |

### Build Type Notes

- For single-config generators, the recommended config is `RelWithDebInfo`
- For multi-config generators, choose the configuration at build time
- Backend and feature options are independent from build type

## Usage

### CLI

The main CLI binary is:

```bash
build/bin/audiocpp_cli
```

High-level command shape:

```bash
audiocpp_cli --task <task> --model <path> [--family <family>] [--backend <backend>] [--mode <mode>] [options]
```

Core selectors:

- `--task vad|asr|diar|sep|gen|tts|clon|vc|s2s|align|vdes|spk|svc`
- `--model <path>`
- `--family <name>` optionally narrows model-loader selection when a model path could match more than one family
- `--backend cpu|cuda|vulkan|metal|best`
- `--mode offline|streaming`; streaming is available for models whose docs list streaming support

Common interface options:

- `--load-option key=value` passes model-load options, such as PocketTTS language selection
- `--session-option key=value` passes session/runtime options, such as backend-specific weight controls
- `--request-option key=value` passes per-request model options
- `--config <id>` selects a discovered config asset
- `--weight <id>` selects a discovered weight asset
- `--device <n>` selects the backend device
- `--threads <n>` sets backend and OpenMP worker threads

Examples:

Text-to-speech:

```bash
build/bin/audiocpp_cli \
  --task tts \
  --family pocket_tts \
  --model /path/to/model \
  --backend cuda \
  --text "audio.cpp is running PocketTTS locally." \
  --voice-ref assets/resources/sample.wav \
  --out build/out/pocket_tts.wav
```

PocketTTS with another language and a built-in voice:

```bash
build/bin/audiocpp_cli \
  --task tts \
  --family pocket_tts \
  --model /path/to/models/pocket-tts \
  --backend cuda \
  --load-option language=spanish \
  --text "Hola, esta es una prueba corta de Pocket TTS." \
  --voice-id alba \
  --out build/out/pocket_tts_spanish.wav
```

ASR:

```bash
build/bin/audiocpp_cli \
  --task asr \
  --family qwen3_asr \
  --model /path/to/model \
  --backend cuda \
  --audio assets/resources/sample_16k.wav
```

Voice conversion:

```bash
build/bin/audiocpp_cli \
  --task vc \
  --family seed_vc \
  --model /path/to/model \
  --backend cuda \
  --audio assets/resources/a.wav \
  --voice-ref assets/resources/b.wav \
  --out build/out/seed_vc.wav
```

Useful CLI features:

- `--help` with `--task` shows task-oriented help
- `--help` with `--model <path>` and optional `--family <family>` shows model-owned request, session, and load options
- `--inspect` prints discovered configs, weights, and capabilities
- `--list-loaders` prints registered model families
- `--batch-text-file <txt>` runs one offline request per non-empty line
- `--batch-text-dir <dir>` runs one offline request per `.txt`, `.md`, or `.json` file, normalizing each file as one paragraph
- `--batch-audio-dir <dir>` runs one offline request per `.wav`
- `--audio-chunk-mode auto` lets ASR/alignment models choose their safe long-audio policy; expert users can override with `fixed`, `vad`, or `none` where supported
- `--request-sequence <json>` runs a multi-request offline session
- `--batch-merge-audio none|concat` controls batch audio merge behavior
- `--batch-manifest-out <json>` writes a batch output manifest
- `--pipeline <json>` runs a workflow instead of a raw task
- `--list-pipelines` prints registered workflows
- `--workflow-input key=value` overrides pipeline inputs
- `--log` streams framework logs to stdout
- `--log-file <path>` streams framework logs to a file in real time
- `--segments-out`, `--turns-out`, and `--words-out` write structured JSON outputs
- `--vad-chunks-out` writes offline VAD-based chunk windows; tune them with `--vad-chunk-max-seconds`, `--vad-chunk-merge-gap-seconds`, and `--vad-chunk-padding-seconds`

### Pipelines

Pipelines are an experimental JSON workflow feature for chaining multiple model and audio-processing steps behind one CLI command. A pipeline can define default inputs, let users override them with `--workflow-input key=value`, split long media into model-sized chunks, merge text or audio outputs back together, write intermediate artifacts under `--out-dir`, and copy the declared `final_audio` to `--out`.

This is the higher-level layer for production-style audio jobs: redubbing, batch cleanup, long-form narration, voice conversion, source-separation workflows, transcription-plus-alignment, and future workflows that combine translation, diarization, denoise, enhancement, or review steps as those model surfaces are wired into the framework.

The included same-language speech redub pipeline transcribes long speech in chunks with Qwen3 ASR, merges the transcript, then regenerates the speech in a target reference voice with Qwen3 TTS. The default test input `assets/resources/speech.wav` is about 418 seconds long and was generated from an 8,091-character speech text, so it exercises long-audio split and merge behavior rather than a short one-shot request:

```bash
build/bin/audiocpp_cli \
  --pipeline assets/pipeline/speech_redub.json \
  --backend cuda \
  --out-dir build/out/speech_redub_pipeline \
  --out build/out/speech_redub_pipeline.wav
```

Override the source speech or target voice without editing the JSON:

```bash
build/bin/audiocpp_cli \
  --pipeline assets/pipeline/speech_redub.json \
  --backend cuda \
  --workflow-input source_audio=/path/to/speech.wav \
  --workflow-input target_voice=/path/to/voice.wav \
  --workflow-input language=English \
  --out-dir build/out/speech_redub_pipeline \
  --out build/out/speech_redub_pipeline.wav
```

### Tools / Model Manager

The repository also ships a model manager at `tools/model_manager.py` for downloading supported model packages into the framework expected `models/` layout.

Dependencies:

- Python 3
- `torch`
- `safetensors`
- `PyYAML`
- network access to the upstream model sources

The tool supports three main commands:

- `list` shows the available package ids
- `info` shows the target layout, required files, and install source for one package
- `install` downloads or converts one package into a models root

Recommended top-level install packages:

`Yes` means Hugging Face has a ready-to-use repo that the framework can download as-is. `No` means the tool must assemble, convert, or post-process files before the framework can use them.

| Package id | Model | HF ready-to-use repo |
|---|---|---|
| `ace_step` | ACE-Step 1.5 Turbo/Base | No |
| `chatterbox` | Chatterbox | **Yes** |
| `citrinet_asr` | Citrinet ASR converted layout | No |
| `heartmula` | HeartMuLa | No |
| `higgs_audio_stt` | Higgs Audio STT | No |
| `higgs_audio_v3_tts_4b` | Higgs Audio v3 TTS 4B | **Yes** |
| `htdemucs` | HTDemucs | No |
| `hviske_asr` | Hviske ASR | **Yes** |
| `irodori_tts_500m_v3` | Irodori-TTS 500M v3 | No |
| `irodori_tts_600m_v3_voice_design` | Irodori-TTS 600M v3 VoiceDesign | No |
| `kokoro_82m_bf16` | Kokoro 82M bf16 | **Yes** |
| `marblenet_vad` | MarbleNet VAD converted layout | No |
| `mel_band_roformer` | Mel-Band RoFormer MLX | **Yes** |
| `miocodec_25hz_44k_v2` | MioCodec 25Hz 44.1kHz v2 | No |
| `miotts_1_7b` | MioTTS 1.7B | No |
| `moss_tts` | MOSS-TTS-Local | No |
| `nemotron_asr` | Nemotron ASR | **Yes** |
| `omnivoice` | OmniVoice | **Yes** |
| `parakeet_tdt_0_6b_v3` | Parakeet TDT 0.6B v3 | **Yes** |
| `pocket_tts` | PocketTTS | **Yes** |
| `qwen3_asr_0_6b` | Qwen3 ASR 0.6B | **Yes** |
| `qwen3_forced_aligner_0_6b` | Qwen3 Forced Aligner 0.6B | **Yes** |
| `qwen3_tts_0_6b_base` | Qwen3 TTS 12Hz 0.6B Base | **Yes** |
| `qwen3_tts_1_7b_base` | Qwen3 TTS 12Hz 1.7B Base | **Yes** |
| `qwen3_tts_1_7b_custom_voice` | Qwen3 TTS 12Hz 1.7B Custom Voice | **Yes** |
| `qwen3_tts_1_7b_voice_design` | Qwen3 TTS 12Hz 1.7B Voice Design | **Yes** |
| `seed_vc` | SeedVC-MLX | **Yes** |
| `sortformer_diar_4spk_v1` | Sortformer diarization 4 speaker v1 | **Yes** |
| `stable_audio_3_medium` | Stable Audio 3 Medium | **Yes** |
| `stable_audio_3_small_music` | Stable Audio 3 Small Music | **Yes** |
| `stable_audio_3_small_sfx` | Stable Audio 3 Small SFX | **Yes** |
| `supertonic_3` | Supertonic 3 | **Yes** |
| `vevo2` | Vevo2 | No |
| `vibevoice_1_5b` | VibeVoice 1.5B | No |
| `vibevoice_7b` | VibeVoice 7B | No |
| `vibevoice_asr` | VibeVoice ASR | No |
| `voxcpm2` | VoxCPM2 | No |

> [!WARNING]
> PocketTTS is hosted in a gated Hugging Face repo, so the model manager needs a Hugging Face token with access to `kyutai/pocket-tts`. It currently downloads only the English model and the built-in `alba` voice.

> [!TIP]
> If you already have the VibeVoice Hugging Face model directory, you do not need to redownload the tokenizer files. Copy `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt` from `assets/model_manager/vibevoice_1_5b/` into `VibeVoice-1.5B/`, `VibeVoice-7B/`, or `VibeVoice-ASR/`.

Examples:

List packages:

```bash
python3 tools/model_manager.py list
```

Show one package:

```bash
python3 tools/model_manager.py info qwen3_tts_1_7b_base
```

Install one package into the default `models/` directory:

```bash
python3 tools/model_manager.py install qwen3_tts_1_7b_base
```

Install into a custom models root:

```bash
python3 tools/model_manager.py install vevo2 --models-root /path/to/models
```

Replace an existing installed package:

```bash
python3 tools/model_manager.py install pocket_tts --overwrite
```

Some packages are direct snapshots, while others are composite installs or local-file utilities. Use `info` first when you want to inspect the expected target directory, required files, or whether a package needs extra local source inputs such as `--source-file` or `--source-dir`.

Run a local-file utility:

```bash
python3 tools/model_manager.py info voxcpm2_audiovae
python3 tools/model_manager.py install voxcpm2_audiovae --source-file models/VoxCPM2/audiovae.pth --models-root models --overwrite
```

### Server

The server binary is:

```bash
build/bin/audiocpp_server
```

Build:

```bash
cmake --build build --parallel --target audiocpp_server
```

Create a config file with your own model paths:

```bash
cat > server.json <<'JSON'
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 1,
  "lazy_load": true,
  "models": [
    {
      "id": "pocket-tts",
      "family": "pocket_tts",
      "path": "/path/to/models/pocket-tts",
      "task": "tts",
      "mode": "offline",
      "load_options": {
        "language": "english"
      },
      "session_options": {
        "language": "english"
      }
    },
    {
      "id": "qwen3-asr",
      "family": "qwen3_asr",
      "path": "/path/to/models/Qwen3-ASR-0.6B",
      "task": "asr",
      "mode": "offline"
    }
  ]
}
JSON
```

Set `"lazy_load": true` to register configured model ids at startup while loading each model only on first use. Use per-model `"lazy": true` or `"lazy": false` to override that default.

Set top-level `"backend"` to `"cuda"`, `"cpu"`, `"vulkan"`, or `"metal"`. CUDA is the optimized path for audio.cpp; CPU, Vulkan, and Metal are intended for portability and testing when the binary is built with that backend, but performance and model coverage may be lower.

> [!WARNING]
> Lazy loading does not unload models after a request. Once a model is first used, the server keeps that model and session in memory for reuse until the server exits.

Start:

```bash
build/bin/audiocpp_server --config server.json
```

The server exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/tasks/run`

More server examples are in [app/server/README.md](app/server/README.md).

## Tests

The repository includes both framework-level parity validation and app-level end-to-end path checks. At a high level, the flow is:

<p align="center">
  <img src="assets/figure/parity_test_flow.png" alt="Parity test flow" width="720" />
</p>

The main harness under `tests/` is `tests/warmbench.py`. It is used for long-lived multi-request validation, parity checks against Python references, and performance-oriented session reuse scenarios. The `tests/` tree also contains model-specific C++ and Python warmbench entrypoints that `warmbench.py` coordinates.

The main app-facing test tooling under `tools/` is `tools/audiocpp_cli/run_audiocpp_cli_path_tests.py`. It drives `audiocpp_cli` through cataloged offline and streaming cases, verifies expected outputs such as audio or JSON artifacts, and is useful for checking real user-facing request paths rather than just lower-level model components. Streaming coverage is model-specific and applies to models documented with streaming support.

The Python-reference side of these tests usually requires more time-consuming setup than the C++ path because different models rely on different Python reference repos and dependency stacks. In practice, the framework-side tooling is fast to iterate on once models are installed, while Python parity runs often need extra environment preparation before they are ready.

## Projects

- [Pocket TTS Browser Engine](https://github.com/jjmlovesgit/pocket-tts-browser-engine) uses audio.cpp to bring fully local PocketTTS voices into Chrome and Edge through the browser TTS API.
- [GuideAnts](https://github.com/Elumenotion/GuideAnts) uses audio.cpp as the default local AI stack path for basic ASR and TTS, with planned reusable skills for audio.cpp scenarios and model configurations.

## Performance Metrics

> [!WARNING]
> These Python-relative numbers were measured for the initial release. Several model paths have improved substantially since then, so the figures below should be read as the original release baseline rather than the latest peak performance.

All performance metrics in this section were measured on Ubuntu with the CUDA backend on an NVIDIA GeForce RTX 5090. The Python-relative one-shot and long-lived-session comparisons come from direct framework/runtime API benchmark calls, not from `audiocpp_cli`; CLI path tests are separate and include app-layer request parsing, output writing, and other user-facing overhead.

**Absolute RTF depends on the GPU and system setup, but the Python-relative speedups are real because audio.cpp and the matching Python reference paths were measured on the same CUDA setup.**

audio.cpp already shows some genuinely exciting wins against the matching Python reference paths, especially on the TTS side, even when using the original model weights without quantization. The headline win is wall time: several TTS paths run **1.8x-5.0x faster** than Python while cutting end-to-end latency by **45%-80%**.

- In one-shot runs, several TTS-family models already land far ahead of Python:
  - `vevo2`: **5.03x faster** with **80.11% less wall time**
  - `pocket tts`: **3.68x faster** with **72.80% less wall time**
  - `miotts`: **2.73x faster** with **63.39% less wall time**
  - `moss tts`: **2.33x faster** with **57.07% less wall time**
  - `qwen3 tts`: **1.83x faster** with **45.34% less wall time**
  - `vibevoice`: **1.40x faster** with **28.75% less wall time**
- In long-lived-session runs, where the same loaded session serves multiple requests in sequence, the gains stay strong:
  - `pocket tts`: **3.22x faster** with **68.91% less wall time**
  - `qwen3 tts`: **2.74x faster** with **63.47% less wall time**
  - `moss tts`: **2.66x faster** with **62.35% less wall time**
  - `miotts`: **2.28x faster** with **56.22% less wall time**
  - `vibevoice`: **1.77x faster** with **43.55% less wall time**
  - `vevo2`: **1.75x faster** with **42.72% less wall time**
- In long-form runs on the shared 6,026-character, 1,028-word passage, the strongest Python-relative wins still show up clearly:
  - `pocket tts`: **3.15x faster** with **68.23% less wall time**
  - `qwen3 tts`: **3.06x faster** with **67.33% less wall time**
  - `vibevoice`: **2.86x faster** with **65.07% less wall time**
  - `vevo2`: **1.77x faster** with **43.51% less wall time**
  - `chatterbox`: **1.58x faster** with **36.83% less wall time**
- These long-lived-session numbers are especially important for real applications, because they reflect the common case where model load, cached state, and reusable runtime setup are amortized across many requests.
- Bars below the 1.0x line are useful too: they spotlight exactly where more optimization work is still worth doing.

<p align="center">
  <img src="assets/figure/perf_one_shot_20260630.svg" alt="One-shot" width="720" />
</p>

<p align="center">
  <img src="assets/figure/perf_long_lived_session_20260630.svg" alt="Long-lived session" width="720" />
</p>

The figures report `Python wall time / audio.cpp wall time`. The 1.0x line means equal wall time; bars above 1.0x mean audio.cpp is faster than Python, and bars below 1.0x mean it is slower.

For TTS-family models, the measured one-shot RTF is:

| model | audio len (s) | wall time (s) | RTF | x faster than real time |
|---|---:|---:|---:|---:|
| chatterbox | 9.72 | 2.45 | 0.252 | 3.97x |
| kokoro tts | 10.15 | 0.64 | 0.063 | 15.90x |
| miotts | 20.40 | 3.30 | 0.162 | 6.18x |
| moss tts | 9.60 | 0.97 | 0.101 | 9.91x |
| omnivoice | 9.00 | 1.32 | 0.146 | 6.84x |
| pocket tts | 8.08 | 0.26 | 0.032 | 31.09x |
| qwen3 tts | 11.44 | 4.46 | 0.390 | 2.56x |
| vevo2 | 8.66 | 2.47 | 0.285 | 3.51x |
| vibevoice | 11.07 | 5.02 | 0.454 | 2.20x |
| voxcpm2 | 5.60 | 3.09 | 0.551 | 1.81x |

For long-form TTS tests, each run uses the same 6,026-character, 1,028-word input text (vibevoice uses 106,310 chars, 18,052 words, 4 speakers). The measured RTF is:

| model | audio len (s) | wall time (s) | RTF | x faster than real time |
|---|---:|---:|---:|---:|
| chatterbox | 391.24 | 58.57 | 0.150 | 6.68x |
| kokoro tts | 371.17 | 7.19 | 0.019 | 51.60x |
| miotts | 399.16 | 66.59 | 0.167 | 5.99x |
| moss tts | 301.36 | 25.00 | 0.083 | 12.06x |
| omnivoice | 357.00 | 17.77 | 0.050 | 20.09x |
| pocket tts | 353.12 | 7.30 | 0.021 | 48.40x |
| qwen3 tts | 327.60 | 72.65 | 0.222 | 4.51x |
| vevo2 | 457.68 | 52.47 | 0.115 | 8.72x |
| voxcpm2 | 315.84 | 72.70 | 0.230 | 4.34x |
| vibevoice | 5615.73 | 1376.84 | 0.245 | 4.08x |

## Runtime Memory Options

Some models expose memory-saver session options such as `ace_step.mem_saver=true`, `heartmula.mem_saver=true`, `stable_audio.mem_saver=true`, `omnivoice.mem_saver=true`, and `voxcpm2.mem_saver=true`. These options keep the default output path unchanged while reducing graph workspace VRAM or releasing staged graph/cache state after request phases; later requests may rebuild released graphs.

## Precision/Quantization Support

Many model sessions expose quantization through `--session-option <family>.weight_type=<mode>`, and some families also expose more specific knobs such as `...conv_weight_type`, `...talker_weight_type`, or `...speech_decoder_weight_type`. The exact supported modes are model-specific rather than global.

Example:

```bash
build/bin/audiocpp_cli --task tts --family qwen3_tts --model /path/to/model --session-option qwen3_tts.weight_type=q8_0
```

In practice, lower precision and quantized modes should be treated as model- and route-specific optimizations rather than universally safe defaults.

- **Safety.** Quantization may not be safe on every path even when a model parser accepts the option. For example, in our ACE-Step 1.5 checks, lower-precision runs could fail at runtime with `ACE-Step planner masked decode found no valid token` while higher-precision settings completed normally.

- **Quality Drop.** Output quality can drop a lot. In our VeVo2 checks, non-`fp32` outputs showed noticeably weaker similarity to the `fp32` reference under the repo's existing waveform and log-mel comparison metrics, and even output length could shift.

- **Performance Gain.** The performance gain may be minor relative to that quality risk. For example, `q8_0` was faster than the default setting by only around 3.8% on Qwen3-TTS and around 3.6% on VeVo2. Other models may benefit more, but the tradeoff should be validated per model and per route rather than assumed.

- **Memory Benefit.** Lower precision and quantized weights can still be useful for reducing weight memory footprint and making larger models easier to fit within device limits. For example, in our Qwen3-TTS checks, switching from the default setting to `q8_0` reduced peak RAM by about 3.7% and peak VRAM by about 25.0%. That benefit is real, but it should be evaluated together with runtime stability, output quality, and end-to-end speed rather than assumed from precision alone.

## Notes

- The repo supports multiple backends, but backend and model coverage are model-dependent.
- GGUF model loading is planned, but not supported yet.
- `Build_xcframework.sh` is outdated; Metal and Apple XCFramework packaging still need to be retested after the framework refactor.
