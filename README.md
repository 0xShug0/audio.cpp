# audio.cpp

`audio.cpp` is a high-performance C++ audio inference framework built on top of `ggml`, designed to make modern local audio models practical, portable, and fast.

Tired of juggling a dozen Conda environments, hundreds of Python packages, and dependency conflicts just to try a few audio models? audio.cpp gives those paths a shared native runtime instead.

> [!IMPORTANT]
> **CUDA performance headline:** multiple TTS paths already run **1.8x-5.0x faster than their Python reference paths** while cutting end-to-end latency by **45%-80%**.
>
> **VibeVoice 1.5B:** generates a **93.9-minute podcast in 18.2 minutes** with **10 diffusion steps** and without quantization, running about **5.15x faster than real time**.
>
> **Supertonic 3:** generates about **10 hours of audio in 3 minutes** on RTX5090. Up to 200x+ real-time on CUDA, 6x+ real-time on CPU, and 47 ms TTFT in CUDA streaming mode.
> [Demo: 10 hours of audio generated in 3 minutes](https://www.reddit.com/r/LocalLLaMA/comments/1uwpvt9/audiocpp_10_hours_of_audio_generated_in_3_minutes/).
>
> **Real-world ASR win:** In [TranscrIA benchmark](https://github.com/Martossien/transcria/blob/main/docs/STT_BENCHMARK_REAL_MEETINGS.md) on messy French meeting audio, audio.cpp’s Nemotron 3.5 ASR matched the same WER as other implementations while using about **1/4 of the wall time**. 

It is built for real end-to-end execution rather than one-off model demos: the same runtime powers TTS, voice cloning, voice conversion, ASR, diarization, VAD, source separation, alignment, codec-style models, and higher-level workflows through a common framework surface.

Highlights:

- **Parity.** Strong parity tooling against Python reference paths.
- **Performance.** Performance-focused execution, reusable sessions, and batch-style offline inference. **Optimized for CUDA**.
- **Portability.** A portable native stack centered on `ggml`, with CLI and server entry points instead of Python-only deployment paths.
- **Pipelines.** Experimental JSON pipeline support for higher-level multi-step workflows.
- **Audio Utilities.** Built-in denoise, enhancement, resampling, and STFT/ISTFT utilities for real production-style task paths.

<p><strong><span style="font-size:1.1em;">The goal of the framework is to provide highly optimized, reusable building blocks for audio-related models, so new model integrations can be brought up faster, shared components can be improved once and benefit many families, and real end-to-end inference paths can stay efficient, maintainable, and portable.</span></strong></p>

audio.cpp would not be moving this quickly without generous contributors bringing in real fixes, new capabilities, and careful polish. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute and for a shout-out to the people already helping shape the project.

> [!TIP]
> **Contribution focus:** the most helpful contributions right now are improvements to the UI, API server, and pipeline/workflow subsystems. These areas make the existing model surface easier to use, serve, compose, and validate. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
>
> **New model PRs:** before starting a new model port, **please check the supported model table because several families are already implemented or under testing**. If you do add a model, follow the validation style in [PR #19](https://github.com/0xShug0/audio.cpp/pull/19): include exact build/run commands, model paths or package ids, generated outputs, parity or path-test results, and relevant performance or memory notes.

## News

> [!IMPORTANT]
> **2026-07-18 - Voxtral Realtime ASR:** Voxtral is now released in audio.cpp with offline and streaming ASR paths. On warmed normal requests, BF16 GGUF runs at **RTF 0.089** (**11.2x realtime**) and Q8_0 at **RTF 0.064** (**15.7x realtime**); CUDA streaming TTFT is about **209 ms** with BF16 and **171 ms** with Q8_0.
>
> **2026-07-14 - Release 0.3:** This release expands audio.cpp with five new TTS families: IndexTTS2, Irodori-TTS, MOSS-TTS-Nano, MOSS-TTS-Local (thanks to [@justinjohn0306](https://github.com/justinjohn0306)), and Supertonic 3. Chatterbox also gains voice-conversion support, extending the existing TTS/voice-cloning path into a fuller speech workflow.
>
> **GGUF support:** audio.cpp now has reusable GGUF loading and conversion support, with tested GGUF paths for multiple ASR and TTS models. Some models can run up to 2× faster with Q8 GGUF, without any parity drift. See [docs/gguf.md](docs/gguf.md) for the current support status. Huge thanks to [@mirek190](https://github.com/mirek190) for driving the core GGUF work and model support forward.

**2026-06-25 to 2026-07-08:** audio.cpp grew from the first released model wave into broad TTS, ASR, music generation, source separation, VAD, diarization, codec, and voice-conversion coverage, with VibeVoice 1.5B/7B, LoRA adapter loading, initial streaming support, and major CUDA Conv1DTransp speedups.

## Supported Models

| Family | Task | Supported language(s) | Supported variant(s) in this repo |
|---|---|---|---|
| **ace_step** | music generation, music editing | 50+ langs | ACE-Step 1.5 Turbo and Base with acestep-5Hz-lm-1.7B |
| **chatterbox** | TTS, voice cloning, voice conversion | ar, da, de, el, en, es, fi, fr, hi, it, ko, ms, nl, no, pl, pt, sv, sw, tr | Chatterbox with 0.5B backbone |
| **citrinet_asr** | ASR | en | Citrinet-256 |
| **fish_audio** | TTS, voice cloning | auto, en, zh | Fish Audio S2 Pro |
| **heartmula** | music generation | zh, en, ja, ko, es | HeartMuLa-oss-3B with HeartCodec-oss |
| **higgs_audio_stt** | ASR | en | Higgs Audio v3 STT |
| **higgs_audio_tts** | TTS, voice cloning | auto | Higgs Audio v3 TTS 4B |
| **htdemucs** | source separation | lang agnostic | HTDemucs, HTDemucs_ft |
| **hviske_asr** | ASR | da | Hviske v5.3 |
| **marblenet_vad** | VAD | lang agnostic | MarbleNet VAD |
| **mel_band_roformer** | vocal separation | lang agnostic | Mel-Band RoFormer MLX vocal separation variants |
| **miocodec** | audio codec, voice conversion backend | lang agnostic | MioCodec v2, 25 Hz, 44.1 kHz |
| **miotts** | TTS, voice cloning | en, ja | MioTTS-1.7B |
| **omnivoice** | TTS, voice cloning, voice design | 646+ langs | OmniVoice, Qwen3-0.6B based |
| **outetts** | TTS, voice cloning | en, ar, zh, nl, fr, de, it, ja, ko, lt, ru, es, pt, be, bn, ka, hu, lv, fa, pl, sw, ta, uk | Llama-OuteTTS-1.0-1B |
| **pocket_tts** | TTS, voice cloning | en, de, it, pt, es | PocketTTS-100M |
| **nemotron_asr** | ASR | 100+ ASR prompt codes incl. auto | Nemotron 3.5 ASR Streaming 0.6B |
| **qwen3_asr** | ASR | zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, ro, hu, mk | Qwen3-ASR-0.6B, Qwen3-ASR-1.7B-hf |
| **qwen3_forced_aligner** | forced alignment | zh, yue, en, de, es, fr, it, pt, ru, ko, ja | Qwen3-ForcedAligner-0.6B |
| **qwen3_tts** | TTS, voice cloning, voice design | zh, en, fr, de, it, ja, ko, pt, ru, es | Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-1.7B-Base, Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| **seed_vc** | voice conversion | lang agnostic | SeedVC XLS-R + HiFT, SeedVC Whisper-small + BigVGAN |
| **silero_vad** | VAD | lang agnostic | Silero VAD |
| **sortformer_diar** | diarization | en | Sortformer-4spk-v1 |
| **stable_audio** | music generation, sound generation, audio editing | en | Stable Audio 3 Small Music, Stable Audio 3 Small SFX, Stable Audio 3 Medium |
| **vevo2** | TTS, singing generation, voice conversion, singing conversion, editing | en, zh | Vevo2 with Qwen2.5-0.5B AR model |
| **vibevoice** | TTS, multi-speaker dialogue TTS | en, zh | VibeVoice-1.5B, VibeVoice-7B |
| **vibevoice_asr** | ASR | auto | VibeVoice ASR |
| **voxtral_realtime** | ASR | auto | Voxtral-Mini-4B-Realtime-2602 |
| **voxcpm2** | TTS, voice cloning, voice design | ar, da, de, el, en, es, fi, fr, he, hi, id, it, ja, km, ko, lo, ms, my, nl, no, pl, pt, ru, sv, sw, th, tl, tr, vi, zh | VoxCPM2-2B, 48 kHz |
| **index_tts2** | TTS, voice cloning, expressive speech | zh, en | IndexTTS-2 |
| **irodori_tts** | TTS, voice cloning, voice design | ja | Irodori-TTS-500M-v3, Irodori-TTS-600M-v3-VoiceDesign |
| **moss_tts_nano** | TTS, voice cloning | auto | MOSS-TTS-Nano-100M |
| **moss_tts_local** | TTS, voice cloning | auto, optional language hint | MOSS-TTS-Local-Transformer-v1.5 |
| **supertonic** | TTS | en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi, na | Supertonic 3 |

## Community Models

Community model ports live under `community_models` to make the ownership boundary clear while keeping them available through the normal audio.cpp CLI and server paths. Huge thanks to the contributors who bring these models in, test them, and keep pushing the framework into new territory. See [docs/community_models/models.md](docs/community_models/models.md) for community-model expectations and current entries.

| Family | Task | Supported language(s) | Contributor | What They Added |
|---|---|---|---|---|
| **outetts** | TTS, voice cloning | en, ar, zh, nl, fr, de, it, ja, ko, lt, ru, es, pt, be, bn, ka, hu, lv, fa, pl, sw, ta, uk | Mirek [@mirek190](https://github.com/mirek190) | Llama-OuteTTS-1.0-1B TTS and voice cloning support |
| **vietneu_tts** | TTS, voice cloning | vi, en | Phuoc [@phuocnguyen90](https://github.com/phuocnguyen90) | [VieNeu-TTS-v3-Turbo](vietneu_tts.md) TTS and voice cloning support |

PocketTTS language selection is a model-load option. When the model path points at the PocketTTS root, the loader uses `english` unless you pass `--load-option language=<name>`. Kyutai's normal non-English PocketTTS releases are smaller distilled language models intended for the fast PocketTTS path. The `_24l` variants are larger 24-layer, undistilled preview models that can sound better but are slower. Kyutai currently publishes French only as `french_24l`, not as a normal distilled `french` language directory, so French is not listed as a normal PocketTTS language here.

## Docker

Docker CPU and CUDA images are available for both CLI and server use. See [Docker.md](Docker.md) for build commands and working Docker examples.

## Build

| OS | Requirements |
|---|---|
| Linux | GCC 13 or newer, CMake, backend toolchain for CUDA or Vulkan builds |
| Windows | Visual Studio Build Tools 2022 or newer with C++ desktop workload, MSVC x64 compiler, Windows SDK, CMake, Ninja, MSVC OpenMP components; official NVIDIA CUDA Toolkit for CUDA builds |
| macOS | Xcode or Xcode Command Line Tools with the Metal compiler available through `xcrun` |

### Linux Build

Use the Linux helper script for CPU, CUDA, or Vulkan builds:

```bash
scripts/build_linux.sh --backend cuda --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend vulkan --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend cpu --target audiocpp_cli --target audiocpp_server
```

The script writes to aligned build directories such as `build/linux-cuda-release`, `build/linux-vulkan-release`, and `build/linux-cpu-release`.

For portable CPU kernels on machines where native ISA flags are not suitable:

```bash
scripts/build_linux.sh --backend cuda --native-cpu OFF --target audiocpp_cli --target audiocpp_server
```

For deployment builds with compiled package specs:

```bash
scripts/build_linux.sh --backend cuda --deployment-build --target audiocpp_cli --target audiocpp_server
```

For direct CMake commands, see [docs/build/linux.md](docs/build/linux.md).

### Windows Build

Use the Windows PowerShell build script:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

Common presets:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_cli
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli
.\scripts\build_windows.ps1 -Target audiocpp_server -Jobs 16
```

From `cmd.exe`, use the wrapper:

```bat
scripts\build_windows.cmd
```

For deployment builds with compiled package specs:

```powershell
.\scripts\build_windows.ps1 -DeploymentBuild -Target audiocpp_cli
```

For requirements, CPU profiles, CUDA packaging, and release zips, see [docs/build/windows.md](docs/build/windows.md).

### Metal Build

On macOS, use the Metal helper script to build against ggml's Metal backend:

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
scripts/build_metal.sh --deployment-build --target audiocpp_cli
```

The built CLI is written to:

```bash
build/macos-metal-release/bin/audiocpp_cli
```

### Build Options

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
| `AUDIOCPP_DEPLOYMENT_BUILD` | Compile package specs into CLI/server binaries for standalone GGUF and package-spec fallback loading. Script builds expose this as `--deployment-build` on Linux/macOS and `-DeploymentBuild` on Windows. | `OFF` |

## Usage

For full setup, CLI, server, and workflow examples, see [docs/usage.md](docs/usage.md).

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
- `--list-loaders` prints registered model families (`--json` for the machine-readable contract)
- `python tools/model_manager.py list --json` prints installable packages; keep it synced with loaders ([docs/maintainers/loader_and_catalog.md](docs/maintainers/loader_and_catalog.md))
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

Some models also have GGUF packages available. Current GGUF repositories include [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf) and [mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp). See [docs/gguf.md](docs/gguf.md) for GGUF support status. A dedicated GGUF model-management tool is under development.

Dependencies:

- Python 3
- `torch`
- `safetensors`
- `PyYAML`
- network access to the upstream model sources

The tool supports three main commands:

- `list` shows the available package ids
- `list --json` prints a machine-readable package catalog
- `info` shows the target layout, required files, and install source for one package
- `info <package> --json` prints machine-readable package details
- `install` downloads or converts one package into a models root

The CLI also exposes the runtime loader catalog with `audiocpp_cli --list-loaders --json`, including task and endpoint metadata added by [PR #74](https://github.com/0xShug0/audio.cpp/pull/74).

Recommended top-level install packages:

`Yes` means Hugging Face has a ready-to-use repo that the framework can download as-is. `No` means the tool must assemble, convert, or post-process files before the framework can use them. Packages whose loaders are not registered in this release tree are listed as **Unavailable** (see [docs/maintainers/loader_and_catalog.md](docs/maintainers/loader_and_catalog.md)).

For shared audio.cpp GGUF packages, the model manager installs the default Q8_0 GGUF. Other precision variants can be downloaded directly from [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf); see [docs/gguf.md](docs/gguf.md) for GGUF support status.

| Package id | Model | HF ready-to-use repo |
|---|---|---|
| `ace_step` | ACE-Step 1.5 Turbo/Base | No |
| `chatterbox` | Chatterbox | **Yes** |
| `citrinet_asr` | Citrinet ASR converted layout | No |
| `fish_audio_s2_pro` | Fish Audio S2 Pro GGUF Q8_0 | **Yes** |
| `heartmula` | HeartMuLa | No |
| `higgs_audio_stt` | Higgs Audio STT | No |
| `higgs_audio_v3_tts_4b` | Higgs Audio v3 TTS 4B GGUF Q8_0 | **Yes** |
| `htdemucs` | HTDemucs | No |
| `hviske_asr` | Hviske ASR | **Yes** |
| `irodori_tts_500m_v3` | Irodori-TTS 500M v3 | No |
| `irodori_tts_600m_v3_voice_design` | Irodori-TTS 600M v3 VoiceDesign | No |
| `index_tts2` | IndexTTS-2 | **Yes** |
| `marblenet_vad` | MarbleNet VAD converted layout | No |
| `mel_band_roformer` | Mel-Band RoFormer MLX | **Yes** |
| `miocodec_25hz_44k_v2` | MioCodec 25Hz 44.1kHz v2 | No |
| `miotts_1_7b` | MioTTS 1.7B | No |
| `moss_audio_tokenizer_nano` | MOSS Audio Tokenizer Nano | No |
| `moss_audio_tokenizer_v2` | MOSS Audio Tokenizer v2 | No |
| `moss_tts_nano_100m` | MOSS-TTS-Nano 100M | No |
| `moss_tts_nano_100m_model` | MOSS-TTS-Nano 100M model subcomponent | No |
| `moss_tts_local_v1_5` | MOSS-TTS-Local Transformer v1.5 | No |
| `nemotron_asr` | Nemotron ASR | **Yes** |
| `omnivoice` | OmniVoice | **Yes** |
| `outetts_1_0_1b` | OuteTTS 1.0 1B with IBM DAC codec and Qwen3-aligned voice cloning | No |
| `pocket_tts` | PocketTTS | **Yes** |
| `qwen3_asr_0_6b` | Qwen3 ASR 0.6B | **Yes** |
| `qwen3_asr_1_7b_hf` | Qwen3 ASR 1.7B HF | **Yes** |
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
| `vietneu_tts_v3_turbo` | VieNeu-TTS v3 Turbo | **Yes** |
| `vibevoice_1_5b` | VibeVoice 1.5B | No |
| `vibevoice_7b` | VibeVoice 7B | No |
| `vibevoice_asr` | VibeVoice ASR | No |
| `voxtral_realtime` | Voxtral Mini 4B Realtime GGUF Q8_0 | **Yes** |
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
cmake --build build -j$(nproc) --target audiocpp_server
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

- [TranscrIA](https://github.com/Martossien/transcria) is a self-hosted meeting transcription platform with diarization and local LLM correction. audio.cpp is integrated as a first-class STT engine in the product.
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
  - `moss_tts_local`: **2.33x faster** with **57.07% less wall time**
  - `qwen3 tts`: **1.83x faster** with **45.34% less wall time**
  - `vibevoice`: **1.40x faster** with **28.75% less wall time**
- In long-lived-session runs, where the same loaded session serves multiple requests in sequence, the gains stay strong:
  - `pocket tts`: **3.22x faster** with **68.91% less wall time**
  - `qwen3 tts`: **2.74x faster** with **63.47% less wall time**
  - `moss_tts_local`: **2.66x faster** with **62.35% less wall time**
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
| miotts | 20.40 | 3.30 | 0.162 | 6.18x |
| moss_tts_local | 9.60 | 0.97 | 0.101 | 9.91x |
| omnivoice | 9.00 | 1.32 | 0.146 | 6.84x |
| pocket tts | 8.08 | 0.26 | 0.032 | 31.09x |
| qwen3 tts | 11.44 | 4.46 | 0.390 | 2.56x |
| vevo2 | 8.66 | 2.47 | 0.285 | 3.51x |
| vibevoice | 11.07 | 5.02 | 0.454 | 2.20x |
| voxcpm2 | 5.60 | 3.09 | 0.551 | 1.81x |

For long-form TTS tests, each run uses the same 6,026-character, 1,028-word input text (vibevoice uses 106,310 chars, 18,052 words, 4 speakers). Rows are CUDA unless marked CPU. The measured RTF is:

| model | audio len (s) | wall time (s) | RTF | x faster than real time |
|---|---:|---:|---:|---:|
| chatterbox | 391.24 | 58.57 | 0.150 | 6.68x |
| index tts2 | 422.12 | 139.95 | 0.332 | 3.02x |
| miotts | 399.16 | 66.59 | 0.167 | 5.99x |
| moss_tts_nano | 391.20 | 43.16 | 0.110 | 9.06x |
| moss_tts_local | 375.44 | 73.84 | 0.197 | 5.08x |
| omnivoice | 357.00 | 17.77 | 0.050 | 20.09x |
| pocket tts | 353.12 | 7.30 | 0.021 | 48.40x |
| qwen3 tts | 327.60 | 72.65 | 0.222 | 4.51x |
| supertonic | 379.32 | 2.02 | 0.005 | 187.62x |
| supertonic (CPU) | 379.40 | 61.40 | 0.162 | 6.18x |
| vevo2 | 457.68 | 52.47 | 0.115 | 8.72x |
| voxcpm2 | 315.84 | 72.70 | 0.230 | 4.34x |
| vibevoice | 5615.73 | 1376.84 | 0.245 | 4.08x |

## Runtime Memory Options

Some models expose memory-saver session options such as `ace_step.mem_saver=true`, `heartmula.mem_saver=true`, `stable_audio.mem_saver=true`, `omnivoice.mem_saver=true`, and `voxcpm2.mem_saver=true`. These options keep the default output path unchanged while reducing graph workspace VRAM or releasing staged graph/cache state after request phases; later requests may rebuild released graphs.

## Precision/Quantization Support

Many model sessions expose quantization through `--session-option <family>.weight_type=<mode>`, and some families also expose more specific knobs such as `...conv_weight_type`, `...talker_weight_type`, or `...speech_decoder_weight_type`. The exact supported modes are model-specific rather than global.

The framework also has a reusable GGUF tensor source and a streaming converter. The
container reader is shared by all model families; a family still has to list a `.gguf`
checkpoint as one of its accepted assets because model configuration and tensor naming
remain architecture-specific. Qwen3 ASR, Qwen3 Forced Aligner, Qwen3 TTS, Nemotron
3.5 ASR, VibeVoice-ASR, Higgs Audio STT, Hviske ASR, Citrinet ASR, and OuteTTS currently accept
`model.gguf` (including `speech_tokenizer/model.gguf` for TTS). The converter recursively embeds sidecar files
up to 64 MiB by default using binary-safe metadata, including nested tokenizer models,
and Qwen3 ASR, Nemotron ASR, VibeVoice-ASR, Higgs Audio STT, Hviske ASR, Citrinet ASR, and OuteTTS
can load the resulting `model.gguf` as a standalone file. The converter embeds the selected
package spec in new GGUF files. Standalone conversion with embedded sidecars is the default
and fails if required package resources are missing. Pass `--no-sidecars` only to explicitly
create a tensor-only container; its package spec is still embedded and validated. A
`model.safetensors.index.json` is also a first-class tensor source and is merged from
its routed shards while converting. Exact original tensor ranks are stored separately
because GGML normally collapses trailing singleton dimensions. Rank-0 safetensors
scalars are stored physically as one-element GGML tensors while their scalar rank is
preserved in the exact-shape metadata.

| Format | Package spec source | External model files |
|---|---|---:|
| Safetensors | Override, deployment binary, or discovered `model_specs` | Yes |
| New standalone GGUF | Embedded in GGUF | No |
| New tensor-only GGUF created with `--no-sidecars` | Embedded in GGUF | Yes, required sidecars |
| Legacy GGUF without embedded spec | Deployment binary or discovered `model_specs` | Depends on sidecars |

At runtime the order is explicit override, GGUF metadata, compiled deployment spec, then
external discovery. Configure with `-DAUDIOCPP_DEPLOYMENT_BUILD=ON` to compile the source
catalog into CLI/server binaries; the option is off by default. For package-layout
development or testing, the CLI and server can explicitly replace every fallback with
`--model-spec-override <json-or-directory>`. When a directory is supplied, the runtime
selects `<directory>/<family>.json`. The server configuration also accepts
`model_spec_override` globally or per model. An override is trusted runtime input and
should only point to a spec you control.

```bash
build/bin/audiocpp_gguf \
  --input models/Qwen3-ASR-1.7B-hf/model.safetensors \
  --family qwen3_asr \
  --output models/Qwen3-ASR-1.7B-hf/model.gguf \
  --type q8_0
```

The converter discovers the spec from `--model-spec`, model `config.json`, the model
root, a discovered external catalog, or its bundled conversion catalog. The converter
catalog is always embedded in `audiocpp_gguf` even when `AUDIOCPP_DEPLOYMENT_BUILD` is
off; that option controls the CLI/server fallback catalog. The converter validates the
requested tensor namespaces and every required GGUF sidecar before writing. Use
`--allow-missing-model-spec` only for a generic tensor archive that is not intended to be
loaded by audio.cpp.

Multi-component checkpoints can be packed into one GGUF with repeated namespaced
inputs. Existing component loaders can open a namespace through
`open_tensor_source(path, "component")`, which strips that prefix from the view:

```bash
build/bin/audiocpp_gguf \
  --input gpt=models/index-tts2-mlx/gpt.safetensors \
  --input s2mel=models/index-tts2-mlx/s2mel.safetensors \
  --input bigvgan=models/index-tts2-mlx/bigvgan/model.safetensors \
  --root models/index-tts2-mlx \
  --sidecar models/shared/preprocessor_config.json=preprocessor_config.json \
  --output models/index-tts2-mlx-GGUF/model.gguf \
  --type q8_0
```

`--root` selects the directory whose non-weight sidecars are embedded. Repeat
`--sidecar source=destination` for required assets outside that root, or to remap an
asset such as Higgs Audio STT's shared Whisper `preprocessor_config.json` into the
standalone model root. For Higgs, that external Whisper file is needed only during GGUF
creation; it is not required when loading the completed standalone GGUF.

Packing is a container feature; it does not by itself wire every model loader to the new
layout. The packed IndexTTS-2 MLX checkpoint has been conversion-tested, including its
CAMPPlus rank-0 counters, but the existing IndexTTS-2 runtime still has to open and consume
the corresponding namespaces before audio.cpp can synthesize directly from that file.

Supported conversion types are `f16`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, and
`q6_k`. Quantized GGUF files use mixed precision: projection matrices are quantized,
while embedding/codebook lookup tables and unsupported shapes retain a backend-safe
type. If both files exist, Qwen loaders prefer `model.gguf` over `model.safetensors`.

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
- GGUF is a container, not a universal architecture adapter. Existing llama.cpp or
  whisper.cpp GGUF files are not automatically compatible unless their tensor names and
  model metadata are mapped to the audio.cpp family implementation.
- `Build_xcframework.sh` is outdated; Metal and Apple XCFramework packaging still need to be retested after the framework refactor.
