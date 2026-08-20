# audio.cpp

[![0xShug0/audio.cpp | Trendshift](https://trendshift.io/api/badge/trendshift/repositories/64983/daily?language=C%2B%2B)](https://trendshift.io/repositories/64983?utm_source=trendshift-badge&utm_medium=badge&utm_campaign=badge-trendshift-64983)

`audio.cpp` is a high-performance C++ audio inference framework built on top of `ggml`, designed to make modern local audio models practical, portable, and fast.

Tired of juggling a dozen Conda environments, hundreds of Python packages, and dependency conflicts just to try a few audio models? audio.cpp gives those paths a shared native runtime instead. Runs on Windows, Linux, and macOS, with support for NVIDIA, AMD, Apple Silicon, and CPU-only machines.

> [!IMPORTANT]
> **2026-08-18**: MiniMax Music3 has been merged from the `preview` branch into `main`. Please use the version from `main` going forward.
>
> **CUDA performance headline:** multiple TTS paths already run **1.8x to up to 8x faster than their Python reference paths** while cutting end-to-end latency by **45%-85%**.
>
> **GGUF performance:** all released model families support GGUF loading, and tested Q8 packages can run up to **1.53x faster** while reducing peak VRAM by up to about **37%** on routes such as Higgs Audio, Fish Audio, and Voxtral. See the [GGUF guide](docs/gguf.md) for support status and the [Q8 performance report](docs/reports/gguf_q8_performance.md) for 16-bit vs Q8 measurements.
>
> **Production deployment example:** Try Fun-ASR-Nano with audio.cpp on the FunASR platform https://www.funasr.com/en/deploy/audio-cpp.html!
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
- **Portability.** A portable native stack centered on `ggml`, with CUDA, HIP/ROCm, Vulkan, Metal, and CPU backends behind shared CLI and server entry points instead of Python-only deployment paths.
- **Pipelines.** Experimental JSON pipeline support for higher-level multi-step workflows.
- **Audio Utilities.** Built-in denoise, enhancement, resampling, and STFT/ISTFT utilities for real production-style task paths.

<p><strong><span style="font-size:1.1em;">The goal of the framework is to provide highly optimized, reusable building blocks for audio-related models, so new model integrations can be brought up faster, shared components can be improved once and benefit many families, and real end-to-end inference paths can stay efficient, maintainable, and portable.</span></strong></p>

audio.cpp would not be moving this quickly without generous contributors bringing in real fixes, new capabilities, and careful polish. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute and for a shout-out to the people already helping shape the project.

> [!TIP]
> **Contribution focus:** the most helpful contributions right now are improvements to the UI, API server, and pipeline/workflow subsystems. These areas make the existing model surface easier to use, serve, compose, and validate. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
>
> **New model PRs:** before starting a new model port, **please check the supported model table because several families are already implemented or under testing**. New ports should start under the [community models](docs/community_models/models.md) surface, where review is lighter than core models but still needs reproducible validation. Please follow the measurement style in [PR #19](https://github.com/0xShug0/audio.cpp/pull/19) and [PR #63](https://github.com/0xShug0/audio.cpp/pull/63): exact build/run commands, model paths or package ids, generated outputs, parity or path-test results, and relevant performance or memory notes.

## News

> [!IMPORTANT]
> **2026-08-18 - Release 0.6.1:** This update adds MiniMax Music 3 docs, DotTTS Edit docs, ACE-Step 1.5 XL notes, and WebUI/Docker launch guidance, bringing audio.cpp to **50** total model families and **70+** model variants.
>
> **2026-08-13 - Release 0.6:** This release adds **5** new model families - DotTTS, NeuTTS, MuScriptor, MiniMax-H3, and SenseVoice - bringing audio.cpp to **49** total model families and **70+** model variants, alongside the new native WebUI from [@mirek190](https://github.com/mirek190), expanded GGUF packaging, and more shared framework runtime pieces.
>
> **2026-07-31 - Release 0.5:** audio.cpp grows to **44 model families** with **9 new additions**: DramaBox, Confucius4-TTS, RVC, BS-RoFormer, GLM-TTS, Kroko ASR, Parakeet-TDT, Inflect v2, and Fun-ASR-Nano.
>
> **HIP/ROCm support:** Platform coverage also takes a big step forward! Early HIP/ROCm support lands for AMD GPUs thanks to [@IIIIIllllIIIIIlllll](https://github.com/IIIIIllllIIIIIlllll), with Nix ROCm/HIP build support from [@francescobozzo](https://github.com/francescobozzo).
>
> **Metal performance boost:** Thanks to [@liuzl](https://github.com/liuzl), Metal ops were optimized, making tested VoxCPM2 end-to-end runs up to **2.56x faster** on Apple Silicon. Many models should benefit from these optimizations.
>
> This release is a major GGUF-first usability pass. The WebUI now uses model-spec package links for downloads, prefers standalone GGUF packages when available, and handles more models directly from the normal UI flow. New schema-v1 specs make model options, packages, metadata, and UI-facing behavior much easier to keep in sync.
>
> **2026-07-23 - Release 0.4:** audio.cpp expanded to **35 model families**, adding Higgs Audio v3 TTS 4B, Fish Audio S2 Pro, Voxtral Realtime ASR, community OuteTTS and VieNeu-TTS, broader GGUF/package-spec support, reusable framework improvements, and the integrated WebUI thanks to [@kigner](https://github.com/kigner) and [@patrickjchen](https://github.com/patrickjchen).
>
> **2026-07-14 - Release 0.3:** This release added IndexTTS2, Irodori-TTS, MOSS-TTS-Nano, MOSS-TTS-Local, Supertonic 3, Chatterbox voice conversion, and the first broad GGUF loading/conversion wave. Thanks to [@justinjohn0306](https://github.com/justinjohn0306) for MOSS-TTS-Local and [@mirek190](https://github.com/mirek190) for driving GGUF forward.

**2026-06-25 to 2026-07-08:** audio.cpp grew from the first released model wave into broad TTS, ASR, music generation, source separation, VAD, diarization, codec, and voice-conversion coverage, with VibeVoice 1.5B/7B, LoRA adapter loading, initial streaming support, and major CUDA Conv1DTransp speedups.

## Supported Models

Task tags: `TTS` text to speech, `Clone` voice cloning, `VC` voice conversion, `S2S` speech-to-speech, `ASR` speech recognition, `Align` forced alignment, `VAD` voice activity detection, `Diar` speaker diarization, `Codec` audio codec, `Sep` source separation, `MIDI` audio-to-symbolic MIDI/events, `Music` music/song generation, `SFX` sound effects, `Video` video generation, `Edit` audio/music editing, `Design` voice design, `Dialogue` multi-speaker dialogue TTS, `Ctrl` TTS/clone voice control such as emotion, style, instruction, caption, or non-verbal tag control.

Runtime tags summarize the supported loading paths. GGUF package precision varies by model and release; check the [audio.cpp GGUF repo](https://huggingface.co/audio-cpp/audio.cpp-gguf) or [docs/gguf.md](docs/gguf.md) for the exact package list. `Bundled` means the tiny runtime asset ships under `assets/framework/models` and needs no separate model download. `Stream` means the family exposes a streaming server/session path.

### Speech Generation And Conversation

| Family | Task | Lang | Variants | Runtime |
|---|---|---|---|---|
| **chatterbox** | TTS, Clone, VC| ar, da, de, el, en, es, fi, fr, hi, it, ko, ms, nl, no, pl, pt, sv, sw, tr | Chatterbox with 0.5B backbone | GGUF 16/Q8 |
| **confucius4_tts** | Clone | zh, en, ja, ko, de, fr, es, id, it, th, pt, ru, ms, vi | Confucius4-TTS multilingual voice cloning | GGUF F32, Stream |
| **dots_tts** | TTS, Clone, Edit, Ctrl | multilingual | DotTTS SOAR, MeanFlow, and Edit | GGUF 16/Q8, Stream |
| **dramabox** | TTS, Clone | en | DramaBox expressive TTS and voice cloning | GGUF Q8 |
| **fish_audio** | TTS, Clone, Ctrl | auto, en, zh | Fish Audio S2 Pro | GGUF 16/Q8 |
| **higgs_audio_tts** | TTS, Clone, Ctrl | auto | Higgs Audio v3 TTS 4B | GGUF 16/Q8 |
| **index_tts2** | TTS, Clone, Ctrl | zh, en, ja, es, ar | IndexTTS-2, IndexTTS-2.5 (variant) | GGUF 16/Q8 |
| **irodori_tts** | TTS, Clone, Design, Ctrl | ja | Irodori-TTS-v4-Small, Irodori-TTS-500M-v3, Irodori-TTS-600M-v3-VoiceDesign | GGUF 16/Q8 |
| **magpie_tts** | TTS | ar-AE, ar-MSA, ar-SA, de, en, es, fr, hi, it, ko, pt-BR, vi, zh | NVIDIA MagpieTTS Multilingual 357M (v2607) with baked speaker prompts and NanoCodec decode | GGUF original/Q8 |
| **miotts** | TTS, Clone | en, ja | MioTTS-1.7B | GGUF 16/Q8 |
| **moss_tts_local** | TTS, Clone, Ctrl | auto, optional language hint | MOSS-TTS-Local-Transformer-v1.5 | GGUF 16/Q8 |
| **moss_tts_nano** | TTS, Clone | auto | MOSS-TTS-Nano-100M | GGUF 16/Q8 |
| **neutts** | TTS, Ctrl | en | NeuTTS 2E with built-in speaker prompts and emotion control | GGUF original precision, Stream |
| **omnivoice** | TTS, Clone, Design, Ctrl | 646+ langs | OmniVoice, Qwen3-0.6B based | GGUF 16/Q8, Stream |
| **personaplex** | Dialogue, S2S | en | PersonaPlex 7B v1 speech-to-speech conversational model with packaged voice/persona prompts | GGUF Q4/Q8, Stream |
| **pocket_tts** | TTS, Clone | en, de, it, pt, es | PocketTTS-100M | GGUF 16/Q8 |
| **qwen3_tts** | TTS, Clone, Design, Ctrl | zh, en, fr, de, it, ja, ko, pt, ru, es | Qwen3-TTS-12Hz-0.6B-Base, Qwen3-TTS-12Hz-1.7B-Base, Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen3-TTS-12Hz-1.7B-VoiceDesign | GGUF 16/Q8 |
| **supertonic** | TTS | en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi, na | Supertonic 3 | GGUF F32, Stream |
| **vibevoice** | TTS, Dialogue | en, zh | VibeVoice-1.5B, VibeVoice-7B | GGUF 16/Q8 |
| **voxcpm2** | TTS, Clone, Design, Ctrl | ar, da, de, el, en, es, fi, fr, he, hi, id, it, ja, km, ko, lo, ms, my, nl, no, pl, pt, ru, sv, sw, th, tl, tr, vi, zh | VoxCPM2-2B, 48 kHz | GGUF 16/Q8, Stream |

### Speech Recognition And Analysis

| Family | Task | Lang | Variants | Runtime |
|---|---|---|---|---|
| **citrinet_asr** | ASR | en | Citrinet-256 | GGUF Q8 |
| **fun_asr_nano** | ASR | auto, zh, en, ja | Fun-ASR-Nano-2512 | GGUF 16/Q8 |
| **higgs_audio_stt** | ASR | en | Higgs Audio v3 STT | GGUF 16/Q8, Stream |
| **hviske_asr** | ASR | da | Hviske v5.3 | GGUF Q8 |
| **marblenet_vad** | VAD | lang agnostic | MarbleNet VAD | Bundled |
| **nemotron_asr** | ASR | 100+ ASR prompt codes incl. auto | Nemotron 3.5 ASR Streaming 0.6B | GGUF 16/Q8, Stream |
| **qwen3_asr** | ASR | zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, ro, hu, mk | Qwen3-ASR-0.6B, Qwen3-ASR-1.7B-hf | GGUF 16/Q8, Stream |
| **qwen3_forced_aligner** | Align | zh, yue, en, de, es, fr, it, pt, ru, ko, ja | Qwen3-ForcedAligner-0.6B | GGUF 16/Q8 |
| **silero_vad** | VAD | lang agnostic | Silero VAD | Bundled, Stream |
| **sortformer_diar** | Diar | en | Sortformer-4spk-v1 | - |
| **vibevoice_asr** | ASR | auto | VibeVoice ASR | GGUF 16/Q8 |
| **voxtral_realtime** | ASR | auto | Voxtral-Mini-4B-Realtime-2602 | GGUF 16/Q8/Q4, Stream |

### Audio Conversion And Processing

| Family | Task | Lang | Variants | Runtime |
|---|---|---|---|---|
| **bs_roformer** | Sep | lang agnostic | BS-RoFormer vocal separation checkpoints | GGUF Q8 |
| **htdemucs** | Sep | lang agnostic | HTDemucs, HTDemucs_ft | GGUF 16/Q8 |
| **meanvc2** | VC | lang agnostic | MeanVC2 120 ms/40 ms zero-shot voice conversion | GGUF F32/Q4, Stream |
| **mel_band_roformer** | Sep | lang agnostic | Mel-Band RoFormer MLX vocal separation variants | GGUF 16/Q8 |
| **miocodec** | Codec, VC | lang agnostic | MioCodec v2, 25 Hz, 44.1 kHz | GGUF 16/Q8 |
| **muscriptor** | MIDI | music | MuScriptor Small audio-to-symbolic transcription | GGUF F32, Stream |
| **rvc** | VC | lang agnostic | RVC F16 GGUF with packaged v1/v2 voices and optional retrieval blending | GGUF 16 |
| **seed_vc** | VC | lang agnostic | SeedVC XLS-R + HiFT, SeedVC Whisper-small + BigVGAN | GGUF 16/Q8 |

### Music, Media, And Editing

| Family | Task | Lang | Variants | Runtime |
|---|---|---|---|---|
| **ace_step** | Music, Edit | 50+ langs | ACE-Step 1.5 Turbo/Base and XL Turbo/SFT with acestep-5Hz-lm-1.7B | GGUF 16 |
| **heartmula** | Music | zh, en, ja, ko, es | HeartMuLa-oss-3B with HeartCodec-oss | GGUF 16/Q8 |
| **minimax_h3** | Video, Music, TTS/Dialogue | auto | MiniMax-H3 Q4_K with optional INT8 ConvRot DiT | GGUF Q4/INT8 |
| **minimax_music3** | Music | auto | MiniMax Music 3 text-to-music generation with lyrics conditioning | GGUF Q4/Q8 |
| **stable_audio** | Music, SFX, Edit | en | Stable Audio 3 Small Music, Stable Audio 3 Small SFX, Stable Audio 3 Medium | GGUF 16/Q8 |
| **vevo2** | TTS, Music, VC, Edit | en, zh | Vevo2 with Qwen2.5-0.5B AR model | GGUF 16 |

Some model families in the supported table started as outside contributions before being promoted into the core release surface. Thanks to Mirek [@mirek190](https://github.com/mirek190) for BS-RoFormer, [@justinjohn0306](https://github.com/justinjohn0306) for MOSS-TTS-Local, and [@LauraGPT](https://github.com/LauraGPT) from the official FunASR team for Fun-ASR-Nano.

## Community Models

Community model ports live under `community_models` to make the ownership boundary clear while keeping them available through the normal audio.cpp CLI and server paths. Some community-contributed models graduate into the core model tree when they become part of the main release surface. Huge thanks to the contributors who bring these models in, test them, and keep pushing the framework into new territory. See [docs/community_models/models.md](docs/community_models/models.md) for community-model expectations and current entries.

| Family | Task | Lang | Runtime | Contributor | What They Added |
|---|---|---|---|---|---|
| **glm_tts** | TTS, Clone | zh, en | GGUF | Mirek [@mirek190](https://github.com/mirek190) | [GLM-TTS](docs/community_models/glm_tts.md) zero-shot synthesis and voice cloning support |
| **inflect_v2** | TTS | en | GGUF FP32 | Jan [@JanWerder](https://github.com/JanWerder) | [Inflect Micro v2 and Nano v2](docs/community_models/inflect_v2.md) native offline synthesis |
| **kroko_asr** | ASR | de, en, es, fr, it, he, nl, pt, sv, tr | Safetensors, GGUF Q8 | Mirek [@mirek190](https://github.com/mirek190) | [Kroko Community ASR](docs/community_models/kroko_asr.md) native offline/streaming Zipformer2/RNN-T transcription with word timestamps |
| **minimax_h3** | Video, Music, TTS/Dialogue | auto | GGUF Q4/INT8 | [@0xShug0](https://github.com/0xShug0) | [MiniMax-H3](docs/community_models/minimax_h3.md) text-to-audio/video generation with Q4_K and optional INT8 ConvRot DiT |
| **minimax_music3** | Music | auto | GGUF Q4/Q8 | [@0xShug0](https://github.com/0xShug0), [@JoeMattie](https://github.com/JoeMattie) | [MiniMax Music 3](docs/community_models/minimax_music3.md) text-to-music generation with lyrics conditioning |
| **moss_tts_local** | TTS, Clone, Ctrl | auto, optional language hint | GGUF | [@justinjohn0306](https://github.com/justinjohn0306) | MOSS-TTS-Local Transformer v1.5 support |
| **outetts** | TTS, Clone | en, ar, zh, nl, fr, de, it, ja, ko, lt, ru, es, pt, be, bn, ka, hu, lv, fa, pl, sw, ta, uk | GGUF | Mirek [@mirek190](https://github.com/mirek190) | Llama-OuteTTS-1.0-1B TTS and voice cloning support |
| **parakeet_tdt** | ASR | auto, bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk | GGUF F32/16/Q8, Stream | [@dleiferives](https://github.com/dleiferives) | [Parakeet-TDT 0.6B v3](docs/community_models/parakeet_tdt.md) offline, long-form, and buffered-streaming ASR support |
| **sense_asr** | ASR | auto, zh, en, yue, ja, ko, pt, ru, es, it, fr, de, nl, pl, tr, ar, hi, vi, th, id, ms, fa, nospeech | GGUF Q8, Stream | Jason Chen [@jasonchen31](https://github.com/jasonchen31), [@LauraGPT](https://github.com/LauraGPT) / FunASR | [SenseVoice-Small](docs/community_models/sense_asr.md) offline/streaming SAN-M + CTC transcription with event/emotion/language tags and ITN |
| **vietneu_tts** | TTS, Clone | vi, en | GGUF | Phuoc [@phuocnguyen90](https://github.com/phuocnguyen90) | [VieNeu-TTS-v3-Turbo](docs/community_models/vietneu_tts.md) TTS and voice cloning support |

## Docker

Docker CUDA and CPU images are available for both CLI and server use. See [docker.md](docs/docker.md) for
available images, build commands and working Docker examples.

## Model Manager and GGUF Downloads

Use `tools/model_manager_v2.py` for normal model downloads. It reads
`model_specs/*.json` and installs the default package for each family, preferring
ready-to-use GGUF packages when they are available.

Native builds configured with `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` also
provide `audiocpp_model_manager`, a standalone headless
frontend over the same reusable C++ package-management library used by the
server and embedded WebUI. It supports list, info, install, clean, and remove
without starting `audiocpp_server`; this is the preferred native path for CLI,
Docker, remote provisioning, and other scripted environments. The Python v2
manager remains available as an alternative during migration. The native path
uses bundled TLS by default and does not require libcurl.

```bash
audiocpp_model_manager list
audiocpp_model_manager install qwen3_asr_0_6b_q8_0 --models-dir models
```

The old safetensors/converter catalog has been renamed to
`tools/model_manager_deprecated.py`. Use it only for legacy model layouts that
have not moved to spec-backed GGUF packages yet.

GGUF downloads:

- Released model packages: [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf)
- Community model package: [mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp)

See the [Model Manager guide](docs/model_manager.md) for model-manager usage and
package notes.

## WebUI
![Maintained by contributors](https://img.shields.io/badge/maintained%20by-contributors-brightgreen)

`audiocpp_server` includes an embedded SvelteKit/TypeScript WebUI for running local TTS, cloning, ASR,
generation, conversion, separation, VAD, diarization, and alignment workflows. The production UI is compiled
into the server binary, so using it requires neither Python nor separate frontend files:

```bash
audiocpp_server --ui --ui-management --backend cuda
```

Open `http://127.0.0.1:8080`. `--ui-management` enables catalog browsing, downloads, temporary
uploads, and dynamic model switching, so this is the easiest way to try audio.cpp without writing a
server config first. For a locked-down config-driven instance, start with `--ui` and a server config
instead; in that mode the UI only offers models declared by the config unless management is enabled.

For Docker UI downloads, mount a writable models directory at `/app/models` and
use `--ui-management`. See [docker.md](docs/docker.md#native-webui).

The native UI also exposes background model download/preparation, long-text split-and-merge synthesis, a
browser-local saved voice library, microphone recording, and near-live ASR input. Some model preparation jobs invoke
the repository's Python model manager because those packages require Hugging Face download or checkpoint conversion;
model inference and the embedded UI remain Python-free. See [webui/README.md](webui/README.md) for launch commands,
model notes, and frontend development instructions.

## Prebuilt Binaries

- **Windows (CUDA / CPU):** official packages on the [Releases page](https://github.com/0xShug0/audio.cpp/releases).
- **Windows (HIP/ROCm, AMD GPUs):** community-maintained packages with the ROCm runtime bundled — no HIP SDK installation required. Published from [@IIIIIllllIIIIIlllll's fork Releases](https://github.com/IIIIIllllIIIIIlllll/audio.cpp/releases) in two tracks: ROCm 6.4 (full coverage incl. RX 7600 / gfx1102) and ROCm 7.1 (recommended for RDNA4). Version numbers follow the upstream releases; see [docs/build/windows-hip-distribution.md](docs/build/windows-hip-distribution.md) for details.

## Build

| OS | Requirements |
|---|---|
| Linux | GCC 13 or newer, CMake, plus the backend toolchain for the build you want: NVIDIA CUDA Toolkit for CUDA, Vulkan SDK for Vulkan, ROCm for HIP |
| Windows | Visual Studio Build Tools 2022 or newer with C++ desktop workload, MSVC x64 compiler, Windows SDK, CMake, Ninja, MSVC OpenMP components; official NVIDIA CUDA Toolkit for CUDA builds, Vulkan SDK for Vulkan builds, AMD HIP SDK for HIP builds |
| macOS | Xcode or Xcode Command Line Tools, plus CMake. Metal builds also require the Metal compiler available through `xcrun` |

### Homebrew Install

On macOS, audio.cpp can be installed from the Homebrew tap:

```bash
brew tap 0xShug0/audio-cpp
brew trust 0xShug0/audio-cpp
brew install audio-cpp
```

For Nix and NixOS builds, see [docs/build/nixos.md](docs/build/nixos.md).

### Composite Builds

Composite builds let you compile only the model families you need. `full` is the default and is what release/Docker builds should use. `custom` registers only the requested loaders while still linking required internal dependencies; `core` builds the runtime without the optional model-family set.

The helper scripts expose this as `--model-set` and `--models` on Linux/macOS, and `-ModelSet` and `-Models` on Windows:

```bash
scripts/build_linux.sh --backend cuda --model-set custom --models qwen3_tts,pocket_tts,qwen3_asr --target audiocpp_cli
```

Direct CMake builds use the same underlying variables:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=qwen3_tts,pocket_tts,qwen3_asr
cmake --build build/debug --target audiocpp_cli -j 8
```

### Linux Build

Use the Linux helper script for CPU, CUDA, Vulkan, or HIP builds:

```bash
scripts/build_linux.sh --backend cuda --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend vulkan --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend hip --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend cpu --target audiocpp_cli --target audiocpp_server
```

The script writes to aligned build directories such as `build/linux-cuda-release`, `build/linux-vulkan-release`, `build/linux-hip-release`, and `build/linux-cpu-release`.

Composite examples:

```bash
scripts/build_linux.sh --backend cuda --model-set full --target audiocpp_cli
scripts/build_linux.sh --backend cuda --model-set custom --models qwen3_tts,pocket_tts,qwen3_asr --target audiocpp_cli
scripts/build_linux.sh --backend cpu --model-set core --target audiocpp_cli
```

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
.\scripts\build_windows.ps1 -Preset windows-vulkan-release -Target audiocpp_cli
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli
.\scripts\build_windows.ps1 -Target audiocpp_server -Jobs 16
.\scripts\build_windows.ps1 -Preset windows-cuda-release -ModelSet custom -Models "qwen3_tts,pocket_tts,qwen3_asr" -Target audiocpp_cli
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

### macOS CPU Build

Apple builds enable Metal by default. To build a CPU-only binary, disable Metal explicitly:

```bash
cmake -S . -B build/macos-cpu-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=OFF \
  -DENGINE_ENABLE_VULKAN=OFF \
  -DENGINE_ENABLE_METAL=OFF \
  -DENGINE_ENABLE_OPENMP=OFF \
  -DGGML_OPENMP=OFF
cmake --build build/macos-cpu-release \
  --parallel "$(sysctl -n hw.logicalcpu)" \
  --target audiocpp_cli audiocpp_server audiocpp_gguf
```

Confirm that the resulting CLI sees the host CPU backend:

```bash
build/macos-cpu-release/bin/audiocpp_cli --list-devices
```

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
scripts/build_metal.sh --model-set custom --models qwen3_tts,pocket_tts --target audiocpp_cli
scripts/build_metal.sh --with-tests --target audio_dsp_test
scripts/build_metal.sh --openmp auto --target audiocpp_cli
scripts/build_metal.sh --native-cpu OFF --target audiocpp_cli
scripts/build_metal.sh --deployment-build --target audiocpp_cli
```

The built CLI is written to:

```bash
build/macos-metal-release/bin/audiocpp_cli
```

### HIP/ROCm Build

On Linux and Windows, HIP builds compile ggml's CUDA backend sources as HIP code for AMD GPUs. `ENGINE_ENABLE_HIP` and `ENGINE_ENABLE_CUDA` are mutually exclusive — configure with exactly one of them.

Linux (the helper script auto-detects ROCm via `ROCM_PATH`/`HIP_PATH`/hipconfig and local GPU targets via `amdgpu-arch`, falling back to `rocminfo`; pass `--gpu-targets` to build for other architectures, or when no AMD GPU is visible, e.g. in a VM or container):

```bash
scripts/build_linux.sh --backend hip --target audiocpp_cli --target audiocpp_server
scripts/build_linux.sh --backend hip --gpu-targets "gfx1100;gfx1103" --target audiocpp_cli
```

Direct CMake:

```bash
cmake -S . -B build_hip \
  -DENGINE_ENABLE_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DCMAKE_C_COMPILER="$(hipconfig -l)/clang" \
  -DCMAKE_CXX_COMPILER="$(hipconfig -l)/clang++" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip -j$(nproc)
```

Windows (the helper script auto-detects ROCm, GPU targets, cmake, and ninja):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_hip.ps1
```

Run with `--backend hip` (`rocm` is accepted as an alias). For GPU target selection, hipBLASLt GEMM notes, iGPU tuning, and known limitations, see [docs/build/HIP.md](docs/build/HIP.md).

### Build Options

| Option | Meaning | Default |
|---|---|---|
| `ENGINE_ENABLE_CUDA` | Enable the ggml CUDA backend. Required for `--backend cuda`. | `OFF` |
| `ENGINE_ENABLE_HIP` | Enable the ggml HIP backend (AMD GPUs). Required for `--backend hip`; mutually exclusive with `ENGINE_ENABLE_CUDA`. | `OFF` |
| `ENGINE_ENABLE_VULKAN` | Enable the ggml Vulkan backend. Required for `--backend vulkan`. | `OFF` |
| `ENGINE_ENABLE_METAL` | Enable the ggml Metal backend. Required for `--backend metal`. | `OFF` on most platforms, `ON` on Apple |
| `ENGINE_ENABLE_LLAMAFILE` | Enable llamafile SGEMM support in ggml CPU builds. | `ON` |
| `ENGINE_ENABLE_CUDA_GRAPHS` | Enable ggml CUDA graphs support when CUDA is enabled. | `ON` |
| `ENGINE_ENABLE_NATIVE_CPU` | Build ggml CPU kernels with native host ISA flags such as `-march=native`. Disable this for portable CPU kernels or toolchains that reject generated CPU instructions. | `ON` |
| `ENGINE_ENABLE_OPENMP` | Enable OpenMP for host-side parallel work. | `ON` |
| `ENGINE_BUILD_EXAMPLES` | Build example binaries. | `OFF` |
| `ENGINE_BUILD_TESTS` | Build framework unit tests. | `OFF` |
| `ENGINE_BUILD_WARMBENCH` | Build warmbench helper binaries. | `OFF` |
| `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER` | Build the standalone native model manager and enable server-side WebUI downloads/install management. This opt-in feature builds the HTTP/TLS dependency. | `OFF` |
| `AUDIOCPP_USE_SYSTEM_OPENSSL` | Use system OpenSSL instead of bundled BoringSSL when native model management is enabled. | `OFF` |
| `AUDIOCPP_DEPLOYMENT_BUILD` | Compile package specs into CLI/server binaries for standalone GGUF and package-spec fallback loading. Script builds expose this as `--deployment-build` on Linux/macOS and `-DeploymentBuild` on Windows. | `OFF` |
| `AUDIOCPP_MODEL_SET` | Model composite to build: `full`, `core`, or `custom`. Script builds expose this as `--model-set` on Linux/macOS and `-ModelSet` on Windows. | `full` |
| `AUDIOCPP_MODELS` | Comma or semicolon separated model target names when `AUDIOCPP_MODEL_SET=custom`, such as `qwen3_tts,pocket_tts,qwen3_asr`. Script builds expose this as `--models` on Linux/macOS and `-Models` on Windows. | empty |

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
- `python tools/model_manager_v2.py list --json` prints installable packages from `model_specs/*.json`
- `--batch-text-file <txt>` runs one offline request per non-empty line
- `--batch-text-dir <dir>` runs one offline request per `.txt`, `.md`, or `.json` file, normalizing each file as one paragraph
- `--batch-audio-dir <dir>` runs one offline request per `.wav`
- `--audio-chunk-mode auto` lets ASR/alignment models choose their safe long-audio policy; expert users can override with `fixed`, `vad`, or `none` where supported
- `--request-sequence <json>` runs a multi-request offline session
- `--batch-merge-audio none|concat` controls batch audio merge behavior
- `--batch-manifest-out <json>` writes a batch output manifest
- `--metrics` prints compact offline wall time, audio duration, RTF, realtime speed, sample rate, and channel metrics
- Use `--request-sequence <json> --metrics` for per-request metrics from one long-lived offline session
- `--pipeline <json>` runs a workflow instead of a raw task
- `--list-pipelines` prints registered workflows
- `--workflow-input key=value` overrides pipeline inputs
- `--log` streams framework logs to stdout
- `--log-file <path>` streams framework logs to a file in real time
- `--segments-out`, `--turns-out`, and `--words-out` write structured JSON outputs
- `--vad-chunks-out` writes offline VAD-based chunk windows; tune them with `--vad-chunk-max-seconds`, `--vad-chunk-merge-gap-seconds`, and `--vad-chunk-padding-seconds`


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

Set top-level `"backend"` to `"cuda"`, `"cpu"`, `"vulkan"`, `"metal"`, or `"hip"`. CUDA is the optimized path for audio.cpp; CPU, Vulkan, Metal, and HIP are intended for portability and testing when the binary is built with that backend, but performance and model coverage may be lower.

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


## Tests

The repository includes both framework-level parity validation and app-level end-to-end path checks. At a high level, the flow is:

<p align="center">
  <img src="assets/figure/parity_test_flow.png" alt="Parity test flow" width="720" />
</p>

The main harness under `tests/` is `tests/warmbench.py`. It is used for long-lived multi-request validation, parity checks against Python references, and performance-oriented session reuse scenarios. The `tests/` tree also contains model-specific C++ and Python warmbench entrypoints that `warmbench.py` coordinates.

The main app-facing test tooling under `tools/` is `tools/audiocpp_cli/run_audiocpp_cli_path_tests.py`. It drives `audiocpp_cli` through cataloged offline and streaming cases, verifies expected outputs such as audio or JSON artifacts, and is useful for checking real user-facing request paths rather than just lower-level model components. Streaming coverage is model-specific and applies to models documented with streaming support.

The Python-reference side of these tests usually requires more time-consuming setup than the C++ path because different models rely on different Python reference repos and dependency stacks. In practice, the framework-side tooling is fast to iterate on once models are installed, while Python parity runs often need extra environment preparation before they are ready.

## Projects

Last update: 2026-08-17

Have a project using audio.cpp? Submit a PR or let me know, and I’ll be happy to add it here.

- [TranscrIA](https://github.com/Martossien/transcria) is a self-hosted meeting transcription platform with diarization and local LLM correction. audio.cpp is integrated as a first-class STT engine in the product.
- [Pocket TTS Browser Engine](https://github.com/jjmlovesgit/pocket-tts-browser-engine) uses audio.cpp to bring fully local PocketTTS voices into Chrome and Edge through the browser TTS API.
- [GuideAnts](https://github.com/Elumenotion/GuideAnts) uses audio.cpp as the default local AI stack path for basic ASR and TTS, with planned reusable skills for audio.cpp scenarios and model configurations.
- [audio.cpp-webui](https://github.com/kigner/audio.cpp-webui) provides a full-task Python/Gradio WebUI for audio.cpp, focused on browser-based model downloads and common TTS, ASR, voice conversion, diarization, music, and audio workflows.
- [audio.cpp-hub](https://github.com/IIIIIllllIIIIIlllll/audio.cpp-hub) is a Web GUI for audio.cpp that packages model-oriented workflows around the native runtime.
- [Delusion](https://github.com/BrokenSource/Delusion) provide Pythonic, strongly typed wrapper classes around audio.cpp usage, including model download/cache helpers and typed request surfaces.
- [AudioCppTray](https://github.com/spicchio72/AudioCppTray) is a Windows tray management tool for `audiocpp_server.exe`, with start/stop/restart controls, notifications, log viewing, log rotation, and server configuration shortcuts.


## Performance Metrics

> [!WARNING]
> These Python-relative numbers were measured for the initial release. Several model paths have improved substantially since then, so the figures below should be read as the original release baseline rather than the latest peak performance.

All performance metrics in this section were measured on Ubuntu with the CUDA backend on an NVIDIA GeForce RTX 5090. The Python-relative one-shot and long-lived-session comparisons come from direct framework/runtime API benchmark calls, not from `audiocpp_cli`; CLI path tests are separate and include app-layer request parsing, output writing, and other user-facing overhead.

**Absolute RTF depends on the GPU and system setup, but the Python-relative speedups are real because audio.cpp and the matching Python reference paths were measured on the same CUDA setup.**

audio.cpp already shows some genuinely exciting wins against the matching Python reference paths, especially on the TTS side, even when using the original model weights without quantization. The headline win is wall time: several TTS paths run **1.8x to up to 10x faster** than Python while cutting end-to-end latency by **45%-90%**.

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

Some models expose memory-saver session options such as `ace_step.mem_saver=true`, `dramabox.mem_saver=true`, `heartmula.mem_saver=true`, `stable_audio.mem_saver=true`, `omnivoice.mem_saver=true`, and `voxcpm2.mem_saver=true`. These options keep the default output path unchanged while reducing graph workspace VRAM or releasing staged graph/cache state after request phases; later requests may rebuild released graphs.

## Precision/Quantization Support

Many model sessions expose quantization through `--session-option <family>.weight_type=<mode>`, and some families also expose more specific knobs such as `...conv_weight_type`, `...talker_weight_type`, or `...speech_decoder_weight_type`. The exact supported modes are model-specific rather than global.

audio.cpp also supports standalone GGUF packages. 

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
