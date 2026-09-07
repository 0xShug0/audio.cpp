# Contributing

Thanks for helping improve audio.cpp. The project is growing quickly, so the most useful contributions are the ones that make the existing model surface easier to use, serve, compose, test, and maintain.

## Preferred Contribution Areas

High-impact areas right now:

- UI and app-facing workflows
- API server behavior, especially OpenAI-compatible serving
- Pipeline and workflow subsystem improvements
- Model documentation, examples, and validation reports
- Cross-platform build and packaging polish

These areas help many model families at once. If you are unsure where to start, improving one of these shared surfaces is usually more valuable than adding another copy of an already-supported model.

## Before Adding a Model

Please check the supported model table in [README.md](README.md) before starting a new model port. Some model families are already released, and others are implemented but still marked as testing while they are validated, polished, or promoted into the broader released surface.

If you want to add support for a model family that is already listed, please focus on improving the existing implementation instead of opening a duplicate port.

When a loader is registered (or parked), keep the **package catalog** in sync. Installable `ModelPackage` entries must not advertise families that `audiocpp_cli --list-loaders` does not expose. Follow the checklist in [docs/maintainers/loader_and_catalog.md](docs/maintainers/loader_and_catalog.md) and run:

```bash
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py
```

Do not leave a live Hugging Face `SnapshotSource` for a loader that is commented out of `registry.cpp` — mark it `UnsupportedSource` (or remove it) and update the README package table.
Good follow-up work for existing model families includes:

- Better CLI or server examples
- More complete path tests
- Clearer model-manager package entries
- Backend coverage improvements
- Memory, latency, or portability improvements
- Documentation for real user workflows

## New Model PRs

New standalone model ports should normally start under `community_models/`. This keeps ownership clear and lets useful model ports land with a lighter review bar than core framework models. Models can graduate into the core model tree later after they are validated, polished, and maintained as part of the main release surface.

Even for community models, PRs should include enough evidence for maintainers and users to understand exactly what was tested. Follow the validation style shown in [PR #19](https://github.com/0xShug0/audio.cpp/pull/19) and [PR #63](https://github.com/0xShug0/audio.cpp/pull/63).

Please include:

- Exact build commands
- Exact run commands
- Model paths or model-manager package ids
- Generated output artifacts or paths
- Path-test or parity-test results
- Backend tested, such as CPU, CUDA, Vulkan, or Metal
- Relevant timing, RTF, RSS, VRAM, or resident-memory notes
- Known limitations

For TTS-style community models, the most useful validation includes a long-lived session with multiple requests, a long-form request through the framework text chunker, cache/graph reuse logs when relevant, and peak VRAM under repeated requests. These measurements do not need to be perfect on the first PR, but they make review much faster and help avoid a large cleanup pass after merge.

For models with Python references, include parity evidence when practical. For models without a clean reference path, include reproducible generated outputs and enough setup detail for another contributor to repeat the run.

Please be prepared to help maintain new model contributions as framework APIs evolve. Keeping model code aligned with the shared framework surface is part of making the implementation useful long term.

## Framework Modules (High Risk)

Framework modules are a high-impact but higher-risk contribution area because the internal framework APIs are still evolving quickly. Changes here can affect many model families at once, so please prefer additive work over modifying behavior that existing models already depend on.

If your contribution is a variant of an existing module, add it as a separate experimental module, for example `xxxExp`, instead of branching inside or rewriting the existing module. The new module can replace the existing one later, after it has shown no regressions across all models that rely on the current implementation.

## Pull Request Notes

Keep PRs focused. A model port, a server change, a pipeline change, and a broad refactor are easier to review and validate when they are separate.

When changing shared framework code, explain which model families or routes were checked. When changing model behavior, explain whether the change affects outputs, performance, memory, or only docs/build wiring.

If a PR intentionally leaves a model under testing, say what remains before it should be marked released.

## Acknowledgments

audio.cpp is moving faster because people keep showing up with real fixes, careful testing, and useful pressure on the parts that matter. Thank you to:

- [@mirek190](https://github.com/mirek190) for pushing the GGUF work forward across the converter, standalone package-spec loading, ASR GGUF support, the platform-neutral `audiocpp_gguf` binary, Qwen decoder improvements, OuteTTS, GLM-TTS, BS-RoFormer, Kroko ASR, Windows GGUF loading performance, model-spec v1 migrations, and Q8 CUDA residual graph fusion in [#8](https://github.com/0xShug0/audio.cpp/pull/8), [#43](https://github.com/0xShug0/audio.cpp/pull/43), [#45](https://github.com/0xShug0/audio.cpp/pull/45), [#46](https://github.com/0xShug0/audio.cpp/pull/46), [#53](https://github.com/0xShug0/audio.cpp/pull/53), [#62](https://github.com/0xShug0/audio.cpp/pull/62), [#63](https://github.com/0xShug0/audio.cpp/pull/63), [#68](https://github.com/0xShug0/audio.cpp/pull/68), [#79](https://github.com/0xShug0/audio.cpp/pull/79), [#97](https://github.com/0xShug0/audio.cpp/pull/97), [#98](https://github.com/0xShug0/audio.cpp/pull/98), [#114](https://github.com/0xShug0/audio.cpp/pull/114), [#122](https://github.com/0xShug0/audio.cpp/pull/122), [#123](https://github.com/0xShug0/audio.cpp/pull/123), [#124](https://github.com/0xShug0/audio.cpp/pull/124), [#126](https://github.com/0xShug0/audio.cpp/pull/126), [#140](https://github.com/0xShug0/audio.cpp/pull/140), [#141](https://github.com/0xShug0/audio.cpp/pull/141), [#142](https://github.com/0xShug0/audio.cpp/pull/142), and [#154](https://github.com/0xShug0/audio.cpp/pull/154).
- [@justinjohn0306](https://github.com/justinjohn0306) for VibeVoice 7B, LoRA/fine-tune adapter loading, and the initial MOSS-TTS-Local model family implementation in [#14](https://github.com/0xShug0/audio.cpp/pull/14) and [#19](https://github.com/0xShug0/audio.cpp/pull/19).
- [@patrickjchen](https://github.com/patrickjchen) for CUDA build polish, safer constant tensor allocation, the server busy guard that keeps later requests from hanging behind a stuck model, the WebUI integration, English WebUI docs, the release 0.4 WebUI pass, and model-download fixes in [#72](https://github.com/0xShug0/audio.cpp/pull/72), [#73](https://github.com/0xShug0/audio.cpp/pull/73), [#75](https://github.com/0xShug0/audio.cpp/pull/75), [#87](https://github.com/0xShug0/audio.cpp/pull/87), [#90](https://github.com/0xShug0/audio.cpp/pull/90), [#108](https://github.com/0xShug0/audio.cpp/pull/108), and [#130](https://github.com/0xShug0/audio.cpp/pull/130).
- [@lapy](https://github.com/lapy) for the machine-readable loader/package catalog exports, loader-catalog sync checks, and model-spec dependency option checks that keep package metadata honest in [#74](https://github.com/0xShug0/audio.cpp/pull/74), [#86](https://github.com/0xShug0/audio.cpp/pull/86), and [#161](https://github.com/0xShug0/audio.cpp/pull/161).
- [@fedeizzo](https://github.com/fedeizzo) for the cross-platform Nix flake, follow-up Nix documentation polish, Strix Halo ROCm inference optimization, and Qwen3-TTS prompt modes without ICL reference codes in [#82](https://github.com/0xShug0/audio.cpp/pull/82), [#83](https://github.com/0xShug0/audio.cpp/pull/83), [#201](https://github.com/0xShug0/audio.cpp/pull/201), and [#202](https://github.com/0xShug0/audio.cpp/pull/202).
- [@phuocnguyen90](https://github.com/phuocnguyen90) for bringing VieNeu-TTS v3 Turbo into the community model surface in [#80](https://github.com/0xShug0/audio.cpp/pull/80).
- [@mosujiba](https://github.com/mosujiba) for adding configurable CORS handling to the server path in [#85](https://github.com/0xShug0/audio.cpp/pull/85).
- [@adambenhassen](https://github.com/adambenhassen) for PocketTTS runtime fixes and upstream-aligned English defaults in [#76](https://github.com/0xShug0/audio.cpp/pull/76) and [#77](https://github.com/0xShug0/audio.cpp/pull/77).
- [@vicenteliu](https://github.com/vicenteliu) for hardening the server against client disconnects by ignoring `SIGPIPE` in [#78](https://github.com/0xShug0/audio.cpp/pull/78).
- [@Cr4xy](https://github.com/Cr4xy) for improving multipart upload handling and removing temporary-file writes from that path in [#61](https://github.com/0xShug0/audio.cpp/pull/61).
- [@kevin-ho](https://github.com/kevin-ho) for making single-model server voice discovery work cleanly when the model parameter is omitted in [#64](https://github.com/0xShug0/audio.cpp/pull/64).
- [@xashr](https://github.com/xashr) for Dockerfiles, Docker examples, Docker documentation, portable CPU Docker builds, published-image docs, GGUF/Qwen3-TTS examples, CI workflow polish, CUDA probe cleanup, and portable CUDA architecture selection in [#30](https://github.com/0xShug0/audio.cpp/pull/30), [#51](https://github.com/0xShug0/audio.cpp/pull/51), [#81](https://github.com/0xShug0/audio.cpp/pull/81), [#84](https://github.com/0xShug0/audio.cpp/pull/84), [#107](https://github.com/0xShug0/audio.cpp/pull/107), [#109](https://github.com/0xShug0/audio.cpp/pull/109), and [#280](https://github.com/0xShug0/audio.cpp/pull/280).
- [@5uck1ess](https://github.com/5uck1ess) for improving Citrinet CTC decoding through the SentencePiece model, hardening PocketTTS FlowLM step allocation, and adding live PCM transcription ingest to the server in [#49](https://github.com/0xShug0/audio.cpp/pull/49), [#59](https://github.com/0xShug0/audio.cpp/pull/59), and [#144](https://github.com/0xShug0/audio.cpp/pull/144).
- [@dkruyt](https://github.com/dkruyt) for the first multipart transcription upload support in [#25](https://github.com/0xShug0/audio.cpp/pull/25).
- [@CaptainArni](https://github.com/CaptainArni) for fixing PocketTTS empty output when switching cached voices, keeping the Windows CUDA build path healthy, and adding ACE-Step 1.5 XL DiT variants in [#22](https://github.com/0xShug0/audio.cpp/pull/22), [#93](https://github.com/0xShug0/audio.cpp/pull/93), and [#235](https://github.com/0xShug0/audio.cpp/pull/235).
- [@IIIIIllllIIIIIlllll](https://github.com/IIIIIllllIIIIIlllll) for the experimental ROCm/HIP backend, Linux HIP build path, Windows HIP/ROCm distribution preparation, HIP build documentation, VibeVoice HIP enablement, backend device listing, Vulkan AMD integer-dot guard, IndexTTS-2.5 support, base64 voice references in server requests, IndexTTS2 speech-rate controls, IndexTTS2 HIP/text-normalization fixes, and BreezeTTS 2 performance and VRAM improvements in [#48](https://github.com/0xShug0/audio.cpp/pull/48), [#148](https://github.com/0xShug0/audio.cpp/pull/148), [#153](https://github.com/0xShug0/audio.cpp/pull/153), [#159](https://github.com/0xShug0/audio.cpp/pull/159), [#164](https://github.com/0xShug0/audio.cpp/pull/164), [#168](https://github.com/0xShug0/audio.cpp/pull/168), [#171](https://github.com/0xShug0/audio.cpp/pull/171), [#193](https://github.com/0xShug0/audio.cpp/pull/193), [#210](https://github.com/0xShug0/audio.cpp/pull/210), [#226](https://github.com/0xShug0/audio.cpp/pull/226), [#239](https://github.com/0xShug0/audio.cpp/pull/239), [#247](https://github.com/0xShug0/audio.cpp/pull/247), [#259](https://github.com/0xShug0/audio.cpp/pull/259), [#299](https://github.com/0xShug0/audio.cpp/pull/299), [#305](https://github.com/0xShug0/audio.cpp/pull/305), [#393](https://github.com/0xShug0/audio.cpp/pull/393), and [#431](https://github.com/0xShug0/audio.cpp/pull/431).
- [@francescobozzo](https://github.com/francescobozzo) for Nix ROCm/HIP backend support, selectable model targets, and Nix CI/package fixes in [#162](https://github.com/0xShug0/audio.cpp/pull/162), [#163](https://github.com/0xShug0/audio.cpp/pull/163), and [#172](https://github.com/0xShug0/audio.cpp/pull/172).
- [@patrickvonplaten](https://github.com/patrickvonplaten) for Metal backend fixes, Voxtral Realtime streaming speedups, live audio streaming from stdin, and incremental transcript deltas in [#102](https://github.com/0xShug0/audio.cpp/pull/102), [#116](https://github.com/0xShug0/audio.cpp/pull/116), [#118](https://github.com/0xShug0/audio.cpp/pull/118), and [#127](https://github.com/0xShug0/audio.cpp/pull/127).
- [@dleiferives](https://github.com/dleiferives) for Parakeet-TDT 0.6B v3 ASR support and follow-up standalone GGUF validation/docs in [#111](https://github.com/0xShug0/audio.cpp/pull/111) and [#139](https://github.com/0xShug0/audio.cpp/pull/139).
- [@JanWerder](https://github.com/JanWerder) for adding Inflect Micro v2 and Nano v2 TTS support in [#125](https://github.com/0xShug0/audio.cpp/pull/125).
- [@chikosan](https://github.com/chikosan) for hardening safetensors parsing and bounding allocations sized by attacker-controlled file and header fields in [#138](https://github.com/0xShug0/audio.cpp/pull/138) and [#143](https://github.com/0xShug0/audio.cpp/pull/143).
- [@LauraGPT](https://github.com/LauraGPT) for adding Fun-ASR-Nano offline ASR, fixing the Linux build script executable bit, and bringing SenseVoice-Small offline/streaming ASR into the community model surface with Jason Chen and FunASR Ops in [#155](https://github.com/0xShug0/audio.cpp/pull/155), [#156](https://github.com/0xShug0/audio.cpp/pull/156), and [#219](https://github.com/0xShug0/audio.cpp/pull/219).
- [@liuzl](https://github.com/liuzl) for speeding up Metal `conv_transpose_1d` dispatch in [#149](https://github.com/0xShug0/audio.cpp/pull/149).
- [@JayDataEngineer](https://github.com/JayDataEngineer) for fixing PocketTTS `clone_audio_path` option typing in [#147](https://github.com/0xShug0/audio.cpp/pull/147).
- [@jasonchen31](https://github.com/jasonchen31) for adding server-side voice library folder support for name-based voice cloning, helping land SenseVoice-Small, fixing crashes on binaries built for older GPUs, adding Audio8 TTS, and fixing VoxCPM1 WebUI download/Yue language handling in [#191](https://github.com/0xShug0/audio.cpp/pull/191), [#219](https://github.com/0xShug0/audio.cpp/pull/219), [#240](https://github.com/0xShug0/audio.cpp/pull/240), [#333](https://github.com/0xShug0/audio.cpp/pull/333), and [#424](https://github.com/0xShug0/audio.cpp/pull/424).
- [@mirek190](https://github.com/mirek190) for the native WebUI follow-up work around model management, package handling, request controls, package labels, new GGUF package surfacing, CUDA BF16 cuBLAS output on Ampere, MiniMax-H3 WebUI polish, retiring the legacy Python WebUI, adding the reusable native model package manager, clearing stale WebUI service workers, clarifying ACE-Step GGUF package labels, documenting server host/port options, and fixing Qwen compact-logits graph reuse in [#199](https://github.com/0xShug0/audio.cpp/pull/199), [#206](https://github.com/0xShug0/audio.cpp/pull/206), [#207](https://github.com/0xShug0/audio.cpp/pull/207), [#208](https://github.com/0xShug0/audio.cpp/pull/208), [#211](https://github.com/0xShug0/audio.cpp/pull/211), [#213](https://github.com/0xShug0/audio.cpp/pull/213), [#229](https://github.com/0xShug0/audio.cpp/pull/229), [#230](https://github.com/0xShug0/audio.cpp/pull/230), [#257](https://github.com/0xShug0/audio.cpp/pull/257), [#258](https://github.com/0xShug0/audio.cpp/pull/258), [#381](https://github.com/0xShug0/audio.cpp/pull/381), and [#426](https://github.com/0xShug0/audio.cpp/pull/426).
- [@nikich340](https://github.com/nikich340) for adding explicit server model unloading support in [#197](https://github.com/0xShug0/audio.cpp/pull/197).
- [@utsl42](https://github.com/utsl42) for fixing CUDA linking on NixOS in [#214](https://github.com/0xShug0/audio.cpp/pull/214).
- [@yegorius](https://github.com/yegorius) for fixing PocketTTS handling in the model manager in [#205](https://github.com/0xShug0/audio.cpp/pull/205).
- [@noctrex](https://github.com/noctrex) for fixing the Windows build under MSYS2 / MinGW GCC 16 in [#135](https://github.com/0xShug0/audio.cpp/pull/135).
- [@panw3i](https://github.com/panw3i) for fixing PowerShell parse errors in `build_windows.ps1` in [#110](https://github.com/0xShug0/audio.cpp/pull/110).
- [@robotokpro](https://github.com/robotokpro) for fixing WebUI model download handling after `list_hf_files` began returning 3-tuples in [#106](https://github.com/0xShug0/audio.cpp/pull/106).
- [@jamesweiym-ops](https://github.com/jamesweiym-ops) for supporting TheRock LLVM layout in the Windows HIP build script in [#160](https://github.com/0xShug0/audio.cpp/pull/160).
- [@Blakeolson21](https://github.com/Blakeolson21) for documenting macOS CPU-only builds and fixing HIP backend classification in tests in [#237](https://github.com/0xShug0/audio.cpp/pull/237) and [#238](https://github.com/0xShug0/audio.cpp/pull/238).
- [@odest](https://github.com/odest) for adding Vulkan backend support to Windows builds in [#242](https://github.com/0xShug0/audio.cpp/pull/242).
- [@JoeMattie](https://github.com/JoeMattie) for bringing up MiniMax Music3, fixing BF16 convolution issues, repairing conversion, wiring native WebUI support, and honoring required request options in Studio in [#243](https://github.com/0xShug0/audio.cpp/pull/243) and [#245](https://github.com/0xShug0/audio.cpp/pull/245).
- [@Orion-zhen](https://github.com/Orion-zhen) for adding `--voice-dir` parsing to the server CLI in [#246](https://github.com/0xShug0/audio.cpp/pull/246).
- [@derekja](https://github.com/derekja) for multipart ASR recognition prompts, clearer large-request errors, ASR graph allocation fixes, and CUDA graph-cache cleanup for concurrent ASR pressure in [#262](https://github.com/0xShug0/audio.cpp/pull/262), [#264](https://github.com/0xShug0/audio.cpp/pull/264), [#276](https://github.com/0xShug0/audio.cpp/pull/276), and [#293](https://github.com/0xShug0/audio.cpp/pull/293).
- [@kawshikbuet17](https://github.com/kawshikbuet17) for adding OmniVoice weight-type benchmark coverage, validation reports, and Python-to-C++ tests in [#269](https://github.com/0xShug0/audio.cpp/pull/269).
- [@tareko](https://github.com/tareko) for adding F5-TTS community scaffolding with Habibi Arabic aliases in [#275](https://github.com/0xShug0/audio.cpp/pull/275).
- [@jrohde](https://github.com/jrohde) for adding the MOSS-VoiceGenerator community model in [#278](https://github.com/0xShug0/audio.cpp/pull/278).
- [@LysanderdeJong](https://github.com/LysanderdeJong) for adding the MMS-300M-1130 forced aligner community model in [#279](https://github.com/0xShug0/audio.cpp/pull/279).
- [@drzsdrtfg](https://github.com/drzsdrtfg) for the tag-driven prebuilt release pipeline, Supertonic voice-preset request-option fix, Soprano TTS community model, and Qwen cached-graph prefill input fix in [#286](https://github.com/0xShug0/audio.cpp/pull/286), [#302](https://github.com/0xShug0/audio.cpp/pull/302), [#323](https://github.com/0xShug0/audio.cpp/pull/323), and [#331](https://github.com/0xShug0/audio.cpp/pull/331).
- [@Hi5808](https://github.com/Hi5808) for documenting Jetson Orin bring-up and correcting native architecture wording in [#288](https://github.com/0xShug0/audio.cpp/pull/288).
- [@SelfRef](https://github.com/SelfRef) for making embedded WebUI work behind a path-prefix reverse proxy and adding server model LRU controls in [#297](https://github.com/0xShug0/audio.cpp/pull/297) and [#298](https://github.com/0xShug0/audio.cpp/pull/298).
- [@bjhengen](https://github.com/bjhengen) for dropping out-of-span chunk speech metadata instead of aborting the whole run in [#301](https://github.com/0xShug0/audio.cpp/pull/301).
- [@gqf2008](https://github.com/gqf2008) for adding server idle unload and pre-load memory guard behavior, tightening indeterminate-footprint handling, adding Audio8 ASR, and improving default CJK text chunking in [#306](https://github.com/0xShug0/audio.cpp/pull/306), [#308](https://github.com/0xShug0/audio.cpp/pull/308), [#337](https://github.com/0xShug0/audio.cpp/pull/337), and [#441](https://github.com/0xShug0/audio.cpp/pull/441).
- [@ampersandru](https://github.com/ampersandru) for adding IBM Granite Speech 5.0 470M TurboCTC ASR as a community model in [#311](https://github.com/0xShug0/audio.cpp/pull/311).
- [@iamwavecut](https://github.com/iamwavecut) for MiniMax Music3 performance work around native RoPE/SwiGLU lowering, opt-in CFG reuse, chunk hop support, batched ensemble takes, and Q4_K depth decoder support in [#321](https://github.com/0xShug0/audio.cpp/pull/321).
- [@XythQ](https://github.com/XythQ) for resolving model contracts once per loaded model instead of per request in [#328](https://github.com/0xShug0/audio.cpp/pull/328).
- [@pannagaps](https://github.com/pannagaps) for adding Chatterbox Turbo TTS as a community model in [#394](https://github.com/0xShug0/audio.cpp/pull/394).
- [@gsaon](https://github.com/gsaon) for fixing Granite Speech ASR mel filterbank construction on the continuous frequency axis in [#384](https://github.com/0xShug0/audio.cpp/pull/384).
- [@reezex0-ux](https://github.com/reezex0-ux) for adding the HIP Fast-AR top-k sampler path for Fish Audio in [#386](https://github.com/0xShug0/audio.cpp/pull/386).
- [@feng19](https://github.com/feng19) for adding `HF_ENDPOINT` mirror support to model downloads in [#397](https://github.com/0xShug0/audio.cpp/pull/397).
- [@CryptVenture](https://github.com/CryptVenture) for adding richer WAV output options, correcting the shared text chunk mode preset, accepting the MOSS VoiceGenerator instruction spelling used by the speech route, and forwarding `language` only when a model contract accepts it in [#358](https://github.com/0xShug0/audio.cpp/pull/358), [#370](https://github.com/0xShug0/audio.cpp/pull/370), [#371](https://github.com/0xShug0/audio.cpp/pull/371), and [#400](https://github.com/0xShug0/audio.cpp/pull/400).
