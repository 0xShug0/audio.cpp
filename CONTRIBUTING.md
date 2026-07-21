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

New model PRs should include enough evidence for maintainers and users to understand exactly what was tested. Follow the validation style shown in [PR #19](https://github.com/0xShug0/audio.cpp/pull/19).

Please include:

- Exact build commands
- Exact run commands
- Model paths or model-manager package ids
- Generated output artifacts or paths
- Path-test or parity-test results
- Backend tested, such as CPU, CUDA, Vulkan, or Metal
- Relevant timing, RTF, RSS, VRAM, or resident-memory notes
- Known limitations

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

- [@mirek190](https://github.com/mirek190) for pushing the GGUF work forward across the converter, standalone package-spec loading, ASR GGUF support, the platform-neutral `audiocpp_gguf` binary, and Qwen decoder improvements in [#8](https://github.com/0xShug0/audio.cpp/pull/8), [#43](https://github.com/0xShug0/audio.cpp/pull/43), [#45](https://github.com/0xShug0/audio.cpp/pull/45), [#46](https://github.com/0xShug0/audio.cpp/pull/46), [#53](https://github.com/0xShug0/audio.cpp/pull/53), [#62](https://github.com/0xShug0/audio.cpp/pull/62), and [#68](https://github.com/0xShug0/audio.cpp/pull/68).
- [@justinjohn0306](https://github.com/justinjohn0306) for VibeVoice 7B, LoRA/fine-tune adapter loading, and the initial MOSS-TTS-Local model family implementation in [#14](https://github.com/0xShug0/audio.cpp/pull/14) and [#19](https://github.com/0xShug0/audio.cpp/pull/19).
- [@patrickjchen](https://github.com/patrickjchen) for CUDA build polish, safer constant tensor allocation, and the server busy guard that keeps later requests from hanging behind a stuck model in [#72](https://github.com/0xShug0/audio.cpp/pull/72), [#73](https://github.com/0xShug0/audio.cpp/pull/73), and [#75](https://github.com/0xShug0/audio.cpp/pull/75).
- [@lapy](https://github.com/lapy) for the machine-readable loader and package catalog exports in [#74](https://github.com/0xShug0/audio.cpp/pull/74).
- [@Cr4xy](https://github.com/Cr4xy) for improving multipart upload handling and removing temporary-file writes from that path in [#61](https://github.com/0xShug0/audio.cpp/pull/61).
- [@kevin-ho](https://github.com/kevin-ho) for making single-model server voice discovery work cleanly when the model parameter is omitted in [#64](https://github.com/0xShug0/audio.cpp/pull/64).
- [@xashr](https://github.com/xashr) for Dockerfiles, Docker examples, and Docker documentation in [#30](https://github.com/0xShug0/audio.cpp/pull/30).
- [@dkruyt](https://github.com/dkruyt) for the first multipart transcription upload support in [#25](https://github.com/0xShug0/audio.cpp/pull/25).
- [@CaptainArni](https://github.com/CaptainArni) for fixing PocketTTS empty output when switching cached voices in [#22](https://github.com/0xShug0/audio.cpp/pull/22).
