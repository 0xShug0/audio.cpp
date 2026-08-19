# Community Models

Community model ports live under `community_models` to make the ownership boundary clear while keeping them available through the normal audio.cpp CLI and server paths. Some community-contributed models graduate into the core model tree when they become part of the main release surface.

The review bar for community models is intentionally lighter than core model integrations, so contributors can share useful ports earlier. The model does not need to be fully promoted into the core model tree on day one, but it should still be reproducible and honest about its limits.

Practical expectations:

- RTF should be below 1.0.
- VRAM usage should stay stable across multiple requests. If memory needs to be optimized, use `mem_saver` to balance performance and VRAM instead of hiding leaks.
- Long-form generation should work correctly. The shared long-form TTS/clone test cases live in `tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json`.
- Use existing framework modules and patterns as much as possible.
- Include exact build/run commands, generated WAVs or output artifacts, backend coverage, parity or path-test results when available, and timing/memory notes. [PR #19](https://github.com/0xShug0/audio.cpp/pull/19) and [PR #63](https://github.com/0xShug0/audio.cpp/pull/63) are good examples of contributors providing enough detail for maintainers to reproduce and review the model.

## Current Community Models

| Family | Task | Supported language(s) | Contributor | What They Added |
|---|---|---|---|---|
| **glm_tts** | TTS, voice cloning | zh, en | Mirek [@mirek190](https://github.com/mirek190) | [GLM-TTS](glm_tts.md) zero-shot synthesis and voice cloning support |
| **inflect_v2** | TTS | en | Community | [Inflect Micro v2 and Nano v2](inflect_v2.md) native FP32 offline synthesis |
| **kroko_asr** | ASR | de, en, es, fr, it, he, nl, pt, sv, tr | Mirek [@mirek190](https://github.com/mirek190) | [Kroko Community ASR](kroko_asr.md) native offline/streaming Zipformer2/RNN-T transcription with word timestamps |
| **minimax_h3** | Video, Music, TTS/Dialogue | auto | [@0xShug0](https://github.com/0xShug0) | [MiniMax-H3](minimax_h3.md) text-to-audio/video generation with Q4_K and optional INT8 ConvRot DiT |
| **minimax_music3** | Music | auto | [@0xShug0](https://github.com/0xShug0) | [MiniMax Music 3](minimax_music3.md) text-to-music generation with lyrics conditioning |
| **moss_tts_local** | TTS, voice cloning | auto, optional language hint | [@justinjohn0306](https://github.com/justinjohn0306) | [MOSS-TTS-Local Transformer v1.5](../models/moss_tts.md) support in the core model tree |
| **outetts** | TTS, voice cloning | en, ar, zh, nl, fr, de, it, ja, ko, lt, ru, es, pt, be, bn, ka, hu, lv, fa, pl, sw, ta, uk | Mirek [@mirek190](https://github.com/mirek190) | [Llama-OuteTTS-1.0-1B](outetts.md) TTS and voice cloning support |
| **parakeet_tdt** | ASR | auto, bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk | [@dleiferives](https://github.com/dleiferives) | [Parakeet-TDT 0.6B v3](parakeet_tdt.md) offline, long-form, and buffered-streaming ASR support |
| **sense_asr** | ASR | auto, zh, en, yue, ja, ko, pt, ru, es, it, fr, de, nl, pl, tr, ar, hi, vi, th, id, ms, fa, nospeech | Jason Chen [@jasonchen31](https://github.com/jasonchen31), [@LauraGPT](https://github.com/LauraGPT) / FunASR | [SenseVoice-Small](sense_asr.md) offline/streaming SAN-M + CTC transcription with event/emotion/language tags and ITN |
| **vietneu_tts** | TTS, voice cloning | vi, en | Phuoc [@phuocnguyen90](https://github.com/phuocnguyen90) | [VieNeu-TTS-v3-Turbo](vietneu_tts.md) TTS and voice cloning support |
