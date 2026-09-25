# LFM2.5-Audio (work in progress)

LFM2.5-Audio is Liquid AI's end-to-end speech and text model. One checkpoint handles speech recognition, text-to-speech, and interleaved text-and-audio replies for speech-to-speech conversation; the system prompt and the generation mode select the task.

Upstream: [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) · [LiquidAI/LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) · reference implementation: [liquid-audio](https://github.com/Liquid4All/liquid-audio)

**Status: work in progress.** Nothing runs in audio.cpp yet: there is no model spec, loader, or package. Progress is tracked in [#683](https://github.com/0xShug0/audio.cpp/pull/683).

| Field | Value |
|---|---|
| Family | `lfm2_audio` (planned) |
| Tasks (planned) | `asr`, `tts`, `s2s` |
| Modes (planned) | offline; streaming for TTS and speech-to-speech later |
| Languages | en (`LFM2.5-Audio-1.5B`), ja (`LFM2.5-Audio-1.5B-JP`) |
| Input | Speech (resampled to 16 kHz) and/or text |
| Output | Text and/or 24 kHz speech |
| License | LFM Open License v1.0 |

## Architecture

- NeMo-style 128-bin log-mel frontend and a FastConformer encoder (17 layers, 8x depthwise-striding subsampling, relative-position attention), followed by an MLP adapter. The encoder and frontend configuration, including the mel filterbank and window, match `canary_asr` (canary-180m-flash).
- LFM2 hybrid backbone: 10 gated short-convolution blocks and 6 grouped-query attention blocks, with RMSNorm, QK-norm, and SwiGLU. The text head is tied to the token embedding.
- A 6-layer depth transformer that predicts 8 audio codebooks per 12.5 Hz frame from the backbone hidden state.
- An LFM2-based detokenizer (short-convolution plus 30-frame sliding-window attention) with a log-magnitude/phase head and an ISTFT (n_fft 1280, hop 320) that produces 24 kHz audio.

| Task | System prompt | Generation |
|---|---|---|
| `asr` | `Perform ASR.` (`Perform ASR in japanese.` for the JP model) | Sequential, text output |
| `tts` | `Perform TTS. Use the {US,UK} {male,female} voice.` (`Perform TTS in japanese.` for the JP model) | Sequential, audio output |
| `s2s` | `Respond with interleaved text and audio.` | Interleaved text and audio |

## Packaging

Liquid AI already publishes llama.cpp-format GGUFs in [LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) and [LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF). The same repositories ship `llama-liquid-audio` runners that consume these files, built from [ggml-org/llama.cpp#18641](https://github.com/ggml-org/llama.cpp/pull/18641), which is not merged upstream yet. Each quantization has four files, where `<model>` is `LFM2.5-Audio-1.5B` or `LFM2.5-Audio-1.5B-JP`:

| File | Contents |
|---|---|
| `<model>-<quant>.gguf` | Backbone, text tokenizer, chat template |
| `mmproj-<model>-<quant>.gguf` | Conformer encoder, adapter, and the input embedding for generated audio codes |
| `vocoder-<model>-<quant>.gguf` | Depth transformer, detokenizer input embedding, ISTFT window |
| `tokenizer-<model>-<quant>.gguf` | Detokenizer backbone and output head |

`lfm2_audio` packages will point at these repositories, pinned to revisions, instead of publishing separate GGUFs. Loading will follow `auk`: `--model` takes the package directory, and a session option picks each component file, so quantizations can be mixed without modifying the upstream GGUFs.

| Session option (planned) | Selects |
|---|---|
| `lfm2_audio.model_gguf` | `<model>-<quant>.gguf` |
| `lfm2_audio.mmproj_gguf` | `mmproj-<model>-<quant>.gguf` |
| `lfm2_audio.vocoder_gguf` | `vocoder-<model>-<quant>.gguf` |
| `lfm2_audio.detokenizer_gguf` | `tokenizer-<model>-<quant>.gguf` |

Paths are relative to the package directory, and each option has a default. The model manager will offer each component quantization as its own download into the same directory.

## Quantization

The English repository ships F16, Q8_0, and Q4_0; the Japanese repository also ships F32. Both use the same per-tensor layout:

| Tensors | F32 | F16 | Q8_0 | Q4_0 |
|---|---|---|---|---|
| Backbone and detokenizer matmuls (attention, short-conv projections, FFN, detokenizer output head) | F32 | F16 | Q8_0 | Q4_0 |
| Backbone and detokenizer token embeddings (the backbone embedding is also the tied text head) | F32 | F16 | Q8_0 | Q6_K |
| Encoder and adapter matmuls (attention, relative-position projection, FFN, subsampling output projection, adapter linear layers) | F32 | F16 | Q8_0 | Q4_0 |
| Encoder pointwise convolutions | F32 | F32 (EN), F16 (JP) | Q8_0 | Q4_0 |
| Depth transformer, its input projection, per-codebook embeddings and heads, detokenizer code embedding | F32 | F16 | Q8_0 | Q4_0 |
| Norms (including the adapter LayerNorm), biases, depthwise convolutions, subsampling Conv2d, folded BatchNorm, relative-position biases, audio-code input embedding, ISTFT window | F32 | F32 | F32 | F32 |

The loader will keep each tensor's stored type, so every package needs F32, F16, Q8_0, Q4_0, and Q6_K matmul and `get_rows` support. Tensors that inference does not use will not be loaded: `a.embd_to_logits` and `a.position_embd_norm` in the mmproj file, the duplicate `audio_embedding.*` in the vocoder file, and the detokenizer's text `token_embd`.

## Milestones

Each milestone is gated on stage-by-stage parity against liquid-audio, with exact build and run commands and RTF and memory numbers on CUDA, Metal, and CPU, matching the evidence bar in #54. Reference dumps come from a CUDA machine because liquid-audio loads its detokenizer with `.cuda()`.

| Milestone | Scope | Status |
|---|---|---|
| M0 | This page, the packaging plan, and questions for maintainers | done |
| M1 | ASR: model spec v1, loader with the component options above, a model-local tokenizer adapter that reads the GGUF `tokenizer.ggml.*` metadata, mel frontend, encoder and adapter, LFM2 hybrid backbone, sequential text generation, audio chunking | not started |
| M2 | TTS: depth transformer, detokenizer and ISTFT, built-in voices, long-form text through the framework text chunker | not started |
| M3 | Speech-to-speech: interleaved generation and a streaming session (follow-up pull request) | not started |
| M4 | Server: `POST /v1/audio/speech/live?return_text=true` streams the text along with the audio; without it the route stays audio-only (separate pull request) | not started |
