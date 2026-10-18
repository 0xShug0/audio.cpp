# Fun-ASR-Nano Native Runtime Design

## Goal

Add production-grade offline transcription for `FunAudioLLM/Fun-ASR-Nano-2512`
to audio.cpp under the family id `fun_asr_nano`. The first complete release must
load the official Hugging Face safetensors checkpoint, convert it with the normal
audio.cpp GGUF path, run on CPU and CUDA, and serve through the existing CLI and
OpenAI-compatible server.

## Scope

The first release includes:

- 16 kHz mono audio input with Kaldi-compatible 80-bin fbank extraction;
- pre-emphasis `0.97`, 25 ms frames, 10 ms shift, and LFR `m=7`, `n=6`;
- the 70-block SenseVoice SAN-M encoder (1 projection stem, 49 main blocks,
  20 timestamp-prediction blocks);
- the 512 -> 2048 -> 1024 projector and two-layer bidirectional adaptor;
- the bundled Qwen3-0.6B causal decoder with KV cache;
- Chinese, English, and Japanese transcription plus the model's Chinese
  dialect and regional-accent coverage;
- prompt-controlled inverse text normalization through `fun_asr_nano.itn`;
- native safetensors and audio.cpp-owned F16, Q8_0, and Q4_K_M GGUF packages;
- deterministic greedy decoding, timing metrics, and explicit input limits;
- reference parity against Transformers and the MIT-licensed transcribe.cpp
  implementation pinned at `c87109304c42707867f926fb7b9c378d9f46df8a`.

Streaming, built-in timestamps, translation, diarization, and the unpublished
CTC path are outside the first release. The published checkpoint contains no
complete CTC decoder, so the runtime is intentionally LLM-only.

## Source Of Truth

The implementation uses these pinned references:

- official HF checkpoint: `FunAudioLLM/Fun-ASR-Nano-2512-hf` at
  `854d88f94205cd17d2afdb24332130d86fbe654a`;
- Transformers integration PR #46180 and its `fun_asr_nano` config, frontend,
  model, and conversion mapping;
- transcribe.cpp at `c87109304c42707867f926fb7b9c378d9f46df8a`
  for independently validated SAN-M, adaptor, and frontend math;
- transcribe.cpp BF16 result of 1.78% WER on LibriSpeech test-clean as an
  external acceptance reference, not as a replacement for audio.cpp parity.

Adapted MIT code must retain attribution in the affected source files and in
the repository's third-party notices. Model packages retain the FunASR Model
Open Source License Agreement v1.1.

## Architecture

### Assets And Configuration

`FunAsrNanoAssets` owns a `ResourceBundle`, parsed configuration, tokenizer
assets, and one tensor source. The parser accepts the current nested
Transformers config and the legacy flat adaptor fields already published by
the HF checkpoint. Validation rejects mismatched dimensions before allocating
weights.

The runtime reads the converted HF tensor namespaces directly:

- `model.audio_tower.stem`;
- `model.audio_tower.layers`;
- `model.audio_tower.timestamp_prediction_layers`;
- `model.audio_tower.layer_norm` and
  `model.audio_tower.timestamp_prediction_layer_norm`;
- `model.multi_modal_projector` and `model.audio_adaptor.blocks`;
- `model.language_model`, `lm_head`, and tied token embeddings.

The audio.cpp GGUF converter preserves these logical names and embeds config,
processor config, generation config, tokenizer JSON, chat template, and the
typed `fun_asr_nano` model spec. This keeps safetensors and GGUF on one loader
path.

### Shared Frontend

A parameterized Kaldi fbank frontend belongs in `engine/framework/audio`, not
inside the model. It supports the exact options needed by Fun-ASR-Nano and can
later replace duplicated SenseVoice-style frontend code:

- pre-emphasis and Hamming window;
- HTK mel scale and natural-log output;
- configurable snip-edges behavior;
- centered LFR stacking with first/last frame replication;
- optional CMVN, disabled for Fun-ASR-Nano.

The model wrapper fixes the published values and returns time-major LFR frames,
an attention mask, and valid lengths.

### Shared SAN-M Module

The SAN-M graph primitives belong in
`engine/framework/modules/speech_encoders/sanm`. A non-owning weight view feeds
three operations:

- the 560 -> 512 projection stem;
- a residual SAN-M block;
- affine layer normalization.

Each block computes projected multi-head self-attention, the depthwise FSMN
branch over value states with kernel size 11, a residual connection, and the
ReLU feed-forward branch. F16 matrix multiplication requests F32 accumulation
where the validated reference requires it.

### Encoder And Adaptor

`FunAsrNanoEncoderRuntime` owns typed weight storage and a shape-keyed graph
cache. It applies `sqrt(512)` scaling, sinusoidal positions, the stem, 49 main
blocks, normalization, 20 timestamp-prediction blocks, and final normalization.

`FunAsrNanoAdaptorRuntime` applies the two projector linears and two pre-norm
bidirectional Transformer blocks. The adaptor preserves one output embedding
per valid LFR frame; padded frames are masked and removed before prompt splice.

### Decoder And Prompt

The text path reuses audio.cpp's `QwenCausalDecoderModule`, Qwen3 tokenizer,
sampling code, and KV-cache conventions. The Fun-ASR wrapper supplies the
1024-hidden, 28-layer, 16-query/8-KV-head configuration with RoPE theta
`1e6` and tied embeddings.

The prompt builder emits the checkpoint chat template, inserts one audio token
per adaptor output, and supports the model's two Chinese normalization prompts.
Greedy decoding is the default. Existing request sampling fields remain
available without creating model-specific duplicates.

### Session And Serving

`FunAsrNanoSession` implements `IOfflineVoiceTaskSession`. It validates task and
run mode, resamples input to 16 kHz, executes frontend -> encoder -> adaptor ->
decoder -> tokenizer, and publishes per-stage timings. It uses the framework's
existing audio chunking for long files, with 30-second default chunks and no
claim of model-native streaming.

Registration through `audiocpp_add_model` automatically exposes the family to
`audiocpp_cli`, `audiocpp_server`, loader inspection, and generic GGUF tooling.

## Model Spec And Packages

`model_specs/fun_asr_nano.json` is the source of truth. It advertises:

- task `asr`, mode `offline`, languages `zh`, `en`, and `ja`;
- official HF safetensors package from
  `FunAudioLLM/Fun-ASR-Nano-2512-hf`;
- audio.cpp-hosted F16, Q8_0, and Q4_K_M packages after parity validation;
- request option `language` and session option `itn`;
- explicit lack of timestamps and streaming.

The package license and attribution are displayed in the model documentation
and release metadata.

## Verification

Tests proceed in increasing cost:

1. config, tensor-name, tokenizer, and model-spec unit tests;
2. frontend parity on fixed waveforms, including short and padded audio;
3. encoder stage parity at stem, main block 0/24/48, and timestamp block
   0/10/19;
4. adaptor parity after each projector and adaptor block;
5. decoder prefill and first three decode-step logits parity;
6. end-to-end transcript parity on Chinese, English, and Japanese fixtures;
7. CPU and CUDA smoke tests for safetensors and GGUF;
8. WER evaluation on LibriSpeech test-clean before publishing quantized files.

F16/BF16 stage tensors target `atol=2e-3`, `rtol=2e-3`; end-to-end greedy
transcripts must match the HF reference on the fixture set. Quantized packages
must remain within 0.20 absolute WER percentage points of the F16 audio.cpp
baseline.

## Rollback

Work stays on `codex/add-fun-asr-nano-audiocpp-20260729` with a bundle and
patch backup before each commit. Each commit is independently buildable. The
model is registered only in the final end-to-end commit, so earlier commits do
not expose an incomplete family to users.
