# NeuTTS

NeuTTS is an experimental English TTS family with built-in speaker prompts and
emotion-token control. The current package is the 2E variant, and the default
download is its standalone GGUF package; the original safetensors layout is
still supported for local development.

```bash
python3 tools/model_manager_v2.py install neutts
```

```bash
audiocpp_cli --task tts --family neutts \
  --model models/NeuTTS-2E-GGUF/neutts-2e-orig.gguf \
  --backend cuda \
  --text "The release checklist is almost complete, and the baseline run looks healthy." \
  --request-option voice_id=emily \
  --request-option emotion=neutral \
  --out out.wav
```

Streaming mode emits generated audio chunks and a final merged WAV:

```bash
audiocpp_cli --task tts --mode streaming --family neutts \
  --model models/NeuTTS-2E-GGUF/neutts-2e-orig.gguf \
  --backend cuda \
  --text "This longer request is split into generated segments and returned through the streaming pull-event path." \
  --request-option voice_id=paul \
  --request-option emotion=happy \
  --out stream.wav \
  --out-dir stream_chunks
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option voice_id=<name>` | `dave`, `emily`, `greta`, `jo`, `juliette`, `mateo`, `paul`, `sophie`, `steven` | `emily` | Built-in speaker prompt. |
| `--request-option emotion=<name>` | `angry`, `disgusted`, `sad`, `happy`, `fearful`, `neutral`, `surprised` | `neutral` | Optional emotion token. |
| `--max-tokens` / `--request-option max_tokens=<n>` | integer | `0` | Maximum generated speech tokens; `0` uses the remaining context. |
| `--request-option min_tokens=<n>` | integer | `50` | Minimum generated speech tokens before EOS may stop generation. |
| `--temperature` | float | `1.0` | AR sampling temperature. |
| `--top-k` | integer | `50` | AR top-k sampling limit. |
| `--seed` | integer | random | Sampling seed. |
| `--text-chunk-size` | chars | `600` | Long-form chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |
| `--session-option neutts.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared backbone and codec matmul weight storage type. |
| `--session-option neutts.generator_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | backend-dependent | Backbone matmul weight storage type. |
| `--session-option neutts.codec_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `neutts.weight_type` or `native` | NeuCodec decoder matmul weight storage type. |
| `--session-option neutts.codec_conv_weight_type=<type>` | `native`, `f32`, `f16` | `native` | NeuCodec convolution weight storage type. |
| `--session-option neutts.runtime_graph_arena_mb=<mb>` | integer MiB | `1024` | Reusable graph arena size. |
