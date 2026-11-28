# Checkpoint-dependent runtime lifecycle tests

These targets are built only with `ENGINE_BUILD_MODEL_TESTS=ON`. They call
framework runtimes directly, without model sessions or reference/output caches.
They are not checkpoint-free unit tests.

Each suite compares A, repeated A, same-shape B, return to A, changed-shape C,
return to A and destruction/recreation against separately constructed runtimes.
Where exposed by the API, it also checks explicit release/reprepare. These are
lifecycle comparisons, not independent references for the model mathematics.

Build in Debug:

```bash
cmake --build build/debug --target framework_encoder_lifecycle_test fsq_codec_lifecycle_test -j8
```

Run `framework_encoder_lifecycle_test` with `--model`, `--component`, `--prefix`,
`--audio`, `--backend cuda --threads 8 --log --log-file <path>`:

| Component | Compatible checkpoint | Prefix | Additional arguments |
|---|---|---|---|
| `qwen3` | M5-TTS GGUF | `codec` | `--attention eager` or `--attention flash` |
| `s3` | Chatterbox GGUF | `s3gen` | `--construction factory` or `--construction weights` |
| `wav2vec2` | ControlFoley GGUF | `mert` | Uses the published MERT configuration with graph retention enabled |
| `musicgen_style` | ControlFoley GGUF | `mert` | `--aux-prefix musicgen_style` |
| `clap` | ControlFoley GGUF | `clap` | Uses batch two for C because waveform length is padded to a fixed graph size |

Use a mono speech recording such as `assets/resources/b.wav`. MERT expects
24 kHz audio; the tests use up to four seconds, matching the conditioner input
budget. The `weights` S3 construction path exercises its public constructor
with valid weights loaded through the factory.

MusicGen's `--abort-retry --backend cpu` case uses GGML's CPU abort callback
to interrupt the style graph halfway through its observed callback count,
then compares a same-runtime retry with fresh execution. This checks failed-call
recovery separately from the normal successful-call lifecycle.

CLAP's `--memory-cycles --backend cuda` case runs four create/encode/destroy
cycles with two encodes each, leaving both graph allocators live before
destruction. It reports device memory before and after destruction through the
framework memory API. Backend caches can retain memory independently of these
graph buffers.

Run `fsq_codec_lifecycle_test --model <NeuTTS-GGUF> --backend cuda --threads 8
--log --log-file <path>`. It uses the model adapter only to obtain checkpoint
bindings/configuration, then calls the shared FSQ codec directly with generated
valid code IDs. It compares raw decoder-head values before waveform processing.

Run cases sequentially. No checkpoints, tensor dumps or generated WAVs belong
in this test directory.
