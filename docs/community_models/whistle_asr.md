# Whistle ASR (experimental)

The `whistle_asr` family runs [Cactus Compute Whistle](https://huggingface.co/Cactus-Compute/whistle)
locally through audio.cpp. The current implementation supports **offline**
transcription of 16 kHz mono WAV files up to 30 seconds on the CPU and Metal
backends. It detects one of the
model's seven languages (English, German, French, Spanish, Italian, Dutch, or
Polish); `--language en`, `de`, `fr`, `es`, `it`, `nl`, or `pl` forces a language.
These are supported language codes, not seven-language validation evidence.
The focused validation report records the English and German checks separately.
French, Spanish, Italian, Dutch, and Polish each have one ElevenLabs synthetic
clip checked through automatic and forced-language CLI requests. See the
[five-language spot checks](../../tests/whistle_asr/MULTILINGUAL_VALIDATION.md).
This limited coverage does not qualify general accuracy in all seven languages.

There is no downloadable audio.cpp GGUF package yet. Convert the official
model locally. Use the exact revision
`b358ddadd89b7a713b5aa131f23032d3cca1b251` and put
`checkpoints/whistle.safetensors` in a dedicated directory as
`whistle.safetensors`, together with `config.json` and `LICENSE`. Keep
`whistle.cact` outside that directory. The converter checks their byte lengths
and SHA-256 hashes, extracts the tokenizer, mel filterbank, and Hadamard
permutations from the official `.cact`, and embeds those six sidecars plus the
Whistle model spec in the GGUF:

```bash
cmake --build build/release --target audiocpp_gguf audiocpp_cli
python3 tools/community_models/whistle_asr/convert.py \
  --source /path/to/whistle-source \
  --cact /path/to/whistle.cact \
  --converter build/release/bin/audiocpp_gguf \
  --output /path/to/whistle-f32.gguf
audiocpp_cli --task asr --family whistle_asr --backend cpu \
  --model /path/to/whistle-f32.gguf \
  --audio assets/resources/sample_16k.wav
```

The official `.cact` uses Cactus Quants. The GGUF stores the published
FP32 Safetensors checkpoint instead. The two weight sets
can yield different spellings or number formatting for the same speech.
The runtime accepts FP32 tensors only, including for an extracted Safetensors
directory with the same five required assets. Raw upstream Safetensors alone
are insufficient. Other GGUF precisions are rejected.
Conversion and inference do not require the Cactus engine, ONNX Runtime, or
Python at recognition time.

Streaming, keyword biasing, word timestamps, speech embeddings, resampling, and
longer recordings are not supported by this family. CUDA, HIP, and Vulkan are
not validated and are rejected at session creation.
The runtime rejects unsupported sample rates, channel counts, and durations
rather than silently changing the input.
Decoding uses the checkpoint's 320-token ceiling; a non-terminating decode
fails explicitly instead of reporting a truncated transcript as complete.
No model is downloaded automatically.

Constant-RMS speech is passed to the model rather than rejected by an
amplitude-based noise gate. Digital silence returns an empty transcript.
Inputs shorter than 40 milliseconds also return an empty transcript.
This family has no speech activity detector, so noise-only recordings may
produce words.

## Runtime structure

The mel frontend reuses the shared `NemoMelFrontend` (512-point FFT, 400-sample
symmetric Hann window, 160-sample hop, constant padding, natural-log mel energies,
per-bin mean and unbiased-variance normalization). Whistle's 99th-percentile block
RMS gain normalization is applied locally before it.

The encoder runs as one GGML graph on the selected backend. It is built from the
framework `Conv2dModule`, `DepthwiseConv2dModule`, `DepthwiseConv1dModule`,
`LinearModule`, `RMSNormModule`, `GemmaRMSNormModule`, `RoPEModule`, `SiluModule`,
`SigmoidModule`, `SoftmaxModule`, and `GLUModule`. The stem does not use the shared
`DepthwiseConvSubsamplingModule` because that module hardcodes ReLU and Whistle's
stem uses SiLU. Whistle-specific math stays model-local in `encoder.cpp`: the
four-lane manifold hyper-connection mixing with its Sinkhorn normalization, the
conditioned Kronecker-factored Hadamard MLP with its two fixed permutations, and
grouped-query attention with 48-wide queries/keys and 64-wide values, which the
shared attention modules cannot express because they assume one head size.
Checkpoint tensors are JAX `[in, out]` kernels; the loader transposes them once into
the backend weight store.

The autoregressive decoder (one token per step, n-gram engram lookups, three-tap
query/key/value mixing, cross-attention over the encoder output) still runs on the
host CPU with its own FP32 math in `runtime.cpp`. That host code keeps its own
implementations of the linear, RMS norm, Hadamard, Sinkhorn, mHC, and RoPE math that
the encoder graph now also expresses; the duplication goes away when the decoder
moves onto a GGML step graph. Its encoder-side cross-attention key/value projections
are already computed inside the encoder graph.

## Build and request options

Follow the platform [build instructions](../build/windows.md) on Windows.
For a focused build, select `AUDIOCPP_MODEL_SET=custom` and
`AUDIOCPP_MODELS=whistle_asr`. A Windows helper example is:

```powershell
.\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models "whistle_asr" -Target audiocpp_cli
cmake --build build/windows-cpu-release --target audiocpp_gguf
```

On macOS the Metal helper configures the same model set:

```bash
scripts/build_metal.sh --build-dir build/whistle --build-type Release --openmp auto --with-tests --deployment-build --model-set custom --models whistle_asr --target audiocpp_cli --target audiocpp_gguf
```

Use `--backend cpu` or `--backend metal`. `--threads` sets the ggml CPU backend
thread count and, when the build has OpenMP, the host decoder's worker count; the
runtime accepts 1 to 64. Use `--backend cpu --threads 1` for a portable inference
baseline. The spec exposes only the optional `language` request option, no
model-specific session or load options. Use `--language de` or
`--request-option language=de` to force German. Omit the language option for model
language selection. Decoding is greedy, with at most 320 text tokens. The model
manager has no Whistle package entry because `packages` is empty.

## Provenance and licensing

Use the pinned [upstream revision](https://huggingface.co/Cactus-Compute/whistle/tree/b358ddadd89b7a713b5aa131f23032d3cca1b251).
The converter preserves the upstream `LICENSE` inside the GGUF. The repository
license does not grant rights to the weights or extracted assets. See the
[provenance record](../../tools/community_models/whistle_asr/PROVENANCE.md) and
[model license table](../model_licenses.md) for source-specific terms and
unresolved questions. No weights or extracted assets are included in this port.

## Validation references

- [Official model, checkpoint, and license](https://huggingface.co/Cactus-Compute/whistle)
- [Published Whistle reference reconstruction](https://huggingface.co/mrfakename/whistle-ONNX)
- `whistle_frontend_test` checks mel features against independently computed values.
- `whistle_tokenizer_test` checks byte-fallback UTF-8 replacement and metaspace decoding without model weights.
- `whistle_assets_test <gguf> assets/resources/sample_16k.wav
  "Some call me Nature."` checks a converted GGUF against the public
  sample's first three seconds. `--backend cpu|metal` and `--threads n` select
  the backend; `--full` uses the whole recording.
- `whistle_assets_test <gguf> <wav> --reference <dir> [--tolerance f]` compares
  the mel features, encoder memory, and cross-attention keys/values against
  `.f32` reference dumps in `<dir>` and reports the largest difference as a
  fraction of each reference tensor's largest magnitude (default tolerance 2e-3;
  Metal's half-precision matmul operand staging needs `--tolerance 5e-3`).
  The dumps used in [the validation report](../../tests/whistle_asr/VALIDATION.md)
  came from the earlier host-only implementation at commit `6e0a40c4`.

To run the native tests through CTest, configure with
`-DENGINE_BUILD_TESTS=ON` and
`-DAUDIOCPP_WHISTLE_TEST_MODEL=<absolute-path-to-converted-GGUF>`, then build
`whistle_frontend_test`, `whistle_tokenizer_test`, and `whistle_assets_test`, then run
`ctest --test-dir <build-directory> -R whistle_ --output-on-failure`. The
opt-in tests check both the three-second excerpt and the entire public WAV,
plus quiet and constant-RMS speech, silence, the 30-second limit, and decoder reuse.

See [validation evidence](../../tests/whistle_asr/VALIDATION.md)
for exact local commands, observed results, performance, and readiness blockers.
Comparisons with the quantized official runtime are transcript checks, not exact
FP32 numerical parity.
