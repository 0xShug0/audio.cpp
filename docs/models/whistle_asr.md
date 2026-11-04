# Whistle ASR (experimental)

The `whistle_asr` family runs [Cactus Compute Whistle](https://huggingface.co/Cactus-Compute/whistle)
locally through audio.cpp. The current implementation supports **offline**
transcription of 16 kHz mono WAV files up to 30 seconds. The encoder uses the
selected backend and the decoder runs on the host CPU. It detects one of the
model's seven languages (English, German, French, Spanish, Italian, Dutch, or
Polish); `--language en`, `de`, `fr`, `es`, `it`, `nl`, or `pl` forces a language.
These are supported language codes, not seven-language validation evidence.
The focused validation report records the English and German checks separately.
French, Spanish, Italian, Dutch, and Polish each have one ElevenLabs synthetic
clip checked through automatic and forced-language CLI requests. See the
[five-language spot checks](../../tests/whistle_asr/MULTILINGUAL_VALIDATION.md).
This limited coverage does not qualify general accuracy in all seven languages.

## Model

| Field | Value |
|---|---|
| Family | `whistle_asr` |
| Tasks | `asr` |
| Modes | `offline` |
| Languages | `en`, `de`, `fr`, `es`, `it`, `nl`, `pl` |
| Default package | `whistle_asr_f32` |
| Input | Speech audio |
| Output | Transcript text |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `language` | `string` | Model/session default | Force one supported language; omit to detect the language. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `whistle_asr.exec_plan` | `string` | `{}` | JSON component execution plans. Component decoder accepts impl native/neon (default native), both on CPU independently of the encoder backend. The experimental neon implementation accelerates the vocabulary projection with the framework NEON library and requires ENGINE_ENABLE_NEON=ON and AArch64 NEON. No artifacts or params are used. |

## Quick Start

The default package is `whistle_asr_f32`, containing
`Whistle-GGUF/whistle-f32.gguf` in
[`audio-cpp/audio.cpp-gguf`](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Whistle-GGUF)
(about 211 MiB).

```bash
audiocpp_cli --task asr --family whistle_asr --backend cuda --threads 4 \
  --model models/Whistle-GGUF/whistle-f32.gguf \
  --audio assets/resources/sample_16k.wav --log --metrics
```

Use `--backend vulkan`, `cpu`, or `metal` for those backends. The decoder
runs on the host CPU in every case.

## Local conversion

To convert the official model locally, use the exact revision
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
  --audio assets/resources/sample_16k.wav --log --metrics
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
longer recordings are not supported by this family. CPU, Metal, CUDA, and Vulkan
are supported encoder backends. HIP has not been validated.
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

Use `--backend` to select an available backend: `cpu`, `metal`, `cuda`, or `vulkan`.
`--threads` sets the CPU encoder's
thread count and bounds the host decoder's OpenMP workers; the runtime accepts
1 to 64. Build with OpenMP enabled for parallel host decoding. Without OpenMP,
the host decoder runs serially. An experimental decoder implementation reuses
the framework NEON library for the FP32 vocabulary projection, retaining the
same host attention, mHC, Hadamard and cache logic. Build with
`-DENGINE_ENABLE_NEON=ON` on AArch64 and opt in with:

```bash
--session-option 'whistle_asr.exec_plan={"decoder":{"impl":"neon"}}'
```

The default is `native`; `impl: "default"` also selects it. Both implementations
run on CPU independently of the encoder backend and use the session thread
count. No artifacts or parameters are accepted. Existing embedded GGUF specs
can use this option without reconversion. The spec exposes the optional `language`
request option and `exec_plan` session option. Use `--language de` or
`--request-option language=de` to force German. Omit the language option for model
language selection. Decoding is greedy, with at most 320 text tokens.

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
  sample's first three seconds. `--backend cpu|metal|cuda|vulkan` and `--threads n` select
  the backend; `--full` uses the whole recording.
- `whistle_assets_test <gguf> <wav> --reference <dir> [--tolerance f]` compares
  the mel features, encoder memory, and cross-attention keys/values against
  `.f32` reference dumps in `<dir>` and reports the largest difference as a
  fraction of each reference tensor's largest magnitude (default tolerance 2e-3;
  Metal's half-precision matmul operand staging needs `--tolerance 5e-3`).
  The same modes also run the decoder on the full recording.
  `--dump` writes `decoder_logits_<position>.f32` (the full 8199-wide logits of
  each step), `decoder_tokens.i32` (input tokens, BOS and language token first)
  and `decoder_transcript.txt`. `--reference` compares every logits file with the
  same scaled metric and requires identical tokens and transcript.
  In [the validation report](../../tests/whistle_asr/VALIDATION.md), the encoder
  dumps came from the earlier host-only implementation at commit `6e0a40c4`. The
  decoder dumps came from the host decoder, before the step graph replaced it.

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
