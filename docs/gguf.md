# GGUF Models

audio.cpp can load audio.cpp-native GGUF checkpoints for model families that have a
package spec. GGUF is a container for tensors and sidecar files; it is not a universal
adapter for arbitrary llama.cpp or whisper.cpp GGUF files. The tensor names and embedded
metadata still have to match the selected `--family`.

## Build The Converter

```bash
cmake --build build/debug --parallel --target audiocpp_gguf
```

Check the converter interface:

```bash
audiocpp_gguf --help
```

Current shape:

```bash
audiocpp_gguf --input [namespace=]<weights> [--input namespace=<weights> ...] \
  --output <weights.gguf> \
  --type <orig|f16|bf16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> \
  [--root <model-dir>] \
  [--sidecar <source>=<destination>] \
  [--overwrite] \
  [--no-sidecars]

audiocpp_gguf --inspect <model.gguf>
```

## Convert A Single Tensor Source

Use `--root` when the model has tokenizer, config, processor, or other non-weight files
that should be embedded into the GGUF.

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type f16 \
  --overwrite
```

Safetensors shard indexes are accepted directly:

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors.index.json \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type q8_0 \
  --overwrite
```

## Convert A Multi-Component Model

Use repeated namespaced `--input` entries when a model has multiple tensor components.
The namespace must match the model's package spec.

```bash
audiocpp_gguf \
  --input model_weights=/path/to/model/model.safetensors \
  --input codec_weights=/path/to/model/codec/model.safetensors \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type f16 \
  --overwrite
```

## Add External Sidecars

Use `--sidecar <source>=<destination>` when a runtime file is needed but does not live
under `--root`, or when it should be embedded at a different path inside the GGUF.

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors.index.json \
  --root /path/to/model \
  --sidecar /path/to/shared/preprocessor_config.json=preprocessor_config.json \
  --output /path/to/model-gguf/model.gguf \
  --type q8_0 \
  --overwrite
```

Pass `--no-sidecars` only when you intentionally want a tensor-only container.

## Inspect And Run

Inspect the finished package before using it:

```bash
audiocpp_gguf --inspect /path/to/model-gguf/model.gguf
```

If the GGUF embeds all required sidecars, it can be passed directly as `--model`:

```bash
audiocpp_cli --task asr --family qwen3_asr --model /path/to/model-gguf/model.gguf --backend cuda --audio speech.wav
```

A directory containing `model.gguf` is also accepted by supported package specs:

```bash
audiocpp_cli --task tts --family qwen3_tts --model /path/to/model-gguf --backend cuda --text "Hello." --out out.wav
```

## Type Notes

| Type | Meaning |
|---|---|
| `orig` | Preserve the original safetensors storage type where possible. |
| `f16` | Convert eligible tensors to FP16. |
| `bf16` | Convert eligible tensors to BF16. Useful for BF16 source models. |
| `q8_0` | Quantize eligible tensors to Q8_0; unsupported tensors remain in a backend-safe type. |
| `q2_k`/`q3_k`/`q4_k`/`q5_k`/`q6_k` | Lower-bit quantized formats. Treat as experimental per model and backend. |

Quantized GGUF support is model- and route-specific. A model may load successfully but
still drift in length, waveform similarity, or recognized text, so validate the exact
route you plan to ship.

## Support And Test Status

Status labels:

| Label | Meaning |
|---|---|
| `Done` | Package-spec refactor is in place for this family. |
| `No` | Package-spec refactor is not done, or the tested format is not usable. |
| `Pass` | Covered by the path-test matrix with acceptable output. |
| `Pass (drift)` | Loads and runs, with known acceptable output drift. |
| `No (...)` | Known unsupported, failing, or too much output drift. |
| `---` | Not tested in the current GGUF path-test matrix. |

| Family | Package-spec refactor | Safetensors tested after refactor | 16-bit GGUF tested | `orig` GGUF tested | `q8_0` GGUF tested |
|---|---|---|---|---|---|
| `ace_step` | No | --- | --- | --- | --- |
| `chatterbox` | No | --- | --- | --- | --- |
| `citrinet_asr` | Done | Pass | --- | --- | Pass |
| `heartmula` | No | --- | --- | --- | --- |
| `higgs_audio_stt` | Done | Pass | Pass | --- | Pass |
| `htdemucs` | No | --- | --- | --- | --- |
| `hviske_asr` | Done | Pass | --- | --- | Pass |
| `index_tts2` | Done | Pass | Pass (drift) | Pass | No (similarity drift, frame drift, text minor drift) |
| `irodori_tts` | Done | Pass | Pass | --- | Pass (drift) |
| `marblenet_vad` | No | --- | --- | --- | --- |
| `mel_band_roformer` | No | --- | --- | --- | --- |
| `miocodec` | No | --- | --- | --- | --- |
| `miotts` | No | --- | --- | --- | --- |
| `moss_tts_local` | Done | Pass | Pass | --- | No (similarity drift, frame drift, text minor drift) |
| `moss_tts_nano` | Done | Pass | Pass | --- | No (similarity drift, frame drift, text large drift) |
| `nemotron_asr` | Done | Pass | Pass | --- | Pass (minor filler drift) |
| `omnivoice` | Done | Pass | No (runtime assert, no audio) | --- | No (runtime assert, no audio) |
| `pocket_tts` | No | --- | --- | --- | --- |
| `qwen3_asr` | Done | Pass | Pass | --- | Pass |
| `qwen3_forced_aligner` | Done | Pass | Pass | --- | Pass |
| `qwen3_tts` base | Done | Pass | No (similarity drift, frame drift, text minor drift) | --- | No (similarity drift, frame drift, text minor drift) |
| `qwen3_tts` custom voice | Done | Pass | Pass (minor log-mel drift) | --- | No (similarity drift, frame drift, text minor drift) |
| `qwen3_tts` voice design | Done | Pass | Pass (minor log-mel drift) | --- | No (similarity drift, frame drift, text minor drift) |
| `seed_vc` | No | --- | --- | --- | --- |
| `silero_vad` | No | --- | --- | --- | --- |
| `sortformer_diar` | No | --- | --- | --- | --- |
| `stable_audio` | No | --- | --- | --- | --- |
| `supertonic` | Done | Pass | --- | Pass | No (unsupported weight dtype) |
| `vevo2` | No | --- | --- | --- | --- |
| `vibevoice` | No | --- | --- | --- | --- |
| `vibevoice_asr` | Done | Pass | Pass | --- | Pass |
| `voxcpm2` | No | --- | --- | --- | --- |
