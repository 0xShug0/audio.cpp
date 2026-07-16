# GGUF Models

audio.cpp can load audio.cpp-native GGUF checkpoints for model families that have a
package spec. GGUF is a container for tensors and sidecar files; it is not a universal
adapter for arbitrary llama.cpp or whisper.cpp GGUF files. The tensor names and embedded
metadata still have to match the selected `--family`.

Package specs are maintained as `model_specs/*.json`. New GGUFs contain the selected
spec in `audiocpp.model_spec.*` metadata, so a standalone GGUF does not depend on a
`model_specs` directory or on the binary having that family compiled into its spec
catalog.

Runtime resolution is deterministic:

```text
explicit --model-spec-override
              |
              v
package spec embedded in the selected GGUF
              |
              v
compiled catalog (AUDIOCPP_DEPLOYMENT_BUILD=ON)
              |
              v
external model_specs/<family>.json discovery
```

An explicit override is useful for testing a modified layout without rebuilding or
reconverting:

```bash
audiocpp_cli --inspect --family qwen3_asr --model /path/to/model.gguf \
  --model-spec-override /path/to/qwen3_asr.json
```

The override may also be a directory containing `<family>.json`. The server supports
the same command-line option and a `model_spec_override` field either at the top level
or inside an individual model entry. A per-model field takes precedence over the
server-wide value.

## Build The Converter

```bash
cmake --build build/debug --parallel --target audiocpp_gguf
```

Normal builds leave `AUDIOCPP_DEPLOYMENT_BUILD` off. Enable it when one binary must also
carry fallback specs for safetensors packages or legacy GGUFs that predate embedded spec
metadata:

```bash
cmake -S . -B build/deploy -DAUDIOCPP_DEPLOYMENT_BUILD=ON
cmake --build build/deploy --parallel
```

`audiocpp_gguf` always carries the conversion catalog. This is separate from the optional
CLI/server deployment catalog, and keeps a copied converter executable usable when the
source checkout and its `model_specs` directory are not present.

Check the converter interface:

```bash
audiocpp_gguf --help
```

Current shape:

```bash
audiocpp_gguf --input [namespace=]<weights> [--input namespace=<weights> ...] \
  --output <weights.gguf> \
  --type <orig|f16|bf16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> \
  [--family <family>] \
  [--model-spec <json-or-directory>] \
  [--root <model-dir>] \
  [--sidecar <source>=<destination>] \
  [--overwrite] \
  [--no-sidecars] \
  [--allow-missing-model-spec]

audiocpp_gguf --inspect <model.gguf>
```

## Convert A Single Tensor Source

Standalone conversion is the default. The converter embeds non-weight files recursively
from the first tensor source's directory, or from `--root` when it is supplied. Use
`--root` when the model has tokenizer, config, processor, or other non-weight files in
a different model root. It also finds, validates, and embeds the package spec. Conversion
fails before writing when the tensor namespaces or required sidecars do not match that
spec.

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

If the default pipeline cannot find any sidecars, conversion fails instead of silently
creating a tensor-only file. Supply the correct `--root` and any required external
`--sidecar` mappings. Pass `--no-sidecars` only when you intentionally want a tensor-only
container; place that GGUF and all package-spec-required sidecars together in one model
directory when loading it. `--no-sidecars` does not remove the embedded package spec and
does not disable build-time validation.

## Package Spec Discovery During Conversion

The converter selects the first valid source at the highest available priority:

1. `--model-spec <json-or-directory>` (also accepted as `--model-spec-override`).
2. A spec object, JSON string, or relative path in the model's `config.json`.
3. `model_spec.json` or `model_specs/*.json` below the model root.
4. A discovered `model_specs/*.json` directory from the working directory upward.
5. The converter's bundled source catalog.

Higher-priority inputs are authoritative. If an explicit override, model-config spec, or
local spec is present but does not match the tensor namespaces and required files, the
converter reports that error instead of silently falling back to a lower-priority layout.

Use `--family <family>` to disambiguate models whose configuration does not identify the
audio.cpp family. A model configuration can declare it directly:

```json
{
  "audiocpp_family": "qwen3_asr",
  "audiocpp_model_spec": "model_spec.json"
}
```

`audiocpp_model_spec` may instead be a JSON string or a path relative to `config.json`.
The nested forms `audiocpp.family`, `audiocpp.model_spec`, and
`audiocpp.package_spec` are also accepted. The converter additionally recognizes known
upstream `model_type` values.

`--allow-missing-model-spec` is an explicit escape hatch for creating a tensor archive
that audio.cpp is not expected to load as a model. It is not recommended for deployable
GGUFs.

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

Compatibility summary:

| Format | Where its package spec comes from | Other model files |
|---|---|---|
| Safetensors | Override, compiled deployment catalog, or external discovery | Required |
| New standalone GGUF | Embedded in GGUF | None |
| New tensor-only GGUF (`--no-sidecars`) | Embedded in GGUF | Required sidecars |
| Legacy GGUF without embedded spec | Compiled deployment catalog or external discovery | Depends on embedded sidecars |

Compatibility with older binaries:

`After PR #53` refers to tag `release-0.3-gguf-v2`, commit
`bf1ac678758aee4caafa7bb25fc0e6db9c25228f`.

| Build | GGUF package | Runtime context | Result |
|---|---|---|---|
| Before PR #53 (`14e9258`) | Legacy GGUF without embedded spec | Repo checkout or external `model_specs` visible | Pass |
| Before PR #53 (`14e9258`) | New standalone GGUF | Repo checkout or external `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), normal build | Legacy GGUF without embedded spec | Repo checkout or external `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), normal build | New standalone GGUF | Repo checkout or external `model_specs` visible | Pass |
| Before PR #53 (`14e9258`) | Legacy GGUF without embedded spec | No `model_specs` visible | Fail |
| Before PR #53 (`14e9258`) | New standalone GGUF | No `model_specs` visible | Fail |
| After PR #53 (`release-0.3-gguf-v2`), normal build | Legacy GGUF without embedded spec | No `model_specs` visible | Fail; use `--model-spec-override`, a deployment build, or external specs |
| After PR #53 (`release-0.3-gguf-v2`), normal build | New standalone GGUF | No `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), deployment build | Legacy GGUF without embedded spec | No `model_specs` visible | Pass through compiled package specs |

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
| `Skip (...)` | Package-spec refactor is intentionally skipped. |
| `Pass` | Covered by the path-test matrix with acceptable output. |
| `Pass (TTS + clone)` | Both no-reference TTS and reference-audio voice cloning run successfully. |
| `Pass (drift)` | Loads and runs, with known acceptable output drift. |
| `No (...)` | Known unsupported, failing, or too much output drift. |
| `---` | Not tested in the current GGUF path-test matrix. |

| Family | Package-spec refactor | Safetensors tested after refactor | `orig` GGUF tested | 16-bit GGUF tested | `q8_0` GGUF tested |
|---|---|---|---|---|---|
| `ace_step` | No | --- | --- | --- | --- |
| `chatterbox` | No | --- | --- | --- | --- |
| `citrinet_asr` | Done | Pass | --- | --- | Pass |
| `heartmula` | No | --- | --- | --- | --- |
| `higgs_audio_stt` | Done | Pass | --- | Pass | Pass |
| `htdemucs` | Done | Pass | --- | Pass | Pass (drift) |
| `hviske_asr` | Done | Pass | --- | --- | Pass |
| `index_tts2` | Done | Pass | Pass | Pass (drift) | No (similarity drift, frame drift, text minor drift) |
| `irodori_tts` | Done | Pass | --- | Pass | Pass (drift) |
| `marblenet_vad` | Skip (tiny model) | --- | --- | --- | --- |
| `mel_band_roformer` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `miocodec` | Done | Pass | Pass | Pass (drift) | Pass (drift) |
| `miotts` | Done | Pass | Pass | Pass (drift) | No (similarity drift, frame drift) |
| `moss_tts_local` | Done | Pass | --- | Pass | No (similarity drift, frame drift, text minor drift) |
| `moss_tts_nano` | Done | Pass | --- | Pass | No (similarity drift, frame drift, text large drift) |
| `nemotron_asr` | Done | Pass | --- | Pass | Pass (minor filler drift) |
| `omnivoice` | Done | Pass | --- | No (runtime assert, no audio) | No (runtime assert, no audio) |
| `pocket_tts` | No | --- | --- | --- | --- |
| `qwen3_asr` | Done | Pass | --- | Pass | Pass |
| `qwen3_forced_aligner` | Done | Pass | --- | Pass | Pass |
| `qwen3_tts` base | Done | Pass | Pass | No (similarity drift, frame drift, text minor drift) | No (similarity drift, frame drift, text minor drift) |
| `qwen3_tts` custom voice | Done | Pass | --- | Pass (drift) | No (similarity drift, frame drift, text minor drift) |
| `qwen3_tts` voice design | Done | Pass | --- | Pass (drift) | No (similarity drift, frame drift, text minor drift) |
| `seed_vc` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `silero_vad` | Skip (tiny model) | --- | --- | --- | --- |
| `sortformer_diar` | No | --- | --- | --- | --- |
| `stable_audio` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `supertonic` | Done | Pass | Pass | --- | No (unsupported weight dtype) |
| `vevo2` | Done | Pass | Pass | Pass (drift) | No (similarity drift, frame drift) |
| `vibevoice` | No | --- | --- | --- | --- |
| `vibevoice_asr` | Done | Pass | --- | Pass | Pass |
| `voxcpm2` | Done | Pass | Pass | No (clone similarity drift, frame drift) | No (similarity drift, frame drift) |
