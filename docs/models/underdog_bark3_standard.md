# Underdog Bark 3.0

[Conway Research's standard Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0)
is an English ASR checkpoint in the existing `parakeet_tdt` community family.
The 163 MB entropy-coded download expands to ordinary native GGUF weights;
it is not the runtime RAM/VRAM requirement. The explicit
`underdog_bark3_standard` variant shares the Bark architecture/runtime with Pro
and Plus, including the publisher's 250 ms prefix and original-sample timestamps.

The converter/reference harness reuse the packed reader introduced for Plus
with an immutable standard-checkpoint profile. Checkpoint hashes, variants,
metadata and staging remain separate; no module-global settings are mutated.
Existing Pro/Plus defaults and stock Parakeet/Phonon paths are preserved.
No shared backend/kernel or multi-slot changes are required.

## Build and convert

Use the [Bark/Parakeet build commands](underdog_bark3.md#convert) for the CLI,
server, GGUF tool and warm benchmark. CPU-only builds disable both GPU flags.
Conversion needs Python, `numpy`, `safetensors` and `constriction==0.5.0`;
native inference needs none of these Python dependencies.

```bash
hf download ConwayResearch/Underdog-Bark-3.0 \
  --revision 85db2ad18ab41e534a222ee253c29191d77ed919 \
  --include model.bark bark_loader.py mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir models/Underdog-Bark-3.0-source
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json \
    tokenizer_config.json generation_config.json \
  --local-dir models/Underdog-Bark-3.0-source
python tools/community_models/convert_underdog_bark3_standard.py \
  --source models/Underdog-Bark-3.0-source \
  --staging models/Underdog-Bark-3.0-prepared \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type mixed-f16 \
  --output models/underdog-bark-3.0-mixed-f16.gguf
```

Adjust executable paths for the generator/platform (`.exe` on Windows).
Reuse staging with `--pack-only --type f32`, `q8_0` or `bf16` and another output
path. `--prepare-only` writes staging; `--overwrite` replaces generated GGUFs.
The source is never changed. Every GGUF embeds its English spec v1,
tokenizer/configs, tensor hashes, conversion provenance, NOTICE and full
CC-BY-4.0 license. No unpublished model-manager download is advertised.

Mixed-F16 preserves all staged F32 values, retaining non-representable scale
products, normalization and compensated projections in F32. Q8 uses the
protected mixed policy: eligible attention matrices use Q8_0, feed-forward and
pointwise matrices keep exact F16, and non-representable products retain F32.
The standard BF16 profile retains the predictor, encoder-to-joint projection
and output head in F32 to protect capitalization/duration decisions; Plus's
existing BF16 policy is unchanged. Q8/BF16 still add rounding and must be
assessed separately from the source checkpoint.
Keep `matmul_weight_type`/`conv_weight_type` at `native` to retain this storage
policy; explicit weight-type overrides can round protected tensors again.

## Run

```bash
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model-spec-override embedded --backend cuda --threads 8 \
  --model models/underdog-bark-3.0-mixed-f16.gguf \
  --audio input-16k-mono.wav --text-out transcript.txt --words-out words.json --metrics
build/bark3/bin/audiocpp_server --config examples/server/underdog-bark3-standard-asr.json
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=bark3standard -F file=@input-16k-mono.wav
```

Edit the example paths; use `--backend cpu` or `vulkan` for other devices.
Select the intended GPU with `--device` if multiple Vulkan devices are present.
Select the embedded spec explicitly, since the family catalog otherwise
describes stock multilingual Parakeet. The ordinary legacy server is used.
Full context is the default. Prefix correction, opt-in offline Silero VAD,
native `hotwords`/`hotwords_score`, and raw-PCM HTTP/SSE buffered streaming use
the existing [Bark input/context controls](underdog_bark3.md#input-context-and-timestamps).

English only. Full-context attention grows with duration and retains the encoder
frame limit. Buffered windows/VAD can change recognition relative to full-context
decoding. Streaming is buffered, not native cache-aware. VAD is offline-only;
timestamps use the native TDT emission policy rather than forced alignment.
Metal/HIP and corpus-level WER are unvalidated.
The original model emits `Yeah.` on the one-second silence test; unfiltered
native inference preserves this behavior. Opt-in VAD can skip non-speech spans.

## Reference validation

Install PyTorch 2.8.0, Transformers 5.19.0, soundfile, soxr and librosa in a
separate Python 3.11 reference environment, plus the converter dependencies.
The original loader can fetch NVIDIA processor/generation metadata on first
use, but does not download NVIDIA model weights.

```bash
python tests/parakeet_tdt/bark3_standard_reference.py \
  --source models/Underdog-Bark-3.0-source \
  --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
  --device cpu --check-official-transcribe --output build/bark3/standard-reference.json
python tests/parakeet_tdt/validate_bark3.py --bin-dir build/bark3/bin \
  --model models/underdog-bark-3.0-mixed-f16.gguf --backend cpu \
  --reference build/bark3/standard-reference.json --output-dir build/bark3/standard-check
python -m unittest discover -s tests/parakeet_tdt -p test_bark3_standard_conversion.py
```

See [the validation record](../validation/underdog_bark3_standard.md) for actual
backends, precision differences, time/memory and artifact paths. Matching
particular recordings does not establish corpus WER or universal rounded parity.

Weights are CC-BY-4.0. Credit Conway Research, NVIDIA Parakeet v3 and Moondream
Parakeet Ultra; retain NOTICE/license and disclose conversion/rounding changes.
