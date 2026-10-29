# Underdog Bark 3.0 Plus

[Conway Research's Bark 3.0 Plus](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Plus)
is an English ASR checkpoint in the existing `parakeet_tdt` community family.
It has the same 0.6B FastConformer/TDT architecture as Bark Pro, with weights
distributed in an entropy-coded `model.bark` container. The 180 MB download is
not its expanded RAM/VRAM requirement. This port uses the existing native
Parakeet operations, not a Python bridge or new packed-weight GPU kernels.

The explicit `underdog_bark3_plus` variant reuses Bark's 250 ms prefix and original
sample-clock correction, CPU graph-allocation fix, offline VAD, native hotwords,
and buffered streaming frontend cache. Stock Parakeet, Phonon and Pro retain
their behavior. Full context is the default. Buffered windows/VAD can change
recognition relative to the publisher's whole-recording context; streaming is
not native cache-aware inference. English only; Metal/HIP are not validated.

## Convert

Use the Parakeet/Bark [build instructions](underdog_bark3.md#convert), including
`audiocpp_cli`, `audiocpp_server`, `audiocpp_gguf` and `parakeet_warm_bench`.
The native converter needs `numpy`, `safetensors`, `constriction==0.5.0` and Python;
inference needs none of those. Download just the pinned data and license files:

```bash
hf download ConwayResearch/Underdog-Bark-3.0-Plus \
  --revision d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb \
  --include model.bark mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir models/Underdog-Bark-3.0-Plus-source
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json \
    tokenizer_config.json generation_config.json \
  --local-dir models/Underdog-Bark-3.0-Plus-source

python tools/community_models/convert_underdog_bark3_plus.py \
  --source models/Underdog-Bark-3.0-Plus-source \
  --staging models/Underdog-Bark-3.0-Plus-prepared \
  --gguf-tool build/bark3/bin/audiocpp_gguf --type mixed-f16 \
  --output models/underdog-bark-3.0-plus-mixed-f16.gguf
```

Adjust executable paths for the generator/platform (`.exe` on Windows). Reuse
staging with `--pack-only --type f32`, `bf16` or `q8_0` and a different output.
`--prepare-only` writes staging without packing. `--overwrite` replaces an
existing generated GGUF; source files are never changed.

The converter verifies the container SHA-256 and pinned base sidecars, decodes
the ANS/FP16 levels directly without executing downloaded Python, and prepares
699 inference tensors after dropping 24 training-only counters. Projection and
BatchNorm compensation match the existing Parakeet loader. Each GGUF embeds its
English spec v1, tokenizer/configs, tensor hashes, conversion provenance, upstream
NOTICE and the full CC-BY-4.0 license. There are no advertised unpublished
model-manager downloads.

`mixed-f16` preserves every staged F32 value: tensors that cannot round-trip
through F16 stay F32, as do normalization and compensated projection tensors.
Integer times FP16 row-scale products can require F32; treating this checkpoint
like Pro's BF16 source would silently round them. Q8/BF16 are additionally rounded
derivatives and must be assessed separately from the original packed checkpoint.

## CLI and server

```bash
build/bark3/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model-spec-override embedded --backend cuda \
  --model models/underdog-bark-3.0-plus-mixed-f16.gguf \
  --audio input-16k-mono.wav --text-out transcript.txt --words-out words.json --metrics
```

Use `--backend cpu` or `--backend vulkan` for other builds/devices. Select the
embedded spec explicitly; the family catalog otherwise describes stock
multilingual Parakeet. The [server example](../../examples/server/underdog-bark3-plus-asr.json)
uses the same contract and the ordinary legacy server runtime:

```bash
build/bark3/bin/audiocpp_server --config examples/server/underdog-bark3-plus-asr.json
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=bark3plus -F file=@input-16k-mono.wav
```

The existing [Bark input/context controls](underdog_bark3.md#input-context-and-timestamps)
also apply: prefix correction, explicit bounded windows, offline Silero VAD,
`hotwords`/`hotwords_score`, contiguous mono 16 kHz raw-PCM streaming and HTTP/SSE.
Reported word times follow the native TDT emission policy, not a forced aligner.
Long full-context attention grows with duration and retains the encoder frame
limit. No parallel-runtime or multi-slot changes are part of this port.

## Validation and attribution

See [the validation record](../validation/underdog_bark3_plus.md) for exact source
reconstruction, original-runner comparisons, backends, timing and limitations.
Matching test recordings is not a corpus-level WER claim.

The model is CC-BY-4.0. Credit Conway Research, NVIDIA Parakeet v3 and Moondream
Parakeet Ultra, retain NOTICE/license, and disclose conversion/rounding changes
when redistributing derivatives.
