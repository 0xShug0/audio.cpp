# Phonon-2 (Parakeet TDT variant)

English-only, experimental Phonon-2 weights use the existing `parakeet_tdt`
family, loader and backend arithmetic. There is no CUDA precision-policy API
dependency, separate model family, or packed five-value runtime.

The model-local additions are optional hotwords, rolling live transcription,
and `word_timestamp_mode: "token_duration"`. Other Parakeet packages retain
their defaults. Shared transport support is limited to an optional revisable
transcript snapshot; no scheduler or multislot admission changes are included.

## Convert

Download the official Fermion Research artifact at revision
`7e153e4054c0a10db6d47d6fff5c062d4c77c154`, including the archive, NOTICE and
license files. Parakeet config/tokenizer sidecars use revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. Install `numpy`, `safetensors` and
`zstandard`; conversion does not require PyTorch or NeMo.

```bash
python tools/community_models/convert_phonon2.py \
  --source-dir models/Phonon-2-source \
  --reference-dir models/Phonon-2-source/parakeet-reference \
  --output-dir models/Phonon-2-Parakeet-F32 \
  --converter build/bin/audiocpp_gguf \
  --gguf-type mixed-f16 \
  --gguf-output models/Phonon-2-Parakeet-GGUF/phonon-2-mixed-f16.gguf
```

The example selects the lossless mixed-F16 package described below. For F32,
BF16 or Q8 derivatives, choose `--gguf-type orig`, `bf16` or `q8_0` and distinct
output paths. F32 expands the published trained checkpoint; it is not a pre-quantization
FP32 checkpoint. The converter verifies the container, retains attribution,
expands 699 inference tensors and omits 24 training-only counters. It compensates
the projection by 1/32 to cancel the normal loader's input-scale fold and adjusts
BatchNorm gamma in the two channels affected by its variance floor. These are
documented conversion transforms, not byte-identical original tensor claims.
The F32 audit checks every transformed tensor and the resulting encoder output.

BF16/Q8 are rounded derivatives. Projection tensors remain F32 to preserve their
scale compensation; Q8 also keeps the duration-sensitive joint head in F32.
The dense packages are approximately 2.51/1.26/0.94 GB, not the original 164 MB
packed package. No public download/package-manager URL is advertised here.

The converter sets `variant=phonon2` in the runtime configuration to select its
hotword and rolling-streaming behavior. `word_timestamp_mode` controls timestamp
formatting only. Regenerate unpublished development packages with this converter;
there is no compatibility fallback for earlier Phonon metadata or option names.

`--gguf-type mixed-f16` offers lossless storage of the staged values: tensors
use F16 only when conversion back to F32 is exact. Other tensors, normalization
and compensated projections stay F32. The package is about 1.30 GB. It does
not override backend precision policy; different storage types can select
different native kernels, so exact weights do not guarantee identical duration
decisions. To reproduce the F32 package's runtime weight types,
use existing session options `parakeet_tdt.matmul_weight_type=f32` and
`parakeet_tdt.conv_weight_type=f32`; this restores its working weight memory.

`--gguf-type mixed-q8` creates `phonon-2-mixed-q8.gguf` (about 1.18 GB).
It keeps feed-forward and pointwise matrices in exact F16, normalization and
non-F16-representable values in F32, and quantizes the remaining matrices to
Q8_0. It is a mixed-precision, rounded derivative, not lossless whole-model
storage. Across the 21 local clips on each of CPU, CUDA and Vulkan, all 63
transcripts match the dense reference, with 60/63 word timestamp matches
versus 51/63 for the previous Q8. One CPU clip acquires a 240 ms end shift;
the sampled improvement does not establish full timestamp parity or corpus
WER equivalence. See the [mixed-storage validation](../reports/phonon2_validation.md#mixed-gguf-storage-follow-up).
The original `q8_0` conversion remains available with its previous policy.

TDT timestamps advance on an 80 ms grid. Small score differences can change
blank/token selection or the predicted duration, producing whole-frame shifts
even when the final transcript matches. See the
[duration investigation](../reports/phonon2_validation.md#duration-diagnostics) for
decoder replays, reference-mask checks and conversion tradeoffs.

Earlier development packages used unscaled projections and special runtime
precision controls. Reconvert them; the loader rejects their obsolete metadata
with an actionable error rather than silently applying incompatible scaling.

## CLI and hotwords

```bash
audiocpp_cli --task asr --family parakeet_tdt \
  --model models/Phonon-2-Parakeet-GGUF/phonon-2-mixed-f16.gguf \
  --backend cuda --device 0 --threads 8 --audio recording.wav \
  --request-option 'hotwords=["Ada Lovelace","CUDA"]' \
  --request-option hotwords_score=2
```

Use `cpu`, or `vulkan --device 1` on the tested RTX 3090 setup. The device index
depends on the machine. For Windows executable paths add `.exe`.

Hotwords can be a JSON string array, `{ "word": "...", "spoken": ["..."] }`
entries, or comma/newline/semicolon-separated terms. Without separators, whitespace
separates terms. The model uses recognition-context text as a vocabulary fallback;
explicit hotwords, including an empty list, take precedence.
The policy accepts at most 25 distinct terms, strength 0–100 (default 2), 16 KiB
input, 256-byte terms and at most eight spoken alternatives. It does not bias
blank/duration logits or guarantee correction of every named entity. Request
state is cleared on reset/finalization. Stock Parakeet does not opt into it.

## Streaming

Use the same CLI command with `--mode streaming`, or configure a legacy server
model with `family: "parakeet_tdt"`, the new GGUF path, `task: "asr"` and
`mode: "streaming"`. Transcription uploads with `stream=true` and
`POST /v1/audio/transcriptions/live` (mono 16 kHz PCM) use existing transports.
JSON requests and batch multipart uploads can pass `hotwords` and
`hotwords_score` through the existing generic `options` object. Single-file
multipart uploads and live requests use the existing `prompt` field or query
parameter for vocabulary hints. JSON requests use `text` for recognition context.
There are no dedicated top-level hotword fields or live hotword query parameters.

The Phonon policy processes 50 ms energy blocks, allows a first provisional
decode after 350 ms of a voiced phrase, then every 500 ms of new audio, and
commits after 700 ms silence, 30 seconds, or input completion. It re-encodes the
growing phrase; this is not a cache-aware encoder or an offline speedup.

`partial_text_snapshot` and SSE `transcript.text.partial` contain the complete
current hypothesis and may revise previous words. Replace the provisional
display. `partial_text` and `transcript.text.delta` contain committed segments;
their concatenation equals final text. Committed timestamps have absolute
sample offsets. The original Phonon WebSocket protocol is not implemented.

## Validation and limits

See the [current validation report](../reports/phonon2_validation.md) and
[portable evidence](../reports/phonon2_validation.json) for exact inputs,
backend/storage results, numerical checks, route tests and build commands.
Backend/storage differences are measured rather than hidden by mandatory F32
kernel controls. Exact reference agreement is diagnostic; quantized storage,
provisional text and segmentation boundaries can differ. The sampled comparisons
do not establish corpus-level WER equivalence. RTX 5090, Metal, HIP and MUSA
runtime behavior was not tested locally.

Vulkan state/graph allocation optimizations are reviewed separately and are not
part of this minimal model PR. Raw previous and current evidence remains in
`outputs/phonon2-review-20261010/` and the earlier local Phonon artifact folders.
