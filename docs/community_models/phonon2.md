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
  --gguf-output models/Phonon-2-Parakeet-GGUF/phonon-2-f32.gguf
```

For a new conversion choose `--gguf-type bf16` or `q8_0` and distinct output
paths. F32 expands the published trained checkpoint; it is not a pre-quantization
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

Earlier development packages used unscaled projections and special runtime
precision controls. Reconvert them; the loader rejects their obsolete metadata
with an actionable error rather than silently applying incompatible scaling.

## CLI and hotwords

```bash
audiocpp_cli --task asr --family parakeet_tdt \
  --model models/Phonon-2-Parakeet-GGUF/phonon-2-q8_0.gguf \
  --backend cuda --device 0 --threads 8 --audio recording.wav \
  --request-option 'hotwords=["Ada Lovelace","CUDA"]' \
  --request-option hotword_lambda=2
```

Use `cpu`, or `vulkan --device 1` on the tested RTX 3090 setup. The device index
depends on the machine. For Windows executable paths add `.exe`.

Hotwords can be a JSON string array, `{ "word": "...", "spoken": ["..."] }`
entries, or comma/newline/semicolon-separated terms. Without separators, whitespace
separates terms. The original `prompt` vocabulary fallback is accepted by the
transcription API; explicit hotwords, including an empty list, take precedence.
The policy accepts at most 25 distinct terms, strength 0–100 (default 2), 16 KiB
input, 256-byte terms and at most eight spoken alternatives. It does not bias
blank/duration logits or guarantee correction of every named entity. Request
state is cleared on reset/finalization. Stock Parakeet does not opt into it.

## Streaming

Use the same CLI command with `--mode streaming`, or configure a legacy server
model with `family: "parakeet_tdt"`, the new GGUF path, `task: "asr"` and
`mode: "streaming"`. Transcription uploads with `stream=true` and
`POST /v1/audio/transcriptions/live` (mono 16 kHz PCM) use existing transports.
JSON/multipart uploads accept `hotwords`, `hotword_lambda` and repeated
`hotwords[]`; the live route accepts query options.

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
