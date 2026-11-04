# Phonon-2 ASR

Phonon-2 is an English speech-recognition model exposed as `phonon2_asr`.
Upstream: [FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2).
It shares the Parakeet implementation but has its own model contract, optional
hotwords, and phrase-based buffered streaming. It is distinct from the
[Gradium Phonon TTS model](gradium_phonon.md).

## Install

```bash
python tools/model_manager_v2.py install phonon2_asr_q8_0 --models-root models
```

Use `phonon2_asr_f16` for the mixed-F16 package. Published packages are in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Phonon-2-GGUF).

## Quick Start

```bash
audiocpp_cli --task asr --family phonon2_asr \
  --model models/Phonon-2-GGUF/phonon-2-mixed-q8.gguf \
  --backend cuda --audio recording.wav \
  --text-out transcript.txt --words-out words.json --log
```

Add `--mode streaming` for incremental transcription. Use the same family in
server configuration:

```json
{
  "id": "phonon-2",
  "family": "phonon2_asr",
  "path": "models/Phonon-2-GGUF/phonon-2-mixed-q8.gguf",
  "task": "asr",
  "mode": "streaming"
}
```

## Model

| Field | Value |
|---|---|
| Family | `phonon2_asr` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Language | English |
| Input | Speech audio |
| Output | Transcript text and optional word timestamps |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model GGUF file or package directory. |
| `--backend` | Backend supported by the selected build. |
| `--audio` | Input recording; streaming raw PCM uses `--audio -`. |
| `--mode` | `offline` or `streaming`. |
| `--text-out` | Save the transcript. |
| `--words-out` | Save word timestamps. |
| `--log` | Enable runtime logging. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `return_timestamps` | `bool` | Model/session default | Accepted for CLI --words-out compatibility. Word timestamps are always returned, including when false or omitted. |
| `max_tokens` | `int` | `0` | Maximum TDT generated tokens; 0 or omitted uses the model-derived limit. Declared range: minimum 0. |
| `keep_language_tags` | `bool` | `false` | Keep language tag tokens in decoded text; default false. |
| `audio_chunk_mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline audio chunking mode. fixed selects overlapping long-form windows regardless of offline_mode; vad uses the configured VAD model (Silero by default) to skip silence; auto follows offline_mode. Offline input is downmixed and resampled as needed. |
| `audio_chunk_duration_sec` | `float` | Model/session default | VAD maximum chunk duration; omitted inherits the session duration. Fixed/long_form windows use the session setting, not this request override. Declared range: minimum 0.001. |
| `hotwords` | `string` | Model/session default | Optional native vocabulary bias: comma-separated terms or a JSON list; up to 25 terms. |
| `hotwords_score` | `float` | `2.0` | Phonon-2/Bark native matching-piece logit bonus; duration scores and blank are not biased. Declared range: minimum 0, maximum 100. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `phonon2_asr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type; default native. |
| `phonon2_asr.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Encoder and decoder matmul weight storage type; defaults to weight_type, which defaults to native. Q8_0 measured 1.79x faster on the tested CPU and changed roughly 8 percent of transcripts without moving aggregate word error rate. |
| `phonon2_asr.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage type; default native. |
| `phonon2_asr.perf_mode` | `off`, `flash_attention` | `off` | Encoder attention implementation. Default off uses the validated relative-attention path; flash_attention enables the fused implementation, which was numerically validated but slower on the tested hardware. |
| `phonon2_asr.weight_context_mb` | `int` | `3072` | Weight context arena size in MiB; default 3072. Declared range: minimum 1. |
| `phonon2_asr.encoder_graph_arena_mb` | `int` | `1024` | Encoder graph arena size in MiB; default 1024. Declared range: minimum 1. |
| `phonon2_asr.decoder_graph_arena_mb` | `int` | `256` | Decoder graph arena size in MiB; default 256. Declared range: minimum 1. |
| `phonon2_asr.audio_chunk_duration_sec` | `float` | `2` | Center-region duration for buffered streaming in seconds; default 2. Fixed context windows are re-encoded rather than cache-aware. Also sizes offline fixed/long_form windows. Declared range: minimum 0.001. |
| `phonon2_asr.left_context_sec` | `float` | `10` | Past context included when re-encoding each buffered-streaming window in seconds; default 10. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `phonon2_asr.right_context_sec` | `float` | `2` | Future lookahead included when re-encoding each buffered-streaming window in seconds; default 2 and adds equivalent partial-result latency. Also applies to offline fixed/long_form windows. Declared range: minimum 0. |
| `phonon2_asr.streaming_attention_mode` | `full_context` | `full_context` | Attention policy inside each buffered window. full_context preserves bidirectional attention over the bounded window. |
| `phonon2_asr.offline_mode` | `full_context`, `long_form`, `auto` | `full_context` | Offline encoder scheduling. full_context encodes the whole utterance, long_form uses bounded overlapping windows, and auto selects long_form beyond audio_chunk_threshold_sec. |
| `phonon2_asr.audio_chunk_threshold_sec` | `float` | `30.0` | Duration threshold used by offline_mode=auto before switching to bounded-window long-form execution; default 30 seconds. Declared range: minimum 0.001. |
| `phonon2_asr.vad_model_path` | `path` | `assets/framework/models/silero_vad` | VAD model path used by audio_chunk_mode=vad (any built-in VAD family); default assets/framework/models/silero_vad. |

## Hotwords

```bash
--request-option 'hotwords=["Ada Lovelace","CUDA"]' \
--request-option hotwords_score=2
```

Hotwords accept JSON strings, objects with `word` and optional `spoken` forms,
or comma/newline/semicolon-separated terms. Without separators, whitespace
separates terms. Explicit hotwords, including an empty list, take precedence
over recognition-context text. At most 25 distinct terms and eight spoken
alternatives per term are accepted. Strength is 0–100; the default is 2.
Hotwords bias token selection, not blank or duration logits, and do not
guarantee correction of a named entity.

## Streaming

The rolling policy uses 50 ms energy blocks, first attempts decoding after
350 ms of voiced audio, updates after each additional 500 ms, and commits after
700 ms of silence, 30 seconds, or input completion. It re-encodes the growing
phrase; this is not a cache-aware encoder or an offline speedup.

For the server, use `stream=true` transcription uploads or
[`/v1/audio/transcriptions/live`](../../app/server/api.md#post-v1audiotranscriptionslive)
with mono 16 kHz PCM. JSON requests accept hotwords in `options`. Multipart
and live requests use the existing `prompt` field for vocabulary hints.

`partial_text_snapshot` and SSE `transcript.text.partial` contain the full
provisional hypothesis and may revise earlier words: replace the displayed
partial text. `partial_text` and `transcript.text.delta` contain committed
segments; concatenating them gives the final transcript. Committed timestamps
use absolute sample offsets. The original Phonon WebSocket protocol is not
implemented.

## Conversion and Storage

Use `tools/community_models/convert_phonon2.py` with the official Fermion
Research artifact at revision `7e153e4054c0a10db6d47d6fff5c062d4c77c154` and
Parakeet config/tokenizer sidecars at revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. Conversion needs NumPy,
safetensors and zstandard, but not PyTorch or NeMo.

```bash
python tools/community_models/convert_phonon2.py \
  --source-dir models/Phonon-2-source \
  --reference-dir models/Phonon-2-source/parakeet-reference \
  --output-dir models/Phonon-2-F32 \
  --converter build/debug/bin/audiocpp_gguf \
  --gguf-type mixed-f16 \
  --gguf-output models/Phonon-2-GGUF/phonon-2-mixed-f16.gguf
```

Mixed-F16 preserves the converted F32 values using exact F16 storage where
possible and F32 for protected values. Mixed-Q8 rounds selected matrices;
use `--gguf-type mixed-q8` with its own output filename. Keep
`phonon2_asr.matmul_weight_type` and `phonon2_asr.conv_weight_type` at `native`
to preserve the package's storage policy.

TDT timestamps advance on an 80 ms grid. Storage precision can change duration
choices even when transcripts match. See the
[storage and timestamp validation report](../reports/phonon2_validation.md)
for measured differences. Converted packages are larger than the original
compressed artifact; that download size is not a RAM or VRAM estimate.

Older development GGUFs need conversion with the dedicated `phonon2_asr` spec.
The registered family selects the behavior; changing a filename or a `variant`
label does not update embedded model metadata.
