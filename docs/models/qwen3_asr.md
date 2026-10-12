# Qwen3 ASR

For shared conversion guidance, see
[Qwen3 GGUF checkpoints](qwen3.md#gguf-checkpoints).

Qwen3 ASR transcribes speech audio. Word timestamps are produced by running the
recognized transcript through Qwen3 Forced Aligner, so `--words-out` requires
the forced aligner model path. Long audio is split inside the model session
before ASR inference; word timestamps are shifted back onto the original audio
timeline.

Streaming mode accepts live audio chunks and emits buffered transcript deltas.
Timestamp output remains offline-only because word alignment runs after the full
transcript is known.

Offline `audio_chunk_mode=auto` is the default. For transcript-only ASR, Qwen3 ASR uses
fixed chunks. When word timestamps are requested, it uses bundled Silero VAD
internally to choose speech-aware chunks before running ASR and alignment.
Streaming uses duration-based windows regardless of `audio_chunk_mode`.

| Field | Value |
|---|---|
| Family | `qwen3_asr` |
| Model directory | `models/Qwen3-ASR-0.6B` or `models/Qwen3-ASR-1.7B-hf` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Input | Speech WAV through `--audio` |
| Output | Text to stdout or `--text-out`; optional word JSON through `--words-out` with a forced aligner |

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-0.6B --backend cuda --audio speech_16k.wav --text "" --text-out transcript.txt
```

```bash
audiocpp_cli --task asr --mode streaming --family qwen3_asr --model models/Qwen3-ASR-0.6B --backend cuda --audio speech_16k.wav --text "" --request-option audio_chunk_seconds=5 --text-out transcript.txt
```

The native Hugging Face Transformers layout of `Qwen/Qwen3-ASR-1.7B-hf` is
also accepted directly. It uses `processor_config.json`, `tokenizer.json`, and
the `model.audio_tower` / `model.language_model` tensor namespaces; no model
conversion is required:

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-1.7B-hf --backend cuda --audio speech_16k.wav --text "" --text-out transcript.txt
```

## QwenCleo-ASR (Egyptian Arabic)

[QwenCleo-ASR](https://huggingface.co/mohammedaly22/QwenCleo-ASR) is a
Qwen3-ASR-1.7B fine-tune for Egyptian Arabic and Arabic-English code-switching.
It uses the same `qwen3_asr` family, loader, and options; only the weights
differ. The checkpoint is saved with tied word embeddings and therefore has no
`thinker.lm_head.weight`; the loader reuses `thinker.model.embed_tokens.weight`
for the output projection in that case. Model and integration by
[@Mohammedaly22](https://github.com/Mohammedaly22).

| Package | Format | Notes |
|---|---|---|
| `qwencleo_asr_q8_0` | GGUF Q8_0 | Standalone GGUF, recommended for local inference. |
| `qwencleo_asr_f16` | GGUF F16 | Closest to the original BF16 weights. |
| `qwencleo_asr_safetensors` | Safetensors | Original upstream checkpoint, loaded directly. |

Pass `--language Arabic`, as recommended by the model card; quantized packages
are less reliable at automatic language detection.

```bash
audiocpp_model_manager install qwencleo_asr_q8_0 --models-dir models
audiocpp_cli --task asr --family qwen3_asr --model models/QwenCleo-ASR-GGUF --backend cuda --audio egyptian_16k.wav --language Arabic --text "" --text-out transcript.txt
```

To build the GGUF packages from the upstream checkpoint:

```bash
audiocpp_gguf --input models/QwenCleo-ASR/model.safetensors --root models/QwenCleo-ASR   --output models/QwenCleo-ASR-GGUF/qwencleo-asr-q8_0.gguf --family qwen3_asr --type q8_0 --overwrite
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech audio; use 16 kHz WAV for the examples. |
| `--text` | text | empty string | Context prompt. |
| `--language` | language code | empty string | Recognition language hint. |
| `--max-tokens` | integer | `512` | Maximum decode tokens. |
| `--audio-chunk-seconds` | seconds | `30` for streaming/transcription; `15` for offline timestamps/VAD | Target/max chunk length; must be > 0 when chunking or streaming. Ignored with `audio_chunk_mode=none` offline. |
| `--audio-chunk-mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Let the model choose, force fixed chunks, force internal VAD chunks, or disable model-side chunking. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--words-out` | JSON path | not set | Word timestamp output. Requires `qwen3_asr.forced_aligner_path`. |

The language, token limit, chunk mode, and chunk duration can also be supplied
through request options `language`, `max_tokens`, `audio_chunk_mode`, and
`audio_chunk_duration_sec`, respectively.

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `return_timestamps` | bool | `false` | Offline only: run the configured forced aligner after ASR. Streaming rejects true, even with an aligner configured. CLI `--words-out` enables this automatically; server requests must enable it explicitly or through `default_request_options`. |
| `clamp_timestamps_to_audio` | bool | `false` | Opt-in guard for `--words-out`: keep repaired forced-aligner word spans inside each local audio chunk. Default preserves existing timestamp repair behavior. |
| `preserve_punctuation` | bool | `false` | Preserve ASR punctuation in timestamped, chunked transcript text instead of rebuilding text from aligned words. Legacy `qwen3_asr.preserve_punctuation` remains accepted. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `qwen3_asr.forced_aligner_path` | model path | not set | Qwen3 Forced Aligner model used to generate word timestamps after ASR. |
| `qwen3_asr.vad_model_path` | model directory | `assets/framework/models/silero_vad` | Optional internal VAD model override for timestamp-safe chunking. Legacy `qwen3_asr.vad_path` remains accepted. |
| `qwen3_asr.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Text decoder weight storage fallback. |
| `qwen3_asr.audio_encoder_weight_type` | `native`, `f32`, `f16` | `native` | Audio encoder weight storage. |
| `qwen3_asr.text_decoder_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | inherits `weight_type` | Explicit text decoder weight storage override. |
| `qwen3_asr.audio_encoder_graph_arena_mb` | positive integer | `128` | Audio encoder graph arena in MiB. |
| `qwen3_asr.text_decoder_prefill_graph_arena_mb` | positive integer | `256` | Text decoder prefill graph arena in MiB. |
| `qwen3_asr.text_decoder_decode_graph_arena_mb` | positive integer | `256` | Text decoder cached-step graph arena in MiB. |
| `qwen3_asr.text_decoder_weight_context_mb` | positive integer | `64` | Text decoder weight tensor context in MiB. |

Legacy session names `qwen3_asr.forced_aligner_model_path` (also
`qwen3_asr.aligner_model_path`) and `qwen3_asr.vad_model_path` remain accepted.
The normalized `qwen3_asr.text_decoder_weight_type`,
`qwen3_asr.text_decoder_prefill_graph_arena_mb`,
`qwen3_asr.text_decoder_decode_graph_arena_mb`, and
`qwen3_asr.text_decoder_weight_context_mb` also accept their existing `thinker_*`
spellings. Normalized names take precedence if both are supplied. These aliases
work with existing GGUFs without reconversion.

## Server: Transcript and Word Timestamps

Configure the aligner and VAD with paths visible to the server. Put request
defaults, including `language`, in `default_request_options`, not `session_options`.

```json
{
  "id": "qwen3-asr",
  "family": "qwen3_asr",
  "path": "/path/to/qwen3-asr-0.6b-q8_0.gguf",
  "task": "asr",
  "mode": "offline",
  "session_options": {
    "qwen3_asr.forced_aligner_path": "/path/to/qwen3-forced-aligner-0.6b-q8_0.gguf",
    "qwen3_asr.vad_model_path": "/path/to/assets/framework/models/silero_vad"
  },
  "default_request_options": {
    "language": "English",
    "return_timestamps": true
  }
}
```

Add that entry to the server config's `models` array, then upload the recording:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions/details \
  -F model=qwen3-asr \
  -F language=English \
  -F file=@input.wav
```

The response includes `text`, `words` with `start_sample`/`end_sample`, and
`sample_rate`. Divide sample offsets by `sample_rate` for seconds. The ordinary
`/v1/audio/transcriptions` endpoint returns text and timing only, even when the
model generates word timestamps. Timestamp generation uses Silero VAD in the
default `audio_chunk_mode=auto` path.

