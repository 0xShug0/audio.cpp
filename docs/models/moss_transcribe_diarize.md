# MOSS-Transcribe-Diarize

MOSS-Transcribe-Diarize transcribes audio with speaker labels and segment timestamps.
It supports 50+ languages without requiring a language hint.

## Install

```bash
python3 tools/model_manager_v2.py install moss_transcribe_diarize_bf16
```

For a smaller download, install `moss_transcribe_diarize_q8_0` or
`moss_transcribe_diarize_q4_k` instead and use the corresponding `-q8_0.gguf` or
`-q4_k.gguf` filename below.

## Quick Start

```bash
audiocpp_cli \
  --task asr \
  --family moss_transcribe_diarize \
  --model models/MOSS-Transcribe-Diarize-GGUF/moss-transcribe-diarize-bf16.gguf \
  --backend cuda \
  --audio recording.wav \
  --text-out transcript.txt \
  --segments-out segments.json \
  --turns-out turns.json \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `moss_transcribe_diarize` |
| Task | `asr` |
| Modes | `offline`, `streaming` (text output only) |
| Model directory | `models/MOSS-Transcribe-Diarize-GGUF` |
| Default weights | `moss-transcribe-diarize-bf16.gguf` |
| Other weights | `moss-transcribe-diarize-q8_0.gguf`, `moss-transcribe-diarize-q4_k.gguf` |
| Input | Recording through `--audio` |
| Output | Transcript, speech segments, and speaker turns |

Each GGUF includes the tokenizer and configuration; no separate encoder or
diarization model is required.

Use `--backend cpu --threads 8` for CPU inference or `--backend vulkan` for Vulkan.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Recording to transcribe. |
| `--mode` | `offline`, `streaming` | `offline` | Emit the final result or text deltas during generation. |
| `--instruct` | text | upstream instruction | Replace the transcription instruction; also accepted as `--request-option instruct=<text>`. |
| `--text-out` | TXT path | not set | Save the timestamped, speaker-labelled transcript. |
| `--segments-out` | JSON path | not set | Save speech segments. |
| `--turns-out` | JSON path | not set | Save speaker turns. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `instruct` | text | upstream instruction | Replace the transcription instruction. |
| `max_tokens` | integer > 0 | `5120` | Maximum generated transcript tokens. |

Increase `max_tokens` if a long recording reaches the output-token limit.

There are no model-specific session options. Select weight precision through
the GGUF filename passed to `--model`.

## Text Streaming

Add `--mode streaming` to the quick-start command to print text as it is
generated. The complete recording is processed before text generation starts;
live microphone input is not supported. Output files are written when the
transcription finishes.
