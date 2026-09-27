# GigaAM ASR

[GigaAM v3](https://huggingface.co/ai-sage/GigaAM-v3) and
[GigaAM Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual)
use the `gigaam_asr` family. Both are released under the MIT license.

| Field | Value |
|---|---|
| Task | `asr` |
| Mode | `offline` |
| Output | Transcript text, longform segments, optional word timestamps |
| Long-form audio | Automatic quiet-energy chunking |
| Languages | v3: Russian; Multilingual: Russian, Kazakh, Kyrgyz, Uzbek, English |

## Variants

| Upstream repository | Revision | GGUF filename |
|---|---|---|
| `ai-sage/GigaAM-v3` | `ctc` | `gigaam-v3-ctc-f16.gguf` |
| `ai-sage/GigaAM-v3` | `rnnt` | `gigaam-v3-rnnt-f16.gguf` |
| `ai-sage/GigaAM-v3` | `e2e_ctc` | `gigaam-v3-e2e-ctc-f16.gguf` |
| `ai-sage/GigaAM-v3` | `e2e_rnnt` | `gigaam-v3-e2e-rnnt-f16.gguf` |
| `ai-sage/GigaAM-Multilingual` | `ctc` | `gigaam-multilingual-ctc-f32.gguf` |
| `ai-sage/GigaAM-Multilingual` | `ctc` | `gigaam-multilingual-ctc-f16.gguf` |
| `ai-sage/GigaAM-Multilingual` | `large_ctc` | `gigaam-multilingual-large-ctc-f32.gguf` |
| `ai-sage/GigaAM-Multilingual` | `large_ctc` | `gigaam-multilingual-large-ctc-f16.gguf` |

E2E variants restore punctuation and capitalization. Select the checkpoint by
passing its GGUF file directly. Language is inferred from the audio, not selected
by a request option.

## Convert

Download a fine-tuned ASR checkpoint, not the SSL-only encoder:

```bash
hf download ai-sage/GigaAM-v3 --revision ctc --local-dir models/GigaAM-v3-ctc
python tests/gigaam_asr/convert_gguf.py models/GigaAM-v3-ctc \
  models/GigaAM-ASR-GGUF/gigaam-v3-ctc-f16.gguf \
  --staging-dir models/GigaAM-v3-ctc/converted
```

The converter requires PyTorch, safetensors, and `build/debug/bin/audiocpp_gguf`.
It preserves checkpoint tensor dtypes and embeds configuration, model spec, and
the SentencePiece tokenizer where needed. In v3 packages, the encoder is F16
while frontend and decoder tensors retain their original precision.
Use the repository, revision, and filename from the table for other variants.
For an F16 multilingual package, add `--type f16` and use `-f16.gguf` as the
filename suffix. The frontend, normalization weights, and biases remain F32.

## CLI

```bash
audiocpp_cli --task asr --family gigaam_asr \
  --model models/GigaAM-ASR-GGUF/gigaam-v3-ctc-f16.gguf \
  --backend cuda --audio speech.wav --text-out transcript.txt --log
```

For CPU inference, use `--backend cpu --threads 8`. Long recordings are split
automatically and the chunk transcripts are joined in order.
Add `--words-out words.json` to enable and save word timestamps. Add
`--segments-out segments.json` to save the longform chunk boundaries and text.
Word timestamps come from decoder token emission frames, not forced alignment.

For VAD-based segmentation, set `--request-option audio_chunk_mode=vad` and
`--session-option vad_model_path=/path/to/vad-model`. The selected
VAD must support the input sample rate. For example, the bundled Silero weights
are in `assets/framework/models/silero_vad` and require 16 kHz input. The VAD is
loaded only when requested. This uses the framework VAD chunker; it does not
reproduce the official Python pipeline's external Pyannote segment boundaries.

Use `--session-option weight_type=<type>` only to override the packaged tensor
storage type at load time.

## Server Timestamps

Send JSON to `/v1/audio/transcriptions/details` with the configured model ID:

```json
{
  "model": "gigaam",
  "audio_path": "/path/to/speech.wav",
  "options": {"return_timestamps": true}
}
```

The response contains `words`, and `segments` for chunked longform audio.
Their `start_sample` and `end_sample` values use the response's `sample_rate`.
The multipart transcription route does not forward custom `options`; use the
JSON request above to select timestamp or chunking options.

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `return_timestamps` | `false` | Include word start/end times; enabled automatically by CLI `--words-out`. |
| `audio_chunk_mode` | `auto` | `auto` and `quiet_energy` split near quiet points; `fixed` uses equal-duration chunks; `vad` uses the selected VAD; `none` processes the whole recording at once. |
| `audio_chunk_duration_sec` | `25` | Maximum chunk duration, from 0.02 to 25 seconds. Ignored by `none`; disabling chunking increases memory use on long recordings. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `weight_type` | `native` | Weight storage override. |
| `vad_model_path` | Empty | Offline VAD model path, required only for `audio_chunk_mode=vad`. |

Translation, speaker labels, and live audio streaming are not
exposed by this port.
