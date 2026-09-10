# VibeVoice ASR Models

audio.cpp supports two VibeVoice ASR families:

| Model | Family | Mode(s) | Recommended package |
|---|---|---|---|
| VibeVoice ASR | `vibevoice_asr` | offline | `vibevoice_asr_q8_0` |
| VibeVoice ASR Streaming 7B | `vibevoice_asr_streaming` | offline, streaming | `vibevoice_asr_streaming_7b_q8_0` |

## VibeVoice ASR

VibeVoice ASR is an offline ASR model with greedy, sampling, and beam-search
decode paths. It can return transcription text and structured segment or
speaker-turn output when the model produces timestamps.

A fully quantized port of the same model — INT8 activations through the encoder,
ternary BitNet weights in the decoder — lives under community models as
`vibeasr`: see [VibeASR](../community_models/vibeasr.md). It is not a separate
model, only a CPU-only alternative numeric pipeline for the same weights.

| Field | Value |
|---|---|
| Family | `vibevoice_asr` |
| Model directory | `models/VibeVoice-ASR-GGUF` |
| Task | `asr` |
| Modes | `offline` |
| Required tokenizer files | Embedded in the standalone GGUF |
| Output | Transcription text; optional segments through `--segments-out`; optional speaker turns through `--turns-out` |
| Streaming | Not supported |
| Timestamps | Segment and speaker-turn timestamps when produced |

Install:

```bash
python3 tools/model_manager_v2.py install vibevoice_asr_q8_0
```

Offline CLI:

```bash
audiocpp_cli --task asr \
  --family vibevoice_asr \
  --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf \
  --backend cuda \
  --audio assets/resources/sample_16k.wav \
  --text-out transcript.txt \
  --metrics \
  --log
```

Structured output:

```bash
audiocpp_cli --task asr \
  --family vibevoice_asr \
  --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf \
  --backend cuda \
  --audio meeting.wav \
  --text "The recording is a meeting conversation." \
  --text-out transcript.txt \
  --segments-out segments.json \
  --turns-out turns.json \
  --metrics \
  --log
```

With VAD chunking, provide the bundled Silero VAD model:

```bash
audiocpp_cli --task asr \
  --family vibevoice_asr \
  --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf \
  --backend cuda \
  --audio assets/resources/sample_16k.wav \
  --audio-chunk-mode vad \
  --session-option vibevoice_asr.vad_model_path=assets/framework/models/silero_vad \
  --text-out transcript.txt \
  --metrics \
  --log
```

Convert from safetensors:

```powershell
audiocpp_gguf.exe --input models\VibeVoice-ASR\model.safetensors.index.json --output models\VibeVoice-ASR-Q8_0\model.gguf --type q8_0
```

Configuration and tokenizer assets are embedded by default, so the output
directory may contain only `model.gguf`.

Options:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--text` | text | empty string | Context prompt for the ASR request. |
| `--language` | language code | `auto` | ASR language label. |
| `--max-tokens` | integer | model default | Maximum generated transcript tokens. |
| `--temperature` | float | model default | Sampling temperature; `0` uses deterministic decoding. |
| `--top-p` | float | model default | Nucleus sampling probability. |
| `--top-k` | integer | model default | Top-k sampling limit; `0` disables top-k filtering. |
| `--num-beams` | integer | `1` | Beam count for deterministic beam search. |
| `--repetition-penalty` | float | model default | Generation repetition penalty. |
| `--seed` | integer | random if omitted | Sampling seed. |
| `--audio-chunk-mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Long-audio chunking mode. `auto` uses fixed chunks. |
| `--audio-chunk-seconds` | float seconds | `1200` | Fixed audio chunk duration. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--segments-out` | JSON path | not set | Write structured ASR segments when produced. |
| `--turns-out` | JSON path | not set | Write speaker turns when produced. |
| `--session-option vibevoice_asr.vad_model_path=<path>` | model directory | `assets/framework/models/silero_vad` | Internal VAD model used by `--audio-chunk-mode vad`. |

## VibeVoice ASR Streaming 7B

VibeVoice ASR Streaming 7B is the streaming VibeVoice ASR model. It keeps a
persistent decoder state and can emit speaker-attributed transcript deltas as
audio arrives. It can also run in offline mode through the same family.

The upstream streaming model emits speaker-attributed text such as
`Speaker 0: ...`, but it does not emit timestamped segments. In audio.cpp,
`--turns-out` can expose those speaker/text turns when the model emits speaker
labels; `--segments-out` is not populated by this streaming model. Use the
non-streaming `vibevoice_asr` family when timestamped segment output is needed.

> [!WARNING]
> Speaker-turn segmentation can differ from the offline VibeVoice ASR family.
> In local validation, the streaming 7B model merged adjacent speech into fewer
> speaker turns than the offline model; the same behavior was reproduced with
> the official Python streaming reference.

The recommended audio.cpp package is the Q8_0 GGUF in the dedicated model repo:
<https://huggingface.co/audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF>. BF16 and
Q4_K GGUF variants are also available in the same repo.

Upstream lists ten supported language codes for this streaming model: `en`,
`zh`, `es`, `pt`, `de`, `ja`, `ko`, `fr`, `ru`, and `it`.

| Field | Value |
|---|---|
| Family | `vibevoice_asr_streaming` |
| Model package | `vibevoice_asr_streaming_7b_q8_0` |
| Model directory | `models/VibeVoice-ASR-Streaming-7B-GGUF` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Required tokenizer files | Embedded in the standalone GGUF |
| Output | Transcript text; speaker/text turns when produced by the model |
| Timestamps | Not produced by the streaming model |
| Streaming | Live audio chunks over the `/v1/audio/transcriptions/live` endpoint |

Install:

```bash
python3 tools/model_manager_v2.py install vibevoice_asr_streaming_7b_q8_0
```

Offline CLI:

```bash
audiocpp_cli --task asr \
  --family vibevoice_asr_streaming \
  --model models/VibeVoice-ASR-Streaming-7B-GGUF/vibevoice-asr-streaming-7b-q8_0.gguf \
  --backend cuda \
  --threads 8 \
  --audio assets/resources/sample_16k.wav \
  --text-out transcript.txt \
  --turns-out speaker_turns.json \
  --metrics \
  --log
```

Server config for live streaming:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "threads": 8,
  "models": [
    {
      "id": "vibevoice-streaming-7b",
      "family": "vibevoice_asr_streaming",
      "path": "models/VibeVoice-ASR-Streaming-7B-GGUF/vibevoice-asr-streaming-7b-q8_0.gguf",
      "task": "asr",
      "mode": "streaming"
    }
  ]
}
```

Start the server:

```bash
audiocpp_server --config server.json --log
```

Live streaming request with 16 kHz mono signed 16-bit PCM:

```bash
ffmpeg -hide_banner -loglevel error -i input.wav -f s16le -ac 1 -ar 16000 - \
  | curl -N -X POST \
      -H 'Content-Type: application/octet-stream' \
      -H 'Transfer-Encoding: chunked' \
      -H 'Expect:' \
      -T - \
      'http://127.0.0.1:8080/v1/audio/transcriptions/live?model=vibevoice-streaming-7b&sample_rate=16000&channels=1&sample_format=s16le'
```

Common request options:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language` | language label | `auto` | ASR language label. |
| `--request-option context=<text>` | text | empty string | Extra context or hotwords injected into the streaming prompt. |
| `--max-tokens` | integer | `256` | Maximum generated transcript tokens per chunk. |
| `--temperature` | float | `0` | Sampling temperature; `0` uses deterministic decoding. |
| `--top-p` | float | `1` | Nucleus sampling probability. |
| `--top-k` | integer | `0` | Top-k sampling limit; `0` disables top-k filtering. |
| `--num-beams` | integer | `1` | Beam count for deterministic beam search. |
| `--repetition-penalty` | float | `1` | Generation repetition penalty. |
| `--audio-chunk-mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Offline audio chunking mode. |
| `--audio-chunk-seconds` | float seconds | `1200` | Offline chunk duration for fixed and VAD chunking. |
