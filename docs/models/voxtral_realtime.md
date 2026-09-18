# Voxtral Realtime

Voxtral Realtime is a Mistral realtime ASR model with offline and streaming
sessions. The model manager installs the Q8_0 standalone GGUF package by
default; native Hugging Face directories and other standalone GGUF variants can
also be used when provided directly. A Q4_K GGUF package is available for lower
memory use and faster CUDA runs; in a quick path check its transcripts matched
Q8_0 except for one capitalization-only difference.

## Model

| Field | Value |
|---|---|
| Family | `voxtral_realtime` |
| Model path | `models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf` when installed through the model manager |
| GGUF variants | `bf16`, `q8_0`, `q4_k` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Output | Transcription text |
| Streaming input | Audio chunks |
| Timestamps | Not exposed |

## CLI

Offline:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --audio assets/resources/sample.wav --text-out transcript.txt
```

Sampling and token-cap options can be passed through request options:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --audio assets/resources/sample.wav --text-out transcript.txt --request-option max_new_tokens=256 --do-sample false --temperature 1.0 --top-p 1.0 --top-k 50 --seed 1234
```

Streaming:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --mode streaming --audio assets/resources/sample.wav --text-out transcript.txt
```

## Live Streaming Input

`--audio -` reads raw, headerless interleaved PCM from stdin and feeds it to the
model chunk by chunk as it arrives, so the audio is never buffered up front and
does not have to exist as a file. Any capture tool that can write PCM to a pipe
works as the source:

```bash
# Microphone (macOS; use -f alsa on Linux or -f dshow on Windows)
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --mode streaming --audio -
```

```bash
# Any file or network stream, decoded to PCM on the fly
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f s16le - \
  | audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --mode streaming --audio -
```

Stdin input requires `--mode streaming`, and the PCM format must be described
up front because a live stream carries no header. The defaults (`s16le`, 16 kHz,
mono) match what the model expects. The chosen interpretation is echoed as an
`audio_input=stdin` line.

Each update carries only the text decoded since the last one, matching the
other streaming ASR models, so the updates concatenate into the transcript. On
a terminal they are appended unlabelled and the transcript scrolls like
ordinary output. When stdout is redirected, each update is written as its own
`partial_text=` line and flushed as it is produced, so pipes and logs stay
parseable. The complete transcript is also printed once at the end as
`text_output=`.

An update covers one decoded chunk, so `stream_batch_tokens=<n>` reports every
`n`th token's worth of text in a single update rather than making the updates
`n` times shorter. Whatever the batch size, concatenating the updates
reproduces `text_output=` exactly.

Emitting deltas rather than restating the transcript matters for long runs,
where the restated form is quadratic in the transcript length: a one-hour
session writes roughly 364 MB restated against about 54 KB as deltas.

To capture the transcript itself rather than the update stream, use
`--text-out`, which writes the complete transcript and nothing else:

```bash
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --mode streaming --audio - --text-out transcript.txt
```

`--text-out` and the `text_output=` line are both written when the stream ends,
so an interrupted session leaves neither. The `partial_text=` lines are flushed
as they are produced, so a log of them survives an interrupted run and
concatenates back into the transcript:

```bash
grep '^partial_text=' session.log | sed 's/^partial_text=//' | tr -d '\n' > transcript.txt
```

## Live PCM Over HTTP

The same live source is available to an HTTP client through
`POST /v1/audio/transcriptions/live`: raw PCM goes up in a chunked request body
while transcript deltas come back as SSE on the same connection. This is the
server equivalent of `--audio -`, and the only way to get capture-time partials
without the CLI. See [the server README](../../app/server/README.md) for
parameters and examples.

```bash
ffmpeg -f alsa -i default -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/transcriptions/live?model=voxtral-realtime'
```

Use `-T -`, not `--data-binary @-`: the latter reads stdin to EOF before it
connects, so a live capture would be uploaded as a finished file and no partial
could arrive early.

Whether text appears while the speaker is still talking depends on the model's
streaming policy rather than on the transport. `voxtral_realtime` decodes as
audio arrives and emits throughout the utterance; `nemotron_asr` consumes the
full utterance in its encoder first, so its deltas arrive only once the audio
ends. Both are supported here; the difference is what the transcript looks
like mid-sentence.

## Streaming Throughput

A streaming step always advances 80 ms of audio, so a step has to cost under
80 ms to keep up with a realtime source. Measured on an Apple M3 Air (Metal,
Q8_0):

| Config | Short clip, cool | Sustained 7 min |
|---|---:|---:|
| default | 78 ms/step (0.98x) | 88 ms/step (1.10x) |
| `stream_batch_tokens=4` | 60 ms/step (0.76x) | 74 ms/step (0.92x) |

The default splits roughly 48 ms for the text decoder and 30 ms for the audio
encoder; batching takes the encoder to about 13 ms. The second column is what a
long session actually gets on a fanless machine: a short clip run immediately
after the seven-minute one still measured 88 ms/step, so the gap is the machine
staying warm rather than anything that resets between sessions. Budget for the
sustained column, and prefer `stream_batch_tokens=4` if the source is realtime.

The decoder runs one step per 80 ms whether the audio holds speech or silence,
so a session that falls behind stays behind. The lag is monotonic and does not
recover during pauses. Measure your own hardware before relying on a live
source.

## Server

Streaming server config:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 8,
  "lazy_load": true,
  "models": [
    {
      "id": "voxtral-stream",
      "family": "voxtral_realtime",
      "path": "/path/to/voxtral-mini-4b-realtime-2602-q8_0.gguf",
      "task": "asr",
      "mode": "streaming"
    }
  ]
}
```

Streaming server request:

```bash
curl -N http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=voxtral-stream \
  -F stream=true \
  -F file=@assets/resources/sample.wav
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path or `-` | required | Speech input. `-` streams raw PCM from stdin and requires `--mode streaming`. |
| `--input-format` | `s16le`, `f32le` | `s16le` | Sample format of raw PCM read from stdin. Ignored for file input. |
| `--input-rate` | integer Hz | `16000` | Sample rate of raw PCM read from stdin. Ignored for file input. |
| `--input-channels` | integer | `1` | Channel count of raw PCM read from stdin. Ignored for file input. |
| `--mode` | `offline`, `streaming` | `offline` | Full-context or streaming session. |
| `--do-sample` | bool | `false` | Enable sampling instead of greedy decode. |
| `--temperature` | float | `1.0` | Sampling temperature. |
| `--top-p` | float | `1.0` | Nucleus sampling limit. |
| `--top-k` | integer | `50` | Top-k sampling limit; `0` disables top-k. |
| `--seed` | integer | `1234` | Sampling seed. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `max_new_tokens` | integer | model-derived limit | Maximum generated transcript tokens. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `voxtral_realtime.stream_batch_tokens` | integer >= 1 | `1` | Audio tokens per encoder forward. The decoder still runs one step per 80 ms; batching amortizes the encoder's fixed per-forward cost at the price of delaying each partial by up to `n * 80 ms`. |
| `voxtral_realtime.stream_decode_cache_steps` | integer >= 1 | `1024` | Decoder KV cache size in 80 ms steps, about 82 seconds of context. The cache ring wraps in place during long sessions. Lower values trade context for memory, not speed. |
| `voxtral_realtime.weight_type` | `native`, `f32`, `f16`, `bf16`, `q4_0`, `q4_k`, `q5_k`, `q6_k`, `q8_0` | `native` | Shared matmul weight storage type. |
| `voxtral_realtime.audio_encoder_weight_type` | same as above | shared setting | Audio encoder matmul weight storage type. Leave at `native` for streaming: the encoder is not bandwidth-bound there, so quantizing it makes it slower. |
| `voxtral_realtime.text_decoder_weight_type` | same as above | shared setting | Text decoder matmul weight storage type. `q4_k` roughly halves the streaming decoder step cost. |
| `voxtral_realtime.audio_encoder_graph_arena_mb` | integer MiB | `512` | Audio encoder graph arena size. |
| `voxtral_realtime.audio_encoder_weight_context_mb` | integer MiB | `128` | Audio encoder weight context arena size. |
| `voxtral_realtime.text_decoder_prefill_graph_arena_mb` | integer MiB | `512` | Text decoder prefill graph arena size. |
| `voxtral_realtime.text_decoder_decode_graph_arena_mb` | integer MiB | `512` | Text decoder cached-step graph arena size. |
| `voxtral_realtime.text_decoder_weight_context_mb` | integer MiB | `128` | Text decoder weight context arena size. |

Weight storage types are applied when the model loads, so requesting a type the
GGUF does not already hold requantizes on the CPU before the first token. This
takes around three minutes for `q4_k` from the shipped Q8_0 package. Prefer the
published `q4_k` GGUF variant, which needs no load-time conversion. See
[GGUF](../gguf.md).

For backend weight-type controls, use
`audiocpp_cli --inspect --model <model-dir> --family <family>`.
