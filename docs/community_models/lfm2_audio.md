# LFM2.5-Audio

LFM2.5-Audio is Liquid AI's end-to-end speech and text model. audio.cpp runs its
speech recognition, text-to-speech and spoken chat (speech-to-speech) as the
community family `lfm2_audio`, directly from the GGUFs Liquid AI publishes, for
the English and the Japanese checkpoint.

Upstream: [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) ·
[LiquidAI/LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) ·
reference implementation: [liquid-audio](https://github.com/Liquid4All/liquid-audio)

| Field | Value |
|---|---|
| Family | `lfm2_audio` |
| Tasks | `asr`, `tts`, `s2s` |
| Modes | `offline`; `streaming` for TTS and S2S |
| Languages | `en` (`LFM2.5-Audio-1.5B`), `ja` (`LFM2.5-Audio-1.5B-JP`); each checkpoint transcribes, speaks and chats in its own language |
| Audio input (ASR, S2S) | WAV at any sample rate; channels are averaged and the audio is resampled to 16 kHz |
| Speech output (TTS, S2S) | 24 kHz mono speech; S2S also returns the reply's text |
| Voices | TTS: `us_male` (default), `us_female`, `uk_male`, `uk_female` for English; the Japanese checkpoint has one voice. S2S replies in the checkpoint's own voice |
| Backends tested | CPU, CUDA, Metal |
| License | LFM Open License v1.0 |

## Architecture

- NeMo 128-bin log-mel frontend and a FastConformer encoder (17 layers, 8x
  depthwise-striding subsampling, relative-position attention) based on
  canary-180m-flash, followed by an MLP adapter. The frontend configuration is
  the same as `canary_asr`.
- LFM2 hybrid backbone: 10 gated short-convolution blocks and 6 grouped-query
  attention blocks, with RMSNorm, QK-norm and SwiGLU. The text head is tied to
  the token embedding.
- ASR: greedy decoding of the chat prompt liquid-audio builds: the system
  prompt `Perform ASR.` (`Perform ASR in japanese.` for the JP checkpoint),
  then the audio embeddings as the user turn, until `<|im_end|>`.
- TTS: the system prompt `Perform TTS. Use the US male voice.` (one per voice;
  `Perform TTS in japanese.` for the JP checkpoint) and the text as the user
  turn. After `<|audio_start|>` each backbone step yields one 80 ms audio
  frame: a 6-layer depthformer predicts its 8 codes one codebook at a time,
  and the frame goes back into the backbone as the sum of its code embeddings,
  until a frame starts with the end-of-audio code. A frame that picks
  end-of-audio for another codebook goes back in too but has no sound, so, as
  in liquid-audio's demo, it is not decoded.
- S2S: liquid-audio's interleaved generation. The system prompt `Respond with
  interleaved text and audio.` and the user's audio as the user turn; the reply
  alternates 6 text tokens and 12 audio frames (9 for the JP checkpoint, whose
  vocoder GGUF records its blocks) until `<|text_end|>`, continues with audio
  until end-of-audio, and ends at `<|im_end|>`. Text is greedy; audio is sampled
  at temperature 1.0 with top-k 4, as in liquid-audio's README and chat demo.
- An LFM2 detokenizer (8 layers, causal sliding-window attention over 30 steps)
  turns the mean of each frame's code embeddings, repeated 6 times, into
  log-magnitude and phase, and an ISTFT (n_fft 1280, hop 320) gives 24 kHz
  audio.

## Packages

Liquid AI publishes each quantization as four llama.cpp-format GGUFs in
[LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/tree/7d525f883a077e20afb782f2ff618edcae0e39e4)
and [LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/tree/64b96718b341dbd5650f9e85627cecdcbd4ac61b).
The packages point at those repositories, pinned to the linked revisions:

| File | Contents | Used by |
|---|---|---|
| `<model>-<quant>.gguf` | LFM2 backbone and text tokenizer | ASR, TTS, S2S |
| `mmproj-<model>-<quant>.gguf` | Audio encoder and adapter; the embedding of generated audio codes | ASR, TTS, S2S |
| `vocoder-<model>-<quant>.gguf` | Depthformer; the detokenizer's code embedding and ISTFT window | TTS, S2S |
| `tokenizer-<model>-<quant>.gguf` | Audio detokenizer | TTS, S2S |

| Package | Checkpoint | Weights | Download |
|---|---|---|---|
| `lfm2_audio_1_5b_q8_0` (default) | EN | Q8_0 | 1.8 GB |
| `lfm2_audio_1_5b_f16` | EN | F16 | 3.3 GB |
| `lfm2_audio_1_5b_q4_0` | EN | Q4_0 | 1.1 GB |
| `lfm2_audio_1_5b_jp_q8_0` | JP | Q8_0 | 1.8 GB |
| `lfm2_audio_1_5b_jp_f16` | JP | F16 | 3.3 GB |
| `lfm2_audio_1_5b_jp_f32` | JP | F32 | 6.5 GB |
| `lfm2_audio_1_5b_jp_q4_0` | JP | Q4_0 | 1.1 GB |

```bash
python3 tools/model_manager_v2.py install lfm2_audio_1_5b_q8_0 --models-root models
python3 tools/model_manager_v2.py install lfm2_audio_1_5b_jp_q8_0 --models-root models
```

All quantizations of a checkpoint install into one directory,
`models/LFM2.5-Audio-1.5B-GGUF` or `models/LFM2.5-Audio-1.5B-JP-GGUF`, so
components of different quantizations can be combined with the session options
below. The WebUI lists both checkpoints under ASR, TTS and speech-to-speech with
their Q8_0 and F16 packages.

## Run

```bash
audiocpp_cli --task asr --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda --audio speech.wav

audiocpp_cli --task tts --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda \
  --text "The next train leaves in ten minutes." --voice-id uk_female --out speech.wav

audiocpp_cli --task s2s --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda \
  --audio question.wav --out reply.wav --text-out reply.txt
```

`--model` is the package directory: the published GGUFs do not embed an
audio.cpp model spec, so a GGUF file path is rejected. With one quantization
installed, its backbone is used with the `mmproj-`, `vocoder-` and
`tokenizer-` files of the same name. With several, choose the backbone:

```bash
audiocpp_cli --task asr --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda --audio speech.wav \
  --session-option lfm2_audio.model_gguf=LFM2.5-Audio-1.5B-F16.gguf
```

For Japanese, pass `models/LFM2.5-Audio-1.5B-JP-GGUF` and no voice. `language`
can be left out; a value other than the checkpoint's language is rejected.
Speech is sampled like the README's example (temperature 0.8, top-k 64); pass
`--seed` for repeatable audio, or `--temperature 0` for greedy decoding.

S2S answers a spoken turn with a spoken reply and its text: the CLI prints the
text (`text_output=`) and `--text-out` writes it. Each request is a new
conversation under liquid-audio's chat system prompt, `Respond with interleaved
text and audio.`; `--text` replaces that prompt, which the checkpoints were
trained with, so leave it out unless experimenting. The reply is sampled like
liquid-audio's README and demo (temperature 1.0, top-k 4) and may run to 1024
steps, text tokens and audio frames together, about a minute of speech.

The server takes the same directory and session options:

```json
{"models": [{"id": "lfm2-audio-asr", "family": "lfm2_audio", "task": "asr", "mode": "offline",
  "path": "models/LFM2.5-Audio-1.5B-GGUF",
  "session_options": {"lfm2_audio.model_gguf": "LFM2.5-Audio-1.5B-Q8_0.gguf"}}]}
```

```bash
audiocpp_server --backend cuda --config server.json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F model=lfm2-audio-asr -F file=@speech.wav
```

A TTS entry is the same with `"task": "tts"`, served at `/v1/audio/speech`:

```bash
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lfm2-audio-tts", "input": "The next train leaves in ten minutes.", "voice": "us_female"}' -o speech.wav
```

An S2S entry (`"task": "s2s"`) answers at `/v1/tasks/run` with the reply's
`text` and its audio as base64 WAV (`audio`):

```bash
curl http://127.0.0.1:8080/v1/tasks/run -H 'Content-Type: application/json' \
  -d '{"model": "lfm2-audio-s2s", "request": {"audio": "/path/to/question.wav"}}'
```

### Streaming TTS

With `--mode streaming` (CLI) or `"mode": "streaming"` (server entry), speech
comes out as it is generated. Like liquid-audio's demo, each audio frame is
decoded as soon as the depthformer picks it. The detokenizer is causal and
carries each layer's state from frame to frame (the attention layers' last 29
keys and values and the short-conv layers' last inputs), so a frame costs
only its own six steps, and the ISTFT releases a sample once no later window
reaches it. Each event carries one frame (80 ms, `stream_frames_per_event` to
change it), plus a last 20 ms event per text chunk; the events add up to the
offline speech for the same seed. A text chunk that reaches `max_tokens` ends
its speech there, streamed as offline, with the same warning.

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lfm2-audio-tts-stream", "input": "The next train leaves in ten minutes.", "voice": "us_female", "stream": true}'
```

The response is server-sent events with base64 16-bit PCM deltas at 24 kHz;
`"stream_format": "audio"` returns the raw PCM instead.

Time to the first audio and real-time factor through the server, for a
two-sentence English text (6.5 s of speech) and a Japanese one (8.5 s):

| Backend | First audio, F16 / Q8_0 / Q4_0 / JP F32 | RTF, 1 frame per event | RTF, 4 frames per event |
|---|---|---|---|
| CUDA, NVIDIA A10 | 39 / 27 / 24 / 45 ms | 0.09-0.21 | 0.09-0.21 |
| Metal, Apple M3 Ultra | 40 / 37 / 35 / 52 ms | 0.19-0.29 | 0.19-0.27 |
| CPU, Apple M3 Ultra, 16 threads | 150 / 106 / 113 / 308 ms | 0.18-0.57 | 0.18-0.57 |

A frame per event costs no more than four, and streaming about what offline
speech does. Across the backends, the streamed audio differs from offline by
5e-4 to 6e-3 (relative RMS) on the test's short text, the detokenizer's
arithmetic in graphs of other sizes. CUDA's kernels round differently for each
size, so over a minute of speech it differs by 2e-2 to 3e-2, and by 0.2 with
Q4_0, whose coarse detokenizer weights amplify it.

### Streaming S2S

With `--mode streaming` (CLI) or `"mode": "streaming"` (server entry), the
question comes in as audio chunks, live PCM included (`--audio -` on the CLI).
The reply starts when the input ends and streams like TTS: each event carries
the audio of one frame (`stream_frames_per_event`) and the text written since
the last event, in whole characters; together they are the offline reply for
the same seed. The server's live route, `/v1/audio/speech/live`, takes the
question as chunked raw PCM and requires an `input` query parameter, which
becomes the system prompt, so pass liquid-audio's:

```bash
ffmpeg -i question.wav -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/speech/live?model=lfm2-audio-s2s-stream&sample_rate=16000&channels=1&sample_format=s16le&input=Respond%20with%20interleaved%20text%20and%20audio.'
```

It returns the reply's audio as server-sent events; the text is not returned
on this route. Time from the end of a 7.5 s English question, streamed in real
time, to the first audio of the reply:

| Backend | F16 | Q8_0 | Q4_0 |
|---|---|---|---|
| CUDA, NVIDIA A10 | 107 ms | 106 ms | 73 ms |
| Metal, Apple M3 Ultra | 183 ms | 168 ms | 153 ms |
| CPU, Apple M3 Ultra, 16 threads | 478 ms | 360 ms | 371 ms |

That covers encoding the whole question, the prompt, the first text block and
the first frame. With the JP F32 package on CUDA, the first audio came 113 ms
after a 2.6 s Japanese question. Offline, `/v1/tasks/run` returned a 13.1 s
English reply in 2.1 s on CUDA (F16).

## Request Options (use with `--request-option`)

| Option | Task | Default | Meaning |
|---|---|---|---|
| `language` | all | The checkpoint's | `en` or `ja`; must match the checkpoint. |
| `max_tokens` | all | `512`; S2S `1024` | ASR: transcript tokens per audio chunk. TTS: 80 ms audio frames per text chunk. S2S: text tokens and audio frames of the reply together. A transcript or a text chunk's speech that reaches it is cut off there and kept, as liquid-audio keeps it, and a warning goes to stderr; the other chunks go on. A reply that needs more fails the request rather than returning a cut-off result. Transcript tokens need not end on a character boundary, so a cut can fall inside a character: liquid-audio then shows U+FFFD for the partial character, while audio.cpp drops it and ends the transcript at the last whole character. |
| `audio_chunk_mode` | ASR | `auto` | `auto`, `vad`, `fixed` or `none`; see [Long audio](#long-audio). |
| `audio_chunk_seconds` | ASR | `30` | Longest chunk in seconds, at least 1. |
| `temperature` | TTS, S2S | `0.8`; S2S `1.0` | Audio code sampling temperature; 0 is greedy. |
| `top_k` | TTS, S2S | `64`; S2S `4` | Sample from the k most likely codes; 0 keeps all, 1 is greedy. |
| `seed` | TTS, S2S | Random | Sampling seed; TTS text chunk i uses seed + i. |
| `text_chunk_mode` | TTS | `japanese` for JP, else `default` | How long text is split; see [Long text](#long-text). |
| `text_chunk_size` | TTS | `200` | Unicode codepoints per text chunk. |
| `stream_frames_per_event` | TTS, S2S streaming | `1` | Audio frames (80 ms) per streaming event. |

Each task rejects the options it does not take.

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `lfm2_audio.model_gguf` | The directory's only backbone | Backbone GGUF, relative to the model directory. |
| `lfm2_audio.mmproj_gguf` | `mmproj-<backbone file>`, else the only mmproj | Audio encoder GGUF, relative to the model directory. |
| `lfm2_audio.vocoder_gguf` | `vocoder-<backbone file>`, else the only vocoder | TTS depthformer GGUF, relative to the model directory. |
| `lfm2_audio.detokenizer_gguf` | `tokenizer-<backbone file>`, else the only tokenizer | TTS audio detokenizer GGUF, relative to the model directory. |
| `lfm2_audio.vad_model_path` | `assets/framework/models/silero_vad` | Silero VAD model used to split long audio. |

## Long Audio

liquid-audio transcribes a whole file in one pass, which the model handles up
to about a minute; longer audio loses words and then repeats itself. audio.cpp
keeps short input whole and splits longer input:

| Mode | Behavior |
|---|---|
| `auto` | Input up to `audio_chunk_seconds` is transcribed whole, as liquid-audio does. Longer input is split at pauses found by the bundled Silero VAD, and silence between speech is skipped. Without the VAD model, it falls back to `fixed`. |
| `vad` | Always splits at pauses; fails without the VAD model. |
| `fixed` | Cuts every `audio_chunk_seconds`; a last piece shorter than 1 s joins the previous chunk. |
| `none` | One pass over the whole input, as liquid-audio does. |

Each chunk is transcribed on its own, and the transcripts are joined with a
space between English words and without one in Japanese text.

Word error rate by input length, EN F16 on CUDA, 8 files per length built from
consecutive LibriSpeech test-clean utterances joined with 0.3 s gaps:

| Length | liquid-audio (one pass) | `auto` | `fixed` |
|---|---|---|---|
| 30 s | 2.8% | 2.3% | 2.9% |
| 60 s | 2.1% | 2.0% | 2.8% |
| 90 s | 9.4% | 1.3% | 2.7% |
| 120 s | 83% | 1.9% | 2.6% |
| 180 s | 296% | 1.4% | 2.4% |

`none` follows liquid-audio up to 90 s (2.6%, 2.0% and 9.4%). At 120 s it
finishes all 8 files at 29%, 6 of them word for word as liquid-audio. At 180 s,
7 of the 8 reach `max_tokens` and come back cut off there, repetitions included,
with a warning.

## Long Text

liquid-audio speaks a whole text in one turn. audio.cpp splits text longer than
`text_chunk_size` codepoints at sentence boundaries with the framework's text
chunker and speaks each chunk as its own turn in the same voice, then joins the
audio. At the default 200 codepoints a chunk is about 13 s of English speech.
The shared 6,000-character long-form test text (`tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json`)
comes out as about 345 s of speech that transcribes back through the ASR task
at 1.5-1.6% WER with the EN F16, Q8_0 and Q4_0 packages on CUDA.

Each chunk is its own take: the voice stays, but pitch, pace and loudness can
shift from one chunk to the next (up to about 3 semitones, 20% and 4 dB on a
7-chunk English text), which can sound like a new recording. liquid-audio's
chunks shift the same way, and the model cannot carry context across them. In
one turn, liquid-audio speaks English reliably up to about 430 characters and
starts to add or drop words from about 600; given the previous chunk as an
earlier turn, it answered in text instead of speaking on up to 2 of 7 chunks.
For English, a larger `text_chunk_size` means fewer joins: 300, 400 and 600
codepoints transcribed back as well as 200 (0.5-1.5% WER). Keep the default for
Japanese: at 400 its chunks ran past `max_tokens`.

## Validation

### ASR

Transcripts were compared with liquid-audio
([`19e65845`](https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa),
fp32, greedy) on 200 English utterances from LibriSpeech test-clean and 200
Japanese utterances from the Common Voice 8.0 ja test set, for every package,
on CPU and CUDA (NVIDIA A10) and on CPU and Metal (Apple M3 Ultra):

| Weights | Transcripts identical to liquid-audio |
|---|---|
| F32 (JP), F16 | 198-200 of 200 |
| Q8_0 | 193-197 of 200 |
| Q4_0 | 160-170 of 200 |

At F32 and F16, the few differences are near-ties in liquid-audio's own
output, where its two best tokens are within 0.04 in log-probability. Below
that, the quantized weights change close token choices.

Stage by stage on CPU with the F32 weights, the adapter output matches
liquid-audio within 1.4e-6 relative error and the first-step logits within 9e-7.
`test_lfm2_audio_asr` (`ENGINE_BUILD_MODEL_TESTS`) checks the prompt, the
tokenizer, the stage numbers and the transcripts against liquid-audio; it runs
when `lfm2_audio_1_5b_f16` is installed in `models/` and skips otherwise.
The `lfm2_audio_*_test` unit tests run on small synthetic GGUFs.

Real-time factor over each 200-utterance set (processing time divided by audio
length, with the model loaded; the CPU runs used 16 threads):

| Backend | EN F16 | JP F32 | EN Q4_0 |
|---|---|---|---|
| CUDA, NVIDIA A10 | 0.027 | 0.037 | 0.017 |
| Metal, Apple M3 Ultra | 0.034 | 0.042 | 0.021 |
| CPU, Linux x86-64 | 0.11 | 0.14 | 0.077 |
| CPU, Apple M3 Ultra | 0.10 | 0.17 | 0.059 |

### TTS

liquid-audio (fp32, CUDA with TF32 off) was dumped at every stage for three
cases: "The quick brown fox jumps over the lazy dog." (US female), "What is
this obsession people have with books?" (UK male) and a Japanese sentence,
with greedy decoding (41, 32 and 45 frames):

| Stage | JP F32, CPU | F16, Linux CPU and CUDA |
|---|---|---|
| Backbone output at `<|audio_start|>` | 1.4e-7 relative | 2e-4 to 9e-4 |
| First frame's depthformer logits | 5e-7 | 4e-4 to 3e-3 |
| Greedy frames identical to liquid-audio | all | all for UK male and Japanese; US female until a near-tie (top two logits 0.002 and 0.000 apart) |
| Detokenizer head on the reference frames | 1.4e-6 | 3e-4 to 5e-3 |
| Waveform | 6e-6 | 1.3e-3 to 1.9e-2 |

The detokenizer's log-magnitudes reach 4.7, a magnitude of 114, which the
framework's Vocos ISTFT would clamp at 100, so the ISTFT here is the
reference's. Long audio is detokenized in chunks that overlap by the model's
receptive field (97 steps); with F32 weights the chunked output matches one
pass to 3e-5. A stream carries each layer's state instead, and with F32
weights its waveform matches one pass to 1e-6 (relative RMS).
`test_lfm2_audio_tts` checks the prompt, the stage numbers, the greedy frames,
the chunking, the stream, a round trip through ASR, greedy and sampled, and
speech cut off at `max_tokens`, offline and streamed; it runs when
`lfm2_audio_1_5b_f16` is installed in `models/`.

Real-time factor through `audiocpp_server` (processing time divided by audio
length, model loaded; short = three sentences of 3-7 s, long = the 345 s
long-form text; the CPU runs used 16 threads):

| Backend | EN F16 short / long | EN Q8_0 short / long | EN Q4_0 short / long | JP F32 short |
|---|---|---|---|---|
| CUDA, NVIDIA A10 | 0.19 / 0.15 | 0.16 / 0.13 | 0.16 / 0.086 | 0.21 |
| Metal, Apple M3 Ultra | 0.21 / 0.21 | 0.18 / 0.18 | 0.15 / 0.15 | 0.26 |
| CPU, Linux x86-64 | 0.42 / 0.41 | 0.29 / 0.30 | 0.22 | 0.65 |
| CPU, Apple M3 Ultra | 0.32 | 0.22 | 0.17 | 0.56 |

Each frame takes a backbone step and eight small depthformer steps. On GPUs
per-step overhead dominates, so the weights' size matters little; on the CPU
each frame reads the backbone once and the depthformer eight times, which
shows in F16. Transcribed back by the ASR task, the long-form speech scored
1.4-1.9% WER on every backend and package measured, mostly homophones and
spellings such as "their" and "harbour"; the two Japanese test sentences came
back exactly.

With the Q4_0 package on Metal on an Apple M3 Max, the server's footprint was 1.17 GB and
stayed within 4 MB of that over 12 requests alternating 2.5 s and 57 s of
speech.

### S2S

liquid-audio's `generate_interleaved` (fp32, CUDA with TF32 off) was dumped at
every step for its README's two questions (English and Japanese) and for
`assets/resources/c.wav`, greedy and sampled like the README (temperature 1.0,
top-k 4, seed 0), and the same steps were replayed through audio.cpp:

| Stage | JP F32, CPU | EN F16, CPU / CUDA |
|---|---|---|
| Prompt ids and audio positions | identical | identical |
| Text logits at every text step | 4e-6 relative | 7e-4 / 3e-3 |
| Backbone output at every audio step | 4e-7 | 5e-4 / 1.4e-3 |
| Depthformer logits at every frame | 7e-6 | 1.5e-3 / 3e-3 |
| Free-running steps identical to liquid-audio | all 277 greedy and 308 sampled | the whole 250-step greedy reply to `c.wav` on the CPU, 248 on CUDA; sampled replies until a near-tie |
| Waveform of the reference frames | 5e-5 | 9e-3 / 0.12 |

The sampled replies were compared through torch's own draws: fed
liquid-audio's logits, the framework's torch-compatible sampler picks every
code torch drew. audio.cpp itself samples with its own generator, so a seed
does not reproduce a liquid-audio reply. The CUDA waveform differs more because
CUDA accumulates F16 products in half precision.

A reply's audio says what its text says, with rare exceptions. Over 30 seeds
of replies to `c.wav` with the F16, Q8_0 and Q4_0 packages, on the CPU and Metal
(Apple M3 Ultra) and on the CPU and CUDA (x86-64, NVIDIA A10), the ASR task
heard the text at a median 4-7% WER, the names the model makes up aside.
Whisper large-v3-turbo heard nearly every reply at 18% or less; one x86 CPU
reply spoke a question its text did not have. The ASR task sometimes answers a
reply, paraphrases it or runs on instead of transcribing it: on 0 to 6 of 30
replies per backend and package, and on 5 of 60 with liquid-audio fp32. It can
also add a sentence that was never spoken to speech that sounds like an
assistant: on one Japanese reply, every precision and liquid-audio fp32 added
「何かご質問はありますか？」. With Q4_0, 2 of 30 replies on the M3 Ultra CPU and 2
of 30 on CUDA repeated a sentence until `max_tokens` (2 of 60 on the x86 CPU,
none on Metal). F16 and Q8_0 did not, though one Q8_0 reply (M3 Ultra CPU, 8
threads) finished its text but not its audio within `max_tokens`.
`test_lfm2_audio_s2s` checks the prompt, the first text and audio blocks, the
round trip of three replies through ASR (their median, a failed reply counting
as a miss), and streaming against offline; it runs when `lfm2_audio_1_5b_f16`
is installed in `models/`.

### Memory

Memory is dominated by the weights, about the package size. Between chunks and
requests the encoder also keeps its graph and compute buffer, at most what a
30 s chunk needs: about 194 MiB, which took about 190 to 250 MiB of device
memory on CUDA, depending on the requests before. The buffer of a longer chunk
is freed after it. With the Q4_0 package on an Apple M3 Ultra, the server's
memory footprint on Metal, kept buffer included, was 1168 MB after a first 3.5 s
request and 1267 to 1310 MB over the next 59, alternating 70 s and 3.5 s of
audio. The Q4_0 files store the token embedding as Q6_K, which CUDA cannot
gather rows from, so on CUDA the backbone also keeps a 256 MiB F16 copy of it.

## Limitations

- S2S answers one turn per request, as a new conversation; liquid-audio's
  demo also keeps the earlier turns.
- `/v1/audio/speech/live` returns an S2S reply's audio but not its text.
- ASR is offline only; TTS and S2S also stream.
- TTS speaks with the built-in voices only; there is no voice cloning.
- On CPU, quantized weights run without repacked kernels. On Apple Silicon,
  llama.cpp transcribes the same Q4_0 files up to 1.8x faster.
