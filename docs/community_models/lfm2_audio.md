# LFM2.5-Audio

LFM2.5-Audio is Liquid AI's end-to-end speech and text model. audio.cpp runs its
speech recognition and text-to-speech as the community family `lfm2_audio`,
directly from the GGUFs Liquid AI publishes, for the English and the Japanese
checkpoint. Speech-to-speech is planned; progress is tracked in
[#683](https://github.com/0xShug0/audio.cpp/pull/683).

Upstream: [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) ·
[LiquidAI/LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) ·
reference implementation: [liquid-audio](https://github.com/Liquid4All/liquid-audio)

| Field | Value |
|---|---|
| Family | `lfm2_audio` |
| Tasks | `asr`, `tts` |
| Modes | `offline`; `streaming` for TTS |
| Languages | `en` (`LFM2.5-Audio-1.5B`), `ja` (`LFM2.5-Audio-1.5B-JP`); each checkpoint transcribes and speaks its own language |
| ASR input | WAV at any sample rate; channels are averaged and the audio is resampled to 16 kHz |
| TTS output | 24 kHz mono speech |
| Voices | `us_male` (default), `us_female`, `uk_male`, `uk_female` for English; the Japanese checkpoint has one voice |
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
  until a frame starts with the end-of-audio code.
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
| `<model>-<quant>.gguf` | LFM2 backbone and text tokenizer | ASR, TTS |
| `mmproj-<model>-<quant>.gguf` | Audio encoder and adapter; the embedding of generated audio codes | ASR, TTS |
| `vocoder-<model>-<quant>.gguf` | Depthformer; the detokenizer's code embedding and ISTFT window | TTS |
| `tokenizer-<model>-<quant>.gguf` | Audio detokenizer | TTS |

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
below. The WebUI lists both checkpoints under ASR and TTS with their Q8_0 and
F16 packages.

## Run

```bash
audiocpp_cli --task asr --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda --audio speech.wav

audiocpp_cli --task tts --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda \
  --text "The next train leaves in ten minutes." --voice-id uk_female --out speech.wav
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

### Streaming TTS

With `--mode streaming` (CLI) or `"mode": "streaming"` (server entry), speech
comes out as it is generated. Like liquid-audio's demo, each audio frame is
decoded as soon as the depthformer picks it: the detokenizer is causal, so a
frame needs only the 17 frames before it, and the ISTFT releases a sample once
no later window reaches it. Each event carries one frame (80 ms,
`stream_frames_per_event` to change it), plus a last 20 ms event per text
chunk; the events add up to the offline speech for the same seed. A text chunk
that reaches `max_tokens` ends its speech there, streamed as offline, with the
same warning.

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lfm2-audio-tts-stream", "input": "The next train leaves in ten minutes.", "voice": "us_female", "stream": true}'
```

The response is server-sent events with base64 16-bit PCM deltas at 24 kHz;
`"stream_format": "audio"` returns the raw PCM instead.

Time to the first audio and real-time factor through the server, for a
two-sentence English text (about 14 s of speech) and a Japanese one (8.5 s):

| Backend | First audio, F16 / Q8_0 / Q4_0 / JP F32 | RTF, 1 frame per event | RTF, 4 frames per event |
|---|---|---|---|
| CUDA, NVIDIA A10 | 31 / 26 / 24 / 44 ms | 0.12-0.23 | 0.09-0.21 |
| Metal, Apple M3 Ultra | 44 / 41 / 38 / 55 ms | 0.20-0.31 | 0.16-0.27 |
| CPU, Apple M3 Ultra, 16 threads | 153 / 97 / 109 / 334 ms | 0.33-0.89 | 0.22-0.64 |

In these runs the streamed audio differed from offline by 1e-4 to 6e-3
(relative RMS), the detokenizer's arithmetic in graphs of other sizes; CLI runs
with other texts and seeds reached 1.5e-2 on the CPU and 4.7e-2 on CUDA, with
the same frames.

## Request Options (use with `--request-option`)

| Option | Task | Default | Meaning |
|---|---|---|---|
| `language` | both | The checkpoint's | `en` or `ja`; must match the checkpoint. |
| `max_tokens` | both | `512` | ASR: transcript tokens per audio chunk. TTS: 80 ms audio frames per text chunk. A transcript or a text chunk's speech that reaches it is cut off there and kept, as liquid-audio keeps it, and a warning goes to stderr; the other chunks go on. Transcript tokens need not end on a character boundary, so a cut can fall inside a character: liquid-audio then shows U+FFFD for the partial character, while audio.cpp drops it and ends the transcript at the last whole character. |
| `audio_chunk_mode` | ASR | `auto` | `auto`, `vad`, `fixed` or `none`; see [Long audio](#long-audio). |
| `audio_chunk_seconds` | ASR | `30` | Longest chunk in seconds, at least 1. |
| `temperature` | TTS | `0.8` | Audio code sampling temperature; 0 is greedy. |
| `top_k` | TTS | `64` | Sample from the k most likely codes; 0 keeps all, 1 is greedy. |
| `seed` | TTS | Random | Sampling seed; text chunk i uses seed + i. |
| `text_chunk_mode` | TTS | `japanese` for JP, else `default` | How long text is split; see [Long text](#long-text). |
| `text_chunk_size` | TTS | `200` | Unicode codepoints per text chunk. |
| `stream_frames_per_event` | TTS streaming | `1` | Audio frames (80 ms) per streaming event. |

Each task rejects the other task's options.

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
pass to 3e-5. `test_lfm2_audio_tts` checks the prompt, the stage numbers, the
greedy frames, the chunking, a round trip through ASR, greedy and sampled, and
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

- Speech-to-speech is planned.
- ASR is offline only; TTS also streams.
- TTS speaks with the built-in voices only; there is no voice cloning.
- On CPU, quantized weights run without repacked kernels. On Apple Silicon,
  llama.cpp transcribes the same Q4_0 files up to 1.8x faster.
