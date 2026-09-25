# LFM2.5-Audio

LFM2.5-Audio is Liquid AI's end-to-end speech and text model. audio.cpp runs its
speech recognition as the community family `lfm2_audio`, directly from the GGUFs
Liquid AI publishes, for the English and the Japanese checkpoint. Text-to-speech
and speech-to-speech are planned; progress is tracked in
[#683](https://github.com/0xShug0/audio.cpp/pull/683).

Upstream: [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) ·
[LiquidAI/LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) ·
reference implementation: [liquid-audio](https://github.com/Liquid4All/liquid-audio)

| Field | Value |
|---|---|
| Family | `lfm2_audio` |
| Task | `asr` |
| Mode | `offline` |
| Languages | `en` (`LFM2.5-Audio-1.5B`), `ja` (`LFM2.5-Audio-1.5B-JP`); each checkpoint transcribes its own language |
| Input | WAV at any sample rate; channels are averaged and the audio is resampled to 16 kHz |
| Output | Transcript text |
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
- Greedy decoding of the chat prompt liquid-audio builds: the system prompt
  `Perform ASR.` (`Perform ASR in japanese.` for the JP checkpoint), then the
  audio embeddings as the user turn, until `<|im_end|>`.

## Packages

Liquid AI publishes each quantization as four llama.cpp-format GGUFs in
[LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/tree/7d525f883a077e20afb782f2ff618edcae0e39e4)
and [LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF/tree/64b96718b341dbd5650f9e85627cecdcbd4ac61b).
The packages point at those repositories, pinned to the linked revisions, and
download the two files transcription needs: the backbone (`<model>-<quant>.gguf`,
which also holds the text tokenizer) and the audio encoder
(`mmproj-<model>-<quant>.gguf`). The `vocoder-` and `tokenizer-` files are for
audio output and are not downloaded.

| Package | Checkpoint | Weights | Download |
|---|---|---|---|
| `lfm2_audio_1_5b_q8_0` (default) | EN | Q8_0 | 1.5 GB |
| `lfm2_audio_1_5b_f16` | EN | F16 | 2.8 GB |
| `lfm2_audio_1_5b_q4_0` | EN | Q4_0 | 0.9 GB |
| `lfm2_audio_1_5b_jp_q8_0` | JP | Q8_0 | 1.5 GB |
| `lfm2_audio_1_5b_jp_f16` | JP | F16 | 2.8 GB |
| `lfm2_audio_1_5b_jp_f32` | JP | F32 | 5.4 GB |
| `lfm2_audio_1_5b_jp_q4_0` | JP | Q4_0 | 0.9 GB |

```bash
python3 tools/model_manager_v2.py install lfm2_audio_1_5b_q8_0 --models-root models
python3 tools/model_manager_v2.py install lfm2_audio_1_5b_jp_q8_0 --models-root models
```

All quantizations of a checkpoint install into one directory,
`models/LFM2.5-Audio-1.5B-GGUF` or `models/LFM2.5-Audio-1.5B-JP-GGUF`, so
components of different quantizations can be combined with the session options
below. The WebUI lists both checkpoints under ASR with their Q8_0 and F16
packages.

## Run

```bash
audiocpp_cli --task asr --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda --audio speech.wav
```

`--model` is the package directory: the published GGUFs do not embed an
audio.cpp model spec, so a GGUF file path is rejected. With one quantization
installed, its backbone is used with the `mmproj-` file of the same name. With
several, choose the backbone:

```bash
audiocpp_cli --task asr --family lfm2_audio \
  --model models/LFM2.5-Audio-1.5B-GGUF --backend cuda --audio speech.wav \
  --session-option lfm2_audio.model_gguf=LFM2.5-Audio-1.5B-F16.gguf
```

For Japanese, pass `models/LFM2.5-Audio-1.5B-JP-GGUF`. `language` can be left
out; a value other than the checkpoint's language is rejected.

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

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `language` | The checkpoint's | `en` or `ja`; must match the checkpoint. |
| `max_tokens` | `512` | Transcript tokens allowed per chunk. A chunk that needs more fails the request rather than returning a cut-off transcript. |
| `audio_chunk_mode` | `auto` | `auto`, `vad`, `fixed` or `none`; see [Long audio](#long-audio). |
| `audio_chunk_seconds` | `30` | Longest chunk in seconds, at least 1. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `lfm2_audio.model_gguf` | The directory's only backbone | Backbone GGUF, relative to the model directory. |
| `lfm2_audio.mmproj_gguf` | `mmproj-<backbone file>`, else the only mmproj | Audio encoder GGUF, relative to the model directory. |
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

`none` follows liquid-audio up to 90 s (2.6%, 2.0% and 9.4%). Beyond that it
fails with a `max_tokens` error instead of returning the repetitions.

## Validation

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

Memory is dominated by the weights, about the package size. With the Q4_0
package on an Apple M3 Max, the server's memory footprint was 1.06 GB on Metal
and stayed within 7 MB of that over 12 requests alternating 3.5 s and 70 s of
audio. The Q4_0 files store the token embedding as Q6_K, which CUDA cannot
gather rows from, so on CUDA the backbone also keeps a 256 MiB F16 copy of it.

## Limitations

- Speech recognition only; text-to-speech and speech-to-speech are planned.
- Offline only, no streaming.
- On CPU, quantized weights run without repacked kernels. On Apple Silicon,
  llama.cpp transcribes the same Q4_0 files up to 1.8x faster.
