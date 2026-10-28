# LFM3-Audio

LFM3-Audio is Liquid AI's end-to-end speech and text model. audio.cpp runs its
speech recognition (with a spoken-question chat mode), text-to-speech in 38
built-in voices and spoken chat (speech-to-speech) as the community family
`lfm3_audio`.

It loads the llama.cpp-format GGUFs Liquid AI publishes in
[LiquidAI/LFM3-Audio-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF),
or the upstream [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B)
safetensors directory.

| Field | Value |
|---|---|
| Family | `lfm3_audio` |
| Status | experimental |
| Tasks | `asr`, `tts`, `s2s` |
| Modes | `offline`; `streaming` for TTS and S2S |
| Languages | `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `sv`, `tr`, `zh` |
| Audio input (ASR, S2S) | WAV at any sample rate. Channels are averaged and the audio is resampled to 16 kHz with libsoxr (see [Limitations](#limitations)) |
| Speech output (TTS, S2S) | 24 kHz mono speech. S2S also returns the reply's text |
| Voices | TTS: `<language>_female` and `<language>_male` for 19 languages (38 voices); `american_english_female` is the default. S2S replies in the checkpoint's own voice |
| Weights | Liquid's GGUFs: Q4_0, Q4_K_M, Q8_0, F16, BF16. The upstream safetensors directory (BF16) |
| Backends tested | CPU (x86 and Apple M3 Ultra), CUDA (NVIDIA H100), Metal (Apple M3 Ultra) |
| License | [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/main/LICENSE): commercial use only for organizations under USD 10M yearly revenue ([model licenses](../model_licenses.md)) |

## Architecture

- Frontend: 128-bin log-mel at 16 kHz (25 ms window, 10 ms hop). A
  three-stage strided conv2d downsampler (8x in time) feeds a 17-layer LFM2
  encoder (768 wide), and an MLP projector maps it to the backbone's width.
  One audio token is 80 ms of input.
- LFM3 backbone: 24 layers, 2048 wide. 13 gated short-convolution layers
  (kernel 7), 6 full-attention and 5 sliding-window (256) attention layers with
  grouped-query attention, QK-norm, a per-head sigmoid gate and sandwich norms.
  The text head is tied to the token embedding.
- Depth decoder: a 3-layer bidirectional transformer that predicts one audio
  frame of 12 codes (12.5 Hz) from one backbone state with MaskGIT, revealing 4
  codebooks per pass in an order drawn per frame. A frame goes back into the
  backbone as the sum of its code embeddings.
- LiquidCodec decoder: FSQ codes, an 8-layer LFM3 stack (causal, sliding window
  30) over the frames repeated 6 times, then log-magnitude and phase and an
  ISTFT (n_fft 1280, hop 320) to 24 kHz audio. Offline decoding uses the
  streaming decoder in chunks of up to 128 frames.
- ASR: the prompt is `Perform ASR in <language>.` (`Perform ASR.` with
  `language=auto`) and the audio as the user turn, decoded greedily. With
  `chat=true` the system turn is left out and the model answers the spoken
  input as text.
- TTS: the prompt is `Perform TTS in <language> using a <female|male> voice.`
  and the text as the user turn. Each step after `<|audio_start|>` gives one
  80 ms frame until a frame holds the end-of-audio code.
- S2S: the system prompt `Respond with interleaved text and audio.` and the
  question as the user turn. The reply alternates 6 text tokens and 12 audio
  frames until `<|text_end|>`, keeps speaking until an end-of-audio frame, and
  ends at `<|im_end|>`.

## Install

Liquid AI publishes each quantization as two llama.cpp-format GGUFs in
[LiquidAI/LFM3-Audio-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/tree/fcfae9027c00a6f3f872c48a841cfeb77e2b5a1c).
The packages point at that repository, pinned to the linked revision:

| File | Contents |
|---|---|
| `LFM3-Audio-2B-<quant>.gguf` | LFM3 backbone and text tokenizer |
| `mmproj-LFM3-Audio-2B-<quant>.gguf` | Audio encoder and projector, depth decoder and codebook embeddings, LiquidCodec decoder (always F16) |

| Package | Backbone | mmproj | Download |
|---|---|---|---|
| `lfm3_audio_2b_q8_0` (default) | Q8_0 | Q8_0 | 2.2 GB |
| `lfm3_audio_2b_f16` | F16 | F16 | 4.0 GB |
| `lfm3_audio_2b_bf16` | BF16 | BF16 | 4.0 GB |
| `lfm3_audio_2b_q4_k_m` | Q4_K_M | Q8_0 | 1.4 GB |
| `lfm3_audio_2b_q4_0` | Q4_0 | Q8_0 | 1.4 GB |

```bash
python3 tools/model_manager_v2.py install lfm3_audio_2b_q8_0 --models-root models
```

All packages install into `models/LFM3-Audio-2B-GGUF`. `--model` takes a
backbone GGUF, or the directory when it holds one backbone. The mmproj is
`mmproj-<backbone file>` when present, else `mmproj-LFM3-Audio-2B-Q8_0.gguf`
(which Liquid pairs with the 4-bit backbones), else the only mmproj there. The
published GGUFs do not embed an audio.cpp model spec: it comes from
`model_specs/lfm3_audio.json`, as for [LFM2.5-Audio](lfm2_audio.md#run).

audio.cpp reads the tensors under the upstream checkpoint's names, and rebuilds
what llama.cpp's converter turns into metadata or constants: the configs, the
tokenizer (the backbone's vocabulary gives the same ids as the upstream
`tokenizer.json` for all 128000 entries) and the codec's ISTFT window. In the
BF16 GGUFs every backbone, encoder and depth-decoder tensor equals the upstream
safetensors bit for bit; the codec decoder is F16 in every published GGUF.

## Convert (optional)

The model also loads from the upstream directory as it is. Other types come
from llama.cpp's tools, which write the published layout:
`convert_hf_to_gguf.py` converts the upstream directory to the backbone GGUF
(with `--mmproj`, to the mmproj), and `llama-quantize` makes other backbone
types. Their LFM3-Audio support came with
[ggml-org/llama.cpp#31257](https://github.com/ggml-org/llama.cpp/pull/31257).
Only the five published types are tested.

## Run

```bash
audiocpp_cli --task asr --family lfm3_audio \
  --model models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf --backend cuda \
  --audio speech.wav --language en

audiocpp_cli --task tts --family lfm3_audio \
  --model models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf --backend cuda \
  --text "The next train leaves in ten minutes." --voice-id british_english_male --out speech.wav

audiocpp_cli --task s2s --family lfm3_audio \
  --model models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf --backend cuda \
  --audio question.wav --out reply.wav --text-out reply.txt
```

From the upstream directory, pass it as `--model models/LFM3-Audio-2B`.
Pass `--family lfm3_audio` to select this loader explicitly; automatic detection
follows registry order and the safetensors layout is shared with other models.
A server config entry needs `"family": "lfm3_audio"`.

### ASR

`language` (CLI `--language`) is one of the 18 codes above, or `auto` (the
default) to let the model find the language. Audio up to
`audio_chunk_duration_sec` (300 s by default) is transcribed in one pass;
longer audio is split at Silero VAD pauses (`audio_chunk_mode=auto`), or into
fixed chunks when the VAD model is missing. A chunk's transcript may run to
`max_tokens`, by default 512 or 8 tokens per second of audio, whichever is more,
and never past the 8192-step limit ([Limitations](#limitations)). A transcript
that reaches it is cut off there and a warning goes to stderr.

`chat=true` answers the spoken input instead of transcribing it, in one pass
(no chunking); `language` has no effect on the answer, and the result names no
language. `enable_thinking=true` (needs `chat=true`) lets the model think
first and returns only the text after `</think>`. If thinking reaches
`max_tokens` (512 for chat, 2048 with thinking), or the model ends its turn,
without closing `</think>`, the answer is empty, as upstream, and a warning says
so.

```bash
audiocpp_cli --task asr --family lfm3_audio \
  --model models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf --backend metal \
  --audio question.wav --request-option chat=true --request-option enable_thinking=true
```

### TTS

`--voice-id` takes any of the 38 voices: `arabic`, `danish`, `german`, `greek`,
`british_english`, `american_english`, `finnish`, `french`, `hebrew`,
`italian`, `japanese`, `korean`, `dutch`, `polish`, `portuguese`, `spanish`,
`swedish`, `turkish` or `chinese`, followed by `_female` or `_male`. Without a
voice, `language` picks that language's female voice (`en` gives
`american_english_female`; British English is reached only by voice id). Long
text is split into chunks of about 200 characters (`text_chunk_size`,
`text_chunk_mode`; `japanese` mode for Japanese and Chinese voices), and each
chunk is its own turn of up to `max_tokens` frames (512, 41 s). Audio codes are
sampled at temperature 1.0 from the top 128, with repetition penalty 1.1 over
each codebook's last 4096 frames: the checkpoint's `generation_config.json`
(`clip.gen.audio.sampling.*` in the published mmproj), which the model card
recommends. `temperature`, `top_k` and `repetition_penalty` override it. Chunk *i*
uses `seed` + *i*, so pass `--seed` for repeatable speech. Reference audio is
rejected: there is no voice cloning.

With `--mode streaming`, each event carries `stream_frames_per_event` frames
(1 by default, 80 ms). The codec is causal and keeps its state between frames,
so the stream adds up to the offline speech for the same seed, up to the
arithmetic of graphs of other sizes ([Validation](#validation)).

### S2S

S2S answers a spoken question with a spoken reply and its text: the CLI prints
the text and `--text-out` writes it. `--text` replaces the system prompt, which
the checkpoint was trained with, so leave it out unless experimenting. S2S takes
no voice and no language other than `auto`. The reply may run to `max_tokens`
(1024) steps, text tokens and frames together. After the reply's text ends, at
most `max_audio_tail_frames` frames (375, 30 s; 0 for no cap) are sampled before
the turn is closed with an end-of-audio frame, because greedy audio
(`--temperature 0`) can keep speaking. Text is greedy unless `text_temperature`
is above 0.

With `--mode streaming` the question comes in as audio chunks (live PCM with
`--audio -`); it is encoded and prefilled when the input ends, and the reply
starts then. Each event carries one frame's audio (`stream_frames_per_event`)
and the text written since the last event, in whole characters.

### Server

The server takes the same paths and options:

```json
{"models": [
  {"id": "lfm3-asr", "family": "lfm3_audio", "task": "asr", "mode": "offline",
   "path": "models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf"},
  {"id": "lfm3-tts", "family": "lfm3_audio", "task": "tts", "mode": "streaming",
   "path": "models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf"},
  {"id": "lfm3-s2s", "family": "lfm3_audio", "task": "s2s", "mode": "streaming",
   "path": "models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf"}]}
```

```bash
audiocpp_server --backend cuda --config server.json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F model=lfm3-asr -F file=@speech.wav
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lfm3-tts", "input": "The next train leaves in ten minutes.", "voice": "german_female"}' -o speech.wav
curl http://127.0.0.1:8080/v1/tasks/run -H 'Content-Type: application/json' \
  -d '{"model": "lfm3-s2s", "request": {"audio": "/path/to/question.wav"}}'
```

The TTS entry streams, and a request without `stream_format` still gets the
whole WAV. With `"stream_format": "sse"`, speech comes back as it is decoded:
`speech.audio.delta` events of base64 16-bit PCM at 24 kHz, which add up to
the WAV, then `speech.audio.done` with the time to the first audio
([server API](../../app/server/api.md#post-v1audiospeech)).

```bash
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lfm3-tts", "input": "The next train leaves in ten minutes.", "voice": "german_female", "stream_format": "sse"}'
```

`/v1/tasks/run` returns the S2S reply's `text` and its audio as base64 WAV.
`/v1/tasks/stream` with `"stream_format": "sse"` streams it from a streaming
entry. See the [server README](../../app/server/README.md).

The live route, `/v1/audio/speech/live`, takes the question as chunked raw PCM
for a streaming entry, and the reply starts when the input ends. Each live
request is a first turn. Without `input` the reply uses the default system
prompt; an `input` replaces it, as `--text` does.

```bash
ffmpeg -i question.wav -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/speech/live?model=lfm3-s2s&sample_rate=16000&channels=1&sample_format=s16le&return_text=true'
```

It returns the reply's audio as server-sent events and, with `return_text=true`,
its text ([server API](../../app/server/api.md#post-v1audiospeech)). The query
also takes the request options `seed`, `temperature`, `top_k`,
`repetition_penalty`, `text_temperature`, `text_top_k` and `max_tokens`. The
route does not forward `max_audio_tail_frames`,
`stream_frames_per_event` or `return_codes`, so those keep their defaults; use
`/v1/tasks/stream` for the reply artifact. The first audio of a reply comes
after encoding the whole question, the prompt, the first text block and the
first frame.

### WebUI

The WebUI lists LFM3-Audio under ASR, TTS and speech-to-speech, each entry
reading `models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf` from the
`lfm3_audio_2b_q8_0` package, which the WebUI can download. ASR has the `chat`
and `enable_thinking` options, TTS offers the 38 voices, and speech-to-speech
has no text box or language, so replies use the default system prompt.

The speech-to-speech page holds a conversation: each Run sends the recorded or
chosen question to `/v1/tasks/stream` with the earlier turns, as
[Conversations](#conversations) describes, plays the reply as it streams in
and adds the turn to a list, and New conversation starts over. With
`--ui-management` the page loads the entry in streaming mode itself. With a
config file, live playback needs the entry in `"mode": "streaming"`; an
offline entry gives each reply when it is done.

## Conversations

audio.cpp keeps no conversation state: the client sends the earlier turns with
each new question, as the request's input artifacts, in order. For each earlier
turn, its question and then its reply:

| Artifact id | Kind | Payload |
|---|---|---|
| `lfm3_audio.question` | `custom` | The question's audio as a WAV file |
| `lfm3_audio.reply` | `acoustic_tokens` | The reply artifact that turn's result returned, payload and meta unchanged |

With `return_codes=true`, an S2S result, offline or streamed, also returns its
reply as an `lfm3_audio.reply` artifact. The payload is little-endian int32
values, one step after another: a text token is its id, and an audio frame is
-1 followed by its 12 codes. It holds every step, the end-of-audio frame
included, and goes back into the prompt verbatim, which equals continuing the
upstream model's KV cache. A reply that was cut off (`ended=false`) gets an
end-of-audio frame and `<|im_end|>` appended when replayed. Its meta:

| Key | Value |
|---|---|
| `format` | `lfm3_audio.reply/1` |
| `codebooks`, `audio_vocab_size`, `text_vocab_size` | `12`, `2017` (2016 codes and end-of-audio), `128000` |
| `steps` | Text tokens and frames in the payload |
| `ended` | `true` when the reply ended; `false` when `max_tokens`, or a stream finished early, cut it off |

A server request sends the history in its `artifacts` array; each artifact
takes a base64 `payload` or a `path`:

```json
{"model": "lfm3-s2s", "request": {
  "audio": "/path/to/turn2.wav",
  "options": {"return_codes": true},
  "artifacts": [
    {"id": "lfm3_audio.question", "kind": "custom", "path": "/path/to/turn1.wav"},
    {"id": "lfm3_audio.reply", "kind": "acoustic_tokens", "payload": "<turn 1's base64>",
     "meta": {"format": "lfm3_audio.reply/1", "codebooks": "12", "audio_vocab_size": "2017",
              "text_vocab_size": "128000", "steps": "<turn 1's steps>", "ended": "<turn 1's ended>"}}
  ]}}
```

The CLI's `--request-sequence` JSON takes the same `artifacts` array, and the C
API takes them with `audiocpp_request_add_artifact` and
`audiocpp_request_set_artifact_meta`, as for [LFM2.5-Audio](lfm2_audio.md#conversations).
Artifacts out of turn, with other ids or meta, from another format version, or
sent to ASR or TTS are request errors (HTTP 400).

The prompt and `max_tokens` may take 8192 steps together (an 80 ms audio token,
a text token or a frame is one step). Past that the turn fails with a
CapacityError and the client leaves out the oldest turns. Every turn encodes all
its questions and prefills the whole conversation again (in 256-step blocks), so
later turns take longer.

## Request Options (use with `--request-option`)

| Option | Task | Default | Meaning |
|---|---|---|---|
| `language` | all | `auto` | ASR: the spoken language, or `auto`. TTS: picks that language's female voice when no voice is given. S2S: only `auto`. |
| `max_tokens` | all | ASR: max(512, 8 per second); chat 512, 2048 with thinking. TTS: 512. S2S: 1024 | ASR: text tokens per chunk. TTS: 80 ms frames per text chunk. S2S: text tokens and frames of the reply together. |
| `audio_chunk_mode` | ASR | `auto` | `auto`, `fixed`, `vad` or `none`. |
| `audio_chunk_duration_sec` | ASR | `300` | Longest chunk in seconds, at least 1; never above `lfm3_audio.max_pass_duration_sec`. |
| `chat` | ASR | `false` | Answer the spoken input as text instead of transcribing it. |
| `enable_thinking` | ASR | `false` | With `chat=true`, think first; only the text after `</think>` is returned. |
| `temperature` | TTS, S2S | `1.0` | Audio code temperature; 0 picks the most likely code. The default is the checkpoint's. |
| `top_k` | TTS, S2S | `128` | Sample audio codes from the k most likely; 0 keeps all. The default is the checkpoint's. |
| `repetition_penalty` | TTS, S2S | `1.1` | Audio repetition penalty over the last 4096 frames of each codebook; 1 disables it. The default and the window are the checkpoint's. |
| `text_temperature` | ASR, S2S | `0` | Text token temperature; 0 is greedy. |
| `text_top_k` | ASR, S2S | `0` | Sample text tokens from the k most likely; 0 keeps all. |
| `seed` | all | Random | Sampling seed; TTS text chunk i uses seed + i. |
| `text_chunk_mode` | TTS | `japanese` for Japanese and Chinese voices, else `default` | `default`, `tag_aware`, `japanese` or `endline`. |
| `text_chunk_size` | TTS | `200` | Target characters per text chunk. |
| `stream_frames_per_event` | TTS, S2S streaming | `1` | Audio frames (80 ms) per event. |
| `max_audio_tail_frames` | S2S | `375` | Most frames after the reply's text ends before the turn is closed with an end-of-audio frame; 0 disables the cap. |
| `return_codes` | S2S | `false` | Also return the `lfm3_audio.reply` artifact for the next turn. |

Each task rejects the options it does not take.

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `lfm3_audio.weight_type` | `native` | `native` keeps the file's types (BF16 is loaded as F16 where the backend has no fast BF16 matmul: Metal and the arm64 CPU). `f16` converts the weights to F16 at load. `f32` loads them as F32 and also uses F32 KV caches and explicit attention: the parity mode. |
| `lfm3_audio.cpu_repack` | `true` | On the CPU, repack matmul weights for ggml's extra buffer kernels. Other backends ignore it. |
| `lfm3_audio.vad_model_path` | `assets/framework/models/silero_vad` | VAD model (any built-in VAD family) used by `audio_chunk_mode` `auto` and `vad`. |
| `lfm3_audio.max_pass_duration_sec` | `300` | Most audio, in seconds, the encoder takes in one pass (an ASR chunk, the S2S question or an earlier question); longer audio is rejected before it is encoded. |

## Developers

### Parity mode and stage dumps

`--session-option lfm3_audio.weight_type=f32` runs every weight, KV cache and
attention in F32, the mode the stage checks compare against the fp32 reference.
These environment variables are for parity work:

| Variable | Effect |
|---|---|
| `LFM3_AUDIO_DUMP_DIR=<dir>` | Write each stage to `<dir>/<name>.npy` (float32 or int32, C order). |
| `LFM3_AUDIO_DUMP_FRAMES=<n>` | Frames whose per-pass depth logits are dumped (default 4). |
| `LFM3_AUDIO_MASKGIT_ORDERS=<file.npy>` | Int32 `[frames, 12]`; each row, a permutation of 0..11, replaces the MaskGIT order of that frame in every generation. With greedy sampling this reproduces a reference run exactly. |

Dumps: `wave16k`, `features`, `downsample`, `encoder`, `projector`,
`prompt_ids`, `prefill_hidden`, `prefill_logits`, `steps`
(`[S, 13]`: the token or -1, then 12 codes or -1), `maskgit_orders`,
`depth_logits_f<f>_p<p>`, `codes`, `codec_spectrum` and `audio24k`.

### Fixtures

`tools/community_models/lfm3_audio/make_fixtures.py` runs the upstream model in
fp32 on CUDA (needs torch, transformers, librosa) and writes the reference
arrays and `manifest.json`, with the bf16-vs-fp32 noise floor of each stage:

```bash
python3 tools/community_models/lfm3_audio/make_fixtures.py --model models/LFM3-Audio-2B \
  --assets assets/resources --out models/LFM3-Audio-2B/fixtures
```

The fixtures come from the model weights. Keep them, and every dump, out of
git.

### Tests

The 13 unit tests (`tests/community_models/lfm3_audio/`) build a tiny synthetic
package and need no weights. They build with the unit tests
(`-DENGINE_BUILD_TESTS=ON`):

```bash
ctest --test-dir build -R lfm3_audio
```

The real-weight tests `test_lfm3_audio_parity`, `test_lfm3_audio_asr`,
`test_lfm3_audio_tts` and `test_lfm3_audio_s2s` build with
`-DENGINE_BUILD_MODEL_TESTS=ON`. They read every expected value from the fixture
directory at run time and exit 125 (skipped) without the model or the fixtures:

```bash
build/bin/test_lfm3_audio_parity --model models/LFM3-Audio-2B \
  --fixtures models/LFM3-Audio-2B/fixtures --backend cpu --weight-type f32 --threads 16
build/bin/test_lfm3_audio_asr --model models/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf \
  --fixtures models/LFM3-Audio-2B/fixtures --backend best --weight-type native
```

`--model` takes the upstream directory or a GGUF (a published backbone GGUF
brings its mmproj), `--backend` is `cpu` or `best`, and `--weight-type` is
`f32` (the default) or `native`. The tolerance follows the weights: F32 on the
CPU 1e-5 rel per stage (3e-5 for the encoder stages run on the host's features,
1e-4 for the 418 s clip and the codec waveform) with identical greedy output;
F32 on a GPU 5e-3 (waveform 1e-2), where greedy output may differ only at a
near-tie; BF16 and F16 max(2e-2, 2x the stage's noise floor), with the same
argmax and the same ASR transcript; Q8_0 max(5e-2, 3x the floor); the 4-bit
types (Q4_K_M, Q4_0) are reported only. The tolerance is picked from the file
name, so keep the quantization in it.

`lfm3_audio_tokenizer_test --model <upstream dir> --fixtures <dir> --gguf
<backbone GGUF>` also checks a published backbone's vocabulary against
`tokenizer.json`, entry by entry.

## Validation

Real-weight runs on 2026-10-27 with the public checkpoint: CPU and CUDA on an
NVIDIA H100 host, CPU and Metal on an Apple M3 Ultra. F32 rows load the
upstream safetensors directory with `weight_type=f32`; the others load Liquid's
GGUFs with their own types. rel is the relative L2 error against the fp32
reference. Stages: the encoder output (projector), the last-position logits of
an ASR prompt and of a 418 s one-pass prompt, the depth decoder's logits on the
reference's inputs (40 frames, 1440 codes), and the codec waveform for real and
random codes.

| Config | Projector | ASR logits | 418 s logits | Depth logits | Depth argmax | Codec wave (TTS / random) |
|---|---|---|---|---|---|---|
| CPU F32 (x86) | 1.5e-6 | 2.1e-6 | 4.8e-6 | 3.5e-7 | 1440/1440 | 2.7e-6 / 2.4e-5 |
| CPU F32 (M3 Ultra) | 1.5e-6 | 1.9e-6 | 4.0e-6 | 3.6e-7 | 1440/1440 | 2.6e-6 / 2.4e-5 |
| CPU BF16 (x86) | 3.9e-3 | 6.0e-3 | 2.0e-3 | 1.6e-3 | 1425/1440 | 3.0e-4 / 7.0e-4 |
| CPU BF16 as F16 (M3 Ultra) | 2.1e-3 | 2.8e-3 | 1.6e-3 | 8.5e-4 | 1434/1440 | 2.7e-3 / 1.5e-2 |
| CUDA F32 | 4.9e-4 | 3.7e-4 | 4.4e-4 | 6.0e-4 | 1437/1440 | 1.1e-3 / 8.1e-3 |
| CUDA BF16 | 4.1e-3 | 2.1e-3 | 1.3e-3 | 1.5e-3 | 1435/1440 | 1.3e-3 / 8.8e-3 |
| CUDA F16 | 5.0e-4 | 5.6e-4 | 1.0e-3 | 2.5e-4 | 1438/1440 | 1.3e-3 / 8.8e-3 |
| CUDA Q8_0 | 2.9e-2 | 1.3e-2 | 8.6e-3 | 8.6e-3 | 1378/1440 | 1.3e-3 / 8.8e-3 |
| CUDA Q4_K_M | 2.9e-2 | 8.1e-2 | 0.42 | 8.6e-3 | 1378/1440 | 1.3e-3 / 8.8e-3 |
| CUDA Q4_0 | 2.9e-2 | 0.13 | 0.78 | 8.6e-3 | 1378/1440 | 1.3e-3 / 8.8e-3 |
| Metal F32 | 4.8e-4 | 3.0e-4 | 3.4e-4 | 1.8e-4 | 1438/1440 | 7.2e-4 / 7.6e-3 |
| Metal BF16 = F16 | 4.6e-4 | 3.6e-4 | 5.1e-4 | 1.8e-4 | 1439/1440 | 7.9e-4 / 6.7e-3 |
| Metal Q8_0 | 1.7e-2 | 8.8e-3 | 8.6e-3 | 4.8e-3 | 1403/1440 | 7.9e-4 / 6.7e-3 |
| Metal Q4_K_M | 1.7e-2 | 8.0e-2 | 0.40 | 4.8e-3 | 1403/1440 | 7.9e-4 / 6.7e-3 |
| Metal Q4_0 | 1.7e-2 | 0.13 | 0.62 | 4.8e-3 | 1403/1440 | 7.9e-4 / 6.7e-3 |

Metal has no BF16 matmul and the arm64 CPU has no SIMD BF16 dot product, so
the BF16 GGUF loads as F16 on both. On Metal it gives the same numbers as the
F16 GGUF. Every published GGUF keeps the codec at F16, and the Q4_K_M and Q4_0
packages share the Q8_0 mmproj, so the encoder, depth decoder and codec columns
repeat across those rows. The published BF16 GGUF with `weight_type=f32` gives
the CUDA F32 row's numbers except the codec (1.1e-3 / 8.9e-3).

- ASR: the short-clip transcripts, with `language=en` and `auto`, equal the
  fp32 reference's with F32, BF16, F16 and Q8_0 on every backend; the 4-bit
  backbones change words. On a 418 s English recording (`language=auto`),
  WER against a whisper-large-v3 transcript is 0.22% with the default
  VAD-split passes and in one pass, for F32 on CUDA and the M3 Ultra's CPU and
  every published type on CUDA, Metal and both CPUs, except Q4_0: 0.07% with
  VAD passes and 0.29% in one pass.
- Thinking chat: identical tokens to the reference on 4 clips with F32 on the
  H100 host's CPU and on CUDA and with the BF16 GGUF on CUDA.
- TTS: with injected MaskGIT orders and greedy codes, F32 on the CPU and on
  Metal gives the reference's frames exactly; the other configs diverge only
  where the top two codes are within 1e-2, or are reported only (quantized).
  Text read back by ASR (5 sentences) gives 0% WER on every config, and all 38
  voices speak and end. Streamed speech differs from offline by at most 2.5e-3
  rel.
- S2S: F32 on either CPU gives the reference's greedy reply step for step (232
  steps, and a reply cut at 96). Streamed replies have the same text and reply
  artifact as offline in every config, and audio within 2.6e-3 rel (CUDA F16:
  1.1e-2). For a second turn, the replayed prompt matches the reference token
  for token, and its first logits are within 7.0e-7 rel on the CPUs in F32,
  1.5e-4 to 2.1e-4 on GPUs in F32 and F16, 9.3e-4 to 1.4e-3 with BF16 and 4.6e-3
  to 7.2e-3 in Q8_0. The argmax is the reference's in every config but one
  4-bit case (CUDA Q4_K_M after the cut reply).

## Performance

Liquid's Q8_0 GGUFs with the checkpoint's sampling defaults, seed 42 and
`--threads 8`, on 2026-10-27. Offline numbers are the median of 3 timed runs
after a warm-up; streaming numbers are measured by the client, the median of 3
(live route: 5). TTS reads 15 English sentences and S2S answers 10 spoken
questions. RTF is processing time over audio duration. The hosts were measured
separately, and these methods differ, so do not compare the two columns:

- Live route input: a 2.1 s question on the H100, the 7.5 s clip on the M3 Ultra.
- TTS RTF: the median of the 15 per-sentence RTFs on the H100, total time over
  total audio on the M3 Ultra.
- Load time: the median of 7 ASR processes on the H100; on the M3 Ultra, the
  median of 3 runs on a 1 s clip after a first.
- S2S: each host asked its own 10 questions.
- Peak memory: the device's GPU memory on the H100, the process footprint on the
  M3 Ultra.

| Metric | H100 (CUDA) | M3 Ultra (Metal) |
|---|---|---|
| ASR RTF, 7.5 s / 418 s clip | 0.015 / 0.019 | 0.029 / 0.041 |
| ASR WER, 418 s clip, `auto` / `en` | 0.07% / 4.31% | 0.14% / 4.38% |
| TTS RTF | 0.063 | 0.138 |
| TTS time to first audio | 15 ms | 37 ms |
| S2S time to first audio, turn 1 / turn 2 | 48 ms / 90 ms | 83 ms / 167 ms |
| S2S streaming RTF | 0.090 | 0.182 |
| Live route, first audio after the input ends | 48 ms | 154 ms |
| Model files / load time | 2083 MB / 2.7 s | 2083 MB / 1.1 s |
| Peak memory, ASR 7.5 s / ASR 418 s / TTS | 2683 / 4719 / 2751 MiB | 3084 / 3594 / 3172 MB |

The 418 s clip's reference is a whisper-large-v3 transcript; the ASR RTF on it
is with `language=en`. These WERs fold apostrophes and dashes (and, on the M3
Ultra, spell out numbers) where the [Validation](#validation) numbers do not,
so they differ from its 0.22%. Turn 2 is slower because it encodes and
prefills the whole conversation again.

## Limitations

- ASR is offline only; TTS and S2S also stream. The model card's live ASR
  (`Perform live ASR.`, about 0.96 s chunks transcribed as they arrive) is not
  implemented.
- TTS speaks with the 38 built-in voices only; there is no voice cloning. S2S
  replies in the checkpoint's voice and takes no voice.
- Input at other sample rates is resampled with libsoxr, as the upstream
  processor does through librosa. audio.cpp loads libsoxr at run time; where it
  is not found, as on a stock macOS, it falls back to a sinc resampler that
  differs by about 6% in the top mel bands and can change words, with a
  warning under `--log` (on one 24 kHz stereo clip it gave the same transcript
  as libsoxr with every type but 4-bit). Install it (`brew install libsoxr`,
  `apt install libsoxr0`), or send 16 kHz audio.
- On CUDA, the short convolutions do not use `ssm_conv`, which aborts on the
  backbone's kernel width 7 in ggml-cuda: passes over several steps (prefill,
  the encoder, the codec) run them as shifted row views, and a decode step as
  one multiply and a row sum. The CPU and Metal use `ssm_conv`.
- ASR takes no request text: `--text` is ignored, so the model card's uses that
  pair the audio with a typed instruction (translation, questions about the
  audio, tool calls) are not available.
- A request's prompt and output share at most 8192 steps, audio.cpp's limit, a
  quarter of the model card's 32K-token context.
- The 4-bit backbones lose accuracy: the 418 s prompt's logits are 0.40-0.42
  rel off with Q4_K_M and 0.62-0.78 with Q4_0 (0.009 with Q8_0), and the short
  clip's transcript differs ("Hallway English Council" for "Whole Life English
  Council"). Their packages use the Q8_0 mmproj, so the encoder, depth decoder
  and codec match the Q8_0 package. Use Q8_0 or better.
- The published GGUFs store the codec decoder in F16 where the upstream
  checkpoint has BF16 and F32 tensors. In F32 mode the published BF16 GGUF
  gives the upstream weights everywhere else, and its codec waveform is as
  close to the reference as the upstream directory's on CUDA (1.1e-3 rel for
  TTS codes).
- Every turn of a conversation encodes all its questions and prefills the
  whole conversation again; there is no cache across turns.
- Thinking chat (`enable_thinking`) gives the reference's tokens exactly on 4
  clips with F32 on the H100 host's CPU and on CUDA and with the published BF16
  GGUF on CUDA; Q8_0 diverges partway through, at gaps of 0.007-0.15. Over
  HTTP, an empty answer carries no flag; the warning goes only to the server
  log.
- With `language=en`, ASR of the 418 s clip repeats the passage before the end
  of its last VAD pass (1451 words against 1392; 4.3-4.4% WER in
  [Performance](#performance), against 0.07-0.14% with the default `auto`),
  with Q8_0 and F16 on CUDA and Metal. This is the model, not the port: on
  that 122 s pass the upstream model in fp32 with greedy decoding gives the
  same 562 tokens as audio.cpp with F32 on the CPU and with Q8_0 on CUDA. Where
  the passage ends, the model goes on with "We" (logit 18.1) instead of ending
  its turn (12.3). The pass has speech up to the file's last sample.
  `language=auto` avoids it on this clip.
- A long S2S reply can stop speaking before its text ends: in the H100
  performance runs, one story reply's audio stopped after 16.6 s with 59 of
  its 108 words spoken. It is rare, and the upstream model does it too: on the
  same story question with the checkpoint's sampling defaults, 1 of 40
  upstream replies (bf16) and 0 of 40 audio.cpp replies (Q8_0) had a
  text-vs-audio WER above 15% by whisper-large-v3.
- TTS misreads some numbers ("48 dollars" was heard as "$40" on the H100;
  "1,250" was garbled on the M3 Ultra) that LFM2.5-Audio reads correctly.
- LFM3-Audio costs more than LFM2.5-Audio: the Q8_0 package is 2083 MB against
  1738 MB, it loads in 1.1-2.7 s against 0.5-0.7 s, and it takes 1.2-2x the
  peak memory (H100, 418 s ASR: 4719 against 2405 MiB GPU memory; M3 Ultra:
  3594 against 2208 MB).
