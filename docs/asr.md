# ASR Models

| Model | Family | Mode(s) | Quick Start |
|---|---|---|---|
| Fun-ASR-Nano | `fun_asr_nano` | offline | [Fun-ASR-Nano](#fun-asr-nano) |
| Qwen3 ASR | `qwen3_asr` | offline, streaming | [Qwen3 ASR](#qwen3-asr) |
| Citrinet ASR | `citrinet_asr` | offline | [Citrinet ASR](#citrinet-asr) |
| Kroko Community ASR | `kroko_asr` | offline, streaming | [Kroko Community ASR](#kroko-community-asr) |
| Higgs Audio STT | `higgs_audio_stt` | offline, streaming | [Higgs Audio STT](#higgs-audio-stt) |
| Hviske ASR | `hviske_asr` | offline | [Hviske ASR](#hviske-asr) |
| Nemotron ASR | `nemotron_asr` | offline, streaming | [Nemotron ASR](#nemotron-asr) |
| Parakeet-TDT | `parakeet_tdt` | offline, streaming | [Parakeet-TDT](#parakeet-tdt) |
| SenseVoice-Small | `sense_asr` | offline, streaming | [SenseVoice-Small](#sensevoice-small) |
| VibeVoice ASR | `vibevoice_asr` | offline | [VibeVoice ASR](#vibevoice-asr) |
| Voxtral Realtime | `voxtral_realtime` | offline, streaming | [Voxtral Realtime](#voxtral-realtime) |

This page covers ASR models. Detailed Qwen3 ASR and forced-alignment notes live in [Qwen3 models](models/qwen3.md).

Common CLI shape:

```bash
audiocpp_cli --task asr --family <family> --model <model-dir> --backend cuda --audio <audio.wav> ...
```

When `--mode streaming` is used, the selected model provides its default streaming policy.

## Fun-ASR-Nano

Fun-ASR-Nano provides offline multilingual transcription for Chinese, English,
and Japanese with automatic language selection. The recommended package is the
standalone Q8_0 GGUF published by FunAudioLLM.

```bash
python3 tools/model_manager_v2.py install fun_asr_nano
audiocpp_cli --task asr --family fun_asr_nano \
  --model models/Fun-ASR-Nano-2512-GGUF/fun-asr-nano-2512-q8_0.gguf \
  --backend cuda --audio speech_16k.wav --text-out transcript.txt
```

The runtime supports fixed offline chunking and inverse text normalization.
Streaming and timestamp output are not exposed. See the
[Fun-ASR-Nano model guide](models/fun_asr_nano.md) for package, option, GGUF,
and server details.

## Qwen3 ASR

Qwen3 ASR transcribes speech and can be paired with Qwen3 Forced Aligner when timestamps are needed. Streaming mode accepts live audio chunks and emits buffered transcript deltas; timestamp output remains an offline path. See [Qwen3 models](models/qwen3.md) for the full ASR and alignment manual.

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-1.7B-hf --backend cuda --audio speech_16k.wav --text-out transcript.txt
```

```bash
audiocpp_cli --task asr --mode streaming --family qwen3_asr --model models/Qwen3-ASR-1.7B-hf --backend cuda --audio speech_16k.wav --request-option audio_chunk_seconds=5 --text-out transcript.txt
```

## Citrinet ASR

Citrinet is an offline CTC ASR model. It produces transcription text from speech audio.

| Field | Value |
|---|---|
| Family | `citrinet_asr` |
| Model directory | `models/citrinet` |
| Task | `asr` |
| Modes | `offline` |
| Output | Transcription text |
| Streaming | Not exposed |

```bash
audiocpp_cli --task asr --family citrinet_asr --model models/citrinet --backend cuda --audio speech_16k.wav
```

Create a standalone Q8_0 GGUF from the converted Citrinet safetensors layout:

```powershell
audiocpp_gguf.exe --input models\citrinet\citrinet_256.safetensors --root models\citrinet --output models\citrinet-Q8_0\model.gguf --type q8_0
```

The GGUF embeds `citrinet_256_config.json` and the vocabulary/tokenizer sidecars, so the
completed `model.gguf` can be moved, renamed, and passed directly to `--model`.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. Use 16 kHz WAV for the example path. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Compute backend. |

## Kroko Community ASR

Kroko Community ASR is a Zipformer2/RNN-T model port maintained in
`community_models`. audio.cpp runs its feature frontend, encoder, predictor,
joiner, greedy search, and modified beam search natively without ONNX Runtime.
Blank penalty, natural-text hotwords, and opt-in endpoint segmentation are
available as request options. Public free packages
are available for German, English, Spanish, French, Italian, Hebrew, Dutch,
Portuguese, Swedish, and Turkish. The model manager defaults to the standalone
English Q8_0 GGUF package:

```powershell
python .\tools\model_manager_v2.py install kroko_asr_community_q8_0 --models-root .\models --overwrite
```

```powershell
.\build\windows-cuda-release\bin\audiocpp_cli.exe `
  --task asr --mode streaming --family kroko_asr `
  --model .\models\Kroko-ASR-GGUF\kroko-en-community-64-l-q8_0.gguf `
  --backend cuda --audio .\speech.wav --language en `
  --text-out .\transcript.txt --words-out .\words.json
```

Standalone Q8 GGUF is supported in offline and stateful streaming modes. Partial
transcripts and word timestamps are exposed.
See [Kroko Community ASR](community_models/kroko_asr.md) for package selection,
conversion, GGUF, decoding options, parity, performance, and limitation details.

## Higgs Audio STT

Higgs Audio STT is an ASR model for Higgs Audio v3 STT assets. Offline mode can split long audio before inference. Streaming mode consumes audio chunks and emits partial text for each processed chunk.

| Field | Value |
|---|---|
| Family | `higgs_audio_stt` |
| Model directory | `models/higgs-audio-v3-stt` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Output | Transcription text |
| Streaming input | Audio chunks; preferred chunk duration is 4 seconds |
| Timestamps | Not exposed |

Offline:

```bash
audiocpp_cli --task asr --family higgs_audio_stt --model models/higgs-audio-v3-stt --backend cuda --audio speech_16k.wav --text "Transcribe the speech." --text-out transcript.txt
```

Standalone Q8_0 GGUF conversion uses the two-shard index. Map the shared Whisper
preprocessor configuration into the GGUF so the original directory layout is not required:

```powershell
audiocpp_gguf.exe --input models\higgs-audio-v3-stt\model.safetensors.index.json --root models\higgs-audio-v3-stt --sidecar models\whisper-large-v3\preprocessor_config.json=preprocessor_config.json --output models\higgs-audio-v3-stt-Q8_0\model.gguf --type q8_0
```

The shared `whisper-large-v3/preprocessor_config.json` is required only as an input while
creating the GGUF. Once embedded, the resulting GGUF can be moved to an unrelated
directory, renamed, and passed directly to `--model`; the external Whisper file and
directory are no longer required at runtime.

Streaming:

```bash
audiocpp_cli --task asr --family higgs_audio_stt --model models/higgs-audio-v3-stt --backend cuda --mode streaming --audio speech_16k.wav --text "Transcribe the speech." --text-out transcript.txt
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--text` | text | empty string | Prompt/context text for the ASR request. |
| `--language` | language code | model default (`en`) | Recognition language hint. |
| `--mode` | `offline`, `streaming` | `offline` | Full-context or streaming session. |
| `--max-tokens` | integer | model default | Maximum generated transcript tokens. |
| `--request-option enable_thinking=true\|false` | bool | `true` | Enable the model thinking prompt. |
| `--audio-chunk-mode` | `auto`, `fixed`, `none` | `auto` | Long-audio chunking mode. `auto` uses fixed chunks. |
| `--request-option audio_chunk_duration_sec=<seconds>` / `--audio-chunk-seconds` | float seconds | `4` | Fixed audio chunk duration. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--session-option higgs_audio_stt.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared text decoder weight storage type. |
| `--session-option higgs_audio_stt.audio_encoder_weight_type=<type>` | `native`, `f32`, `f16` | `native` | Audio encoder convolution weight storage type. |
| `--session-option higgs_audio_stt.text_decoder_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `higgs_audio_stt.weight_type` or `native` | Text decoder matmul weight storage type. |

Compatibility aliases are applied before v1 option validation:

| Legacy request option | v1 request option |
|---|---|
| `audio_chunk_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration` | `audio_chunk_duration_sec` |

| Legacy session option | v1 session option |
|---|---|
| `weight_type` | `higgs_audio_stt.weight_type` |
| `audio_encoder_weight_type` | `higgs_audio_stt.audio_encoder_weight_type` |
| `text_decoder_weight_type` | `higgs_audio_stt.text_decoder_weight_type` |
| `audio_encoder_graph_arena_mb` | `higgs_audio_stt.audio_encoder_graph_arena_mb` |
| `text_decoder_prefill_graph_arena_mb` | `higgs_audio_stt.text_decoder_prefill_graph_arena_mb` |
| `text_decoder_decode_graph_arena_mb` | `higgs_audio_stt.text_decoder_decode_graph_arena_mb` |
| `text_decoder_weight_context_mb` | `higgs_audio_stt.text_decoder_weight_context_mb` |

## Hviske ASR

Hviske ASR is an offline Cohere ASR model path. The integration exposes Danish prompt controls, punctuation control, greedy/sampling decode, beam search, and model-side audio chunking.

| Field | Value |
|---|---|
| Family | `hviske_asr` |
| Model directory | `models/hviske-v5.3` |
| Task | `asr` |
| Modes | `offline` |
| Output | Transcription text |
| Streaming | Not exposed |
| Timestamps | Not exposed |

```bash
audiocpp_cli --task asr --family hviske_asr --model models/hviske-v5.3 --backend cuda --audio speech_16k.wav --text-out transcript.txt
```

Create a standalone Q8_0 GGUF:

```powershell
audiocpp_gguf.exe --input models\hviske-v5.3\model.safetensors --root models\hviske-v5.3 --output models\hviske-v5.3-Q8_0\model.gguf --type q8_0
```

Configuration, generation settings, and the SentencePiece tokenizer are embedded. The
completed GGUF can therefore be moved, renamed, and passed directly to `--model`.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--language` | language code | `da` | Recognition language; can be omitted for the Danish model path. |
| `--request-option punctuation=true\|false` | bool | model default | Enable punctuation tokens in the decoder prompt. |
| `--max-tokens` | integer | model default | Maximum generated transcript tokens. |
| `--num-beams` | integer | `1` | Beam-search beam count; `1` uses greedy or sampling decode. |
| `--request-option length_penalty=<float>` | float | model default | Beam-search length penalty. |
| `--do-sample` | bool | `false` | Enable sampling when `--num-beams 1`. |
| `--temperature` | float | model default | Sampling temperature. |
| `--top-k` | integer | model default | Top-k sampling limit; `0` disables top-k. |
| `--top-p` | float | model default | Nucleus sampling limit. |
| `--seed` | integer | random if omitted | Sampling seed. |
| `--audio-chunk-mode` | `auto`, `fixed`, `none` | `auto` | Long-audio chunking mode. `auto` uses the model clip limit and speech-energy boundaries when chunking is needed. |
| `--request-option audio_chunk_duration_sec=<seconds>` | float seconds | model config | Fixed audio chunk duration. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |

Compatibility aliases for existing requests:

| Legacy option | Current option |
|---|---|
| `audio_chunk_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration_seconds` | `audio_chunk_duration_sec` |
| `audio_chunk_duration` | `audio_chunk_duration_sec` |

## Nemotron ASR

Nemotron ASR is an NVIDIA Nemotron 3.5 ASR RNNT model with offline and streaming sessions. It supports language prompts and optional token timestamp output.

| Field | Value |
|---|---|
| Family | `nemotron_asr` |
| Model directory | `models/nemotron-3.5-asr-streaming-0.6b` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Output | Transcription text; optional token timestamps through `--words-out` |
| Streaming input | Audio chunks; preferred chunk size is one second at the model sample rate |
| Timestamps | Token timestamps |

Offline:

```bash
audiocpp_cli --task asr --family nemotron_asr --model models/nemotron-3.5-asr-streaming-0.6b --backend cuda --audio speech_16k.wav --language en-US --text-out transcript.txt
```

Nemotron 3.5 ASR also accepts audio.cpp-native GGUF checkpoints. The converter
embeds its configuration, processor metadata, and tokenizer by default, so the
converted directory may contain only `model.gguf`:

```powershell
audiocpp_gguf.exe --input models\nemotron-3.5-asr-streaming-0.6b\model.safetensors --output models\nemotron-3.5-asr-streaming-0.6b-Q8_0\model.gguf --type q8_0
```

Streaming:

```bash
audiocpp_cli --task asr --family nemotron_asr --model models/nemotron-3.5-asr-streaming-0.6b --backend cuda --mode streaming --audio speech_16k.wav --language en-US --text-out transcript.txt
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--language` | language code, `auto` | model default | ASR prompt language such as `en-US`, `da-DK`, or `auto`. |
| `--mode` | `offline`, `streaming` | `offline` | Full-context or streaming session. |
| `--request-option lookahead_tokens=<n>` | integer | model default | Chunk-limited encoder right context. |
| `--max-tokens` | integer | model-derived limit | Maximum RNNT generated tokens; `0` uses the model-derived limit. |
| `--request-option keep_language_tags=true\|false` | bool | `false` | Keep language tag tokens in decoded text. |
| `--words-out` | JSON path | not set | Write token timestamp output when produced. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--session-option nemotron_asr.mem_saver=true\|false` | bool | `false` | Release the offline encoder graph after each offline request. |

## Parakeet-TDT

Parakeet-TDT is a FastConformer-TDT ASR model for multilingual offline,
long-form, and buffered-streaming transcription. The model manager defaults to
the standalone Q8_0 GGUF package.

```bash
python3 tools/model_manager_v2.py install parakeet_tdt_q8_0 --models-root models
audiocpp_cli --task asr --family parakeet_tdt \
  --model models/Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf \
  --backend cuda --audio speech_16k.wav --text-out transcript.txt
```

Use `parakeet_tdt_f16` for the F16 GGUF variant. See
[Parakeet-TDT 0.6B v3](community_models/parakeet_tdt.md) for long-form,
streaming, conversion, options, validation, and performance details.

## SenseVoice-Small

SenseVoice-Small is a community multilingual ASR port with offline and buffered
streaming sessions, language/event/emotion tags, and optional inverse text
normalization. The recommended package is the standalone Q8 GGUF from
FunAudioLLM.

```bash
audiocpp_cli --task asr --family sense_asr \
  --model models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cuda --audio speech_16k.wav --text-out transcript.txt
```

Streaming:

```bash
audiocpp_cli --task asr --mode streaming --family sense_asr \
  --model models/SenseVoice-Small-GGUF/sensevoice-small-q8-audiocpp-v1.gguf \
  --backend cuda --audio speech_16k.wav \
  --request-option audio_chunk_duration_sec=5 --text-out transcript.txt
```

See [SenseVoice-Small](community_models/sense_asr.md) for language tags,
chunking, server usage, and validation notes.

## VibeVoice ASR

VibeVoice ASR is an offline ASR model with greedy, sampling, and beam-search decode paths. It can return transcription text and structured segment/speaker-turn output when the model produces timestamps.

| Field | Value |
|---|---|
| Family | `vibevoice_asr` |
| Model directory | `models/VibeVoice-ASR` |
| Task | `asr` |
| Modes | `offline` |
| Required tokenizer files | `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt` in the model directory |
| Output | Transcription text; optional segments through `--segments-out`; optional speaker turns through `--turns-out` |
| Streaming | Not supported |
| Timestamps | Segment and speaker-turn timestamps when produced |

```bash
audiocpp_cli --task asr --family vibevoice_asr --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf --backend cuda --audio assets/resources/sample_16k.wav --text-out transcript.txt
```

VibeVoice-ASR also accepts a standalone audio.cpp-native GGUF. Pass the shard
index to merge all eight safetensors files while converting:

```powershell
audiocpp_gguf.exe --input models\VibeVoice-ASR\model.safetensors.index.json --output models\VibeVoice-ASR-Q8_0\model.gguf --type q8_0
```

Configuration and tokenizer assets are embedded by default, so the output
directory may contain only `model.gguf`.

Structured output:

```bash
audiocpp_cli --task asr --family vibevoice_asr --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf --backend cuda --audio meeting.wav --text "The recording is a meeting conversation." --text-out transcript.txt --segments-out segments.json --turns-out turns.json
```

With VAD chunking, provide the bundled Silero VAD model:

```bash
audiocpp_cli --task asr --family vibevoice_asr --model models/VibeVoice-ASR-GGUF/vibevoice-asr-q8_0.gguf --backend cuda --audio assets/resources/sample_16k.wav --audio-chunk-mode vad --session-option vibevoice_asr.vad_model_path=assets/framework/models/silero_vad --text-out transcript.txt
```

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

## Voxtral Realtime

Voxtral Realtime is a Mistral realtime ASR model with offline and streaming sessions. The model manager installs the Q8_0 standalone GGUF package by default; native Hugging Face directories and other standalone GGUF variants can also be used when provided directly. A Q4_K GGUF package is also available for lower memory use and faster CUDA runs; in a quick path check its transcripts matched Q8_0 except for one capitalization-only difference.

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

Offline CLI:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --audio assets/resources/sample.wav --text-out transcript.txt
```

Sampling and token-cap options can be passed through request options:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --audio assets/resources/sample.wav --text-out transcript.txt --request-option max_new_tokens=256 --do-sample false --temperature 1.0 --top-p 1.0 --top-k 50 --seed 1234
```

Streaming CLI:

```bash
audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --threads 8 --mode streaming --audio assets/resources/sample.wav --text-out transcript.txt
```

Live streaming input. `--audio -` reads raw (headerless) interleaved PCM from stdin and feeds it
to the model chunk by chunk as it arrives, so the audio is never buffered up front and does not
have to exist as a file. Any capture tool that can write PCM to a pipe works as the source:

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

Stdin input requires `--mode streaming`, and the PCM format must be described up front because a
live stream carries no header — the defaults (`s16le`, 16 kHz, mono) match what the model expects.
The chosen interpretation is echoed back as an `audio_input=stdin` line.

Each update carries only the text decoded since the last one, matching the other streaming ASR
models, so the updates concatenate into the transcript. On a terminal they are appended unlabelled
and the transcript scrolls like ordinary output. When stdout is redirected, each update is written
as its own `partial_text=` line and flushed as it is produced, so pipes and logs stay parseable.
The complete transcript is also printed once at the end as `text_output=`.

An update covers one decoded chunk, so `stream_batch_tokens=<n>` reports every `n`th token's worth
of text in a single update rather than making the updates `n` times shorter. Whatever the batch
size, concatenating the updates reproduces `text_output=` exactly.

Emitting deltas rather than restating the transcript matters for long runs, where the restated form
is quadratic in the transcript length: a one-hour session writes roughly 364 MB restated against
about 54 KB as deltas.

To capture the transcript itself rather than the update stream, use `--text-out`, which writes the
complete transcript and nothing else:

```bash
ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le - \
  | audiocpp_cli --task asr --family voxtral_realtime --model models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf --backend cuda --mode streaming --audio - --text-out transcript.txt
```

`--text-out` and the `text_output=` line are both written when the stream ends, so a session that is
interrupted leaves neither. The `partial_text=` lines are flushed as they are produced, so a log of
them survives an interrupted run and concatenates back into the transcript:

```bash
grep '^partial_text=' session.log | sed 's/^partial_text=//' | tr -d '\n' > transcript.txt
```

### Live PCM over HTTP

The same live source is available to an HTTP client through
`POST /v1/audio/transcriptions/live`: raw PCM goes up in a chunked request body while transcript
deltas come back as SSE on the same connection. This is the server equivalent of `--audio -`, and
the only way to get capture-time partials without the CLI. See
[the server README](../app/server/README.md) for parameters and examples.

```bash
ffmpeg -f alsa -i default -ar 16000 -ac 1 -f s16le - \
  | curl -N -X POST -H 'Expect:' -T - \
      'http://127.0.0.1:8080/v1/audio/transcriptions/live?model=voxtral-realtime'
```

Use `-T -`, not `--data-binary @-` — the latter reads stdin to EOF before it connects, so a live
capture would be uploaded as a finished file and no partial could arrive early.

Whether text appears while the speaker is still talking depends on the model's streaming policy
rather than on the transport. `voxtral_realtime` decodes as audio arrives and emits throughout the
utterance; `nemotron_asr` consumes the full utterance in its encoder first, so its deltas arrive
only once the audio ends. Both are supported here — the difference is what the transcript looks
like mid-sentence.

> **Throughput.** A streaming step always advances 80 ms of audio, so a step has to cost under
> 80 ms to keep up with a realtime source. Measured on an Apple M3 Air (Metal, q8_0):
>
> | Config | short clip, cool | sustained 7 min |
> |---|---:|---:|
> | default | 78 ms/step (0.98x) | 88 ms/step (1.10x) |
> | `stream_batch_tokens=4` | 60 ms/step (0.76x) | 74 ms/step (0.92x) |
>
> The default splits roughly 48 ms for the text decoder and 30 ms for the audio encoder; batching
> takes the encoder to ~13 ms. The second column is what a long session actually gets on a fanless
> machine: a short clip run immediately after the 7-minute one still measured 88 ms/step, so the
> gap is the machine staying warm rather than anything that resets between sessions. Budget for the
> sustained column, and prefer `stream_batch_tokens=4` if the source is realtime.
>
> The decoder runs one step per 80 ms whether the audio holds speech or silence, so a session that
> does fall behind stays behind — the lag is monotonic and does not recover during pauses. Measure
> your own hardware before relying on a live source.

Streaming session options:

| Option | Default | Meaning |
|---|---:|---|
| `--session-option voxtral_realtime.stream_batch_tokens=<n>` | `1` | Audio tokens per encoder forward. The decoder still runs one step per 80 ms; batching only amortizes the encoder's fixed per-forward cost, which dominates it. `4` takes the encoder from ~30 to ~13 ms/step, at the price of delaying every partial by up to `n * 80 ms`. |
| `--session-option voxtral_realtime.stream_decode_cache_steps=<n>` | `1024` | Decoder KV cache size in 80 ms steps (~82 s of context). Built once when the stream starts, so a long session never stalls on a cache-growth rebuild; the cache ring then wraps in place, and a 7-minute stream stays coherent across five wraparounds. Lower values trade context for memory, not for speed. |

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

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path or `-` | required | Speech input. `-` streams raw PCM from stdin and requires `--mode streaming`. |
| `--input-format` | `s16le`, `f32le` | `s16le` | Sample format of raw PCM read from stdin. Ignored for file input. |
| `--input-rate` | integer Hz | `16000` | Sample rate of raw PCM read from stdin. Ignored for file input. |
| `--input-channels` | integer | `1` | Channel count of raw PCM read from stdin. Ignored for file input. |
| `--mode` | `offline`, `streaming` | `offline` | Full-context or streaming session. |
| `--request-option max_new_tokens=<n>` | integer | model-derived limit | Maximum generated transcript tokens. |
| `--do-sample` | bool | `false` | Enable sampling instead of greedy decode. |
| `--temperature` | float | `1.0` | Sampling temperature. |
| `--top-p` | float | `1.0` | Nucleus sampling limit. |
| `--top-k` | integer | `50` | Top-k sampling limit; `0` disables top-k. |
| `--seed` | integer | `1234` | Sampling seed. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--session-option voxtral_realtime.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q4_0`, `q4_k`, `q5_k`, `q6_k`, `q8_0` | `native` | Shared matmul weight storage type. |
| `--session-option voxtral_realtime.audio_encoder_weight_type=<type>` | same as above | shared setting | Audio encoder matmul weight storage type. Leave at `native` for streaming: the encoder is not bandwidth-bound there, so quantizing it makes it slower. |
| `--session-option voxtral_realtime.text_decoder_weight_type=<type>` | same as above | shared setting | Text decoder matmul weight storage type. `q4_k` roughly halves the streaming decoder step cost. |
| `--session-option voxtral_realtime.audio_encoder_graph_arena_mb=<n>` | MB | `512` | Audio encoder graph arena size. |
| `--session-option voxtral_realtime.text_decoder_prefill_graph_arena_mb=<n>` | MB | `512` | Text decoder prefill graph arena size. |
| `--session-option voxtral_realtime.text_decoder_decode_graph_arena_mb=<n>` | MB | `512` | Text decoder cached-step graph arena size. |

Weight storage types are applied when the model loads, so asking for one the GGUF does not already
hold means requantizing on the CPU before the first token appears — around three minutes for
`q4_k` from the shipped q8_0 package. Prefer the published `q4_k` GGUF variant, which needs no
load-time conversion. See [GGUF](gguf.md).

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.
