# ASR Models

This page covers ASR models that do not have a dedicated page. Qwen3 ASR is documented in [qwen3.md](qwen3.md).

Common CLI shape:

```bash
audiocpp_cli --task asr --family <family> --model <model-dir> --backend cuda --audio <audio.wav> ...
```

When `--mode streaming` is used, the selected model provides its default streaming policy.

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

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. Use 16 kHz WAV for the example path. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Compute backend. |

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
| `--request-option enable_thinking=true|false` | bool | `true` | Enable the model thinking prompt. |
| `--audio-chunk-mode` | `auto`, `fixed`, `none` | `auto` | Long-audio chunking mode. `auto` uses fixed chunks. |
| `--audio-chunk-seconds` | float seconds | `4` | Fixed audio chunk duration. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |

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

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--language` | language code | `da` | Recognition language; can be omitted for the Danish model path. |
| `--request-option punctuation=true|false` | bool | model default | Enable punctuation tokens in the decoder prompt. |
| `--max-tokens` | integer | model default | Maximum generated transcript tokens. |
| `--num-beams` | integer | `1` | Beam-search beam count; `1` uses greedy or sampling decode. |
| `--request-option length_penalty=<float>` | float | model default | Beam-search length penalty. |
| `--do-sample` | bool | `false` | Enable sampling when `--num-beams 1`. |
| `--temperature` | float | model default | Sampling temperature. |
| `--top-k` | integer | model default | Top-k sampling limit; `0` disables top-k. |
| `--top-p` | float | model default | Nucleus sampling limit. |
| `--seed` | integer | random if omitted | Sampling seed. |
| `--audio-chunk-mode` | `auto`, `fixed`, `none` | `auto` | Long-audio chunking mode. `auto` uses the model clip limit and speech-energy boundaries when chunking is needed. |
| `--audio-chunk-seconds` | float seconds | model config | Fixed audio chunk duration. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |

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
| `--request-option keep_language_tags=true|false` | bool | `false` | Keep language tag tokens in decoded text. |
| `--words-out` | JSON path | not set | Write token timestamp output when produced. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--session-option nemotron_asr.mem_saver=true|false` | bool | `false` | Release the offline encoder graph after each offline request. |

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
audiocpp_cli --task asr --family vibevoice_asr --model models/VibeVoice-ASR --backend cuda --audio speech_16k.wav --text-out transcript.txt
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
audiocpp_cli --task asr --family vibevoice_asr --model models/VibeVoice-ASR --backend cuda --audio meeting.wav --text "The recording is a meeting conversation." --text-out transcript.txt --segments-out segments.json --turns-out turns.json
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

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.
