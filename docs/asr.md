# ASR Models

| Model | Family | Mode(s) | Quick Start |
|---|---|---|---|
| Canary 180M Flash | `canary_asr` | offline | [Canary 180M Flash](models/canary_asr.md) |
| Cohere Transcribe | `cohere_asr` | offline | [Cohere Transcribe](models/cohere_asr.md) |
| Fun-ASR-Nano | `fun_asr_nano` | offline | [Fun-ASR-Nano](#fun-asr-nano) |
| Granite Speech 5.0 TurboCTC | `granite5asr` | offline | [Granite Speech 5.0 TurboCTC](community_models/granite5asr.md) |
| Qwen3 ASR | `qwen3_asr` | offline, streaming | [Qwen3 ASR](#qwen3-asr) |
| Confucius4-R2T2 | `confucius4_r2t2` | offline, streaming | [Confucius4-R2T2](community_models/r2t2.md) |
| Citrinet ASR | `citrinet_asr` | offline | [Citrinet ASR](#citrinet-asr) |
| Kroko Community ASR | `kroko_asr` | offline, streaming | [Kroko Community ASR](#kroko-community-asr) |
| Higgs Audio STT | `higgs_audio_stt` | offline, streaming | [Higgs Audio STT](models/higgs_audio_stt.md) |
| Hviske ASR | `hviske_asr` | offline | [Hviske ASR](#hviske-asr) |
| Moonshine Streaming ASR | `moonshine_asr` | offline, streaming | [Moonshine Streaming ASR](models/moonshine_asr.md) |
| MOSS-Transcribe-Diarize | `moss_transcribe_diarize` | offline, text-output streaming | [MOSS-Transcribe-Diarize](models/moss_transcribe_diarize.md) |
| Nemotron ASR | `nemotron_asr` | offline, streaming | [Nemotron ASR](#nemotron-asr) |
| Niagara ASR | `niagara_asr` | offline | [Niagara ASR](#niagara-asr) |
| Parakeet-TDT | `parakeet_tdt` | offline, streaming | [Parakeet-TDT](#parakeet-tdt) |
| SenseVoice-Small | `sense_asr` | offline, streaming | [SenseVoice-Small](#sensevoice-small) |
| VibeVoice ASR | `vibevoice_asr` | offline | [VibeVoice ASR](models/vibevoice_asr.md#vibevoice-asr) |
| VibeVoice ASR Streaming 7B/1.5B | `vibevoice_asr_streaming` | offline, streaming | [VibeVoice ASR Streaming](models/vibevoice_asr.md#vibevoice-asr-streaming-7b) |
| Voxtral Realtime | `voxtral_realtime` | offline, streaming | [Voxtral Realtime](models/voxtral_realtime.md) |

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

## Confucius4-R2T2

Confucius4-R2T2 is a low-latency append-only streaming ASR model: a Qwen3-ASR
1.7B fine-tune with Longest Stable Prefix (LSP) decoding. Committed text is
never revised, and chunk sizes from 80 ms to 2 s are supported. It runs the
same audio tower as Qwen3 ASR, so only the streaming state machine differs.

```bash
audiocpp_cli --task asr --family confucius4_r2t2 --model models/Confucius4-R2T2 \
  --backend metal --audio speech_16k.wav --text-out transcript.txt
```

```bash
audiocpp_cli --task asr --mode streaming --family confucius4_r2t2 \
  --model models/Confucius4-R2T2 --backend metal --audio speech_16k.wav \
  --session-option confucius4_r2t2.chunk_size_ms=320 --text-out transcript.txt
```

Streaming emits append-only partial text; the final transcript is returned when
the stream ends. See the [Confucius4-R2T2 model guide](community_models/r2t2.md) for the
session options, the LSP state machine, and the MPS golden verification recipe.

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

Higgs Audio STT supports offline and streaming transcription with fixed audio
chunking and configurable encoder/decoder storage. See the dedicated
[Higgs Audio STT guide](models/higgs_audio_stt.md) for GGUF conversion,
commands, options, and compatibility aliases.

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

## Niagara ASR

Niagara provides English offline transcription with 19M and 38M Batch checkpoints
under the same `niagara_asr` family. Select the checkpoint through `--model`.

```bash
audiocpp_cli --task asr --family niagara_asr \
  --model models/Niagara-ASR-GGUF/niagara-19m-batch.en-f32.gguf \
  --backend cpu --threads 8 --audio speech.wav \
  --text-out transcript.txt --log
```

Use `niagara-38m-batch.en-f32.gguf` for the 38M checkpoint. These are offline batch
models; streaming and audio chunking are not supported.

## Nemotron ASR

Nemotron ASR is an NVIDIA Nemotron 3.5 ASR RNNT model with offline and streaming sessions. It supports language prompts and optional token timestamp output.

| Field | Value |
|---|---|
| Family | `nemotron_asr` |
| Model directory | `models/nemotron-3.5-asr-streaming-0.6b` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Output | Transcription text; optional token timestamps through `--words-out` |
| Streaming input | Audio chunks; native cache-aware inference with a 320 ms preferred input cadence by default |
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
| `--request-option speaker_probabilities=<path>` | path | not set | Speaker-tagged output from a `nemotron_3_diar` file; see below. |
| `--request-option speaker_mode=masked\|attribution` | enum | `masked` | How speakers are assigned; see below. |
| `--request-option speaker_mask=mel\|audio` | enum | `mel` | Masked mode only: mask log-mel features, or zero the audio of inactive frames. |
| `--request-option speaker_segment_gap_sec=<s>` | float | `1.0` | Start a new segment for a speaker after a longer pause. |
| `--request-option speaker_segment_max_sec=<s>` | float | `10.0` | Streaming only: split speaker-turn events after this long. |

### Speaker-tagged transcription

Nemotron ASR can label its transcript by speaker with the speaker activity from
[Nemotron 3 Diarization](models/nemotron_3_diar.md). Run the two models one after
the other on the same audio file:

```bash
# 1. Diarize. The asr_la13 profile matches the ASR lookahead of 13.
audiocpp_cli --task diar --family nemotron_3_diar --mode streaming \
  --model models/Nemotron-3-Diarization-GGUF/nemotron-3-diarization-bf16.gguf --audio meeting.wav \
  --session-option nemotron_3_diar.latency_profile=asr_la13 \
  --request-option return_frame_probabilities=true --out-dir diar

# 2. Transcribe by speaker. The lookahead comes from the diarizer file.
audiocpp_cli --task asr --family nemotron_asr --model models/nemotron-3.5-asr-streaming-0.6b \
  --audio meeting.wav --request-option speaker_probabilities=diar/speaker_probabilities.safetensors \
  --turns-out turns.json --out-dir asr
```

Both commands also run with `--mode streaming`. The ASR step reads the
diarizer file as it goes, so streaming uses a precomputed file, not a live
diarizer.

There are two modes:

- `masked` (default) runs one ASR stream per active speaker. Each stream hears
  only its speaker: the features of the other frames are masked, as in NeMo's
  masked multitalker ASR. Overlapping speech is transcribed for each speaker.
  It costs more compute when several speakers are active.
- `attribution` transcribes the audio once and gives each word to the speaker
  with the highest activity around it. It cannot transcribe two speakers at
  once, but it is cheaper and accepts a file from any diarizer profile;
  `very_high` gives the best diarization.

In masked mode the masks are exact only when the diarizer ran with the profile
that matches the ASR lookahead:

| ASR lookahead | Diarizer profile |
|---:|---|
| 13 | `asr_la13` |
| 6 | `asr_la6` |
| 3 | `asr_la3` |
| 0 | `asr_la0` |

The ASR reads the lookahead from the file. An explicit `lookahead_tokens` wins.
If the file does not match a lookahead (for example `very_high`), masked mode
uses 13. Both cases print a warning. A smaller lookahead lowers latency and
accuracy.

Outputs:

- speaker turns with text (`--turns-out`), labelled `speaker_0`, `speaker_1`, and
  so on, like the diarizer;
- a `seglst` artifact (`seglst.json` with `--out-dir`) in the SegLST format used
  by NeMo and meeteval: `session_id`, `speaker`, `start_time`, `end_time`, `words`;
- in masked mode, the transcript text as `speaker_k: words` lines: in start
  time order offline, in the order segments finish when streaming.

A segment ends when its speaker pauses for more than `speaker_segment_gap_sec`.
In streaming, finished segments arrive as speaker-turn events and never change
afterwards. A segment longer than `speaker_segment_max_sec` arrives as several
events. In masked mode each event also carries its lines as partial text, so
the server's `transcript.text.delta` events are append-only and add up to the
final transcript. The final speaker turns and SegLST keep the full segments in
start time order.

Validation: on four AMI test meetings (92 min, 14,837 words), masked mode
scores 32.64% cpWER against 32.61% for NeMo Python's masked pipeline, and
attribution scores 35.73%. With NeMo's own diarizer output as input, masked
mode reproduces NeMo's SegLST segment for segment on CPU.

Limitations:

- The diarizer file must describe the same audio: `floor(samples / 160)` frames.
- Offline attribution uses the offline encoder graph, which needs a lot of
  memory for long files. Use `--mode streaming` for long recordings.
- A speaker's stream pauses while that speaker is silent, so the last token of
  a segment (often a punctuation mark or the end of a word) can arrive only when
  the speaker talks again. The final speaker turns and SegLST add it to the
  earlier segment; the streamed event and text were already sent without it.
  On a 1 h 48 min recording this affected 2% of the words.

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

VibeVoice ASR covers the original offline ASR family and the Streaming family.
The streaming family has its own dedicated GGUF repos and supports both offline
and live streaming transcription, in a 7B and a 1.5B size that share one loader.
The streaming models can emit speaker-attributed text, but they do not produce
timestamped segments.

See [VibeVoice ASR models](models/vibevoice_asr.md) for package IDs, conversion
notes, CLI examples, live server configuration, and request options.

## Voxtral Realtime

Voxtral Realtime supports offline transcription, stateful live streaming, raw
PCM input, and bidirectional HTTP streaming. See the dedicated
[Voxtral Realtime guide](models/voxtral_realtime.md) for CLI and server usage,
streaming behavior, throughput guidance, and options.
