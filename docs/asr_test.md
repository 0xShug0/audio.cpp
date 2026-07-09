Date: 2026-07-08

## Setup

TTS input: `tools/audiocpp_cli/audiocpp_cli_longform_tts_clone_cases.json`
case `qwen3_tts_voice_clone_longform`, request `clone_longform`. The input
text has 6026 characters and 1028 normalized words.

ASR input: `assets/resources/qwen3_tts_longform_asr_input.wav`, a
327.600 second WAV generated once from the TTS input with Qwen3 TTS Base. Future
ASR validation must reuse this audio asset and must not regenerate it during the
ASR test. The current audio ends before the full TTS input tail, so the ASR WER
is checked against a Qwen3-ASR CLI transcript of the current WAV (911 normalized words).

Use `tools/streaming/measure_tts_streaming_client.py` for
streaming TTS measurement; it sends text incrementally as 16 chunks, with a
maximum chunk size of 480 characters. Use
`tools/streaming/measure_asr_streaming_client.py` for streaming ASR measurement;
it sends the audio upload as smaller client-side chunks. VRAM is total sampled
GPU memory used during the request, not process-isolated allocation.

Higgs Audio STT Python reference check:
feeds the full ASR WAV, not only the first 4 seconds. The Python collator
resamples the 327.600 second WAV to 16 kHz, splits it into 82 audio chunks
with `chunk_size_seconds=4.0`, and produces `valid_frames_total=32760`.
Despite receiving the full audio, Python Higgs STT stops early: with
`enable_thinking=true` and `max_tokens=32768`, it emits 122 words; with `enable_thinking=false`, it
emits 90 raw words and is worse.

## TTS Results

| Model | Mode | Memory saver | Request shape | TTFT | Wall time | Audio length | RTF | Peak VRAM |
|---|---|---|---|---:|---:|---:|---:|---:|
| VoxCPM2 | Streaming SSE | On | 16 incremental text chunks | 547.464 ms | 56.648 s | 314.560 s | N/A | 7638 MiB |
| VoxCPM2 | Streaming SSE | Off | 16 incremental text chunks | 308.024 ms | 55.752 s | 314.560 s | N/A | 9091 MiB |

## ASR Results

| Model | Mode | Input audio duration | TTFT | Wall time | RTF | WER | Errors | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Nemotron | Offline | 327.600 s | N/A | 2.173 s | 0.00663454 | 3.18% | 29 (S/D/I 24/5/0) | 8294 MiB |
| Nemotron | Streaming SSE | 327.600 s | 307.591 ms | 11.623 s | N/A | 3.18% | 29 (S/D/I 24/5/0) | 4382 MiB |
| Nemotron | Streaming One-shot | 1800.000 s | N/A | 53.660 s | 0.0298111 | 3.45% | 172 (S/D/I 130/33/9) | 4167 MiB |
| Higgs Audio STT | Offline | 327.600 s | N/A | 11.169 s | 0.0341 | 3.95% | 36 (S/D/I 22/6/8) | 12519 MiB |
| Higgs Audio STT | Streaming SSE | 327.600 s | 468.386 ms | 14.457 s | N/A | 3.95% | 36 (S/D/I 22/6/8) | 6945 MiB |
| VibeVoice ASR | Offline | 327.600 s | N/A | 19.235 s | 0.0587143 | 0.66% | 6 (S/D/I 5/0/1) | 25833 MiB |
| VibeVoice ASR | Offline | 1800.000 s | N/A | 123.720 s | 0.0687333 | 1.51% | 75 (S/D/I 32/30/13) | 31209 MiB |
