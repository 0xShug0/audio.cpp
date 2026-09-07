# MiraTTS validation

This directory follows the long-lived validation pattern used for the OuteTTS
port in PR #63. The model is loaded once and exercised by a sequence containing
cold, repeat, changed-prompt, long-form, and post-long-form repeat requests.
Streaming has a separate long-lived sequence so that offline and streaming
sessions do not duplicate model weights in VRAM.

The benchmark reports one JSON object per request with wall time, generated
duration, real-time factor, sample/frame counts, a deterministic audio hash,
and output path. Streaming runs additionally report first-event latency and the
number of progressively emitted audio events.

## Build

```powershell
cmake -S . -B build/windows-cuda-release `
  -DGGML_CUDA=ON -DENGINE_BUILD_WARMBENCH=ON `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/windows-cuda-release --config Release `
  --target mira_tts_warm_bench -j 8
```

## Native long-lived runs

Set paths once:

```powershell
$bench = "build/windows-cuda-release/bin/mira_tts_warm_bench.exe"
$model = "../models_v3_test/MiraTTS-GGUF/mira-tts-bf16.gguf"
$voice = "../models_v3_test/MiraTTS-comparison/00-reference-voice.wav"
$spec = "model_specs/mira_tts.json"
$out = "build/logs/warmbench/mira_tts"
```

Run the five-case offline sequence:

```powershell
& $bench --model $model --model-spec-override $spec --backend cuda `
  --voice-ref $voice --request-file tests/mira_tts/offline_requests.json `
  --audio-out-dir "$out/native-offline" `
  --summary-file "$out/native-offline.json" `
  --log-file "$out/native-offline.log"
```

Run the repeated streaming sequence:

```powershell
& $bench --model $model --model-spec-override $spec --backend cuda `
  --run-mode streaming --voice-ref $voice `
  --request-file tests/mira_tts/streaming_requests.json `
  --audio-out-dir "$out/native-streaming" `
  --summary-file "$out/native-streaming.json" `
  --log-file "$out/native-streaming.log"
```

Validate deterministic repeats, streaming events, durations, hashes, and WAV
readability (with FFprobe when available):

```powershell
python tests/mira_tts/validate_bench.py `
  --offline-summary "$out/native-offline.json" `
  --streaming-summary "$out/native-streaming.json"
```

For a memory measurement, start the benchmark with `--hold-seconds 30` and
sample the process plus the selected GPU while it is holding the loaded
session. Record both peak host RSS and peak device memory; do not report only
the final idle value.

## Trusted Python sequence

Run the same offline cases through a local, revision-pinned upstream snapshot:

```powershell
python tools/community_models/mira_tts_reference_bench.py `
  --model-dir <verified-upstream-snapshot> --reference $voice `
  --request-file tests/mira_tts/offline_requests.json `
  --output-dir "$out/python-offline" `
  --summary-file "$out/python-offline.json"
```

Compare every matched request. Exact frame count is required for the strongest
deterministic parity run:

```powershell
python tools/community_models/compare_mira_tts_outputs.py `
  --cpp-summary "$out/native-offline.json" `
  --python-summary "$out/python-offline.json" `
  --output "$out/parity.json" --require-exact-frames
```

The default gates are waveform cosine >= 0.95 and log-mel cosine >= 0.95.
Threshold changes must be justified by a saved artifact and must not hide a
frame-count, token-boundary, or sampling mismatch.

## Acceptance matrix

| Check | Required evidence |
|---|---|
| Cold/repeat determinism | `clone_cold`, `clone_repeat`, and `clone_repeat_after_longform` have identical hashes |
| Long-lived lifecycle | All offline requests finish in one process without reloading the model |
| Long form | `longform` produces non-empty audio within its token budget |
| Streaming | More than one event, finite first-event latency, merged non-empty WAV |
| Streaming determinism | `stream_cold` and `stream_repeat` have identical hashes and event counts |
| Python/native parity | Per-case WAV cosine, log-mel cosine, and frame counts are recorded |
| Audio validity | Every WAV is readable by FFmpeg and has the reported sample rate/channels |
| Resource use | Peak RSS, peak GPU memory, wall time, duration, and RTF are recorded |
| Backends | CUDA is required; CPU/Vulkan results or explicit limitations are documented |

Run deterministic checks before experimenting with unseeded sampling. If a
repeat hash changes, inspect generated speech-token boundaries, prompt/context
tokens, and component traces before comparing subjective audio quality.
