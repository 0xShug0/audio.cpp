# Nemotron 3 Diarization

`nemotron_3_diar` provides native audio.cpp inference for
[NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization).
It identifies up to eight speakers by arrival order and supports offline,
native-batch, and continuous streaming inference.

## Model

| Field | Value |
|---|---|
| Family | `nemotron_3_diar` |
| Task | `diar` |
| Modes | `offline`, `streaming` |
| Input | 16 kHz WAV audio |
| Output | Speaker turns through `--turns-out` |
| Weights | BF16 GGUF (default), Q8_0 GGUF |

The checkpoint is distributed under [OpenMDW-1.1](https://openmdw.ai/license/1-1/).
GGUF weights are packaged in [Nemotron-3-Diarization-GGUF](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF).

## Convert

Build `audiocpp_gguf`, then convert the original `.nemo` archive:

```bash
python tests/nemotron_3_diar/convert_gguf.py \
  --checkpoint /path/to/Nemotron-3-Diarization.nemo \
  --output-dir /path/to/staging \
  --converter /path/to/audiocpp_gguf \
  --gguf-output /path/to/Nemotron-3-Diarization-GGUF/nemotron-3-diarization-bf16.gguf \
  --type orig
```

The converter reads the original NeMo archive, writes canonical Safetensors
staging files, and packages a self-contained GGUF with the v1 model spec.

## Run

Offline:

```bash
audiocpp_cli --task diar \
  --family nemotron_3_diar \
  --model /path/to/nemotron-3-diarization-bf16.gguf \
  --backend cuda --audio meeting.wav --turns-out turns.json
```

The server exposes native batch inference through repeated multipart files:

```bash
curl -N http://127.0.0.1:8080/v1/batches/transcriptions \
  -F model=nemotron-3-diar \
  -F file=@/path/to/meeting-a.wav \
  -F file=@/path/to/meeting-b.wav
```

The SSE response emits each file's `speaker_turns` as soon as that result is
ready. Use its `index` field to map results back to upload order.

Streaming with an official latency profile:

```bash
audiocpp_cli --task diar --mode streaming \
  --family nemotron_3_diar \
  --model /path/to/nemotron-3-diarization-bf16.gguf \
  --backend cuda --audio meeting.wav --turns-out turns.json \
  --session-option nemotron_3_diar.latency_profile=low
```

## Latency Profiles

In server streaming mode, use `/v1/audio/transcriptions/live` for live PCM or
`/v1/audio/transcriptions` with `stream=true` for a WAV upload. Configure the model
with `task: "diar"`, `mode: "streaming"`, and the desired latency profile in
`session_options`. See the [server streaming API](../../app/server/README.md#post-v1audiotranscriptionslive)
for request and response formats.

Streaming emits completed speaker turns. An ongoing turn appears when it ends
or when the input stream closes, so time to the first turn also depends on the
speech itself. The final response contains all turns, including the last open
turn. The buffer latencies below describe model input buffering, not time to a
completed speaker turn.

All geometry values use 80 ms encoder frames. Input-buffer latency is
`chunk_len + chunk_right_context`.

| Profile | Buffer latency | Speaker cache | FIFO | Chunk | Right context | Cache update |
|---|---:|---:|---:|---:|---:|---:|
| `very_high` | 30.4 s | 264 | 40 | 340 | 40 | 300 |
| `low` | 1.04 s | 264 | 264 | 9 | 4 | 222 |
| `very_low` | 0.64 s | 264 | 264 | 6 | 2 | 222 |
| `ultra_low` | 0.32 s | 264 | 264 | 3 | 1 | 222 |

Use `custom` to set the five geometry controls directly.

## Common Options

| Option | Value | Default | Description |
|---|---|---:|---|
| `--audio` | WAV path | required | Input audio for one request. |
| `--turns-out` | JSON path | not set | Save decoded speaker turns. |
| `--mode` | `offline`, `streaming` | `offline` | Select bounded offline or incremental streaming execution. |
| `--batch-audio-dir` | directory | not set | Submit all WAV files through native offline batching. |

## Request Options

Use these with `--request-option`.

| Option | Value | Default | Description |
|---|---|---:|---|
| `speaker_threshold` | `0.0` to `1.0` | `0.5` | Speaker activity threshold. |
| `speaker_min_frames` | integer >= 0 | `0` | Minimum turn duration in 10 ms output frames. |
| `speaker_pad_frames` | integer >= 0 | `0` | Padding around turns in 10 ms output frames. |
| `return_frame_probabilities` | `true`, `false` | `false` | Attach the raw 10 ms speaker-activity timeline as the `speaker_probabilities` artifact: little-endian F32, `[frames, 8]`. With `--out-dir`, the CLI writes it to `speaker_probabilities.f32`. |

## Session Options

Use these with `--session-option nemotron_3_diar.<name>=<value>`.

| Option | Value | Default | Description |
|---|---|---:|---|
| `latency_profile` | `very_high`, `low`, `very_low`, `ultra_low`, `custom` | `very_high` | Streaming geometry preset. |
| `spkcache_len` | integer >= 16 | checkpoint value | Speaker-cache length for `custom`. |
| `fifo_len` | integer >= 0 | checkpoint value | FIFO length for `custom`. |
| `chunk_len` | integer >= 1 | checkpoint value | Processing chunk length for `custom`. |
| `chunk_right_context` | integer >= 0 | checkpoint value | Future context for `custom`. |
| `spkcache_update_period` | integer >= 1 | checkpoint value | FIFO-to-cache update period for `custom`. |
| `graph_arena_mb` | integer >= 1 | `1024` | GGML graph metadata arena size. |
| `weight_context_mb` | integer >= 1 | `1024` | GGML weight metadata arena size. |
| `weight_type` | storage type | `native` | Runtime weight storage type. |
| `attention` | `auto`, `flash`, `eager` | `auto` | Encoder attention lowering. `auto` uses flash attention on GPUs that support it and eager attention on others, such as Volta and Turing. |

## Validation

The output was compared with NeMo Python (`SortformerEncLabelModel.forward_streaming`,
FP32, CPU, NVIDIA-NeMo/Speech `cf724ac`) and with NVIDIA's C++ port
(NeMo-Speech.cpp `8c15060`), using the same `.nemo` checkpoint
(SHA-256 `867c53f5...`). All numbers use the native 10 ms speaker
probabilities.

| Case | Build | Max \|Δ\| vs NeMo | Mean \|Δ\| | Frame flips at 0.5 |
|---|---|---:|---:|---:|
| AMI EN2002d 20 s (319,990 samples), `very_high` | CPU, F32 weights | 8.7e-4 | 2.1e-5 | 0 |
| AMI EN2002d 20 s (319,990 samples), `low` | CPU, F32 weights | 1.6e-3 | 2.6e-5 | 1 |
| Same clip, NVIDIA C++ port, `very_high` | CPU, F32 | 1.1e-1 | 9.4e-4 | 2 |

The model is sensitive to tiny input changes in long streams. On a 120 s
meeting with the `low` profile, adding inaudible noise (σ = 1e-5) to the
input changes the NeMo output by 1.1% DER-equivalent and this port's output
by 3.2%. This port and NeMo differ by 3.6–4.2% on the same file, which is the
same order. NVIDIA's C++ port shows the same difference. Against NeMo on that
file, every build falls in one band: CPU F32 4.2%, CUDA F32 4.4%, CUDA BF16
5.1%, and CUDA Q8_0 4.3%. Q8_0 therefore shows no quality cliff.

On CUDA, flash attention converts K and V to F16. Compared with the CPU F32
output, eager attention reduces the difference from 74 to 2 flipped frames
on the 120 s stream and costs about 35% more time (900 s of audio in 16.8 s
instead of 12.5 s on an RTX 4070 Ti SUPER). The default stays `auto`.

Known deviation: NeMo pads mel features to a multiple of 16 frames, and
its subpixel convolution reads those padded rows at the last real 80 ms
frame. This port uses zero padding there, so the final 70 ms of a file can
differ by up to 0.02.
