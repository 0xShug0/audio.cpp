# Nemotron 3 Diarization Preview

`nemotron_3_diar` provides native audio.cpp inference for
[NVIDIA Nemotron 3 Diarization Preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview).
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
| Weights | Local F32 GGUF conversion |

The checkpoint is distributed under the [NVIDIA Software and Model Evaluation
License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/).
The package spec therefore does not advertise a downloadable GGUF. Request
access to the original checkpoint and convert it locally.

## Convert

Build `audiocpp_gguf`, then convert the original `.nemo` archive:

```bash
python tests/nemotron_3_diar/convert_gguf.py \
  --checkpoint /path/to/Nemotron-3-Diarization-preview.nemo \
  --output-dir /path/to/Nemotron-3-Diarization-preview-GGUF/staging \
  --converter /path/to/audiocpp_gguf \
  --gguf-output /path/to/Nemotron-3-Diarization-preview-GGUF/nemotron-3-diarization-preview-f32.gguf \
  --type orig
```

The converter reads the original NeMo archive, writes canonical Safetensors
staging files, and packages a self-contained GGUF with the v1 model spec.

## Run

Offline:

```bash
audiocpp_cli --task diar \
  --family nemotron_3_diar \
  --model /path/to/nemotron-3-diarization-preview-f32.gguf \
  --backend cuda --audio meeting.wav --turns-out turns.json
```

Native batch inference reads all WAV files in one directory and emits one
speaker-turn file per input:

```bash
audiocpp_cli --task diar \
  --family nemotron_3_diar \
  --model /path/to/nemotron-3-diarization-preview-f32.gguf \
  --backend cuda --batch-audio-dir /path/to/wavs \
  --batch-audio-role audio --turns-out batch/turns.json \
  --batch-manifest-out batch/manifest.json
```

Streaming with an official latency profile:

```bash
audiocpp_cli --task diar --mode streaming \
  --family nemotron_3_diar \
  --model /path/to/nemotron-3-diarization-preview-f32.gguf \
  --backend cuda --audio meeting.wav --turns-out turns.json \
  --session-option nemotron_3_diar.latency_profile=low
```

## Latency Profiles

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
