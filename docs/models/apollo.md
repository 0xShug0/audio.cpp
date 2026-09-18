# Apollo

Apollo restores compressed music at 44.1 kHz.

## Quick Start

```bash
audiocpp_cli \
  --task s2s \
  --family apollo \
  --model models/Apollo-GGUF/apollo-orig.gguf \
  --backend cuda \
  --audio input_44100.wav \
  --out restored.wav \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `apollo` |
| Task | `s2s` |
| Mode | `offline` |
| Default weights | `models/Apollo-GGUF/apollo-orig.gguf` (F32) |
| Input | 44.1 kHz WAV through `--audio` |
| Output | Restored 44.1 kHz waveform, preserving input channels |

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | 44.1 kHz WAV path | required | Music to restore. |
| `--out` | WAV path | not set | Save the restored waveform. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `audio_chunk_duration_sec` | seconds >= 0 | `0` | Chunk duration; `0` processes the whole recording. |
| `audio_chunk_overlap_sec` | seconds >= 0 | `1` | Crossfade overlap, at most half the chunk duration. |
| `edge_pad_duration_sec` | seconds >= 0 | `0` | Extra context processed and discarded on each side of a chunk. |

For long recordings, add these options to limit per-chunk memory use:

```bash
--request-option audio_chunk_duration_sec=6 \
--request-option audio_chunk_overlap_sec=1 \
--request-option edge_pad_duration_sec=1
```

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `apollo.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | `native` | Weight storage override at session creation. |

## Convert Weights

From the repository root, convert the official Apollo `pytorch_model.bin`:

```bash
python tests/apollo/convert_apollo_gguf.py \
  --checkpoint /path/to/pytorch_model.bin \
  --output models/Apollo-GGUF/apollo-orig.gguf \
  --log apollo-conversion.log
```

The script uses `build/debug/bin/audiocpp_gguf` by default; use
`--audiocpp-gguf /path/to/audiocpp_gguf` for another executable.
