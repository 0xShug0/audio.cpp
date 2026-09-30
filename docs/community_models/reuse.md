# RE-USE

RE-USE restores speech at its input sample rate, from 8 to 48 kHz.
Output may differ slightly from the original Python model.

## Quick Start

```bash
audiocpp_cli \
  --task s2s \
  --family reuse \
  --model models/RE-USE-GGUF/reuse-f32.gguf \
  --backend cuda \
  --audio input.wav \
  --out restored.wav \
  --log
```

## Model

| Field | Value |
|---|---|
| Family | `reuse` |
| Task | `s2s` |
| Mode | `offline` |
| Weights | `reuse-f32.gguf`, `reuse-f16.gguf`, `reuse-q8_0.gguf` |
| Input | 8-48 kHz WAV through `--audio`, longer than 20 ms |
| Output | Restored waveform with the input sample rate, channels, and length |

Channels are processed independently. RE-USE does not resample the input or
provide streaming inference.

Use the F32 package on Vulkan. Native F16 and Q8 storage currently produces
incorrect Vulkan output; the reduced-storage packages are validated on CUDA.

## Convert Weights

Download the [original model](https://huggingface.co/nvidia/RE-USE) separately
from the output GGUF directory, then convert from its safetensors checkpoint:

```bash
hf download nvidia/RE-USE \
  --revision 022e920d727347a64d6c21fbf0628f2a5f37ad78 \
  --local-dir models/RE-USE-original

python tests/reuse/convert_reuse_gguf.py \
  --source models/RE-USE-original \
  --output models/RE-USE-GGUF/reuse-f32.gguf \
  --audiocpp-gguf build/debug/bin/audiocpp_gguf \
  --log build/logs/reuse/conversion.log
```

The converter requires Python packages `gguf`, `numpy`, and `safetensors`.
It embeds the configuration and model spec. The default F32 conversion verifies
every tensor's original shape, type, and bytes. Use `--type f16` or `--type q8_0`
with a corresponding output filename for reduced-storage packages.

The original weights are released under the
[NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE).
Conversion does not change their license.

## Request Options

Use with **`--request-option name=value`**.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `audio_chunk_duration_sec` | seconds >= 0 | `0` | Chunk duration; `0` processes the whole recording. |
| `audio_chunk_overlap_sec` | seconds >= 0 | `1` | Hann-weighted overlap, shorter than the chunk duration. |

For long recordings, add these options to bound per-chunk graph memory:

```bash
--request-option audio_chunk_duration_sec=10 \
--request-option audio_chunk_overlap_sec=1
```

Enabled chunks must exceed 20 ms. Overlap is unused when the recording fits
within one chunk. Chunking changes the bidirectional and normalization context,
so it can change the output. The input and final output are still held in host
memory; chunking does not make file handling streaming.

## Session Options

Use with **`--session-option reuse.name=value`**.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k` | `native` | Weight storage override at session creation. |

## Server Batching

The `/v1/tasks/batch` endpoint supports offline RE-USE restoration. Inputs with
the same sample rate and length share a graph batch; different shapes run in
separate groups. Chunking is not supported inside native batches. See the
[server API guide](../../app/server/README.md) for the request and response format.
