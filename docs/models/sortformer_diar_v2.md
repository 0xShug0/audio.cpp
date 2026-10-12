# NVIDIA Sortformer v2.1 Diarization

`sortformer_diar_v2` provides native audio.cpp inference for NVIDIA's
`diar_streaming_sortformer_4spk-v2.1` four-speaker streaming diarization
checkpoint. It returns speaker turns for mono 16 kHz audio and supports both
bounded offline execution and continuous streaming execution.

The v2 implementation is additive: the existing `sortformer_diar` v1 family
is unchanged.

## Task and modes

- Task: `diar`
- Modes: `offline`, `streaming`
- Input: mono 16 kHz audio
- Output: speaker turns through `--turns-out`
- Speakers: four model channels, reported as `SPEAKER_00` through
  `SPEAKER_03`

Streaming preserves frontend continuity, AOSC FIFO/speaker-cache state, and
arrival-order speaker identities across chunks. For long recordings, use
streaming mode; CUDA offline execution requires more than 16 GiB for the F32
profile on the validation machine.

## License and local conversion

The checkpoint is governed by the [NVIDIA Open Model
License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Redistribution approval for the converted weights is pending, so no public
GGUF download is advertised. Convert a locally obtained `.nemo` checkpoint.
The checkpoint used for validation was:

```text
nvidia/diar_streaming_sortformer_4spk-v2.1
SHA-256: 8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8
```

The converter reads either the official `.nemo` archive or an extracted
`model_config.yaml` plus `model_weights.ckpt` directory:

```powershell
$checkpoint = "path\to\diar_streaming_sortformer_4spk-v2.1.nemo"
$converter = "build\mms\bin\Release\audiocpp_gguf.exe"
$staging = "models\Sortformer-Diar-v2.1-local"

uv run --python 3.12 `
  --with torch --with numpy --with pyyaml --with safetensors `
  python tools\community_models\convert_sortformer_v2_1.py `
  --checkpoint $checkpoint `
  --output-dir $staging `
  --converter $converter `
  --gguf-output "$staging\sortformer-v2.1-f32.gguf" `
  --type orig
```

The accepted smaller profile is reproducible by changing the final options to:

```text
--type f16 --keep-nonmatrix-f32
```

This stores rank-2 weight matrices as F16 and keeps biases, norms, filterbank,
and other non-matrix tensors as F32. The local validation artifact was
263,707,264 bytes (44.05% smaller than F32), SHA-256
`30188fb6f9d9423293c571d39220ba1ae1432b19a83bda329d75d33593d6df26`.

## Run

Offline:

```powershell
build\sortformer-v2-cuda\bin\Release\audiocpp_cli.exe `
  --task diar --family sortformer_diar_v2 --mode offline `
  --model models\Sortformer-Diar-v2.1-local\sortformer-v2.1-f32.gguf `
  --backend cuda --audio meeting_16k.wav --turns-out turns.json
```

Streaming:

```powershell
build\sortformer-v2-cuda\bin\Release\audiocpp_cli.exe `
  --task diar --family sortformer_diar_v2 --mode streaming `
  --model models\Sortformer-Diar-v2.1-local\sortformer-v2.1-f16-mixed.gguf `
  --backend cuda --device 0 --audio meeting_16k.wav --turns-out turns.json
```

## Options

| Scope | Option | Default | Description |
|---|---|---:|---|
| Request | `speaker_threshold` | `0.5` | Speaker activity threshold. |
| Request | `speaker_min_frames` | `0` | Minimum decoded turn duration in 80 ms frames. |
| Request | `speaker_pad_frames` | `0` | Padding around decoded turns in 80 ms frames. |
| Session | `sortformer_diar_v2.geometry` | `model` | Checkpoint geometry or a supported latency preset. |
| Session | `sortformer_diar_v2.graph_arena_mb` | `1024` | Inference graph arena size. |
| Session | `sortformer_diar_v2.weight_context_mb` | `1024` | Weight context size. |
| Session | `sortformer_diar_v2.weight_type` | `f32` | Default weight storage type. |
| Session | `sortformer_diar_v2.matmul_weight_type` | `weight_type` | Matmul storage override. |
| Session | `sortformer_diar_v2.conv_weight_type` | `weight_type` | Convolution storage override. |

## Validation

The F32 profile was checked against the transcribe.cpp reference on 1,500 and
11,250-frame streams. On the 120-second run, the maximum probability delta was
`0.00306` and the mean delta was `0.000210`; all seven AOSC cache selections
matched.

The mixed-F16 profile was validated on a real 900-second audio file with the
Release MSVC/CUDA binary on Windows, CUDA 13.3, and two RTX 4070 Ti SUPER
GPUs (16,375 MiB each). It produced 11,250 frames and 458 speaker turns and exited
successfully. Three repeated streaming CLI runs completed in 3.10–3.14 seconds
(approximately `0.00346` RTF, or 289x realtime). GPU 0 reached a sampled peak
of 5,880 MiB, with a sustained working level around 4,513–4,540 MiB, and
returned to the 1,108–1,136 MiB post-process baseline after each run; GPU 1
remained at its approximately 2,063 MiB baseline.

This is a repeated-process streaming measurement, not a same-process server
session leak test. Relative to the F32 output, the accepted output-turn
evaluation measured per-speaker F1 of `0.989/0.909/0.955/0.912`, union
activity precision/recall of `0.973/0.955`, and 98.58% same-speaker agreement
on single-speaker overlap frames.

Uniform F16 and Q8_0 were rejected for long-stream drift. Mixed Q8/F16/F32
search was also rejected: the quality cliff begins around 55–56% reduction,
while candidates at 60% or more reduction do not preserve speaker identity.

The Release unit tests are:

```text
ctest --test-dir build/sortformer-v2-cuda -C Release \
  -R '^sortformer_v2_(aosc|schedule)_test$' --output-on-failure
```

## Attribution

The tensor mapping is adapted from `handy-computer/transcribe.cpp` under the
MIT License. The AOSC state and streaming scheduler are adapted from
`NVIDIA/NeMo-Speech.cpp` under Apache License 2.0. See
`tools/community_models/SORTFORMER_THIRD_PARTY_NOTICES.md` for the complete
notices and source revisions.
