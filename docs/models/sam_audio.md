# SAM Audio

SAM Audio separates a described sound from a recording. The native port accepts
text descriptions, masked reference images/videos, and positive/negative time
anchors in offline mode.

## Quick Start

```bash
audiocpp_cli \
  --task s2s \
  --family sam_audio \
  --model /path/to/SAM-Audio-GGUF/sam-audio-small-f32.gguf \
  --backend cuda \
  --audio input.wav \
  --text "man speaking" \
  --seed 42 \
  --out-dir outputs/separation \
  --log
```

The output directory contains `target.wav` (the described sound) and
`residual.wav` (the remaining sounds). Both are mono at 48 kHz. The output may
include padding to the next 40 ms codec frame.

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--audio` | audio path | required | Recording to separate; converted to mono at 48 kHz. |
| `--text` | text | required | Description of the target sound. |
| `--out-dir` | directory | not set | Save both separated waveforms. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `reference_image_path` | image path | not set | Masked RGB image of the target sound source, held constant throughout the recording. |
| `reference_video_path` | video path | not set | Masked video of the target sound source, aligned to audio by frame timestamps. |
| `seed` | integer, `-1` for random | `42` | Random seed. Also accepts the common `--seed` flag. |
| `num_inference_steps` | positive integer | `16` | Midpoint solver steps; each step evaluates the denoiser twice. |
| `anchors` | JSON array | not set | Positive (`+`) or negative (`-`) time spans in seconds. |

For example, mark a target occurrence from 0.5 to 2 seconds and an unwanted sound
from 3 to 4 seconds:

```bash
--request-option 'anchors=[["+",0.5,2.0],["-",3.0,4.0]]'
```

Spans require `0 <= start < end`. For overlapping spans, the later entry takes
precedence. The recording is processed as a whole; memory use grows with its length.

For a static visual prompt, add `--request-option reference_image_path=masked.png`.
Mask unwanted regions in the image before supplying it. The model does not create
the mask. Vision weights load on the first visual request and remain available
for later requests in the same session.

For a moving visual prompt, use `--request-option reference_video_path=masked.mp4`
instead. Video decoding uses FFmpeg/libav runtime libraries; no FFmpeg command-line
process is launched. Audio still comes from `--audio`, and should share the video's
time origin. Use either the image or video option, not both. Frames are selected by
nearest timestamp and encoded in bounded batches with a reused graph.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---|---|
| `sam_audio.memory_bounded` | boolean | `false` | Experimental bounded-workspace execution for long recordings; the option name is provisional. |

This mode keeps full-recording diffusion and evaluates codec convolutions in
tiles with complete receptive-field context. Watermark recurrent state is
carried between tiles. It does not split the recording into independent
separation requests or crossfade the results. Different computation shapes can
produce small floating-point differences.

GPU workspace is bounded by fixed codec tiles and the model's configured
diffusion capacity (400 seconds for Small). Intermediate recordings are held
in host RAM, whose use grows with duration. Extra transfers and overlap
computation may be slower. This does not extend the model's context limit.

Until the option name is finalized, existing GGUFs are unchanged. To try it
with the current checkout, add these flags to the command above:

```bash
--model-spec-override model_specs/sam_audio.json \
--session-option sam_audio.memory_bounded=true
```

## Convert Weights

With the official checkpoint and configuration in a local directory:

```bash
python tests/sam_audio/convert_gguf.py \
  --model /path/to/sam-audio-small \
  --output /path/to/prepared-sam-audio \
  --gguf-output /path/to/SAM-Audio-GGUF/sam-audio-small-f32.gguf
```

The converter includes the T5 text encoder, tokenizer, configuration, and model
spec in the GGUF, and checks every tensor against its source. It uses
`build/debug/bin/audiocpp_gguf` by default; `--converter` selects another binary.

Model weights are subject to Meta's SAM License supplied with the original
checkpoint. The bundled T5 encoder retains its upstream license.
