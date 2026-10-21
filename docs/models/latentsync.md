# LatentSync 1.6

LatentSync changes a face's lip motion in an existing video to match supplied
speech. It uses a Whisper audio encoder, a temporal diffusion U-Net, an SD VAE,
and SCRFD/106-point face alignment. The output retains the input video resolution.

GGUF package: [audio-cpp/LatentSync-1.6-GGUF](https://huggingface.co/audio-cpp/LatentSync-1.6-GGUF).

| Package | Precision |
|---|---|
| `latentsync-1.6-q8_0-f16.gguf` (default) | Q8 U-Net linear weights; remaining weights F16. |
| `latentsync-1.6-f16.gguf` | F16. |
| `latentsync-1.6-f32-f16.gguf` | Original mixed F32/F16. |

Install FFmpeg and make sure `ffmpeg` is available on PATH. Native inference does
not require Python or ONNX Runtime.

## Run

```bash
audiocpp_cli --task gen --family latentsync \
  --model /path/to/latentsync-1.6-q8_0-f16.gguf --backend cuda \
  --audio speech.wav \
  --request-option reference_video_file=speaker.mp4 \
  --out-dir outputs/latentsync --log
```

Use a video with a clearly visible face. When several faces are detected, the
largest eligible face is selected. A frame without an eligible face is an error.
If the audio exceeds the video duration, the video repeats forward and backward.

## Common Options

| Option | Meaning |
|---|---|
| `--audio` | Speech audio used to drive lip motion. |
| `--backend` | Inference backend. |
| `--threads` | CPU worker threads. |
| `--out-dir` | Directory for generated video artifacts. |

## Request Options

Use with **`--request-option name=value`**.

| Option | Default | Meaning |
|---|---:|---|
| `reference_video_file` | Required | Video containing the face to synchronize. |
| `num_inference_steps` | `20` | DDIM denoising steps, from `1` to `999`. |
| `cache_interval` | `1` | DeepCache refresh interval. `1` disables it; `3` matches upstream's DeepCache preset. Cached steps are faster but may change lip detail. |
| `guidance_scale` | `1.5` | Audio guidance; values at or below `1` disable CFG. |
| `frames_per_chunk` | `16` | Frames denoised together, from `1` to `24`. Changing this changes temporal context and output. |
| `fps` | `25` | Input sampling and output frame rate. |
| `seed` | `1247` | Latent and VAE posterior sampling seed. |

## Session Options

Use with **`--session-option latentsync.name=value`**.

| Option | Default | Meaning |
|---|---|---|
| `weight_type` | `native` | Weight storage type. |
| `vae_cudnn_f16` | `false` | Opt-in FP16 VAE on CUDA; requires a cuDNN-enabled build. VAE weights use FP16 in this mode. |
| `unet_cudnn_f16` | `false` | Opt-in FP16 U-Net residual path on CUDA; requires a cuDNN-enabled build and FP16 U-Net weights. |
| `mem_saver` | `false` | Release components between pipeline stages and run CFG branches sequentially to reduce peak VRAM. Reloading the VAE for decoding adds some latency. |

Build with `-DGGML_CUDA_CUDNN_CONV2D=ON` and a discoverable cuDNN installation
to use `--session-option latentsync.vae_cudnn_f16=true`. This changes VAE
rounding; leave it off when exact output from the existing path is required.
The same build enables `--session-option latentsync.unet_cudnn_f16=true` with
FP16 U-Net weights. It also changes rounding; leave it off for exact output
from the existing U-Net path.

For faster CUDA inference, use the default Q8/F16 package with native weight
storage and `--request-option cache_interval=3`. The F16 package is also supported.
DeepCache refreshes the full denoiser
every third step and reuses a deep feature on intervening steps. Each new video
chunk starts with a fresh feature. The cached path retains an additional shallow
graph and feature buffer; it trades memory and some output detail for speed.

For lower peak VRAM, add `--session-option latentsync.mem_saver=true`. This
keeps the configured frame chunk size and denoising schedule, but stages model
components and evaluates the two CFG branches sequentially. Expect a small
numerical drift from the default batched-CFG path and some added latency from
moving DeepCache state through host memory and reloading the VAE for decoding.

## Export MP4

The CLI writes raw frames to `latentsync_video_rgb24.rgb` and dimensions/frame
rate to `latentsync_video.json`. It does not encode MP4 or mux audio internally.
Encode the frames with FFmpeg:

```bash
ffmpeg -f rawvideo -pixel_format rgb24 -video_size WIDTHxHEIGHT -framerate FPS \
  -i outputs/latentsync/latentsync_video_rgb24.rgb -i speech.wav \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -movflags +faststart \
  outputs/latentsync/video.mp4
```

Replace `WIDTH`, `HEIGHT`, and `FPS` with values from `latentsync_video.json`. Frame count follows
the upstream audio-feature windowing and can extend slightly beyond the audio.

## Convert Weights

Obtain the [LatentSync 1.6 checkpoint](https://huggingface.co/ByteDance/LatentSync-1.6),
its Whisper Tiny checkpoint, [SD VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse),
and the `det_10g.onnx` and `2d106det.onnx` files from InsightFace's `buffalo_l` package.
Use the [upstream source](https://github.com/bytedance/LatentSync) for the model
configuration and face mask. Preparation requires PyTorch, safetensors,
OmegaConf, NumPy, OpenCV and ONNX.

```bash
python tests/latentsync/prepare_weights.py \
  --checkpoint-dir /path/to/checkpoints --vae-dir /path/to/sd-vae-ft-mse \
  --reference-dir /path/to/LatentSync --face-model-dir /path/to/buffalo_l \
  --output-dir prepared

audiocpp_gguf --input weights=prepared/model.safetensors.index.json \
  --output latentsync-1.6-f32-f16.gguf --type orig --family latentsync \
  --model-spec model_specs/latentsync.json --root prepared \
  --sidecar prepared/detector_config.json=detector_config.json \
  --sidecar prepared/landmark_config.json=landmark_config.json
```

The checkpoint directory must contain `latentsync_unet.pt` and `whisper/tiny.pt`.
The converter preserves tensor precision and embeds the spec, face-model configs,
and mask. Face-network batch normalization is folded during preparation.

## Licenses

LatentSync is Apache-2.0, but that does not cover all supporting weights.
In particular, [InsightFace's pretrained models](https://github.com/deepinsight/insightface#license)
are for non-commercial research. A combined GGUF retains those restrictions;
conversion does not grant commercial rights. Review each upstream weight license
before redistribution or deployment.
