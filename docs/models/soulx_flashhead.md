# SoulX-FlashHead

> [!IMPORTANT]
> **cuDNN is strongly recommended for Lite on CUDA:** about **2× faster warm
> inference** with roughly half the peak VRAM in the RTX 5090 Q8 measurement
> (**9,152 → 5,168 MiB; 44% less VRAM**).
> **Prebuilt binaries do not include cuDNN support.** Build from source with
> cuDNN and `-DGGML_CUDA_CUDNN_CONV3D=ON`, then select
> `--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"cudnn"}}'`.

SoulX-FlashHead generates portrait video from a reference image and speech audio.
The family uses a shared audio-conditioned DiT with variant-specific video VAEs:
Lite uses LTX and Pro uses Wan. Only Lite GGUF weights are provided.

Upstream: [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead).

## Download Lite

GGUF packages: [audio-cpp/SoulX-FlashHead-GGUF](https://huggingface.co/audio-cpp/SoulX-FlashHead-GGUF).
Q8 is the default package; BF16 is also available. Both include all required weights and configuration.

```bash
hf download audio-cpp/SoulX-FlashHead-GGUF SoulX-FlashHead-Lite-Q8_0.gguf \
  --local-dir models/SoulX-FlashHead-GGUF
```

## Convert Lite

Download `Model_Lite/` and `VAE_LTX/` from
[Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B),
and the model and configuration files from
[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h).

```bash
python tools/community_models/soulx_flashhead/convert_gguf.py \
  --source /path/to/SoulX-FlashHead-1_3B \
  --audio-encoder /path/to/wav2vec2-base-960h \
  --type bf16 \
  --output /path/to/SoulX-FlashHead-GGUF/SoulX-FlashHead-Lite-BF16.gguf
```

The GGUF contains the DiT, video VAE, speech encoder, configuration and model
contract. No Python runtime or original checkpoint files are needed for C++ inference.

## Generate

```bash
build/debug/bin/audiocpp_cli \
  --family soulx_flashhead --task gen \
  --model /path/to/SoulX-FlashHead-Lite-Q8_0.gguf \
  --backend cuda --threads 8 \
  --audio /path/to/speech.wav \
  --request-option reference_image_file=/path/to/portrait.png \
  --out-dir outputs/flashhead \
  --log --log-file outputs/flashhead/run.log
```

Use `--mode streaming` to supply audio incrementally. Streaming input must be
mono 16 kHz. Lite emits 24 frames per audio block at 25 FPS, carrying nine motion
frames between blocks. The audio encoder uses an eight-second sliding window.
A partial final audio block is padded with silence, matching upstream.

Output artifacts contain RGB24 frames with width, height, frame count, frame rate
and chunk index metadata. Streaming consumers can display each chunk as it arrives.

| Request option | Default | Meaning |
|---|---:|---|
| `reference_image_file` | required | Reference portrait; resized and center-cropped. |
| `height`, `width` | 512 | Frame size; divisible by the variant's VAE spatial stride. |
| `num_inference_steps` | 4 | Denoising steps for distilled checkpoints. |
| `seed` | 42 | Generation seed. |
| `color_correction_strength` | 1 | Match the portrait's color statistics; zero disables correction. |
| `audio_guidance_scale` | 3 | Audio guidance for non-distilled checkpoints; unused by Lite/Pro. |

Encode the artifacts as an MP4, optionally muxing the source speech:

```bash
python tools/community_models/soulx_flashhead/render_video.py \
  --input outputs/flashhead --audio /path/to/speech.wav \
  --output outputs/flashhead/video.mp4
```

## Variant boundaries

The DiT reads channel counts, transformer dimensions, patch size and temporal/spatial
VAE strides from the embedded configuration. Lite selects the LTX VAE; Pro and the
non-distilled architecture select the shared Wan VAE modules. Non-distilled weights
use Euler updates with `audio_guidance_scale` (default 3); distilled checkpoints use
the upstream re-noising update. Only Lite GGUFs are provided. Pro has also been executed from normalized source
weights. Its Wan VAE passes the forced-latent RGB comparison against upstream.
The unreleased non-distilled checkpoint has not been validated and is not an
advertised package.
The converter accepts `--variant lite|pro|pretrained`; its default is Lite.
Use `--prepare-only` with an output directory to normalize a source checkpoint
without creating a GGUF. The directory can be loaded directly by this family:

```bash
python tools/community_models/soulx_flashhead/convert_gguf.py \
  --source /path/to/SoulX-FlashHead-1_3B \
  --audio-encoder /path/to/wav2vec2-base-960h \
  --variant pro --prepare-only --output /path/to/FlashHead-Pro-normalized
```

Use that directory as `--model` in the generation command above.

Lite uses the session-owned LTX video VAE runtime, which caches its encode/decode
graphs and shares their scratch arena. Video stays on the backend through color
correction and motion encoding; RGB frames are copied to the host for delivery.
CUDA attention uses SageAttention2 when the build and device support its head
dimensions, otherwise Flash Attention.

Enable the optional native cuDNN Conv3D path for Lite CUDA builds:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DENGINE_ENABLE_CUDA=ON -DGGML_CUDA_CUDNN_CONV3D=ON \
  -DCUDNN_ROOT=/path/to/cudnn
cmake --build build/debug --parallel 8
```

When explicitly selected through the execution plan below, this requires cuDNN
headers and libraries. Native Lite convolutions use F16 operands,
F32 accumulation, and F16 output storage; channel bias is added in F32 to the
F16 convolution result before the final F16 cast. Residual storage is also F16.
Normalization accumulates in F32 with F16 output on the native path. DiT
intermediates, color processing, and public VAE inputs/outputs use F32. These paths avoid reproducing every Python BF16 intermediate rounding;
small numerical differences are expected.

## Server streaming

Configure a streaming entry and start `audiocpp_server` with that config:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "threads": 8,
  "lazy_load": true,
  "models": [{
    "id": "flashhead",
    "family": "soulx_flashhead",
    "path": "/path/to/SoulX-FlashHead-Lite-Q8_0.gguf",
    "task": "gen",
    "mode": "streaming"
  }]
}
```

```bash
curl -N http://127.0.0.1:8080/v1/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"flashhead","stream_format":"sse","include_artifacts":true,"request":{"audio":"/path/to/speech-16k-mono.wav","options":{"reference_image_file":"/path/to/portrait.png"}}}'
```

Each `task.stream.event` carries video chunks in `event.artifacts`. Decode the
artifact's Base64 `payload` as RGB24 using its width, height, frames and FPS
metadata. `chunk_index` gives the order. Chunks arrive during generation;
`task.stream.done` ends the request and does not repeat their payloads. For a
complete non-streaming result, configure an `offline` entry and use
`/v1/tasks/run` instead. A streaming entry delivers video through stream events;
`/v1/tasks/run` does not collect those events into its response.

Select the generic VAE with an omitted execution plan or explicitly with
`--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"ggml"}}'`.
For Lite on CUDA, select the optional cuDNN implementation with
`--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"cudnn"}}'`.
This requires the cuDNN Conv3D build option above; an unavailable implementation
is rejected rather than silently substituted.
