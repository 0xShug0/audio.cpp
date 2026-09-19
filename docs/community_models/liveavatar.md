# LiveAvatar

LiveAvatar is an audio-to-video model that generates an animated avatar from a
reference image, an audio file, and a text description. The native runtime uses the Wan2.2 S2V
components with LiveAvatar blockwise generation.

## Run

The following configuration generates two clips at 720p and 16 FPS while using
the low-VRAM denoiser path:

```bash
build/debug/bin/audiocpp_cli \
  --task gen \
  --family liveavatar \
  --model /path/to/Wan2.2-S2V \
  --backend cuda \
  --threads 8 \
  --audio /path/to/reference.wav \
  --text "A detailed description of the speaker and scene." \
  --load-option "liveavatar.denoiser_gguf=models/unet/Qwen 2SV/Wan2.2-S2V-14B-NVFP4-LORA.gguf" \
  --session-option liveavatar.denoiser_weight_streaming=true \
  --request-option generation_mode=liveavatar \
  --request-option reference_image_path=/path/to/reference.jpg \
  --request-option height=720 \
  --request-option width=1280 \
  --request-option seed=420 \
  --request-option num_clips=2 \
  --request-option denoiser_layerwise=true \
  --request-option denoiser_layerwise_batch=16 \
  --request-option target_cache_blocks=1 \
  --request-option vae_encoder_chunk_size=4 \
  --request-option vae_decoder_tile_size=320 \
  --out-dir outputs/liveavatar \
  --log \
  --log-file outputs/liveavatar/run.log
```

Use `num_clips`, not `num_clip`. With the default 48-frame clip size, two clips
produce 93 output frames after block overlap is resolved.

## Memory Control

`liveavatar.denoiser_weight_streaming` is an opt-in session option. It is
selected when the model session is created and cannot be changed per request
without reloading the session.

- `false` keeps the denoiser weights resident on the accelerator and provides
  the fastest path when enough VRAM is available.
- `true` keeps the denoiser transformer-block weights in pinned host memory,
  stages one layer group at a time, and keeps the immutable condition KV cache
  in host memory after it is built. Common denoiser weights and active compute
  state remain on the accelerator.

Weight streaming requires both:

```bash
--request-option generation_mode=liveavatar
--request-option denoiser_layerwise=true
```

`denoiser_layerwise_batch` controls the number of transformer layers staged at
once. Larger values improve throughput but use more VRAM. Start with `16`; lower
it when the accelerator cannot fit that group.

The VAE has separate memory controls. `vae_encoder_chunk_size` limits encoder
work per chunk, while `vae_decoder_tile_size` enables tiled spatial decoding.
For 720p generation near a 16 GiB limit, use encoder chunks of `4`, decoder tiles
of `320`, and `target_cache_blocks=1` as shown above.

## Options

Request options (use with `--request-option`):

| Option | Default | Notes |
|---|---:|---|
| `generation_mode` | `offline` | Use `liveavatar` for blockwise audio-driven generation. |
| `reference_image_path` | required | Reference image used for identity and appearance. |
| `height` / `width` | `704` / `1024` | Output frame dimensions. |
| `num_frames` | `48` | Frames generated per LiveAvatar segment. |
| `num_clips` | `1` | Maximum number of audio-driven clips to generate. |
| `num_inference_steps` | `4` | Euler denoising steps; the LiveAvatar adapter is designed for four steps. |
| `guidance_scale` | `0` | LiveAvatar blockwise generation requires zero guidance. |
| `seed` | `420` | Generation seed. |
| `denoiser_layerwise` | `false` | Required by denoiser weight streaming. |
| `denoiser_layerwise_batch` | `16` | Number of denoiser layers staged and evaluated together. |
| `target_cache_blocks` | `0` | Target KV cache window; `1` keeps one block. |
| `vae_encoder_chunk_size` | `16` | VAE encoder chunk size. |
| `vae_decoder_tile_size` | `0` | VAE decoder tile size; zero disables tiling. |

Session options (use with `--session-option`):

| Option | Default | Notes |
|---|---:|---|
| `liveavatar.denoiser_weight_streaming` | `false` | Stream denoiser block weights and keep the completed condition KV cache in host memory. |
| `support_gguf` | `Wan2.2-S2V-Support-Q4_K_S-F16.gguf` | Support component containing the text and audio encoders. |
| `vae_gguf` | `Wan2.2-S2V-VAE-F16.gguf` | Video VAE component. |
| `denoiser_gguf` | `Wan2.2-S2V-14B-NVFP4-LORA.gguf` | Denoiser component. |

Load options (use with `--load-option`):

| Option | Default | Notes |
|---|---:|---|
| `support_gguf` | package default | Override the support component path relative to the model root. |
| `vae_gguf` | package default | Override the VAE component path relative to the model root. |
| `denoiser_gguf` | package default | Override the denoiser component path relative to the model root. |
