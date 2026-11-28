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
  --request-option reference_image_file=/path/to/reference.jpg \
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

## Model

| Field | Value |
|---|---|
| Family | `liveavatar` |
| Tasks | `sfx` |
| Modes | `offline`, `streaming` |
| Languages | `auto` |
| Default package | `liveavatar_nvfp4_lora` |
| Input | Reference image, speech audio, and generation prompt |
| Output | Portrait video frames; see generation examples |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `reference_image_file` | `string` | required | Reference image path used as the identity/appearance condition. |
| `generation_mode` | `offline`, `liveavatar` | `offline` | Use whole-clip generation or LiveAvatar blockwise generation. |
| `height` | `int` | `480` | Generated frame height; must be a positive multiple of 16. Declared range: minimum 16. |
| `width` | `int` | `832` | Generated frame width; must be a positive multiple of 16. Declared range: minimum 16. |
| `num_frames` | `int` | `48` | Frames generated per LiveAvatar segment; must be positive and a multiple of 4 in blockwise mode (streaming or generation_mode=liveavatar). Declared range: minimum 1. |
| `num_clips` | `int` | `1` | Maximum LiveAvatar clips to generate from the input audio. Declared range: minimum 1. |
| `num_inference_steps` | `int` | `4` | Denoising steps; offline generation uses UniPC, blockwise generation uses Euler. The LiveAvatar DMD adapter is designed for 4 steps. Declared range: minimum 1. |
| `guidance_scale` | `float` | `0.0` | Classifier-free guidance scale; LiveAvatar blockwise generation requires 0. Declared range: minimum 0.0. |
| `batched_cfg` | `bool` | `true` | Run classifier-free guidance in one denoiser graph. |
| `sage_attention` | `bool` | `true` | Use CUDA SageAttention when the backend supports it. |
| `mem_saver` | `bool` | `true` | Reuse immutable condition data and temporary denoiser graph storage across sequential blocks. |
| `denoiser_layerwise` | `bool` | `false` | Run the denoiser in layerwise mode. Offline generation requires batched_cfg=true and guidance_scale > 1. |
| `denoiser_layerwise_batch` | `int` | `16` | Layer group size used by layerwise denoising. Declared range: minimum 1. |
| `vae_encoder_chunk_size` | `int` | `16` | Cached VAE encoder chunk size. Declared range: minimum 1. |
| `vae_decoder_tile_size` | `int` | `0` | Opt-in VAE decoder spatial tile size for large outputs; 0 disables tiling. Enabled tile sizes must be multiples of 8 and larger than the 64-pixel overlap. Declared range: minimum 0. |
| `target_cache_blocks` | `int` | `0` | Target KV cache window in block units; 0 keeps the full clip cache. Declared range: minimum 0. |
| `vae_cache_f16` | `bool` | `true` | Store cached VAE encoder state in F16. |
| `shift` | `float` | `3.0` | Flow scheduler shift; must be > 0. Declared range: minimum 0.0. |
| `negative_prompt` | `string` | Model/session default | Negative prompt for classifier-free guidance. |
| `seed` | `int` | `420` | Generation seed. Declared range: minimum 0. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `liveavatar.denoiser_weight_streaming` | `bool` | `false` | Keep denoiser transformer blocks in host memory and stage layer groups on the accelerator. |

## Load Options (use with `--load-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `liveavatar.support_gguf` | `string` | `Wan2.2-S2V-Support-Q4_K_S-F16.gguf` | Support GGUF containing text_encoder/audio_encoder tensors and the embedded tokenizer sidecar. |
| `liveavatar.vae_gguf` | `string` | `Wan2.2-S2V-VAE-F16.gguf` | Wan video VAE component GGUF file relative to the model root. |
| `liveavatar.denoiser_gguf` | `string` | `Wan2.2-S2V-14B-NVFP4-LORA.gguf` | Denoiser DiT GGUF file relative to the model root. |

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
