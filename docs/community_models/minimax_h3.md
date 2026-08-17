# MiniMax-H3

MiniMax-H3 is a text-to-audio/video generation model. The current audio.cpp port is
packaged as component GGUF files: text encoder, DiT, Audio VAE, Video VAE, config, and
processor sidecars live together in one model directory.

## Package Layout

The package directory must contain the files referenced by `model_specs/minimax_h3.json`:

```text
MiniMax-H3-Q4-GGUF/
  configuration.json
  text_encoder_q4_k.gguf
  dit.gguf
  dit_int8.gguf optional experimental ConvRot DiT
  audio_vae.gguf
  audio_vae_folded_f16.gguf
  video_vae.gguf
  FL2VA/processor/
    merges.txt
    preprocessor_config.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
    ...
```

The runtime uses `audio_vae_folded_f16.gguf`. Keep `audio_vae.gguf` as the original
Audio VAE GGUF so the folded runtime file can be regenerated.

## Run

For performance, memory, and quality tradeoffs across step count, DiT variant,
and acceleration options, see the [MiniMax-H3 performance notes](../reports/minimax_h3_performance.md).

MiniMax-H3 packages contain several component GGUF files. Pass the DiT entry file
explicitly: `dit.gguf` for the default path, or `dit_int8.gguf` for the experimental
ConvRot INT8 path. The runtime uses that file's parent directory as the package root and
resolves the other component files from there.

```bash
PROMPT='A lively four speaker comedy scene in a small radio studio. Speaker one says, "Welcome back, everyone, today we are testing a microphone that only records embarrassing truths." Speaker two says, "That explains why it kept calling me a sandwich with ambition." Speaker three says, "Please focus, the sponsor asked for professionalism and at least one normal sentence." Speaker four says, "Fine. This is a normal sentence, delivered by a person standing next to a haunted coffee machine." The speakers laugh, interrupt each other lightly, and continue with clear natural voices, quick timing, and no background music.'

build/debug/bin/audiocpp_cli \
  --task gen \
  --family minimax_h3 \
  --model models/MiniMax-H3-Q4-GGUF/dit.gguf \
  --model-spec-override model_specs/minimax_h3.json \
  --backend cuda \
  --threads 8 \
  --text "$PROMPT" \
  --seed 20260808 \
  --num-inference-steps 20 \
  --guidance-scale 1.0 \
  --request-option height=32 \
  --request-option width=32 \
  --request-option num_frames=481 \
  --request-option return_video=false \
  --out output.wav \
  --metrics
```

Set `return_video=true` when you also want RGB24 video output as an artifact. Use
`--out-dir <dir>` to write custom artifacts from the CLI.

## Options

Common request options:

| Option | Default | Notes |
|---|---:|---|
| `num_inference_steps` | `50` | Joint DiT denoising steps. Lower values are faster. |
| `seed` | `42` | Deterministic initial latent seed. |
| `height` | `768` | Target video canvas height. Use smaller values for short audio-focused generations. |
| `width` | `1344` | Target video canvas width. Use smaller values for short audio-focused generations. |
| `num_frames` | `124` | Target video frame count before H3 alignment. MiniMax-H3 video is 24 fps. |
| `guidance_scale` | `1.0` | Classifier-free guidance scale. |
| `negative_prompt` | `" "` | Negative prompt used when guidance is active. |
| `sampler` | `euler` | One of `euler`, `res_multistep`, `dpmpp_2m`, or `unipc`. |
| `flow_shift` | `12.0` | Video FlowMatch scheduler shift. |
| `audio_flow_shift` | `3.0` | Audio FlowMatch scheduler shift. |
| `return_video` | `false` | Decode RGB24 video frames in addition to audio. |

Memory and staged-weight options:

| Option | Default | Notes |
|---|---:|---|
| `minimax_h3.weight_context_mb` | `512` | Staged weight context size in MiB. |
| `minimax_h3.mem_saver` | `true` | Release staged DiT, audio VAE, and video VAE weights after their request phase instead of keeping them resident for the whole session. Prompt encoder weights are always request-scoped. Set `false` to keep generation weights cached when VRAM allows. |

Advanced options:

| Option | Default | Notes |
|---|---:|---|
| `text_layerwise` | `false` | Load prompt encoder weights in scoped layer groups. |
| `text_layerwise_batch` | `1` | Prompt encoder layer-group size when `text_layerwise=true`. |
| `dit_layerwise` | `false` | Load DiT weights in scoped prelude/block/final groups. |
| `dit_layerwise_batch` | `1` | DiT block-group size when `dit_layerwise=true`. |
| `dit_mlp_chunk_tokens` | `0` | Token chunk size for high-resolution layerwise MLP. Zero disables chunking. |
| `dit_acceleration` | `none` | Optional acceleration mode: `none`, `first_block_cache`, or `spectrum`. |
| `first_block_cache_threshold` | `0.1` | First-block cache residual-difference threshold. |
| `first_block_cache_start_sigma` | `0.95` | Largest video sigma where first-block cache may be used. |
| `first_block_cache_end_sigma` | `0.1` | Smallest video sigma where first-block cache may be used. |
| `first_block_cache_max_consecutive` | `2` | Maximum consecutive cache hits before forcing a full DiT step. |
| `spectrum_warmup_steps` | `1` | Full DiT warmup steps before spectrum forecasting. |
| `spectrum_initial_window` | `2.0` | Initial spectrum forecast scheduling window. |
| `spectrum_flex_window` | `0.75` | Spectrum forecast window growth after actual DiT steps. |
| `spectrum_degree` | `1` | Chebyshev polynomial degree for spectrum forecasting. |
| `spectrum_history_size` | `8` | Maximum spectrum forecast history rows. |
| `spectrum_ridge_lambda` | `0.1` | Ridge regularization for spectrum forecasting. |

## Convert

Most MiniMax-H3 components use the normal `audiocpp_gguf` converter. The DiT component is
the exception: it needs MiniMax-H3-specific AdaLN curve-table folding, so use the dedicated
DiT script for that component.

### DiT

```bash
python scripts/minimax_h3/convert_dit_gguf.py \
  --input /path/to/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors \
  --output models/MiniMax-H3-Q4-GGUF/dit.gguf \
  --overwrite \
  --type q4_0 --bnb-nf4-type q4_0 \
  --override 'blocks.*.adaln_proj.linear.weight=q4_0' \
  --override 'blocks.*.attn.out_proj.weight=bf16' \
  --override 'final_layer.adaln_proj.linear.weight=q4_0' \
  --override 'condition_proj.weight=bf16' \
  --override 'time_embedder.proj_in.weight=bf16' \
  --override 'time_embedder.proj_out.weight=bf16' \
  --override 'final_layer.audio_out.weight=bf16' \
  --override 'final_layer.video_out.weight=bf16' \
  --override 'blocks.39.mlp.fc2.weight=bf16' \
  --override 'blocks.40.mlp.fc2.weight=bf16' \
  --override 'blocks.41.mlp.fc2.weight=bf16' \
  --override 'blocks.42.mlp.fc2.weight=bf16' \
  --override 'blocks.43.mlp.fc2.weight=bf16' \
  --override 'blocks.44.mlp.fc2.weight=bf16' \
  --override 'blocks.45.mlp.fc2.weight=bf16' \
  --override 'blocks.46.mlp.fc2.weight=bf16' \
  --override 'blocks.47.mlp.fc2.weight=bf16' \
  --override 'blocks.48.mlp.fc2.weight=bf16' \
  --override 'blocks.49.mlp.fc2.weight=bf16' \
  --quantize-scope weights-2d --ineligible native \
  --quantizer python \
  --helper-chunk-rows 128 \
  --h3-adaln-curve-grid 1024 \
  --h3-adaln-curve-rank 64
```

The DiT GGUF should contain 532 tensors: one F32 AdaLN table, BF16 tensors for the
precision-sensitive override list, and Q4_0 tensors for the remaining quantized 2D
weights.

### DiT INT8 ConvRot

The optional `dit_int8.gguf` variant uses ConvRot INT8 linear layers for selected DiT
projections. It is CUDA-only and currently experimental. Keep the normal `dit.gguf` as the
default package DiT unless you explicitly want to test the INT8 path.

```bash
python scripts/minimax_h3/convert_dit_gguf.py \
  --input /path/to/MiniMax-H3-NF4/minimax-h3-fl2va-nf4.safetensors \
  --output models/MiniMax-H3-Q4-GGUF/dit_int8.gguf \
  --overwrite \
  --type q4_0 --bnb-nf4-type q4_0 \
  --overlay-input /path/to/MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors \
  --overlay-include 'blocks.*.attn.out_proj.weight' \
  --overlay-include 'blocks.*.mlp.fc1.weight' \
  --overlay-include 'blocks.*.mlp.fc2.weight' \
  --override 'blocks.*.adaln_proj.linear.weight=q4_0' \
  --override 'final_layer.adaln_proj.linear.weight=q4_0' \
  --override 'condition_proj.weight=bf16' \
  --override 'time_embedder.proj_in.weight=bf16' \
  --override 'time_embedder.proj_out.weight=bf16' \
  --override 'final_layer.audio_out.weight=bf16' \
  --override 'final_layer.video_out.weight=bf16' \
  --override 'blocks.39.mlp.fc2.weight=bf16' \
  --override 'blocks.40.mlp.fc2.weight=bf16' \
  --override 'blocks.41.mlp.fc2.weight=bf16' \
  --override 'blocks.42.mlp.fc2.weight=bf16' \
  --override 'blocks.43.mlp.fc2.weight=bf16' \
  --override 'blocks.44.mlp.fc2.weight=bf16' \
  --override 'blocks.45.mlp.fc2.weight=bf16' \
  --override 'blocks.46.mlp.fc2.weight=bf16' \
  --override 'blocks.47.mlp.fc2.weight=bf16' \
  --override 'blocks.48.mlp.fc2.weight=bf16' \
  --override 'blocks.49.mlp.fc2.weight=bf16' \
  --quantize-scope weights-2d --ineligible native \
  --quantizer python \
  --helper-chunk-rows 128 \
  --h3-adaln-curve-grid 1024 \
  --h3-adaln-curve-rank 64
```

The INT8 overlay is intentionally limited to `attn.out_proj`, `mlp.fc1`, and `mlp.fc2`.
Do not overlay `attn.qkv_proj`: the isolated projection can look numerically close, but the
end-to-end model output is not valid with QKV overlaid. ConvRot stores rotated INT8 weights
and scales in the GGUF; activation rotation and quantization happen in the CUDA runtime.

### Audio VAE

Create the original Audio VAE GGUF first:

```bash
build/debug/bin/audiocpp_gguf \
  --input /path/to/MiniMax-H3-NF4/audio_vae_nf4.safetensors \
  --output models/MiniMax-H3-Q4-GGUF/audio_vae.gguf \
  --type orig \
  --bnb-nf4-type q4_k \
  --family minimax_h3 \
  --root models/MiniMax-H3-Q4-GGUF \
  --overwrite
```

Then create the folded runtime Audio VAE GGUF:

```bash
python scripts/minimax_h3/convert_fold_audio_vae_gguf.py \
  --input models/MiniMax-H3-Q4-GGUF/audio_vae.gguf \
  --output models/MiniMax-H3-Q4-GGUF/audio_vae_folded_f16.gguf \
  --folded-type f16 \
  --overwrite
```

The folded file replaces MiniMax-H3 BigVGAN weight-norm pairs with direct convolution
weights stored as F16. Other tensors are copied from `audio_vae.gguf`.
