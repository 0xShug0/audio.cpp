# Shared video components

Vision/video headers live in `include/engine/framework/vision/`, with matching
implementations in `src/framework/vision/`. This includes LivePortrait, face
modules, OpenCLIP, Synchformer, CAV-MAE-ST, Wan/LTX VAE modules and the media
helpers below. Existing C++
namespaces are preserved. Wan execution remains model-local.

The mirrored header/source directories are grouped by responsibility:

```text
framework/vision/
  media/       # Image/video I/O, resize plans, buffers, tiling and packing
  modules/     # Face and Wan VAE graph builders
  runtimes/    # LivePortrait, LTX VAE, OpenCLIP, Synchformer and CAV-MAE-ST
```

`modules/wan_video_vae_modules.*` contains the existing graph-building modules;
it does not introduce a shared Wan execution runtime.

Keep model generation, conditioning and temporal policies in the model. Reuse
framework media operations and same-architecture runtimes rather than adding
another graph executor or a universal video-model runtime.

| Component | Responsibility | Current callers |
|---|---|---|
| `io::read_video_rgb` / `load_video_rgb` | Native frames, timestamps and owned padded RGB storage | ControlFoley, SAM Audio |
| `video::RgbImage` and resize plans | Image loading, crop/resize coordinates and explicit arithmetic | FlashHead, LiveAvatar, SAM Audio |
| `video::FloatVideo`, spatial tile planning/blending | Planar video storage and overlap math | LiveAvatar, MiniMax-H3 |
| `video::pack_rgb24`, `RgbVideoBuffer` | Layout conversion and common RGB output storage | FlashHead, LiveAvatar, MiniMax-H3, LatentSync |

## Numerical contracts

Resize plans describe operations, not model names. The double-precision bilinear
plan and Pillow-compatible fixed-point bilinear plan have different rounding
contracts. Existing bicubic model preprocessing remains distinct.

RGB packing takes a scalar conversion callable so each caller retains its exact
normalization, rounding and clipping order. Spatial blending likewise preserves
the previous raw tile and overlap endpoints. Changing these equations is a
separate numerical change, not part of adopting the shared layout.

## Native input

ControlFoley and SAM Audio use the existing in-process FFmpeg 6+ decoder.
It does not launch an executable. LatentSync keeps main's model-local FFmpeg
executable input decoder, preserving its automatic stream selection and deployment
requirements. Browser builds reject native video input explicitly. Model artifact
IDs, metadata and output defaults remain unchanged.

## Focused tests

With `ENGINE_BUILD_TESTS=ON`, `video_processing_test` checks independent numerical
image/tile/packing references. Generated media stays outside Git.
