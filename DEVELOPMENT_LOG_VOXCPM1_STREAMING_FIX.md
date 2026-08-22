# VoxCPM1 Streaming TTS Fix - Development Log

## Objective
Fix the VoxCPM1 streaming TTS test failure in the AudioVAE decoder's stateful depthwise convolution.

## Model & Test
- **Model**: VoxCPM1 GGUF at `/tmp/v1/voxcpm-0.5b-q8_0-audiovae-f16.gguf`
- **Test**: `voxcpm1_streaming_tts` fails with exit code -6 (SIGABRT)
- **Error**: `GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed` at `ggml.c:1761`
- **Stack trace**: Crash in `causal_conv1d_dw_stateful` at line 865 when creating `next_state` view
- **Non-streaming test passes**: `voxcpm1_tts` works correctly using non-stateful conv functions
- **Key difference**: Streaming path uses `causal_conv1d_dw_stateful` / `decoder_block_stateful`; non-streaming uses `causal_conv1d_dw` / `decoder_block`

## Work Completed

### Build & Test Verification
- Built audio.cpp successfully with `cmake --build . --target audiocpp_cli`
- Confirmed non-streaming TTS path works: `python3 tools/audiocpp_cli/run_audiocpp_cli_path_tests.py --only voxcpm1_tts --backend cpu --threads 4 --model-path /tmp/v1/voxcpm-0.5b-q8_0-audiovae-f16.gguf --log` → **[OK]**
- Confirmed streaming TTS fails: same command with `--mode streaming --chunk-size 512` → **[FAIL]**

### Crash Analysis
The crash occurs during streaming decoder graph construction in `build_streaming_decoder_graph`:
1. Input tensor created with shape `[1, vae.latent_dim, latent_frames]` (expected: 64 channels)
2. Passed to `causal_conv1d_dw_stateful` for `decoder.model0.depthwise`
3. Inside function: concatenates state (6 frames) + input, runs depthwise conv
4. Attempts to extract `next_state` via `ggml_view_3d` with computed offset
5. **Assertion fails**: `view_offs = 18446744073709551600` (2^64 - 4096 = -4096 as uint64_t)

### Fix Applied (Partial)
Modified `/workspace/audio.cpp/src/community_models/voxcpm1/audiovae.cpp` in `causal_conv1d_dw_stateful`:
```cpp
// Before: used result.tensor->nb[0] which could be wrong for non-contiguous tensors
const int64_t state_offset = (result_frames - state_frames) * result.tensor->nb[0];

// After: ensure contiguous, compute offset from element_size
auto result_contig = core::ensure_backend_addressable_layout(ctx, result);
const size_t element_size = ggml_type_size(GGML_TYPE_F32);
const int64_t state_offset = (result_frames - state_frames) * static_cast<int64_t>(element_size);
```
**Result**: Still fails with same offset (-4096), indicating `result_frames < state_frames`

## Active Investigation

### Channel Mismatch Mystery
- Crash shows tensor has **96 channels** instead of expected **64** (latent_dim)
- Non-streaming test works → uses `build_decoder_graph` with non-stateful convs
- Streaming test fails → uses `build_streaming_decoder_graph` with stateful convs
- Both use same `assets_->config.audio_vae.latent_dim` and `patch_size`
- **Hypothesis**: Weight loading for folded depthwise convs has layout bug, or config not loaded correctly for streaming path

### GGUF Config Missing
- AudioVAE config (`latent_dim`, `decoder_dim`, `patch_size`, `num_patches`, `output_sample_rate`) NOT in GGUF metadata
- `load_audio_vae_config` requires `config.json` via `source.require_json("config.json")`
- GGUF has no `config.json` tensor or metadata key
- Non-streaming test passes → config must load somehow (sidecar? default? TransformingTensorSource magic?)

### Folded Weight Layout Bug (Suspected)
GGUF stores folded conv weights as `[kernel_size, 1, out_channels]` (e.g., `[7, 1, 64]`)
Expected weight_v layout for depthwise: `[out_channels, 1, kernel_size]` (e.g., `[64, 1, 7]`)
`TransformingTensorSource::reshape_tensor_data` does **flat copy** → wrong transpose for depthwise!
```cpp
// Current: just copies flat array - WRONG for depthwise
return data.values;  // Assumes row-major contiguous, but layouts differ
```
Non-streaming test may pass despite wrong weights (audio quality not validated, only CLI exit code)

### State Slot Channel Count
Streaming state init uses `weights_.decoder_first_depthwise.out_channels` for slot spec
If weight loading produces wrong out_channels (96 vs 64), state slots created with 96 channels
But input tensor has 64 channels → mismatch when taking state slot

## Next Moves (Priority Order)

1. **Verify config loading at runtime**
   - Add debug print of `vae.latent_dim` and `patch_size` in `build_streaming_decoder_graph`
   - Check if `config.json` exists as sidecar in model directory

2. **Fix folded weight loading for depthwise convs**
   - In `TransformingTensorSource::reshape_tensor_data` or `folded_weight_v`
   - Transpose from `[kernel, 1, out]` → `[out, 1, kernel]` for depthwise weights
   - Regular convs: `[out, in, kernel]` may already be correct

3. **Validate weight loading output**
   - Print `weights_.decoder_first_depthwise.out_channels` after loading
   - Verify state slot channel count matches

4. **Add tensor shape debug in streaming graph build**
   - Log input tensor shape before depthwise conv
   - Log result tensor shape after depthwise conv
   - Confirm `result_frames` computation

## Relevant Files

| File | Purpose |
|------|---------|
| `src/community_models/voxcpm1/audiovae.cpp` | Stateful conv functions (lines 822-935, 1182-1340), graph builds |
| `src/community_models/voxcpm1/assets.cpp` | Weight loading, `TransformingTensorSource`, config parsing (lines 113-121, 327-415, 602-613) |
| `src/framework/assets/tensor_source.cpp` | GGUF JSON loading (lines 40-53, 77-82) |
| `tools/audiocpp_cli/run_audiocpp_cli_path_tests.py` | Test runner |
| `external/ggml/src/ggml.c` | Assertion failure (line 1761) |

## Commands for Quick Iteration

```bash
# Rebuild
cd /workspace/audio.cpp/build && cmake --build . --target audiocpp_cli -j$(nproc)

# Run streaming test
cd /workspace/audio.cpp && python3 tools/audiocpp_cli/run_audiocpp_cli_path_tests.py --only voxcpm1_streaming_tts --backend cpu --threads 4 --model-path /tmp/v1/voxcpm-0.5b-q8_0-audiovae-f16.gguf --log

# Run non-streaming test (control)
cd /workspace/audio.cpp && python3 tools/audiocpp_cli/run_audiocpp_cli_path_tests.py --only voxcpm1_tts --backend cpu --threads 4 --model-path /tmp/v1/voxcpm-0.5b-q8_0-audiovae-f16.gguf --log

# Inspect GGUF tensors
cd /workspace/audio.cpp && python3 -c "import gguf; r=gguf.GGUFReader('/tmp/v1/voxcpm-0.5b-q8_0-audiovae-f16.gguf'); [print(t.name, t.shape, t.tensor_type) for t in r.tensors if 'audio_vae' in t.name]"
```

## Session State
- Last successful build: ✅
- Last test run: Streaming FAIL, Non-streaming PASS
- Current focus: Weight loading layout bug for folded depthwise convs
- Blocker: Cannot explain 96-channel tensor in streaming path when config says 64

---

## RESOLUTION - Issues Fixed (2025-08-22)

### Root Cause Identified
The crash was caused by **multiple bugs in stateful convolution implementation** combined with an **incorrect weight transpose** that was double-transposing GGUF weights.

### Fixes Applied (2025-08-22)

#### 1. Fixed Stateful Convolution View Creation (`audiovae.cpp`)
**Problem**: Next state was extracted from **conv output** (too few frames) instead of **padded input**.
**Fix**: Extract next state from **padded input** with correct column-major strides using physical tensor dimensions (`tensor->ne[]`, `tensor->nb[]`).

```cpp
// Fixed in both causal_conv1d_dw_stateful and causal_conv1d_stateful:
const int64_t padded_frames = padded.shape.dims[2];
const int64_t state_offset = (padded_frames - state_frames) * padded.tensor->nb[0];

// View using physical tensor dimensions (column-major layout):
const int64_t view_ne0 = state_frames;
const int64_t view_ne1 = padded.tensor->ne[1];  // channels
const int64_t view_ne2 = padded.tensor->ne[2];  // batch
const size_t view_nb1 = view_ne0 * element_size;
const size_t view_nb2 = view_ne0 * view_ne1 * element_size;
```

#### 2. Fixed State Tensor Wrapping (`causal_conv1d_stateful`)
**Problem**: State tensor wrapped with input shape instead of correct `[1, channels, state_frames]`.
**Fix**: Use proper logical shape for state tensor wrapping.

#### 3. Removed Unused SR Bucket Tensor (`audiovae.cpp`)
**Problem**: `streaming_sr_bucket_` created as input but never used → "tensor buffer not set" error.
**Fix**: Removed unused SR bucket (VoxCPM1 uses per-block fixed SR conditioning, not bucket-based).

#### 4. Initialized decoder_stride_ in Constructor
**Problem**: `decoder_stride_` only set in `build_decoder()` (non-streaming path).
**Fix**: Initialize in Impl constructor: `decoder_stride_(product(assets_->config.audio_vae.decoder_rates))`.

#### 5. Reverted Incorrect Weight Transpose (`assets.cpp`)
**Problem**: GGUF weights were already in correct format; transpose was double-transposing → near-silent audio (RMS ~0.009).
**Fix**: Reverted the depthwise weight transpose change. GGUF weights already in `[out_channels, 1, kernel]` format.

---

## Test Results (All Passing)

| Test | Status | Transcription | Audio Quality |
|------|--------|---------------|---------------|
| `voxcpm1_tts` (non-streaming) | ✅ PASS | "This is a test run for the fix." | RMS 0.11 |
| `voxcpm1_streaming_tts` | ✅ PASS | "This is a test run for the fix." | RMS 0.09 |
| Voice cloning (streaming) | ✅ WORKS | Perfect transcription | RMS 0.07 |

### Verification
- Both streaming and non-streaming produce **identical transcriptions** (verified with STT service)
- Audio RMS: 0.07-0.12 (healthy, was ~0.009 before weight transpose fix)
- Streaming produces chunked output with correct state carryover
- Voice cloning with streaming works correctly

---

## KNOWN ISSUES - To Fix Next Session (ALL FIXED AS OF 2025-08-22)

### ⚠️ Voice Glitches at Chunk Boundaries - **FIXED**
**Observation**: While transcriptions are perfect, the generated voice had **minor audible glitches at chunk boundaries** during streaming.

**Root Cause Identified**: Incorrect strides in `ggml_view_3d` calls for state extraction in all three stateful convolution functions. The view was created with assumed contiguous strides (`view_nb1 = view_ne0 * element_size`), but the source tensor (padded input) has column-major layout with `nb[1] = padded_frames * element_size`. This caused the state to be corrupted for channels > 0.

**Fix Applied (2025-08-22)**: Updated `causal_conv1d_stateful`, `causal_conv1d_dw_stateful`, and `causal_conv_transpose1d_stateful` to use the source tensor's actual strides (`padded.tensor->nb[1]`, `padded.tensor->nb[2]`) for the view.

```cpp
// Before (buggy):
const size_t view_nb1 = view_ne0 * element_size;
const size_t view_nb2 = view_ne0 * view_ne1 * element_size;

// After (fixed):
const size_t view_nb1 = padded.tensor->nb[1];
const size_t view_nb2 = padded.tensor->nb[2];
```

**Verification Results**:
- Non-streaming and streaming now produce **IDENTICAL audio statistics** (same duration, RMS, max, silent ratio)
- Only natural sample-to-sample transitions at some chunk boundaries (also present in non-streaming audio)
- Long streaming test (174 chunks, 13.92s): only 2 boundaries with diff > 0.15 out of 173 boundaries
- Overall max sample-to-sample diff in streaming (0.3719) < non-streaming (0.4290)
- Voice cloning streaming works correctly
- All tests pass: `voxcpm1_tts`, `voxcpm1_streaming_tts`

### Other Items for Next Session
1. **Optimize chunk size**: Current default (512 tokens → ~2.8s chunks) may be too small for smooth prosody
2. **Test longer streaming**: Verify quality over extended streaming sessions (many chunks) - **DONE: 174 chunks tested**
3. **Profile memory**: Ensure no state tensor leaks across long sessions

---

## Files Modified in This Fix

| File | Changes |
|------|---------|
| `include/engine/community_models/voxcpm1/audiovae.h` | Added `AudioVAEStreamingDecodeState` struct and streaming API |
| `src/community_models/voxcpm1/audiovae.cpp` | ~600 lines: stateful convs, streaming graph, state management; **Fixed view strides in 3 stateful conv functions** |
| `src/community_models/voxcpm1/session.cpp` | Integrated streaming decoder into streaming request pipeline |
| `src/community_models/voxcpm1/assets.cpp` | Reverted incorrect weight transpose (depthwise convs) |

---

## Next Session Plan

1. **Optimize chunk size**: Current default (512 tokens → ~2.8s chunks) may be too small for smooth prosody
2. **Profile memory**: Ensure no state tensor leaks across long sessions
3. **Consider VoxCPM2 streaming support** if needed

---

## Final Manual Verification (2025-08-22)

All 6 manual test cases generated successfully to `./out/` folder:

| Test | Command | Output File | Chunks | Duration |
|------|---------|-------------|--------|----------|
| TTS #1 | `--task tts --text "This is the first offline TTS test generation."` | `tts_1.wav` | N/A (offline) | ~2.5s |
| TTS #2 | `--task tts --text "This is the second offline TTS test with different content."` | `tts_2.wav` | N/A (offline) | ~2.8s |
| Streaming #1 | `--task tts --text "This is the first streaming test generation." --mode streaming` | `stream_1.wav` | 31 | ~2.5s |
| Streaming #2 | `--task tts --text "This is the second streaming test with different content." --mode streaming` | `stream_2.wav` | 38 | ~3.0s |
| Clone+Stream #1 | `--task tts --text "This is the first voice cloning streaming test." --mode streaming --audio voices/ana.wav --reference-text "This is Anna Neural speaking..."` | `clone_stream_1.wav` | 49 | ~3.9s |
| Clone+Stream #2 | `--task tts --text "This is the second voice cloning streaming test..." --mode streaming --audio voices/andrew.wav --reference-text "This is Andrew speaking..."` | `clone_stream_2.wav` | ~50 | ~4.0s |

**All 6 tests completed successfully** with healthy audio quality (RMS ~0.08-0.13, no crashes, no glitches at chunk boundaries).