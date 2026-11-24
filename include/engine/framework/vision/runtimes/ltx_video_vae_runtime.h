#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::codecs {

struct LtxVideoVaeRuntimeConfig {
    // Native Conv3D cuDNN lowering on CUDA; other backends use generic Conv3D.
    // An unavailable or unsupported selected lowering must throw.
    // Native CUDA packs convolution weights to private ODHWI F16 bytes once
    // on the host at load time; source names, CF metadata and values are retained.
    // Pending packed uploads cost roughly 800 MiB of host RAM for this VAE,
    // plus the largest tensor's processing temporary; upload releases them.
    bool cuda_cudnn = false;
    // Requires cuda_cudnn and a CUDA execution context. Convolution accumulates
    // in F32 and rounds to F16; optional bias is added in F32 before storing F16. Residual
    // additions, temporal padding, and shortcut LayerNorm use F16 storage;
    // fused RMSNorm/SiLU computes in F32 and stores F16 without BF16 emulation
    // or an intermediate F32 activation buffer. Weight storage is unchanged.
    // Public I/O and posterior/statistics arithmetic remain F32 with their
    // existing BF16 rounding. Internal F16 range/rounding can change results.
    bool cuda_f16_activations = false;
};

// Session-owned LTX patch4 / latent128 VAE, with spatial/temporal scales 32/8.
// ExecutionContext must outlive this runtime. Calls must be serialized.
class LtxVideoVaeRuntime {
public:
    LtxVideoVaeRuntime(core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        LtxVideoVaeRuntimeConfig config = {});
    ~LtxVideoVaeRuntime();

    LtxVideoVaeRuntime(const LtxVideoVaeRuntime &) = delete;
    LtxVideoVaeRuntime & operator=(const LtxVideoVaeRuntime &) = delete;
    LtxVideoVaeRuntime(LtxVideoVaeRuntime &&) noexcept;
    LtxVideoVaeRuntime & operator=(LtxVideoVaeRuntime &&) noexcept;

    // Height/width are pixel dimensions, positive multiples of 32, for every API.
    // Encoder T maps to (T-1)/8+1 latent frames; decoder L maps to 8*(L-1)+1.
    // Adds and caches shapes, allocating ALL cached graphs together before capture.
    // Repeating an already prepared set is a no-op. Adding shapes invalidates all
    // borrowed outputs; callers must consume them before adding shapes.
    void prepare(int64_t height, int64_t width,
        const std::vector<int64_t> & encoder_frame_counts,
        const std::vector<int64_t> & decoder_latent_frame_counts);

    // Public inputs/outputs are contiguous F32 in logical [C,T,H,W] order
    // (C=3 or C=128), regardless of the internal activation storage option.
    // Noise has the encoder output shape and is supplied by the caller's RNG.
    // Backend inputs are copied to persistent graph inputs without host readback.
    // Only prepared shapes may execute; execution never builds/allocates graphs.
    // Returned tensors borrow shared arena storage and are valid only until the
    // next encode/decode call (any shape), shape-adding prepare, or destruction,
    // whichever comes first. Consume or copy the output before that next call.
    // An already-prepared prepare call preserves the current borrowed output.
    core::TensorValue encode(const std::vector<float> & pixels,
        const std::vector<float> & posterior_noise,
        int64_t frames, int64_t height, int64_t width);
    core::TensorValue encode(const core::TensorValue & pixels,
        const std::vector<float> & posterior_noise,
        int64_t frames, int64_t height, int64_t width);
    core::TensorValue decode(const std::vector<float> & latents,
        int64_t latent_frames, int64_t height, int64_t width);
    core::TensorValue decode(const core::TensorValue & latents,
        int64_t latent_frames, int64_t height, int64_t width);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::codecs
