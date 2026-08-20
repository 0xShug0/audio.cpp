#pragma once

#include "engine/community_models/echo_tts/config.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::echo_tts {

// Conditioning for one generation: the tokenized text and the speaker latent,
// both already on the host. Masks are 1.0 for real positions and 0.0 for
// padding; the runtime converts them to additive attention masks.
struct EchoConditioning {
    std::vector<int32_t> text_input_ids;
    std::vector<float> text_mask;
    int64_t text_length = 0;

    std::vector<float> speaker_latent;  // (speaker_frames, latent_size), row-major
    std::vector<float> speaker_mask;    // (speaker_frames)
    int64_t speaker_frames = 0;
};

// Owns the DiT weights and the two graphs that use them.
//
// The conditioning encoders run once per request and their per-block key/value
// projections are held in a persistent device buffer. The denoiser graph then
// reads those buffers as leaves, so the text and speaker stacks are not
// re-executed on every sampler step. Upstream gets the same effect by passing
// Python lists of cached tensors into the forward call.
class EchoDitRuntime {
public:
    EchoDitRuntime(
        const EchoTtsConfig & config,
        const assets::TensorSource & source,
        const std::string & tensor_prefix,
        core::ExecutionContext & execution,
        assets::TensorStorageType matmul_storage_type);
    ~EchoDitRuntime();

    EchoDitRuntime(const EchoDitRuntime &) = delete;
    EchoDitRuntime & operator=(const EchoDitRuntime &) = delete;

    const EchoTtsConfig & config() const noexcept;

    // Runs the text and speaker encoders and populates the cached key/value
    // projections. Must be called before sample().
    void prepare_conditioning(const EchoConditioning & conditioning);

    // Runs the dual-CFG Euler sampler and returns the final latent, shaped
    // (sequence_length, latent_size) row-major.
    std::vector<float> sample(const EchoSamplerOptions & options);

    // One conditional denoiser forward at a fixed timestep, bypassing the
    // sampler. Exists so a parity harness can isolate a wrong DiT block from a
    // wrong integration step: feeding the reference's own x and t makes any
    // difference in the result attributable to the graph alone.
    //
    // `x` is (lanes * sequence_length, latent_size) row-major and sets the
    // sequence length for this call. `lanes` is 1 (conditional only) or 3
    // (cond, text-uncond, speaker-uncond). Requires prepare_conditioning().
    std::vector<float> denoise_once(const std::vector<float> & x, float t, int lanes = 1);


private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::echo_tts
