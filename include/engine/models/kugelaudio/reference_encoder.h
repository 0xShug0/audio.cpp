#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::models::kugelaudio {

struct ReferenceEncoderConfig {
    int64_t latent_size = 64;
    int64_t filters = 32;
    std::vector<int> ratios = {8, 5, 5, 4, 2, 2};
    std::vector<int> depths = {3, 3, 3, 3, 3, 3, 8};
    float norm_eps = 1e-5F;
    float fixed_std = 0.5F;
};

// Parsed only for reference encoding. Preset-only generation does not require
// encoder configuration or weights.
ReferenceEncoderConfig parse_reference_encoder_config(const io::json::Value & tokenizer);

// Same 24 kHz mono / -25 dBFS conditioning convention as the upstream acoustic
// processor. The input is bounded before resampling; it is never truncated.
std::vector<float> prepare_reference_audio(const runtime::AudioBuffer & audio);

class ReferenceEncoder {
public:
    ReferenceEncoder(const assets::TensorSource & source, core::ExecutionContext & execution,
                     ReferenceEncoderConfig config);
    ~ReferenceEncoder();
    // Returns [frames, latent_size] means, before sampling/scaling. The existing
    // generator applies fixed-std noise and the trained connector scale once.
    std::vector<float> encode(const std::vector<float> & normalized_mono);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::kugelaudio
