#pragma once

#include "engine/community_models/piper_tts/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::piper_tts {

struct PiperTtsGenerationOptions {
    float speaking_rate = 1.0F;
    float variation = 0.667F;
    float duration_variation = 0.8F;
    uint32_t seed = 1234;
};

class PiperVitsRuntime {
public:
    // `rknn_decoder_path` (optional) routes the decoder stage to a pre-compiled RKNN
    // engine on a Rockchip NPU; empty keeps the ggml decoder.
    // `rknn_encoder_path` (optional) routes the duration encoder to a comma-separated
    // ladder of pre-compiled RKNN engines; utterances longer than the widest bin fall
    // back to the ggml encoder for that step.
    // `duration_encoder_backend` / `duration_flows_backend` (optional) pin the two duration
    // stages to a backend of their own, independently of each other and of the decoder.
    PiperVitsRuntime(
        std::shared_ptr<const PiperTtsAssets> assets,
        core::BackendConfig backend_config,
        std::string rknn_decoder_path = {},
        std::optional<core::BackendType> duration_encoder_backend = {},
        std::optional<core::BackendType> duration_flows_backend = {},
        std::string rknn_encoder_path = {});
    ~PiperVitsRuntime();

    runtime::AudioBuffer synthesize(
        const std::vector<int32_t> & token_ids,
        const PiperTtsGenerationOptions & options);

private:
    struct State;
    std::unique_ptr<State> state_;
};

void apply_piper_tts_edge_fade(
    std::vector<float> & samples,
    int sample_rate,
    float milliseconds = 5.0F);

}  // namespace engine::models::piper_tts
