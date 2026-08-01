#pragma once

#include "engine/models/chatterbox/components.h"
#include "engine/framework/core/backend.h"
#include "engine/models/chatterbox/s3gen_flow.h"
#include "engine/models/chatterbox/s3gen_types.h"

#include <cstdint>
#include <vector>

namespace engine::models::chatterbox {

struct S3GenTimingBreakdown;

class S3GenSessionCache {
public:
    explicit S3GenSessionCache(engine::core::BackendConfig backend = {});
    ~S3GenSessionCache();
    S3GenSessionCache(S3GenSessionCache &&) noexcept;
    S3GenSessionCache & operator=(S3GenSessionCache &&) noexcept;

    S3GenSessionCache(const S3GenSessionCache &) = delete;
    S3GenSessionCache & operator=(const S3GenSessionCache &) = delete;

    void release_runtime_graphs();

private:
    struct State;
    std::unique_ptr<State> state_;

    friend S3Token2MelOutputs compute_s3_token2mel_inference(
        S3GenSessionCache & cache,
        const S3FlowEncoderWeights & encoder_weights,
        const S3FlowDecoderWeights & decoder_weights,
        const EmbedReferenceOutputs & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        int64_t speech_token_count,
        int64_t num_steps,
        float cfg_rate,
        bool cosine_schedule,
        const std::vector<float> & full_noise,
        uint64_t flow_seed,
        engine::core::BackendConfig backend,
        S3GenTimingBreakdown * timing);
    friend S3GenInferenceOutputs compute_s3gen_inference(
        S3GenSessionCache & cache,
        const S3FlowEncoderWeights & encoder_weights,
        const S3FlowDecoderWeights & decoder_weights,
        const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
        const EmbedReferenceOutputs & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        int64_t speech_token_count,
        int64_t num_steps,
        float cfg_rate,
        bool cosine_schedule,
        const std::vector<float> & full_noise,
        uint64_t flow_seed,
        uint64_t vocoder_seed,
        engine::core::BackendConfig backend,
        S3GenTimingBreakdown * timing);
    friend S3GenInferenceOutputs compute_s3gen_inference(
        S3GenSessionCache & cache,
        const S3FlowEncoderWeights & encoder_weights,
        const S3FlowDecoderWeights & decoder_weights,
        const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
        const EmbedReferenceOutputs & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        int64_t speech_token_count,
        engine::core::BackendConfig backend,
        S3GenTimingBreakdown * timing);
};

struct S3GenTimingBreakdown {
    double token2mel_ms = 0.0;
    double token2mel_embed_ms = 0.0;
    double token2mel_encoder_ms = 0.0;
    double token2mel_mu_ms = 0.0;
    double token2mel_cfm_ms = 0.0;
    S3FlowCFMTimingBreakdown token2mel_cfm;
    double vocoder_ms = 0.0;
};

S3Token2MelOutputs compute_s3_token2mel_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps = 10,
    float cfg_rate = 0.7f,
    bool cosine_schedule = true,
    const std::vector<float> & full_noise = {},
    uint64_t flow_seed = 0,
    engine::core::BackendConfig backend = {},
    S3GenTimingBreakdown * timing = nullptr);

S3GenInferenceOutputs compute_s3gen_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    int64_t num_steps = 10,
    float cfg_rate = 0.7f,
    bool cosine_schedule = true,
    const std::vector<float> & full_noise = {},
    uint64_t flow_seed = 0,
    uint64_t vocoder_seed = 0,
    engine::core::BackendConfig backend = {},
    S3GenTimingBreakdown * timing = nullptr);

S3GenInferenceOutputs compute_s3gen_inference(
    S3GenSessionCache & cache,
    const S3FlowEncoderWeights & encoder_weights,
    const S3FlowDecoderWeights & decoder_weights,
    const engine::models::chatterbox::HiFTVocoderComponent & vocoder,
    const EmbedReferenceOutputs & ref_dict,
    const std::vector<int32_t> & speech_tokens,
    int64_t speech_token_count,
    engine::core::BackendConfig backend,
    S3GenTimingBreakdown * timing = nullptr);

}  // namespace engine::models::chatterbox
