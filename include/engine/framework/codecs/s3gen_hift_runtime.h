#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::modules { struct HiftVocoderWeights; }

namespace engine::codecs::s3gen {

struct EmbedReferenceOutputs {
    std::vector<int32_t> prompt_tokens;
    int64_t prompt_token_count = 0;
    std::vector<float> prompt_feat;
    int64_t prompt_feat_frames = 0;
    int64_t prompt_feat_dims = 0;
    std::vector<float> embedding;
    int64_t embedding_size = 0;
    double prompt_mel_ms = 0.0;
    double speaker_ms = 0.0;
    double tokenizer_ms = 0.0;
};

struct HiFTVocoderOutputs {
    std::vector<float> waveform;
    int64_t samples = 0;
    std::vector<float> source;
    int64_t source_channels = 0;
    int64_t source_frames = 0;
    std::vector<float> f0;
    int64_t f0_frames = 0;
    std::vector<float> post;
    int64_t post_frames = 0;
};

struct HiFTVocoderComponentWeights {
    std::shared_ptr<const engine::modules::HiftVocoderWeights> runtime_weights;
};

class HiFTVocoderComponent {
public:
    static HiFTVocoderComponent load_from_source(
        std::shared_ptr<const engine::assets::TensorSource> source,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    HiFTVocoderComponent(
        std::shared_ptr<const HiFTVocoderComponentWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const HiFTVocoderComponentWeights> & weights() const noexcept;
    HiFTVocoderOutputs infer(
        const std::vector<float> & speech_feat,
        int64_t batch,
        int64_t frames,
        uint64_t seed,
        uint64_t prior_noise_values,
        const std::vector<float> & cache_source) const;
    void release_runtime_cache() const;

private:
    struct State;
    std::shared_ptr<const HiFTVocoderComponentWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

}  // namespace engine::codecs::s3gen
