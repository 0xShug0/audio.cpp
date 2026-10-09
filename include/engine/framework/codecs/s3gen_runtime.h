#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

struct S3Token2MelOutputs {
    std::vector<float> mel;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct S3FlowEncoderOutputs {
    std::vector<float> hidden;
    int64_t frames = 0;
    int64_t storage_frames = 0;
    int64_t hidden_size = 0;
};

struct S3GenInferenceOutputs {
    std::vector<float> waveform;
    int64_t samples = 0;
    std::vector<float> source;
    int64_t source_channels = 0;
    int64_t source_frames = 0;
    std::vector<float> mel;
    int64_t mel_channels = 0;
    int64_t mel_frames = 0;
};

struct S3FlowDecoderRunTiming {
    int64_t calls = 0;
    double conditioning_write_ms = 0.0;
    double time_embedding_ms = 0.0;
    double input_write_ms = 0.0;
    double time_write_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
};

struct S3FlowCFMTimingBreakdown {
    int64_t steps = 0;
    int64_t decoder_calls = 0;
    double initial_state_ms = 0.0;
    double schedule_ms = 0.0;
    double zero_conditioning_ms = 0.0;
    double runner_setup_ms = 0.0;
    double host_update_ms = 0.0;
    S3FlowDecoderRunTiming conditioned;
    S3FlowDecoderRunTiming unconditioned;
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

struct S3GenConfig {
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
    std::string vocoder_tensor_prefix = "mel2wav.";
    modules::HiftVocoderWeightLayout vocoder_weight_layout =
        modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
};

class S3GenRuntime {
public:
    S3GenRuntime(std::shared_ptr<const assets::TensorSource> source,
                 const core::ExecutionContext & execution, S3GenConfig config = {});
    ~S3GenRuntime();
    S3GenRuntime(S3GenRuntime &&) noexcept;
    S3GenRuntime & operator=(S3GenRuntime &&) noexcept;
    S3GenRuntime(const S3GenRuntime &) = delete;
    S3GenRuntime & operator=(const S3GenRuntime &) = delete;

    bool is_meanflow() const;
    S3FlowEncoderOutputs encode(
        const std::vector<float> & embeddings, int64_t frames,
        int64_t capacity_frames, int64_t hidden_size) const;
    S3Token2MelOutputs token_to_mel(
        const EmbedReferenceOutputs & reference,
        const std::vector<int32_t> & speech_tokens, int64_t speech_token_count,
        int64_t num_steps = 10, float cfg_rate = 0.7f, bool cosine_schedule = true,
        const std::vector<float> & full_noise = {}, uint64_t flow_seed = 0,
        S3GenTimingBreakdown * timing = nullptr) const;
    modules::HiftVocoderOutput decode_waveform(
        const std::vector<float> & mel, int64_t frames,
        uint64_t seed = 0, uint64_t prior_noise_values = 0) const;
    S3GenInferenceOutputs synthesize(
        const EmbedReferenceOutputs & reference,
        const std::vector<int32_t> & speech_tokens, int64_t speech_token_count,
        int64_t num_steps = 10, float cfg_rate = 0.7f, bool cosine_schedule = true,
        const std::vector<float> & full_noise = {}, uint64_t flow_seed = 0,
        uint64_t vocoder_seed = 0, S3GenTimingBreakdown * timing = nullptr) const;

    const modules::HiftVocoderComponent & vocoder() const;
    void release_flow_graphs() const;
    void release_vocoder_graphs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::codecs::s3gen
