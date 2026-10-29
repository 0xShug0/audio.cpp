#pragma once

#include "engine/community_models/nemotron_transcribe/frontend.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <functional>
#include <vector>

namespace engine::models::nemotron_transcribe {

struct RopeExpertConfig {
    int feature_size = 128;
    int hidden_size = 1280;
    int heads = 10;
    int layers = 32;
    int intermediate_size = 5120;
    int subsampling_factor = 8;
    int rotary_dimensions = 64;
    bool qk_norm = true;
    bool qkv_bias = false;
    bool pre_block_norm = true;
    float layer_norm_epsilon = 1e-5f;
    float rope_base = 10000.0f;
};

struct SpeakerCacheConfig {
    // PEE online-inference overrides in NVIDIA's native reference. These differ
    // from the standalone diarizer's training geometry (264/0/264).
    int cache_frames = 188;
    int fifo_frames = 40;
    int update_period = 300;
    int silence_frames_per_speaker = 1;
    float prediction_score_threshold = 0.25f;
    float latest_score_boost = 0.05f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_positive_score_rate = 0.5f;
};

struct EncoderConfig {
    RopeExpertConfig asr;
    RopeExpertConfig diar = [] {
        RopeExpertConfig c;
        c.hidden_size = 512;
        c.heads = 8;
        c.layers = 31;
        c.intermediate_size = 2048;
        c.qk_norm = false;
        return c;
    }();
    int projection_dim = 2688;
    int speaker_count = 8;
    int speaker_hidden_size = 192;
    int upsample_factor = 8;
    float speaker_activity_threshold = 0.5f;
    // Output frames, each 80 ms with the released model. Export metadata owns
    // these values; do not substitute the separate HF decoder chunk setting.
    int online_inference_length = 500;
    int chunk_left_context = 50;
    int chunk_right_context = 50;
    SpeakerCacheConfig speaker_cache;
    bool use_flash_attention = true;
    double maximum_audio_seconds = 3600.0;
};

struct EncoderWindow {
    int64_t mel_begin = 0;
    int64_t mel_end = 0;
    int64_t left_drop = 0;
    int64_t core_frames = 0;
    int64_t right_mel_frames = 0;
};

struct EncoderOutput {
    int64_t frames = 0;
    int64_t native_speaker_frames = 0;
    int embedding_dim = 0;
    int asr_dim = 0;
    int speaker_count = 0;
    double duration_seconds = 0.0;
    // Row-major [frames, embedding_dim], [frames, asr_dim], [frames, speakers].
    std::vector<float> audio_embeddings;
    std::vector<float> asr_states;
    std::vector<float> speaker_probs80;
    // Native 10 ms grid. Padded subframes are excluded from this public result.
    std::vector<float> speaker_probs10;
};

void validate_encoder_config(const EncoderConfig & config);
std::vector<EncoderWindow> plan_encoder_windows(int64_t mel_frames, const EncoderConfig & config);

class ParallelExpertEncoder {
public:
    // Source is read only while constructing/uploading weights. Execution must
    // outlive the encoder; its backend and graph resources stay framework-owned.
    ParallelExpertEncoder(
        const core::ExecutionContext & execution,
        const assets::TensorSource & source,
        EncoderConfig config = {},
        assets::TensorStorageType storage = assets::TensorStorageType::Native);
    ~ParallelExpertEncoder();
    ParallelExpertEncoder(const ParallelExpertEncoder &) = delete;
    ParallelExpertEncoder & operator=(const ParallelExpertEncoder &) = delete;

    // Cancellation is observed between graph launches and before every window.
    EncoderOutput encode(const MelFeatures & features, const std::function<bool()> & should_stop = {});
    const EncoderConfig & config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::nemotron_transcribe
