#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/chatterbox/assets.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace engine::modules {
struct HiftVocoderWeights;
}  // namespace engine::modules

namespace engine::models::chatterbox {

namespace components {
struct CampplusEncoderWeights;
struct S3TokenizerV2Weights;
}  // namespace components

struct SpeakerEncoderOutputs {
    std::vector<float> embedding;
    int64_t embedding_size = 0;
};

struct TokenizerOutputs {
    std::vector<int32_t> tokens;
    int64_t token_count = 0;
};

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

struct VoiceEncoderConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_size = 400;
    int64_t hop_size = 160;
    int64_t num_mels = 40;
    int64_t partial_frames = 160;
    int64_t partial_rate = 1;
    float min_coverage = 0.75F;
    int64_t hidden_size = 256;
    int64_t speaker_embed_size = 256;
    bool final_relu = true;
};

struct VoiceEncoderLayerWeights {
    std::vector<float> weight_ih;
    std::vector<float> weight_hh;
    std::vector<float> bias_ih;
    std::vector<float> bias_hh;
};

struct VoiceEncoderWeights {
    VoiceEncoderConfig config;
    std::vector<VoiceEncoderLayerWeights> lstm_layers;
    std::vector<float> proj_weight;
    std::vector<float> proj_bias;
    std::vector<float> similarity_weight;
    std::vector<float> similarity_bias;
};

struct CampplusEncoderComponentWeights {
    std::shared_ptr<const components::CampplusEncoderWeights> runtime_weights;
};

struct S3TokenizerComponentWeights {
    std::shared_ptr<const components::S3TokenizerV2Weights> runtime_weights;
};

struct HiFTVocoderComponentWeights {
    std::shared_ptr<const engine::modules::HiftVocoderWeights> runtime_weights;
};

std::shared_ptr<const VoiceEncoderWeights> load_voice_encoder_weights(const std::filesystem::path & model_root);

class VoiceEncoderComponent {
public:
    static VoiceEncoderComponent load_from_model_root(
        const std::filesystem::path & model_root,
        engine::core::BackendConfig backend);

    VoiceEncoderComponent(
        std::shared_ptr<const VoiceEncoderWeights> weights,
        engine::core::BackendConfig backend);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const VoiceEncoderWeights> & weights() const noexcept;
    const VoiceEncoderConfig & config() const noexcept;
    std::vector<float> embed_utterance_from_audio(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const VoiceEncoderWeights> weights_;
    engine::core::BackendConfig backend_;
};

class CAMPPlusEncoderComponent {
public:
    static CAMPPlusEncoderComponent load_from_checkpoint(
        const std::filesystem::path & checkpoint_path,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    CAMPPlusEncoderComponent(
        std::shared_ptr<const CampplusEncoderComponentWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const CampplusEncoderComponentWeights> & weights() const noexcept;
    SpeakerEncoderOutputs embed_from_audio(const runtime::AudioBuffer & audio) const;
    SpeakerEncoderOutputs embed_from_features(
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims) const;

private:
    struct State;
    std::shared_ptr<const CampplusEncoderComponentWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

class S3TokenizerComponent {
public:
    static S3TokenizerComponent load_from_checkpoint(
        const std::filesystem::path & checkpoint_path,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    S3TokenizerComponent(
        std::shared_ptr<const S3TokenizerComponentWeights> weights,
        const engine::core::ExecutionContext & execution_context);

    const engine::core::BackendConfig & backend() const noexcept;
    const std::shared_ptr<const S3TokenizerComponentWeights> & weights() const noexcept;
    TokenizerOutputs tokenize(
        const runtime::AudioBuffer & audio,
        std::optional<int64_t> max_len) const;
    EmbedReferenceOutputs embed_reference(
        const CampplusEncoderComponentWeights & speaker_weights,
        const runtime::AudioBuffer & audio) const;
    EmbedReferenceOutputs embed_reference_from_wavs(
        const CampplusEncoderComponentWeights & speaker_weights,
        const runtime::AudioBuffer & audio_24k,
        const runtime::AudioBuffer & audio_16k) const;

private:
    struct State;
    std::shared_ptr<const S3TokenizerComponentWeights> weights_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<State> state_;
};

class HiFTVocoderComponent {
public:
    static HiFTVocoderComponent load_from_checkpoint(
        const std::filesystem::path & checkpoint_path,
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

}  // namespace engine::models::chatterbox
