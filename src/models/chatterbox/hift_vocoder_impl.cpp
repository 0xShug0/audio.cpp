#include "components/component_weights.h"

#include "engine/framework/modules/vocoders/hift_vocoder.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::chatterbox {
namespace {

engine::modules::HiftVocoderConfig make_chatterbox_hift_config(
    engine::assets::TensorStorageType weight_storage_type) {
    engine::modules::HiftVocoderConfig config;
    config.in_channels = 80;
    config.base_channels = 512;
    config.nb_harmonics = 8;
    config.sampling_rate = 24000;
    config.nsf_alpha = 0.1F;
    config.nsf_sigma = 0.003F;
    config.nsf_voiced_threshold = 10.0F;
    config.upsample_rates = {8, 5, 3};
    config.upsample_kernel_sizes = {16, 11, 7};
    config.istft_n_fft = 16;
    config.istft_hop = 4;
    config.resblock_kernel_sizes = {3, 7, 11};
    config.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.source_resblock_kernel_sizes = {7, 7, 11};
    config.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    config.lrelu_slope = 0.1F;
    config.audio_limit = 0.99F;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    config.weight_storage_type = weight_storage_type;
    config.tensor_prefix = "mel2wav.";
    config.weight_layout = engine::modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    return config;
}

}  // namespace

struct HiFTVocoderComponent::State {
    engine::modules::HiftVocoderComponent component;
};

HiFTVocoderComponent HiFTVocoderComponent::load_from_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto state = std::make_shared<State>();
    state->component = engine::modules::HiftVocoderComponent::load_from_tensor_source(
        std::move(source),
        execution_context.config(),
        make_chatterbox_hift_config(weight_storage_type));

    auto weights = std::make_shared<HiFTVocoderComponentWeights>();
    weights->runtime_weights = state->component.weights();
    HiFTVocoderComponent component(std::move(weights), execution_context);
    component.state_ = std::move(state);
    return component;
}

HiFTVocoderComponent::HiFTVocoderComponent(
    std::shared_ptr<const HiFTVocoderComponentWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context) {}

const engine::core::BackendConfig & HiFTVocoderComponent::backend() const noexcept {
    return execution_context_->config();
}

const std::shared_ptr<const HiFTVocoderComponentWeights> & HiFTVocoderComponent::weights() const noexcept {
    return weights_;
}

HiFTVocoderOutputs HiFTVocoderComponent::infer(
    const std::vector<float> & speech_feat,
    int64_t batch,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> & cache_source) const {
    if (batch != 1) {
        throw std::runtime_error("Chatterbox HiFT expects batch 1");
    }
    if (!cache_source.empty()) {
        throw std::runtime_error("Chatterbox HiFT does not support cache_source");
    }
    const auto result = state_->component.synthesize(speech_feat, frames, seed, prior_noise_values);
    HiFTVocoderOutputs outputs;
    outputs.waveform = result.waveform;
    outputs.samples = result.samples;
    return outputs;
}

void HiFTVocoderComponent::release_runtime_cache() const {
    if (state_ == nullptr) {
        throw std::runtime_error("Chatterbox HiFT is not initialized");
    }
    state_->component.release_runtime_cache();
}

}  // namespace engine::models::chatterbox
