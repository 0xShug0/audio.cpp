#include "engine/models/audiosr/hifigan.h"

#include "engine/framework/modules/vocoders/hifigan_vocoder.h"

#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::audiosr {
namespace {

engine::modules::HifiGanVocoderConfig make_hifigan_config(
    const AudioSRConfig & config,
    engine::assets::TensorStorageType weight_type) {
    engine::modules::HifiGanVocoderConfig out;
    out.sampling_rate = config.sample_rate;
    out.num_mels = config.vocoder_num_mels;
    out.upsample_initial_channel = config.vocoder_upsample_initial_channel;
    out.output_channels = 1;
    out.upsample_rates.assign(std::begin(config.vocoder_upsample_rates), std::end(config.vocoder_upsample_rates));
    out.upsample_kernel_sizes.assign(std::begin(config.vocoder_upsample_kernels), std::end(config.vocoder_upsample_kernels));
    out.resblock_kernel_sizes.assign(std::begin(config.vocoder_resblock_kernels), std::end(config.vocoder_resblock_kernels));
    out.resblock_dilation_sizes = {
        {1, 3, 5},
        {1, 3, 5},
        {1, 3, 5},
        {1, 3, 5},
    };
    out.leaky_relu_slope = 0.1F;
    out.weight_storage_type = weight_type;
    out.tensor_prefix = "first_stage_model.vocoder";
    return out;
}

}  // namespace

struct AudioSRHiFiGanRuntime::Impl {
    explicit Impl(engine::modules::HifiGanVocoderComponent component)
        : component(std::move(component)) {}

    engine::modules::HifiGanVocoderComponent component;
};

AudioSRHiFiGanRuntime::AudioSRHiFiGanRuntime(
    std::shared_ptr<const AudioSRAssets> assets,
    core::ExecutionContext & execution,
    engine::assets::TensorStorageType weight_type) {
    if (assets == nullptr || assets->weights == nullptr) {
        throw std::runtime_error("AudioSR HiFi-GAN requires tensor source");
    }
    impl_ = std::make_unique<Impl>(engine::modules::HifiGanVocoderComponent::load_from_tensor_source(
        assets->weights,
        execution.config(),
        make_hifigan_config(assets->config, weight_type)));
}

AudioSRHiFiGanRuntime::~AudioSRHiFiGanRuntime() = default;

engine::runtime::AudioBuffer AudioSRHiFiGanRuntime::synthesize(const std::vector<float> & mel, int64_t frames) {
    if (impl_ == nullptr) {
        throw std::runtime_error("AudioSR HiFi-GAN runtime is not initialized");
    }
    auto out = impl_->component.synthesize(mel, frames);
    return engine::runtime::AudioBuffer{static_cast<int>(out.sample_rate), 1, std::move(out.waveform)};
}

void AudioSRHiFiGanRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->component.release_runtime_graph();
    }
}

}  // namespace engine::models::audiosr
