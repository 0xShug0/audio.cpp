#include "engine/models/cosyvoice3/hift.h"

#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/errors.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::cosyvoice3 {
namespace {

engine::modules::HiftVocoderConfig make_hift_config(engine::assets::TensorStorageType storage_type) {
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
    config.conv_pre_kernel_size = 5;
    config.causal_convolutions = true;
    config.f0_num_class = 1;
    config.f0_in_channels = 80;
    config.f0_cond_channels = 512;
    config.f0_condnet_kernel_sizes = {4, 3, 3, 3, 3};
    config.weight_storage_type = storage_type;
    config.weight_layout = engine::modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    config.upsample_mode = engine::modules::HiftVocoderUpsampleMode::CausalConv1dNearest;
    config.source_mode = engine::modules::HiftVocoderSourceMode::CausalSineGen2;
    return config;
}

}  // namespace

class CosyVoice3HiftRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType storage_type)
        : assets_(std::move(assets)), execution_(execution) {
        if (assets_ == nullptr) {
            throw std::runtime_error("CosyVoice3 HiFT runtime requires assets");
        }
        component_ = engine::modules::HiftVocoderComponent::load_from_tensor_source(
            assets_->hift_weights,
            execution.config(),
            make_hift_config(storage_type));
    }

    engine::runtime::AudioBuffer synthesize(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed,
        float speed) {
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != frames * 80) {
            throw std::runtime_error("CosyVoice3 HiFT mel shape mismatch");
        }
        std::vector<float> channel_major(mel.size());
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t channel = 0; channel < 80; ++channel) {
                channel_major[static_cast<size_t>(channel * frames + frame)] =
                    mel[static_cast<size_t>(frame * 80 + channel)];
            }
        }
        if (speed != 1.0F) {
            const float scaled_frames = static_cast<float>(frames) / speed;
            if (!std::isfinite(scaled_frames) || scaled_frames < 1.0F ||
                static_cast<double>(scaled_frames) >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
                throw runtime::InvalidRequestError("CosyVoice3 speed produces an invalid mel length");
            }
            const auto output_frames = static_cast<int64_t>(scaled_frames);
            if (!speed_graph_ || speed_input_.tensor->ne[0] != frames ||
                speed_output_.tensor->ne[0] != output_frames) {
                speed_graph_ = std::make_unique<core::PreparedGraph>(execution_,
                    core::PreparedGraphConfig{"cosyvoice3.speed", 1024 * 1024, 128});
                auto & ctx = speed_graph_->build_context();
                speed_input_ = core::make_tensor(ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 80, frames}));
                ggml_set_input(speed_input_.tensor);
                speed_output_ = modules::Interpolate1dModule({output_frames,
                    modules::Interpolate1dMode::Linear, false}).build(ctx, speed_input_);
                ggml_set_output(speed_output_.tensor);
                ggml_build_forward_expand(speed_graph_->graph(), speed_output_.tensor);
                speed_graph_->allocate();
            }
            core::write_tensor_f32(speed_input_, channel_major);
            speed_graph_->compute();
            channel_major = core::read_tensor_f32(speed_output_.tensor);
            frames = output_frames;
        }
        auto out = component_.synthesize(channel_major, frames, seed, 0);
        engine::runtime::AudioBuffer audio;
        audio.sample_rate = static_cast<int>(out.sample_rate);
        audio.channels = 1;
        audio.samples = std::move(out.waveform);
        return audio;
    }

    void release_graphs() {
        component_.release_runtime_cache();
        speed_graph_.reset();
    }

private:
    std::shared_ptr<const CosyVoice3Assets> assets_;
    core::ExecutionContext & execution_;
    engine::modules::HiftVocoderComponent component_;
    std::unique_ptr<core::PreparedGraph> speed_graph_;
    core::TensorValue speed_input_, speed_output_;
};

CosyVoice3HiftRuntime::CosyVoice3HiftRuntime(
    std::shared_ptr<const CosyVoice3Assets> assets,
    engine::core::ExecutionContext & execution,
    engine::assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type)) {}

CosyVoice3HiftRuntime::~CosyVoice3HiftRuntime() = default;

engine::runtime::AudioBuffer CosyVoice3HiftRuntime::synthesize(
    const std::vector<float> & mel,
    int64_t frames,
    uint64_t seed,
    float speed) {
    return impl_->synthesize(mel, frames, seed, speed);
}

void CosyVoice3HiftRuntime::release_graphs() {
    impl_->release_graphs();
}

}  // namespace engine::models::cosyvoice3
