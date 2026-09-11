#include "engine/models/dramabox/audio_vae.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kVaeDecoderWeightContextBytes = 900ull * 1024ull * 1024ull;
constexpr size_t kVaeGraphContextBytes = 384ull * 1024ull * 1024ull;
constexpr size_t kVaeGraphNodeCapacity = 65536;
constexpr float kPixelNormEps = 1.0e-6F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

DramaBoxVaeResnetBlockWeights load_vae_resnet_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels) {
    DramaBoxVaeResnetBlockWeights weights;
    weights.config = {in_channels, out_channels, 3, 1, kPixelNormEps};
    weights.block.conv1 = modules::binding::conv2d_from_source(
        store,
        source,
        prefix + ".conv1.conv",
        storage_type,
        out_channels,
        in_channels,
        3,
        3,
        true);
    weights.block.conv2 = modules::binding::conv2d_from_source(
        store,
        source,
        prefix + ".conv2.conv",
        storage_type,
        out_channels,
        out_channels,
        3,
        3,
        true);
    if (in_channels != out_channels) {
        weights.block.shortcut = modules::binding::conv2d_from_source(
            store,
            source,
            prefix + ".nin_shortcut.conv",
            storage_type,
            out_channels,
            in_channels,
            1,
            1,
            true);
    }
    return weights;
}

core::TensorValue run_downsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv2dWeights & weights,
    int64_t channels) {
    auto padded = modules::Pad2dModule({0, 1, 2, 0}).build(ctx, input);
    return modules::Conv2dModule({
        channels,
        channels,
        3,
        3,
        2,
        2,
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, padded, weights);
}

std::vector<float> stereo_resampled_reference(
    const runtime::AudioBuffer & audio,
    int target_sample_rate,
    float ref_duration) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("DramaBox reference audio is empty");
    }
    std::vector<float> channels[2];
    if (audio.channels == 1) {
        channels[0] = audio.samples;
        channels[1] = audio.samples;
    } else {
        channels[0] = audio::extract_interleaved_channel(audio.samples, audio.channels, 0);
        channels[1] = audio::extract_interleaved_channel(audio.samples, audio.channels, 1);
    }
    const audio::TorchaudioSincHannResampleOptions resample_options{};
    for (auto & channel : channels) {
        if (audio.sample_rate != target_sample_rate) {
            channel = audio::resample_mono_torchaudio_sinc_hann(
                channel,
                audio.sample_rate,
                target_sample_rate,
                resample_options);
        }
        const int64_t target_samples = std::max<int64_t>(
            1,
            static_cast<int64_t>(static_cast<double>(ref_duration) * static_cast<double>(target_sample_rate)));
        if (static_cast<int64_t>(channel.size()) < target_samples) {
            const auto original = channel;
            if (original.empty()) {
                throw std::runtime_error("DramaBox reference audio resampled to empty channel");
            }
            while (static_cast<int64_t>(channel.size()) < target_samples) {
                channel.insert(channel.end(), original.begin(), original.end());
            }
        }
        channel.resize(static_cast<size_t>(target_samples));
    }
    float peak = 0.0F;
    for (const auto & channel : channels) {
        for (const float value : channel) {
            peak = std::max(peak, std::fabs(value));
        }
    }
    if (peak > 0.0F) {
        const float gain = std::pow(10.0F, -4.0F / 20.0F) / peak;
        for (auto & channel : channels) {
            for (float & value : channel) {
                value *= gain;
            }
        }
    }
    std::vector<float> planar(static_cast<size_t>(2 * channels[0].size()), 0.0F);
    std::copy(channels[0].begin(), channels[0].end(), planar.begin());
    std::copy(channels[1].begin(), channels[1].end(), planar.begin() + static_cast<std::ptrdiff_t>(channels[0].size()));
    return planar;
}

}  // namespace

std::vector<float> reference_log_mel(
    const runtime::AudioBuffer & audio_buffer,
    const DramaBoxConfig & config,
    float ref_duration,
    int threads,
    int64_t & frames_out) {
    const auto planar = stereo_resampled_reference(audio_buffer, static_cast<int>(config.audio_vae.sample_rate), ref_duration);
    const int64_t samples = static_cast<int64_t>(planar.size()) / 2;
    const audio::STFTConfig stft_config{
        config.audio_vae.n_fft,
        config.audio_vae.hop_length,
        config.audio_vae.n_fft,
        true,
        audio::STFTPadMode::Reflect,
        audio::STFTFamily::Default,
    };
    const auto & window = audio::get_cached_stft_window(stft_config);
    auto magnitude = audio::STFT().compute_magnitude(
        planar,
        window,
        2,
        samples,
        stft_config,
        static_cast<size_t>(std::max(1, threads)));
    frames_out = magnitude.shape[2];
    const auto filterbank = audio::MelFilterbank().build(
        audio::MelFilterbankConfig{
            config.audio_vae.sample_rate,
            config.audio_vae.n_fft,
            config.audio_vae.mel_bins,
            0.0F,
            static_cast<float>(config.audio_vae.sample_rate) / 2.0F,
            true,
        });
    auto mel = audio::MelFilterbank().compute_custom(
        magnitude.values,
        2,
        magnitude.shape[1],
        magnitude.shape[2],
        filterbank);
    for (float & value : mel.values) {
        value = std::log(std::max(value, 1.0e-5F));
    }
    std::vector<float> out(static_cast<size_t>(2 * frames_out * config.audio_vae.mel_bins), 0.0F);
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t m = 0; m < config.audio_vae.mel_bins; ++m) {
            for (int64_t t = 0; t < frames_out; ++t) {
                const size_t src = static_cast<size_t>(((c * config.audio_vae.mel_bins + m) * frames_out) + t);
                const size_t dst = static_cast<size_t>((c * frames_out + t) * config.audio_vae.mel_bins + m);
                out[dst] = mel.values[src];
            }
        }
    }
    return out;
}

DramaBoxAudioVaeDecoderWeights load_dramabox_audio_vae_decoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config.audio_vae;
    const auto & source = *assets.audio_weights;
    DramaBoxAudioVaeDecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.audio_vae_decoder.weights",
        weight_context_bytes == 0 ? kVaeDecoderWeightContextBytes : weight_context_bytes);
    weights.latent_mean = source.require_f32(
        "audio_vae.per_channel_statistics.mean-of-means",
        {config.latent_channels * config.latent_mel_bins});
    weights.latent_std = source.require_f32(
        "audio_vae.per_channel_statistics.std-of-means",
        {config.latent_channels * config.latent_mel_bins});
    const auto conv_storage_type = weight_storage_type == assets::TensorStorageType::Native
        ? assets::TensorStorageType::F32
        : weight_storage_type;
    const int64_t base = config.ch;
    const int64_t high = base * config.ch_mult.back();
    weights.conv_in = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.decoder.conv_in.conv", conv_storage_type, high, config.latent_channels, 3, 3, true);
    weights.mid_block_1 = load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.mid.block_1", conv_storage_type, high, high);
    weights.mid_block_2 = load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.mid.block_2", conv_storage_type, high, high);
    weights.up.resize(3);
    weights.up[2].channels = high;
    weights.up[2].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.2.block.0", conv_storage_type, high, high));
    weights.up[2].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.2.block.1", conv_storage_type, high, high));
    weights.up[2].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.2.block.2", conv_storage_type, high, high));
    weights.up[2].upsample = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.decoder.up.2.upsample.conv.conv", conv_storage_type, high, high, 3, 3, true);
    weights.up[1].channels = high / 2;
    weights.up[1].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.1.block.0", conv_storage_type, high, high / 2));
    weights.up[1].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.1.block.1", conv_storage_type, high / 2, high / 2));
    weights.up[1].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.1.block.2", conv_storage_type, high / 2, high / 2));
    weights.up[1].upsample = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.decoder.up.1.upsample.conv.conv", conv_storage_type, high / 2, high / 2, 3, 3, true);
    weights.up[0].channels = base;
    weights.up[0].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.0.block.0", conv_storage_type, high / 2, base));
    weights.up[0].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.0.block.1", conv_storage_type, base, base));
    weights.up[0].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.decoder.up.0.block.2", conv_storage_type, base, base));
    weights.conv_out = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.decoder.conv_out.conv", conv_storage_type, config.out_channels, base, 3, 3, true);
    weights.store->upload();
    return weights;
}

DramaBoxAudioVaeEncoderWeights load_dramabox_audio_vae_encoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config.audio_vae;
    const auto & source = *assets.audio_weights;
    DramaBoxAudioVaeEncoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.audio_vae_encoder.weights",
        weight_context_bytes == 0 ? kVaeDecoderWeightContextBytes : weight_context_bytes);
    weights.latent_mean = source.require_f32(
        "audio_vae.per_channel_statistics.mean-of-means",
        {config.latent_channels * config.latent_mel_bins});
    weights.latent_std = source.require_f32(
        "audio_vae.per_channel_statistics.std-of-means",
        {config.latent_channels * config.latent_mel_bins});
    const auto conv_storage_type = weight_storage_type == assets::TensorStorageType::Native
        ? assets::TensorStorageType::F32
        : weight_storage_type;
    const int64_t base = config.ch;
    weights.conv_in = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.encoder.conv_in.conv", conv_storage_type, base, config.out_channels, 3, 3, true);
    weights.down.resize(3);
    weights.down[0].channels = base;
    weights.down[0].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.0.block.0", conv_storage_type, base, base));
    weights.down[0].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.0.block.1", conv_storage_type, base, base));
    weights.down[0].downsample = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.encoder.down.0.downsample.conv", conv_storage_type, base, base, 3, 3, true);
    weights.down[1].channels = base * 2;
    weights.down[1].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.1.block.0", conv_storage_type, base, base * 2));
    weights.down[1].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.1.block.1", conv_storage_type, base * 2, base * 2));
    weights.down[1].downsample = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.encoder.down.1.downsample.conv", conv_storage_type, base * 2, base * 2, 3, 3, true);
    weights.down[2].channels = base * 4;
    weights.down[2].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.2.block.0", conv_storage_type, base * 2, base * 4));
    weights.down[2].blocks.push_back(load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.down.2.block.1", conv_storage_type, base * 4, base * 4));
    weights.mid_block_1 = load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.mid.block_1", conv_storage_type, base * 4, base * 4);
    weights.mid_block_2 = load_vae_resnet_block(*weights.store, source, "audio_vae.encoder.mid.block_2", conv_storage_type, base * 4, base * 4);
    weights.conv_out = modules::binding::conv2d_from_source(
        *weights.store, source, "audio_vae.encoder.conv_out.conv", conv_storage_type, config.latent_channels * 2, base * 4, 3, 3, true);
    weights.store->upload();
    return weights;
}

class DramaBoxAudioVaeDecoderRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxAudioVaeDecoderWeights & weights,
        int64_t batch,
        int64_t latent_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(batch),
          latent_frames_(latent_frames),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox audio VAE backend initialization failed");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t latent_frames) const noexcept {
        return batch == batch_ && latent_frames == latent_frames_;
    }

    DramaBoxDecodedMel decode(
        const std::vector<float> & patch_latents,
        int64_t batch,
        int64_t latent_frames,
        bool read_output) const {
        if (!matches(batch, latent_frames)) {
            throw std::runtime_error("DramaBox audio VAE input shape does not match prepared graph");
        }
        const auto input_start = Clock::now();
        const auto & config = assets_->config.audio_vae;
        const size_t expected = static_cast<size_t>(
            batch_ * latent_frames_ * config.latent_channels * config.latent_mel_bins);
        if (patch_latents.size() != expected) {
            throw std::runtime_error("DramaBox audio VAE patch latent size mismatch");
        }
        std::vector<float> denormalized(expected);
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t c = 0; c < config.latent_channels; ++c) {
                for (int64_t t = 0; t < latent_frames_; ++t) {
                    for (int64_t m = 0; m < config.latent_mel_bins; ++m) {
                        const size_t flat = static_cast<size_t>(c * config.latent_mel_bins + m);
                        const size_t src = static_cast<size_t>(
                            (b * latent_frames_ + t) * config.latent_channels * config.latent_mel_bins + flat);
                        const size_t dst = static_cast<size_t>(
                            ((b * config.latent_channels + c) * latent_frames_ + t) * config.latent_mel_bins + m);
                        denormalized[dst] = patch_latents[src] * weights_.latent_std[flat] + weights_.latent_mean[flat];
                    }
                }
            }
        }
        core::write_tensor_f32(input_, denormalized);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.audio_vae.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.audio_vae");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox audio VAE graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.audio_vae.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        DramaBoxDecodedMel out;
        out.batch = batch_;
        out.channels = assets_->config.audio_vae.out_channels;
        out.frames = output_frames_;
        out.mel_bins = assets_->config.audio_vae.mel_bins;
        out.device_values = output_;
        if (read_output) {
            core::read_tensor_f32_into(output_, out.values);
            debug::timing_log_scalar("dramabox.audio_vae.output_read_ms", debug::elapsed_ms(output_start, Clock::now()));
        }
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config.audio_vae;
        output_frames_ = std::max<int64_t>(latent_frames_ * 4 - 3, 1);
        ggml_init_params params{kVaeGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox audio VAE ggml context initialization failed");
        }
        input_ = core::wrap_tensor(
            ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, config.latent_mel_bins, latent_frames_, config.latent_channels, batch_),
            core::TensorShape::from_dims({batch_, config.latent_channels, latent_frames_, config.latent_mel_bins}),
            GGML_TYPE_F32);
        ggml_set_input(input_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.audio_vae", backend_type_};
        auto x = input_;
        x = modules::SameWidthCausalConv2dModule({config.latent_channels, config.ch * config.ch_mult.back(), 3, true})
                .build(build_ctx, x, weights_.conv_in);
        x = modules::PixelNormCausalConv2dResBlockModule(weights_.mid_block_1.config)
                .build(build_ctx, x, weights_.mid_block_1.block);
        x = modules::PixelNormCausalConv2dResBlockModule(weights_.mid_block_2.config)
                .build(build_ctx, x, weights_.mid_block_2.block);
        for (int64_t level = 2; level >= 0; --level) {
            const auto & stage = weights_.up[static_cast<size_t>(level)];
            for (const auto & block : stage.blocks) {
                x = modules::PixelNormCausalConv2dResBlockModule(block.config).build(build_ctx, x, block.block);
            }
            if (stage.upsample.has_value()) {
                x = modules::CausalConv2dUpsampleModule({stage.channels, 3, 2, 2, 1, 0})
                        .build(build_ctx, x, *stage.upsample);
            }
        }
        x = modules::PixelNormModule({1, kPixelNormEps}).build(build_ctx, x);
        x = modules::SiluModule{}.build(build_ctx, x);
        x = modules::SameWidthCausalConv2dModule({config.ch, config.out_channels, 3, true})
                .build(build_ctx, x, weights_.conv_out);
        x = modules::SliceModule({2, 0, output_frames_}).build(build_ctx, x);
        x = modules::SliceModule({3, 0, config.mel_bins}).build(build_ctx, x);
        output_ = core::ensure_backend_addressable_layout(build_ctx, x).tensor;
        ggml_set_output(output_);
        const auto expand_start = Clock::now();
        graph_ = ggml_new_graph_custom(ctx_.get(), kVaeGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        debug::timing_log_scalar("dramabox.audio_vae.graph.expand_ms", debug::elapsed_ms(expand_start, Clock::now()));
        debug::trace_log_scalar("dramabox.audio_vae.graph.nodes", static_cast<int64_t>(ggml_graph_n_nodes(graph_)));
        const auto alloc_start = Clock::now();
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox audio VAE backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.audio_vae.graph.alloc_ms", debug::elapsed_ms(alloc_start, Clock::now()));
        debug::timing_log_scalar("dramabox.audio_vae.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t batch_ = 0;
    int64_t latent_frames_ = 0;
    int64_t output_frames_ = 0;
    const DramaBoxAudioVaeDecoderWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxAudioVaeDecoderRuntime::DramaBoxAudioVaeDecoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox audio VAE runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox audio VAE runtime requires assets");
    }
}

DramaBoxAudioVaeDecoderRuntime::~DramaBoxAudioVaeDecoderRuntime() = default;

void DramaBoxAudioVaeDecoderRuntime::prepare(int64_t batch, int64_t latent_frames) const {
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxAudioVaeDecoderWeights>(load_dramabox_audio_vae_decoder_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_));
    }
    if (!graph_ || !graph_->matches(batch, latent_frames)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(*execution_, assets_, *weights_, batch, latent_frames);
    }
}

DramaBoxDecodedMel DramaBoxAudioVaeDecoderRuntime::decode(
    const std::vector<float> & patch_latents,
    int64_t batch,
    int64_t latent_frames) const {
    prepare(batch, latent_frames);
    return graph_->decode(patch_latents, batch, latent_frames, true);
}

DramaBoxDecodedMel DramaBoxAudioVaeDecoderRuntime::decode_to_device(
    const std::vector<float> & patch_latents,
    int64_t batch,
    int64_t latent_frames) const {
    prepare(batch, latent_frames);
    return graph_->decode(patch_latents, batch, latent_frames, false);
}

void DramaBoxAudioVaeDecoderRuntime::release_runtime_state() const {
    graph_.reset();
    weights_.reset();
}

class DramaBoxAudioVaeEncoderRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxAudioVaeEncoderWeights & weights,
        int64_t batch,
        int64_t mel_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(batch),
          mel_frames_(mel_frames),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox audio VAE encoder backend initialization failed");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch, int64_t mel_frames) const noexcept {
        return batch == batch_ && mel_frames == mel_frames_;
    }

    DramaBoxEncodedReferenceLatents encode(const std::vector<float> & mel, int64_t batch, int64_t mel_frames) const {
        if (!matches(batch, mel_frames)) {
            throw std::runtime_error("DramaBox audio VAE encoder input shape does not match prepared graph");
        }
        const auto input_start = Clock::now();
        core::write_tensor_f32(input_, mel);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.audio_vae_encoder.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.audio_vae_encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox audio VAE encoder graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.audio_vae_encoder.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        const auto raw = core::read_tensor_f32(output_);
        const auto & config = assets_->config.audio_vae;
        DramaBoxEncodedReferenceLatents out;
        out.tokens = latent_frames_;
        out.values.resize(static_cast<size_t>(batch_ * latent_frames_ * config.latent_channels * config.latent_mel_bins), 0.0F);
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t t = 0; t < latent_frames_; ++t) {
                for (int64_t c = 0; c < config.latent_channels; ++c) {
                    for (int64_t m = 0; m < config.latent_mel_bins; ++m) {
                        const size_t src = static_cast<size_t>(
                            ((b * config.latent_channels + c) * latent_frames_ + t) * config.latent_mel_bins + m);
                        const size_t flat = static_cast<size_t>(c * config.latent_mel_bins + m);
                        const size_t dst = static_cast<size_t>(
                            (b * latent_frames_ + t) * config.latent_channels * config.latent_mel_bins + flat);
                        out.values[dst] = (raw[src] - weights_.latent_mean[flat]) / weights_.latent_std[flat];
                    }
                }
            }
        }
        debug::timing_log_scalar("dramabox.audio_vae_encoder.output_read_ms", debug::elapsed_ms(output_start, Clock::now()));
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config.audio_vae;
        latent_frames_ = std::max<int64_t>(1, (mel_frames_ + 3) / 4);
        ggml_init_params params{2ull * 1024ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox audio VAE encoder ggml context initialization failed");
        }
        input_ = core::wrap_tensor(
            ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, config.mel_bins, mel_frames_, config.out_channels, batch_),
            core::TensorShape::from_dims({batch_, config.out_channels, mel_frames_, config.mel_bins}),
            GGML_TYPE_F32);
        ggml_set_input(input_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.audio_vae_encoder", backend_type_};
        auto x = modules::SameWidthCausalConv2dModule({config.out_channels, config.ch, 3, true})
                     .build(build_ctx, input_, weights_.conv_in);
        for (int64_t level = 0; level < 3; ++level) {
            const auto & stage = weights_.down[static_cast<size_t>(level)];
            for (const auto & block : stage.blocks) {
                x = modules::PixelNormCausalConv2dResBlockModule(block.config).build(build_ctx, x, block.block);
            }
            if (stage.downsample.has_value()) {
                x = run_downsample(build_ctx, x, *stage.downsample, stage.channels);
            }
        }
        x = modules::PixelNormCausalConv2dResBlockModule(weights_.mid_block_1.config)
                .build(build_ctx, x, weights_.mid_block_1.block);
        x = modules::PixelNormCausalConv2dResBlockModule(weights_.mid_block_2.config)
                .build(build_ctx, x, weights_.mid_block_2.block);
        x = modules::PixelNormModule({1, kPixelNormEps}).build(build_ctx, x);
        x = modules::SiluModule{}.build(build_ctx, x);
        x = modules::SameWidthCausalConv2dModule({config.ch * 4, config.latent_channels * 2, 3, true})
                .build(build_ctx, x, weights_.conv_out);
        x = modules::SliceModule({1, 0, config.latent_channels}).build(build_ctx, x);
        x = modules::SliceModule({2, 0, latent_frames_}).build(build_ctx, x);
        x = modules::SliceModule({3, 0, config.latent_mel_bins}).build(build_ctx, x);
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kVaeGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox audio VAE encoder backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.audio_vae_encoder.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t batch_ = 0;
    int64_t mel_frames_ = 0;
    int64_t latent_frames_ = 0;
    const DramaBoxAudioVaeEncoderWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxAudioVaeEncoderRuntime::DramaBoxAudioVaeEncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox audio VAE encoder runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox audio VAE encoder runtime requires assets");
    }
}

DramaBoxAudioVaeEncoderRuntime::~DramaBoxAudioVaeEncoderRuntime() = default;

void DramaBoxAudioVaeEncoderRuntime::prepare(int64_t batch, int64_t mel_frames) const {
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxAudioVaeEncoderWeights>(load_dramabox_audio_vae_encoder_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_));
    }
    if (!graph_ || !graph_->matches(batch, mel_frames)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(*execution_, assets_, *weights_, batch, mel_frames);
    }
}

DramaBoxEncodedReferenceLatents DramaBoxAudioVaeEncoderRuntime::encode(
    const std::vector<float> & mel,
    int64_t batch,
    int64_t mel_frames) const {
    prepare(batch, mel_frames);
    return graph_->encode(mel, batch, mel_frames);
}

void DramaBoxAudioVaeEncoderRuntime::release_runtime_state() const {
    graph_.reset();
    weights_.reset();
}

}  // namespace engine::models::dramabox
