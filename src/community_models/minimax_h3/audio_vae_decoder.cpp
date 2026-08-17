#include "engine/community_models/minimax_h3/audio_vae_decoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"

#include "ggml-alloc.h"

#include <stdexcept>
#include <utility>

namespace engine::models::minimax_h3 {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;

struct AudioVaeGgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

AudioVaeWeightStore::AudioVaeWeightStore(
    core::ExecutionContext & execution_context,
    std::shared_ptr<const assets::TensorSource> tensor_source,
    const MiniMaxH3Config & cfg,
    size_t weight_context_bytes)
    : execution(execution_context),
      source(std::move(tensor_source)),
      store(execution.backend(), execution.backend_type(), "minimax_h3.audio_vae", weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-H3 audio VAE tensor source is missing");
    }
    latent_mean = store.load_tensor(
        *source,
        "latent_mean",
        assets::TensorStorageType::F32,
        {cfg.audio_vae_latent_channels});
    latent_std = store.load_tensor(
        *source,
        "latent_std",
        assets::TensorStorageType::F32,
        {cfg.audio_vae_latent_channels});
    dec_in.weight = store.load_tensor(
        *source,
        "dec_in_proj.weight",
        assets::TensorStorageType::Native,
        {cfg.audio_vae_latent_dim, cfg.audio_vae_latent_channels, 1});
    dec_in.bias = store.load_tensor(
        *source,
        "dec_in_proj.bias",
        assets::TensorStorageType::F32,
        {cfg.audio_vae_latent_dim});
    modules::BigVganVocoderConfig bigvgan;
    bigvgan.sampling_rate = cfg.sample_rate;
    bigvgan.num_mels = cfg.audio_vae_latent_dim;
    bigvgan.n_fft = 1024;
    bigvgan.hop_size = 512;
    bigvgan.win_size = 1024;
    bigvgan.upsample_initial_channel = cfg.audio_vae_decoder_dim;
    bigvgan.snake_logscale = true;
    bigvgan.upsample_rates = cfg.audio_vae_decoder_rates;
    bigvgan.upsample_kernel_sizes = cfg.audio_vae_decoder_kernels;
    bigvgan.resblock_kernel_sizes = {3, 7, 11};
    bigvgan.weight_storage_type = assets::TensorStorageType::Native;
    decoder = modules::load_mono_module_list_bigvgan_from_tensor_source(
        store,
        *source,
        "decoder",
        std::move(bigvgan),
        cfg.audio_vae_bigvgan_weight_norm,
        modules::BigVganActivationLayout::InterleavedPairs);
    store.upload();
    source->release_storage();
}

class AudioVaeDecodeGraph {
public:
    AudioVaeDecodeGraph(AudioVaeWeightStore & weights, const MiniMaxH3Config & cfg)
        : execution_(weights.execution) {
        ctx_.reset(ggml_init({1024 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 audio VAE graph context");
        }
        input_ctx_.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-H3 audio VAE input context");
        }
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_h3.audio_vae.inputs", execution_.backend_type()};
        latent_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({cfg.audio_channels, cfg.audio_vae_latent_channels, cfg.audio_steps}));
        ggml_set_input(latent_.tensor);

        core::ModuleBuildContext build_ctx{ctx_.get(), "minimax_h3.real_audio_vae_decode", execution_.backend_type()};
        auto mean_t = core::reshape_tensor(build_ctx, weights.latent_mean, core::TensorShape::from_dims({1, cfg.audio_vae_latent_channels, 1}));
        auto std_t = core::reshape_tensor(build_ctx, weights.latent_std, core::TensorShape::from_dims({1, cfg.audio_vae_latent_channels, 1}));
        channel_outputs_.reserve(static_cast<size_t>(cfg.audio_channels));
        for (int64_t channel = 0; channel < cfg.audio_channels; ++channel) {
            auto channel_latent = core::reshape_tensor(
                build_ctx,
                core::ensure_backend_addressable_layout(
                    build_ctx,
                    modules::SliceModule({0, channel, 1}).build(build_ctx, latent_)),
                core::TensorShape::from_dims({1, cfg.audio_vae_latent_channels, cfg.audio_steps}));
            auto scaled = core::wrap_tensor(
                ggml_mul(build_ctx.ggml, channel_latent.tensor, std_t.tensor),
                channel_latent.shape,
                GGML_TYPE_F32);
            auto z = core::wrap_tensor(
                ggml_add(build_ctx.ggml, scaled.tensor, mean_t.tensor),
                channel_latent.shape,
                GGML_TYPE_F32);
            z = modules::Conv1dModule({cfg.audio_vae_latent_channels, cfg.audio_vae_latent_dim, 1, 1, 0, 1, true})
                    .build(build_ctx, z, weights.dec_in);
            auto z_2d = core::reshape_tensor(
                build_ctx,
                core::ensure_backend_addressable_layout(build_ctx, z),
                core::TensorShape::from_dims({cfg.audio_vae_latent_dim, cfg.audio_steps}));
            const auto backend_type = weights.execution.backend_type();
            modules::BigVganGraphOptions options;
            // Match padded transposed conv by running padding-zero and cropping on backends
            // that cannot lower padded ConvTranspose1d directly.
            options.lower_padded_conv_transpose_as_crop =
                backend_type == core::BackendType::Cpu ||
                backend_type == core::BackendType::Vulkan ||
                backend_type == core::BackendType::Metal;
            channel_outputs_.push_back(modules::build_bigvgan_graph(
                build_ctx.ggml,
                backend_type,
                weights.decoder,
                z_2d.tensor,
                options));
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : channel_outputs_) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph_, output);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-H3 audio VAE inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-H3 audio VAE graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~AudioVaeDecodeGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    std::vector<std::vector<float>> run(const std::vector<float> & latents) {
        core::write_tensor_f32(latent_, latents);
        core::set_backend_threads(execution_.backend(), 8);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_h3.audio_vae");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-H3 audio VAE graph compute failed");
        }
        std::vector<std::vector<float>> decoded_channels;
        decoded_channels.reserve(channel_outputs_.size());
        for (auto * output : channel_outputs_) {
            decoded_channels.push_back(core::read_tensor_f32(output));
        }
        return decoded_channels;
    }

private:
    core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, AudioVaeGgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, AudioVaeGgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue latent_;
    std::vector<ggml_tensor *> channel_outputs_;
};

std::vector<float> run_audio_vae_decode_graph(
    AudioVaeWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & audio_rows) {
    std::vector<float> latents(static_cast<size_t>(cfg.audio_channels * cfg.audio_vae_latent_channels * cfg.audio_steps));
    for (int64_t channel = 0; channel < cfg.audio_channels; ++channel) {
        for (int64_t t = 0; t < cfg.audio_steps; ++t) {
            const int64_t row = channel * cfg.audio_steps + t;
            for (int64_t d = 0; d < cfg.audio_vae_latent_channels; ++d) {
                latents[static_cast<size_t>((channel * cfg.audio_vae_latent_channels + d) * cfg.audio_steps + t)] =
                    audio_rows[static_cast<size_t>(row * cfg.audio_vae_latent_channels + d)];
            }
        }
    }

    AudioVaeDecodeGraph graph(weights, cfg);
    auto decoded_channels = graph.run(latents);
    if (decoded_channels.empty()) {
        return {};
    }
    const size_t samples = decoded_channels.front().size();
    for (const auto & channel : decoded_channels) {
        if (channel.size() != samples) {
            throw std::runtime_error("MiniMax-H3 audio VAE channel output size mismatch");
        }
    }
    std::vector<float> planar;
    planar.reserve(samples * static_cast<size_t>(cfg.audio_channels));
    for (int64_t channel = 0; channel < cfg.audio_channels; ++channel) {
        const auto & values = decoded_channels[static_cast<size_t>(channel)];
        planar.insert(planar.end(), values.begin(), values.end());
    }
    return engine::audio::interleave_planar_channels(planar, static_cast<int>(cfg.audio_channels), static_cast<int64_t>(samples));
}

}  // namespace engine::models::minimax_h3
