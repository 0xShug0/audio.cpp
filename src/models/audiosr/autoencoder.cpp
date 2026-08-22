#include "engine/models/audiosr/autoencoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::audiosr {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kVaeLevels = 4;
constexpr int64_t kVaeResBlocks = 2;
constexpr int64_t kVaeDecodeBlocks = 3;
constexpr int64_t kVaeGroups = 32;
constexpr float kVaeNormEps = 1.0e-6F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct VaeResBlockWeights {
    modules::NormWeights norm1;
    modules::Conv2dWeights conv1;
    modules::NormWeights norm2;
    modules::Conv2dWeights conv2;
    std::optional<modules::Conv2dWeights> shortcut;
};

struct VaeAttentionWeights {
    modules::NormWeights norm;
    modules::Conv2dWeights q;
    modules::Conv2dWeights k;
    modules::Conv2dWeights v;
    modules::Conv2dWeights proj_out;
};

struct VaeEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights conv_in;
    VaeResBlockWeights down[kVaeLevels][kVaeResBlocks];
    modules::Conv2dWeights downsample[kVaeLevels - 1];
    VaeResBlockWeights mid_block_1;
    VaeAttentionWeights mid_attn;
    VaeResBlockWeights mid_block_2;
    modules::NormWeights norm_out;
    modules::Conv2dWeights conv_out;
    modules::Conv2dWeights quant_conv;
};

struct VaeDecoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights post_quant_conv;
    modules::Conv2dWeights conv_in;
    VaeResBlockWeights mid_block_1;
    VaeAttentionWeights mid_attn;
    VaeResBlockWeights mid_block_2;
    VaeResBlockWeights up[kVaeLevels][kVaeDecodeBlocks];
    modules::Conv2dWeights upsample[kVaeLevels - 1];
    modules::NormWeights norm_out;
    modules::Conv2dWeights conv_out;
};

int64_t level_channels(const AudioSRConfig & config, int64_t level) {
    constexpr int64_t mult[kVaeLevels] = {1, 2, 4, 8};
    return config.vae_base_channels * mult[level];
}

modules::Conv2dWeights conv2d(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    return binding::conv2d_from_source(store, source, prefix, type, out_channels, in_channels, kernel, kernel, true);
}

modules::NormWeights norm(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return binding::norm_from_source(store, source, prefix, channels);
}

VaeResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t in_channels,
    int64_t out_channels) {
    VaeResBlockWeights out;
    out.norm1 = norm(store, source, prefix + ".norm1", in_channels);
    out.conv1 = conv2d(store, source, prefix + ".conv1", type, out_channels, in_channels, 3);
    out.norm2 = norm(store, source, prefix + ".norm2", out_channels);
    out.conv2 = conv2d(store, source, prefix + ".conv2", type, out_channels, out_channels, 3);
    if (in_channels != out_channels) {
        out.shortcut = conv2d(store, source, prefix + ".nin_shortcut", type, out_channels, in_channels, 1);
    }
    return out;
}

VaeAttentionWeights load_attention(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType type,
    int64_t channels) {
    VaeAttentionWeights out;
    out.norm = norm(store, source, prefix + ".norm", channels);
    out.q = conv2d(store, source, prefix + ".q", type, channels, channels, 1);
    out.k = conv2d(store, source, prefix + ".k", type, channels, channels, 1);
    out.v = conv2d(store, source, prefix + ".v", type, channels, channels, 1);
    out.proj_out = conv2d(store, source, prefix + ".proj_out", type, channels, channels, 1);
    return out;
}

VaeEncoderWeights load_encoder_weights(
    const AudioSRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const std::string & prefix,
    engine::assets::TensorStorageType type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("AudioSR VAE encoder requires tensor source");
    }
    VaeEncoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audiosr.vae.encoder.weights",
        2048ull * 1024ull * 1024ull);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    weights.conv_in = conv2d(*weights.store, source, prefix + ".encoder.conv_in", type, config.vae_base_channels, 1, 3);
    int64_t in_channels = config.vae_base_channels;
    for (int64_t level = 0; level < kVaeLevels; ++level) {
        const int64_t out_channels = level_channels(config, level);
        for (int64_t block = 0; block < kVaeResBlocks; ++block) {
            weights.down[level][block] = load_resblock(
                *weights.store,
                source,
                prefix + ".encoder.down." + std::to_string(level) + ".block." + std::to_string(block),
                type,
                in_channels,
                out_channels);
            in_channels = out_channels;
        }
        if (level != kVaeLevels - 1) {
            weights.downsample[level] = conv2d(
                *weights.store,
                source,
                prefix + ".encoder.down." + std::to_string(level) + ".downsample.conv",
                type,
                in_channels,
                in_channels,
                3);
        }
    }
    weights.mid_block_1 = load_resblock(*weights.store, source, prefix + ".encoder.mid.block_1", type, in_channels, in_channels);
    weights.mid_attn = load_attention(*weights.store, source, prefix + ".encoder.mid.attn_1", type, in_channels);
    weights.mid_block_2 = load_resblock(*weights.store, source, prefix + ".encoder.mid.block_2", type, in_channels, in_channels);
    weights.norm_out = norm(*weights.store, source, prefix + ".encoder.norm_out", in_channels);
    weights.conv_out = conv2d(*weights.store, source, prefix + ".encoder.conv_out", type, 2 * config.encoder_z_channels, in_channels, 3);
    weights.quant_conv = conv2d(*weights.store, source, prefix + ".quant_conv", type, 2 * config.latent_channels, 2 * config.encoder_z_channels, 1);
    weights.store->upload();
    return weights;
}

VaeDecoderWeights load_decoder_weights(
    const AudioSRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    engine::assets::TensorStorageType type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("AudioSR VAE decoder requires tensor source");
    }
    VaeDecoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audiosr.vae.decoder.weights",
        2048ull * 1024ull * 1024ull);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    weights.post_quant_conv = conv2d(*weights.store, source, "first_stage_model.post_quant_conv", type, config.encoder_z_channels, config.latent_channels, 1);
    weights.conv_in = conv2d(*weights.store, source, "first_stage_model.decoder.conv_in", type, level_channels(config, 3), config.encoder_z_channels, 3);
    int64_t channels = level_channels(config, 3);
    weights.mid_block_1 = load_resblock(*weights.store, source, "first_stage_model.decoder.mid.block_1", type, channels, channels);
    weights.mid_attn = load_attention(*weights.store, source, "first_stage_model.decoder.mid.attn_1", type, channels);
    weights.mid_block_2 = load_resblock(*weights.store, source, "first_stage_model.decoder.mid.block_2", type, channels, channels);
    for (int64_t level = kVaeLevels - 1; level >= 0; --level) {
        const int64_t out_channels = level_channels(config, level);
        const int64_t stored_level = level;
        for (int64_t block = 0; block < kVaeDecodeBlocks; ++block) {
            weights.up[stored_level][block] = load_resblock(
                *weights.store,
                source,
                "first_stage_model.decoder.up." + std::to_string(stored_level) + ".block." + std::to_string(block),
                type,
                channels,
                out_channels);
            channels = out_channels;
        }
        if (level != 0) {
            weights.upsample[stored_level - 1] = conv2d(
                *weights.store,
                source,
                "first_stage_model.decoder.up." + std::to_string(stored_level) + ".upsample.conv",
                type,
                channels,
                channels,
                3);
        }
    }
    weights.norm_out = norm(*weights.store, source, "first_stage_model.decoder.norm_out", channels);
    weights.conv_out = conv2d(*weights.store, source, "first_stage_model.decoder.conv_out", type, 1, channels, 3);
    weights.store->upload();
    return weights;
}

core::TensorValue silu(core::ModuleBuildContext & ctx, const core::TensorValue & x) {
    return modules::SiluModule().build(ctx, x);
}

core::TensorValue group_norm_2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights,
    int64_t channels,
    int64_t groups,
    float eps) {
    if (input.shape.rank != 4 || input.shape.dims[1] != channels) {
        throw std::runtime_error("AudioSR VAE GroupNorm2d input shape mismatch");
    }
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = core::wrap_tensor(ggml_group_norm(ctx.ggml, x.tensor, groups, eps), x.shape, GGML_TYPE_F32);
    if (!weights.weight.has_value() || !weights.bias.has_value()) {
        throw std::runtime_error("AudioSR VAE GroupNorm2d requires affine weights");
    }
    auto weight = modules::ReshapeModule({core::TensorShape::from_dims({1, channels, 1, 1})}).build(ctx, *weights.weight);
    weight = modules::RepeatModule({x.shape}).build(ctx, weight);
    x = modules::MulModule().build(ctx, x, weight);
    auto bias = modules::ReshapeModule({core::TensorShape::from_dims({1, channels, 1, 1})}).build(ctx, *weights.bias);
    bias = modules::RepeatModule({x.shape}).build(ctx, bias);
    return modules::AddModule().build(ctx, x, bias);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VaeResBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels) {
    auto h = group_norm_2d(ctx, input, weights.norm1, in_channels, kVaeGroups, kVaeNormEps);
    h = silu(ctx, h);
    h = modules::Conv2dModule({in_channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, h, weights.conv1);
    h = group_norm_2d(ctx, h, weights.norm2, out_channels, kVaeGroups, kVaeNormEps);
    h = silu(ctx, h);
    h = modules::Conv2dModule({out_channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, h, weights.conv2);
    auto residual = input;
    if (weights.shortcut.has_value()) {
        residual = modules::Conv2dModule({in_channels, out_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
                       .build(ctx, input, *weights.shortcut);
    }
    return modules::AddModule().build(ctx, residual, h);
}

core::TensorValue vae_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const VaeAttentionWeights & weights,
    int64_t channels) {
    auto h = group_norm_2d(ctx, input, weights.norm, channels, kVaeGroups, kVaeNormEps);
    auto q = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, h, weights.q);
    auto k = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, h, weights.k);
    auto v = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, h, weights.v);
    const int64_t batch = input.shape.dims[0];
    const int64_t height = input.shape.dims[2];
    const int64_t width = input.shape.dims[3];
    const int64_t tokens = height * width;
    q = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, tokens})}).build(ctx, q);
    q = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, q);
    k = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, tokens})}).build(ctx, k);
    v = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, tokens})}).build(ctx, v);
    auto scores = modules::MatMulModule().build(ctx, q, k);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(channels))),
        scores.shape,
        GGML_TYPE_F32);
    scores = modules::SoftmaxModule().build(ctx, scores);
    v = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, v);
    h = modules::MatMulModule().build(ctx, scores, v);
    h = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, h);
    h = core::ensure_backend_addressable_layout(ctx, h);
    h = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, height, width})}).build(ctx, h);
    h = modules::Conv2dModule({channels, channels, 1, 1, 1, 1, 0, 0, 1, 1, true}).build(ctx, h, weights.proj_out);
    return modules::AddModule().build(ctx, input, h);
}

core::TensorValue downsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv2dWeights & weights,
    int64_t channels) {
    auto x = modules::Pad2dModule({0, 1, 0, 1}).build(ctx, input);
    return modules::Conv2dModule({channels, channels, 3, 3, 2, 2, 0, 0, 1, 1, true}).build(ctx, x, weights);
}

core::TensorValue upsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv2dWeights & weights,
    int64_t channels) {
    auto x = modules::NearestUpsample2dModule({input.shape.dims[2] * 2, input.shape.dims[3] * 2}).build(ctx, input);
    return modules::Conv2dModule({channels, channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights);
}

core::TensorValue build_encoder_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioSRConfig & config,
    const VaeEncoderWeights & weights) {
    auto x = modules::Conv2dModule({1, config.vae_base_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, input, weights.conv_in);
    int64_t channels = config.vae_base_channels;
    for (int64_t level = 0; level < kVaeLevels; ++level) {
        const int64_t out_channels = level_channels(config, level);
        for (int64_t block = 0; block < kVaeResBlocks; ++block) {
            x = resblock(ctx, x, weights.down[level][block], channels, out_channels);
            channels = out_channels;
        }
        if (level != kVaeLevels - 1) {
            x = downsample(ctx, x, weights.downsample[level], channels);
        }
    }
    x = resblock(ctx, x, weights.mid_block_1, channels, channels);
    x = vae_attention(ctx, x, weights.mid_attn, channels);
    x = resblock(ctx, x, weights.mid_block_2, channels, channels);
    x = group_norm_2d(ctx, x, weights.norm_out, channels, kVaeGroups, kVaeNormEps);
    x = silu(ctx, x);
    x = modules::Conv2dModule({channels, 2 * config.encoder_z_channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights.conv_out);
    return modules::Conv2dModule({2 * config.encoder_z_channels, 2 * config.latent_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
        .build(ctx, x, weights.quant_conv);
}

core::TensorValue build_decoder_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioSRConfig & config,
    const VaeDecoderWeights & weights) {
    auto x = modules::Conv2dModule({config.latent_channels, config.encoder_z_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
                 .build(ctx, input, weights.post_quant_conv);
    int64_t channels = level_channels(config, 3);
    x = modules::Conv2dModule({config.encoder_z_channels, channels, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights.conv_in);
    x = resblock(ctx, x, weights.mid_block_1, channels, channels);
    x = vae_attention(ctx, x, weights.mid_attn, channels);
    x = resblock(ctx, x, weights.mid_block_2, channels, channels);
    for (int64_t level = kVaeLevels - 1; level >= 0; --level) {
        const int64_t out_channels = level_channels(config, level);
        for (int64_t block = 0; block < kVaeDecodeBlocks; ++block) {
            x = resblock(ctx, x, weights.up[level][block], channels, out_channels);
            channels = out_channels;
        }
        if (level != 0) {
            x = upsample(ctx, x, weights.upsample[level - 1], channels);
        }
    }
    x = group_norm_2d(ctx, x, weights.norm_out, channels, kVaeGroups, kVaeNormEps);
    x = silu(ctx, x);
    return modules::Conv2dModule({channels, 1, 3, 3, 1, 1, 1, 1, 1, 1, true}).build(ctx, x, weights.conv_out);
}

std::vector<float> sample_posterior(
    const std::vector<float> & moments,
    int64_t channels,
    int64_t height,
    int64_t width,
    uint32_t seed) {
    const int64_t latent_count = channels * height * width;
    if (static_cast<int64_t>(moments.size()) != latent_count * 2) {
        throw std::runtime_error("AudioSR VAE posterior moments size mismatch");
    }
    auto noise = engine::sampling::generate_torch_cuda_randn(static_cast<size_t>(latent_count), seed);
    std::vector<float> latent(static_cast<size_t>(latent_count), 0.0F);
    for (int64_t index = 0; index < latent_count; ++index) {
        const float mean = moments[static_cast<size_t>(index)];
        const float logvar = std::clamp(moments[static_cast<size_t>(latent_count + index)], -30.0F, 20.0F);
        latent[static_cast<size_t>(index)] = mean + std::exp(0.5F * logvar) * noise[static_cast<size_t>(index)];
    }
    return latent;
}

}  // namespace

struct AudioSRAutoencoderRuntime::Impl {
    Impl(
        std::shared_ptr<const AudioSRAssets> assets,
        core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type)
        : assets(std::move(assets)),
          execution(&execution) {
        if (this->assets == nullptr) {
            throw std::runtime_error("AudioSR autoencoder requires assets");
        }
        condition_encoder = load_encoder_weights(
            *this->assets,
            execution.backend(),
            execution.backend_type(),
            "cond_stage_models.0.vae",
            weight_type);
        decoder = load_decoder_weights(*this->assets, execution.backend(), execution.backend_type(), weight_type);
    }

    struct EncoderGraph {
        EncoderGraph(
            core::ExecutionContext & execution,
            std::shared_ptr<const AudioSRAssets> assets_in,
            const VaeEncoderWeights & weights,
            int64_t frames)
            : execution(execution),
              assets(std::move(assets_in)),
              frames(frames) {
            if (this->assets == nullptr) {
                throw std::runtime_error("AudioSR VAE encoder graph requires assets");
            }
            if (frames <= 0 || frames % 8 != 0) {
                throw std::runtime_error("AudioSR VAE encoder frames must be positive and divisible by 8");
            }
            const auto build_start = Clock::now();
            ggml_init_params params{1024ull * 1024ull * 1024ull, nullptr, true};
            ctx.reset(ggml_init(params));
            ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
            input_ctx.reset(ggml_init(input_params));
            if (ctx == nullptr || input_ctx == nullptr) {
                throw std::runtime_error("failed to initialize AudioSR VAE encoder graph context");
            }
            core::ModuleBuildContext graph_ctx{ctx.get(), "audiosr.vae.encoder", execution.backend_type()};
            core::ModuleBuildContext input_build_ctx{input_ctx.get(), "audiosr.vae.encoder.inputs", execution.backend_type()};
            input = core::make_tensor(input_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, frames, assets->config.n_mels})).tensor;
            ggml_set_input(input);
            auto x = core::wrap_tensor(input, core::TensorShape::from_dims({1, 1, frames, assets->config.n_mels}), GGML_TYPE_F32);
            x = build_encoder_graph(graph_ctx, x, assets->config, weights);
            latent_height = frames / 8;
            latent_width = assets->config.n_mels / 8;
            output = core::ensure_backend_addressable_layout(graph_ctx, x).tensor;
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_build_forward_expand(graph, output);
            input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (input_buffer == nullptr || gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                clear();
                throw std::runtime_error("failed to allocate AudioSR VAE encoder graph");
            }
            engine::debug::timing_log_scalar(
                "audiosr.vae.encoder.graph.build_ms",
                engine::debug::elapsed_ms(build_start, Clock::now()));
        }

        ~EncoderGraph() {
            clear();
        }

        void clear() {
            if (graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
            if (input_buffer != nullptr) {
                ggml_backend_buffer_free(input_buffer);
                input_buffer = nullptr;
            }
        }

        AudioSRLatent run(const std::vector<float> & mel, uint32_t seed) {
            const int64_t expected = frames * assets->config.n_mels;
            if (static_cast<int64_t>(mel.size()) != expected) {
                throw std::runtime_error("AudioSR VAE encoder mel size mismatch");
            }
            auto start = Clock::now();
            ggml_backend_tensor_set(input, mel.data(), 0, mel.size() * sizeof(float));
            engine::debug::timing_log_scalar(
                "audiosr.vae.encoder.input_upload_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
            core::set_backend_threads(execution.backend(), execution.config().threads);
            start = Clock::now();
            const ggml_status status = core::compute_backend_graph(execution.backend(), graph);
            ggml_backend_synchronize(execution.backend());
            engine::debug::timing_log_scalar(
                "audiosr.vae.encoder.graph.compute_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("AudioSR VAE encoder graph compute failed");
            }
            std::vector<float> moments;
            core::read_tensor_float_into(output, moments);
            AudioSRLatent latent;
            latent.channels = assets->config.latent_channels;
            latent.height = latent_height;
            latent.width = latent_width;
            latent.values = sample_posterior(moments, latent.channels, latent.height, latent.width, seed);
            return latent;
        }

        core::ExecutionContext & execution;
        std::shared_ptr<const AudioSRAssets> assets;
        int64_t frames = 0;
        int64_t latent_height = 0;
        int64_t latent_width = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t input_buffer = nullptr;
    };

    struct DecoderGraph {
        DecoderGraph(
            core::ExecutionContext & execution,
            std::shared_ptr<const AudioSRAssets> assets_in,
            const VaeDecoderWeights & weights,
            int64_t height,
            int64_t width)
            : execution(execution),
              assets(std::move(assets_in)),
              height(height),
              width(width) {
            if (this->assets == nullptr) {
                throw std::runtime_error("AudioSR VAE decoder graph requires assets");
            }
            if (height <= 0 || width <= 0) {
                throw std::runtime_error("AudioSR VAE decoder graph requires positive latent shape");
            }
            const auto build_start = Clock::now();
            ggml_init_params params{1024ull * 1024ull * 1024ull, nullptr, true};
            ctx.reset(ggml_init(params));
            ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
            input_ctx.reset(ggml_init(input_params));
            if (ctx == nullptr || input_ctx == nullptr) {
                throw std::runtime_error("failed to initialize AudioSR VAE decoder graph context");
            }
            core::ModuleBuildContext graph_ctx{ctx.get(), "audiosr.vae.decoder", execution.backend_type()};
            core::ModuleBuildContext input_build_ctx{input_ctx.get(), "audiosr.vae.decoder.inputs", execution.backend_type()};
            input = core::make_tensor(input_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width})).tensor;
            ggml_set_input(input);
            auto x = core::wrap_tensor(input, core::TensorShape::from_dims({1, assets->config.latent_channels, height, width}), GGML_TYPE_F32);
            x = build_decoder_graph(graph_ctx, x, assets->config, weights);
            output = core::ensure_backend_addressable_layout(graph_ctx, x).tensor;
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_build_forward_expand(graph, output);
            input_buffer = ggml_backend_alloc_ctx_tensors(input_ctx.get(), execution.backend());
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (input_buffer == nullptr || gallocr == nullptr ||
                !ggml_gallocr_reserve(gallocr, graph) ||
                !ggml_gallocr_alloc_graph(gallocr, graph)) {
                clear();
                throw std::runtime_error("failed to allocate AudioSR VAE decoder graph");
            }
            engine::debug::timing_log_scalar(
                "audiosr.vae.decoder.graph.build_ms",
                engine::debug::elapsed_ms(build_start, Clock::now()));
        }

        ~DecoderGraph() {
            clear();
        }

        void clear() {
            if (graph != nullptr) {
                core::release_backend_graph_resources(execution.backend(), graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
            if (input_buffer != nullptr) {
                ggml_backend_buffer_free(input_buffer);
                input_buffer = nullptr;
            }
        }

        std::vector<float> run(const AudioSRLatent & latent) {
            if (latent.channels != assets->config.latent_channels ||
                latent.height != height ||
                latent.width != width) {
                throw std::runtime_error("AudioSR VAE decoder latent shape mismatch");
            }
            std::vector<float> decoder_input = latent.values;
            const float inverse_scale = 1.0F / assets->config.vae_scale_factor;
            for (float & value : decoder_input) {
                value *= inverse_scale;
            }
            auto start = Clock::now();
            ggml_backend_tensor_set(input, decoder_input.data(), 0, decoder_input.size() * sizeof(float));
            engine::debug::timing_log_scalar(
                "audiosr.vae.decoder.input_upload_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
            core::set_backend_threads(execution.backend(), execution.config().threads);
            start = Clock::now();
            const ggml_status status = core::compute_backend_graph(execution.backend(), graph);
            ggml_backend_synchronize(execution.backend());
            engine::debug::timing_log_scalar(
                "audiosr.vae.decoder.graph.compute_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("AudioSR VAE decoder graph compute failed");
            }
            std::vector<float> mel_btcf;
            core::read_tensor_float_into(output, mel_btcf);
            const int64_t frames = height * 8;
            const int64_t bins = width * 8;
            if (bins != assets->config.n_mels || static_cast<int64_t>(mel_btcf.size()) != frames * bins) {
                throw std::runtime_error("AudioSR VAE decoder output shape mismatch");
            }
            std::vector<float> mel_bft(static_cast<size_t>(bins * frames), 0.0F);
            for (int64_t t = 0; t < frames; ++t) {
                for (int64_t f = 0; f < bins; ++f) {
                    mel_bft[static_cast<size_t>(f * frames + t)] =
                        mel_btcf[static_cast<size_t>(t * bins + f)];
                }
            }
            return mel_bft;
        }

        core::ExecutionContext & execution;
        std::shared_ptr<const AudioSRAssets> assets;
        int64_t height = 0;
        int64_t width = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_buffer_t input_buffer = nullptr;
    };

    std::shared_ptr<const AudioSRAssets> assets;
    core::ExecutionContext * execution = nullptr;
    VaeEncoderWeights condition_encoder;
    VaeDecoderWeights decoder;
    std::unique_ptr<EncoderGraph> encoder_graph;
    std::unique_ptr<DecoderGraph> decoder_graph;
};

AudioSRAutoencoderRuntime::AudioSRAutoencoderRuntime(
    std::shared_ptr<const AudioSRAssets> assets,
    core::ExecutionContext & execution,
    engine::assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, weight_type)) {}

AudioSRAutoencoderRuntime::~AudioSRAutoencoderRuntime() = default;

AudioSRLatent AudioSRAutoencoderRuntime::encode_condition(
    const std::vector<float> & mel,
    int64_t frames,
    uint32_t seed) {
    if (impl_->execution == nullptr) {
        throw std::runtime_error("AudioSR VAE execution context is missing");
    }
    if (impl_->encoder_graph == nullptr || impl_->encoder_graph->frames != frames) {
        impl_->encoder_graph.reset();
        impl_->encoder_graph = std::make_unique<Impl::EncoderGraph>(
            *impl_->execution,
            impl_->assets,
            impl_->condition_encoder,
            frames);
    }
    return impl_->encoder_graph->run(mel, seed);
}

std::vector<float> AudioSRAutoencoderRuntime::decode_first_stage(const AudioSRLatent & latent) {
    if (impl_->execution == nullptr) {
        throw std::runtime_error("AudioSR VAE execution context is missing");
    }
    if (impl_->decoder_graph == nullptr ||
        impl_->decoder_graph->height != latent.height ||
        impl_->decoder_graph->width != latent.width) {
        impl_->decoder_graph.reset();
        impl_->decoder_graph = std::make_unique<Impl::DecoderGraph>(
            *impl_->execution,
            impl_->assets,
            impl_->decoder,
            latent.height,
            latent.width);
    }
    return impl_->decoder_graph->run(latent);
}

void AudioSRAutoencoderRuntime::release_encoder_graph() {
    impl_->encoder_graph.reset();
}

void AudioSRAutoencoderRuntime::release_decoder_graph() {
    impl_->decoder_graph.reset();
}

void AudioSRAutoencoderRuntime::release_runtime_graphs() {
    release_encoder_graph();
    release_decoder_graph();
}

}  // namespace engine::models::audiosr
