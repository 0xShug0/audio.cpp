#include "engine/community_models/latentsync/sd_vae.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ggml-alloc.h>
#include <stdexcept>

namespace engine::models::latentsync {

SdVaeTowerWeights load_sd_vae_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const LatentSyncSdVaeConfig & config,
    bool encoder, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    const std::string root = encoder ? "vae.encoder" : "vae.decoder";
    SdVaeTowerWeights weights;
    const auto load_resnet = [&](const std::string & prefix, int64_t in, int64_t out) {
        SdVaeResnetWeights block;
        block.norm1 = binding::norm_from_source(store, source, prefix + ".norm1", in);
        block.norm2 = binding::norm_from_source(store, source, prefix + ".norm2", out);
        block.conv1 = binding::conv2d_from_source(store, source, prefix + ".conv1", type, out, in, 3, 3, true);
        block.conv2 = binding::conv2d_from_source(store, source, prefix + ".conv2", type, out, out, 3, 3, true);
        if (in != out) {
            block.shortcut = binding::conv2d_from_source(store, source, prefix + ".conv_shortcut", type, out, in, 1, 1, true);
        }
        return block;
    };
    int64_t current = encoder ? config.channels.front() : config.channels.back();
    weights.input = binding::conv2d_from_source(store, source, root + ".conv_in", type,
        current, encoder ? 3 : config.latent_channels, 3, 3, true);
    for (size_t index = 0; index < config.channels.size(); ++index) {
        const int64_t channels = config.channels[encoder ? index : config.channels.size() - index - 1];
        const std::string prefix = root + (encoder ? ".down_blocks." : ".up_blocks.") + std::to_string(index);
        SdVaeStageWeights stage;
        for (int64_t layer = 0; layer < config.layers_per_block + (encoder ? 0 : 1); ++layer) {
            stage.resnets.push_back(load_resnet(prefix + ".resnets." + std::to_string(layer), current, channels));
            current = channels;
        }
        if (index + 1 < config.channels.size()) {
            stage.resample = binding::conv2d_from_source(store, source,
                prefix + (encoder ? ".downsamplers.0.conv" : ".upsamplers.0.conv"),
                type, channels, channels, 3, 3, true);
        }
        weights.stages.push_back(std::move(stage));
    }
    const int64_t mid = config.channels.back();
    for (size_t i = 0; i < weights.mid_resnets.size(); ++i) {
        weights.mid_resnets[i] = load_resnet(root + ".mid_block.resnets." + std::to_string(i), mid, mid);
    }
    const auto attention = root + ".mid_block.attentions.0";
    weights.attention_norm = binding::norm_from_source(store, source, attention + ".group_norm", mid);
    weights.attention.query = binding::linear_from_source(store, source, attention + ".query", type, mid, mid, true);
    weights.attention.key = binding::linear_from_source(store, source, attention + ".key", type, mid, mid, true);
    weights.attention.value = binding::linear_from_source(store, source, attention + ".value", type, mid, mid, true);
    weights.attention.out = binding::linear_from_source(store, source, attention + ".proj_attn", type, mid, mid, true);
    weights.output_norm = binding::norm_from_source(store, source, root + ".conv_norm_out", current);
    const int64_t out = encoder ? config.latent_channels * 2 : 3;
    weights.output = binding::conv2d_from_source(store, source, root + ".conv_out", type, out, current, 3, 3, true);
    const int64_t quant = encoder ? config.latent_channels * 2 : config.latent_channels;
    weights.quant = binding::conv2d_from_source(store, source,
        encoder ? "vae.quant_conv" : "vae.post_quant_conv", type, quant, quant, 1, 1, true);
    return weights;
}

LatentSyncSdVae::LatentSyncSdVae(LatentSyncSdVaeConfig config) : config_(config) {
    if (config_.latent_channels <= 0 || config_.layers_per_block <= 0 || config_.norm_groups <= 0) {
        throw std::runtime_error("Invalid LatentSync SD VAE dimensions");
    }
    for (const int64_t channels : config_.channels) {
        if (channels <= 0 || channels % config_.norm_groups != 0) {
            throw std::runtime_error("Invalid LatentSync SD VAE group normalization channels");
        }
    }
}

core::TensorValue LatentSyncSdVae::build_resnet(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const SdVaeResnetWeights & weights) const {
    using namespace modules;
    const int64_t in = input.shape.dims[1];
    const int64_t out = weights.conv1.weight.shape.dims[0];
    auto x = GroupNormModule({in, config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, input, weights.norm1);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({in, out, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.conv1);
    x = GroupNormModule({out, config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.norm2);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({out, out, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.conv2);
    auto residual = input;
    if (weights.shortcut) {
        residual = Conv2dModule({in, out, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, input, *weights.shortcut);
    }
    return AddModule(config_.cuda_f16_activations).build(ctx, residual, x);
}

core::TensorValue LatentSyncSdVae::build_mid(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    const auto x = build_resnet(ctx, input, weights.mid_resnets[0]);
    const int64_t batch = x.shape.dims[0], channels = x.shape.dims[1];
    const int64_t steps = x.shape.dims[2] * x.shape.dims[3];
    auto attention_input = config_.cuda_f16_activations
        ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
    auto sequence = GroupNormModule({channels, config_.norm_groups, 1e-6F}).build(ctx, attention_input, weights.attention_norm);
    sequence = core::reshape_tensor(ctx, sequence, core::TensorShape::from_dims({batch, channels, steps}));
    sequence = TransposeModule({{0, 2, 1}, 3}).build(ctx, sequence);
    const auto project = [&](const LinearWeights & projection) {
        const auto result = LinearModule({channels, channels, true}).build(ctx, sequence, projection);
        return core::reshape_tensor(ctx, result, core::TensorShape::from_dims({batch, 1, steps, channels}));
    };
    // SD VAE uses one 512-dimensional head, outside the CUDA flash kernel's supported sizes.
    auto attended = ScaledDotProductAttentionModule({channels, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
        .build(ctx, project(weights.attention.query), project(weights.attention.key), project(weights.attention.value));
    attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended),
        core::TensorShape::from_dims({batch, steps, channels}));
    attended = LinearModule({channels, channels, true}).build(ctx, attended, weights.attention.out);
    attended = TransposeModule({{0, 2, 1}, 3}).build(ctx, attended);
    attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended), x.shape);
    if (config_.cuda_f16_activations) {
        attended = core::wrap_tensor(ggml_cast(ctx.ggml, attended.tensor, GGML_TYPE_F16), attended.shape, GGML_TYPE_F16);
    }
    return build_resnet(ctx, AddModule(config_.cuda_f16_activations).build(ctx, x, attended), weights.mid_resnets[1]);
}

core::TensorValue LatentSyncSdVae::build_encoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    auto x = Conv2dModule({3, config_.channels.front(), 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, image, weights.input);
    for (const auto & stage : weights.stages) {
        for (const auto & block : stage.resnets) x = build_resnet(ctx, x, block);
        if (stage.resample) {
            const auto pad_input = config_.cuda_f16_activations
                ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
            x = Pad2dModule({0, 1, 0, 1}).build(ctx, pad_input);
            x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 2, 2}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
        }
    }
    x = build_mid(ctx, x, weights);
    x = GroupNormModule({config_.channels.back(), config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.output_norm);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({config_.channels.back(), config_.latent_channels * 2, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.output);
    x = Conv2dModule({config_.latent_channels * 2, config_.latent_channels * 2, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.quant);
    return config_.cuda_f16_activations ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
}

core::TensorValue LatentSyncSdVae::build_decoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & latent, const SdVaeTowerWeights & weights) const {
    using namespace modules;
    auto x = Conv2dModule({config_.latent_channels, config_.latent_channels, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, latent, weights.quant);
    x = Conv2dModule({config_.latent_channels, config_.channels.back(), 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.input);
    x = build_mid(ctx, x, weights);
    for (const auto & stage : weights.stages) {
        for (const auto & block : stage.resnets) x = build_resnet(ctx, x, block);
        if (stage.resample) {
            const auto upsample_input = config_.cuda_f16_activations
                ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
            x = NearestUpsample2dModule({x.shape.dims[2] * 2, x.shape.dims[3] * 2}).build(ctx, upsample_input);
            x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
        }
    }
    x = GroupNormModule({config_.channels.front(), config_.norm_groups, 1e-6F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights.output_norm);
    x = SiluModule(config_.cuda_f16_activations).build(ctx, x);
    x = Conv2dModule({config_.channels.front(), 3, 3, 3, 1, 1, 1, 1}, config_.cuda_cudnn, config_.cuda_f16_activations).build(ctx, x, weights.output);
    return config_.cuda_f16_activations ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
}

namespace {
constexpr float kLatentScale = 0.18215F;

class LatentSyncSdVaeGraph {
public:
    LatentSyncSdVaeGraph(core::ExecutionContext & execution, const SdVaeTowerWeights & weights,
        int64_t resolution, bool encoder, bool cuda_cudnn, bool cuda_f16_activations,
        ggml_gallocr_t shared_allocator = nullptr, ggml_cgraph ** active_graph = nullptr)
        : backend_(execution.backend()), allocator_(shared_allocator), active_graph_(active_graph),
          owns_allocator_(shared_allocator == nullptr) {
        constexpr size_t nodes = 4096;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("LatentSync VAE graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "latentsync.vae", execution.backend_type()};
        const int64_t side = encoder ? resolution : resolution / 8;
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, encoder ? 3 : 4, side, side}));
        ggml_set_input(input_.tensor);
        LatentSyncSdVaeConfig config;
        config.cuda_cudnn = cuda_cudnn;
        config.cuda_f16_activations = cuda_f16_activations;
        const LatentSyncSdVae vae(config);
        output_ = encoder ? vae.build_encoder(ctx, input_, weights) : vae.build_decoder(ctx, input_, weights);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        if (owns_allocator_) allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("LatentSync VAE graph allocation failed");
        if (active_graph_) *active_graph_ = graph_;
        debug::timing_log_scalar(encoder ? "latentsync.vae_encoder.graph_buffer_mb" : "latentsync.vae_decoder.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(allocator_, 0) / 1048576.0);
    }
    ~LatentSyncSdVaeGraph() {
        core::release_backend_graph_resources(backend_, graph_, true);
        if (owns_allocator_) ggml_gallocr_free(allocator_);
    }
    std::vector<float> run(const std::vector<float> & input) {
        if (!owns_allocator_ && *active_graph_ != graph_) {
            if (!ggml_gallocr_alloc_graph(allocator_, graph_)) {
                throw std::runtime_error("LatentSync VAE graph allocation failed");
            }
            *active_graph_ = graph_;
        }
        core::write_tensor_f32(input_, input);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("LatentSync VAE execution failed");
        return core::read_tensor_f32(output_.tensor);
    }
private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    ggml_gallocr_t allocator_ = nullptr;
    ggml_cgraph ** active_graph_ = nullptr;
    bool owns_allocator_ = false;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_, output_;
};
}

struct LatentSyncSdVaeRuntime::Impl {
    core::ExecutionContext & execution;
    int64_t resolution;
    bool cuda_cudnn;
    bool cuda_f16_activations;
    bool share_graph_workspace;
    core::BackendWeightStore store;
    SdVaeTowerWeights encoder_weights, decoder_weights;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;
    ggml_cgraph * active_graph = nullptr;
    std::unique_ptr<LatentSyncSdVaeGraph> encoder, decoder;
    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        assets::TensorStorageType type, int64_t resolution_in, bool cuda_cudnn_in,
        bool cuda_f16_activations_in, bool share_graph_workspace_in)
        : execution(execution_in), resolution(resolution_in), cuda_cudnn(cuda_cudnn_in),
          cuda_f16_activations(cuda_f16_activations_in), share_graph_workspace(share_graph_workspace_in),
          store(execution.backend(), execution.backend_type(), "latentsync.vae.weights", 4 * 1024 * 1024),
          allocator(share_graph_workspace
                  ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()))
                  : nullptr,
              &ggml_gallocr_free) {
        if (cuda_f16_activations && !cuda_cudnn) {
            throw std::runtime_error("LatentSync VAE FP16 activations require cuDNN lowering");
        }
#ifndef GGML_CUDA_CUDNN_CONV2D
        if (cuda_cudnn) throw std::runtime_error("LatentSync VAE cuDNN lowering is not enabled in this build");
#endif
        if (cuda_cudnn && execution.backend_type() != core::BackendType::Cuda) {
            throw std::runtime_error("LatentSync VAE cuDNN lowering requires CUDA");
        }
        if (resolution <= 0 || resolution % 8) throw std::runtime_error("LatentSync VAE resolution must be a positive multiple of 8");
        encoder_weights = load_sd_vae_weights(store, source, {}, true, type);
        decoder_weights = load_sd_vae_weights(store, source, {}, false, type);
        store.upload();
    }

    void ensure_shared_graphs() {
        if (encoder) return;
        // Allocate the larger decoder plan first so the shared backend buffer never has to grow.
        decoder = std::make_unique<LatentSyncSdVaeGraph>(execution, decoder_weights, resolution,
            false, cuda_cudnn, cuda_f16_activations, allocator.get(), &active_graph);
        encoder = std::make_unique<LatentSyncSdVaeGraph>(execution, encoder_weights, resolution,
            true, cuda_cudnn, cuda_f16_activations, allocator.get(), &active_graph);
    }
};

LatentSyncSdVaeRuntime::LatentSyncSdVaeRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, assets::TensorStorageType weight_type, int64_t resolution,
    bool cuda_cudnn, bool cuda_f16_activations, bool share_graph_workspace)
    : impl_(std::make_unique<Impl>(execution, source, weight_type, resolution, cuda_cudnn,
          cuda_f16_activations, share_graph_workspace)) {}
LatentSyncSdVaeRuntime::~LatentSyncSdVaeRuntime() = default;

std::vector<float> LatentSyncSdVaeRuntime::encode_moments(const std::vector<float> & image) {
    const auto start = std::chrono::steady_clock::now();
    if (impl_->share_graph_workspace) impl_->ensure_shared_graphs();
    else if (!impl_->encoder) impl_->encoder = std::make_unique<LatentSyncSdVaeGraph>(impl_->execution,
        impl_->encoder_weights, impl_->resolution, true, impl_->cuda_cudnn, impl_->cuda_f16_activations);
    auto output = impl_->encoder->run(image);
    debug::timing_log_scalar("latentsync.vae_encoder.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncSdVaeRuntime::encode(const std::vector<float> & image, const std::vector<float> & posterior_noise) {
    const auto moments = encode_moments(image);
    const size_t count = moments.size() / 2;
    if (posterior_noise.size() != count) throw std::runtime_error("LatentSync VAE posterior noise size mismatch");
    std::vector<float> latent(count);
    for (size_t i = 0; i < count; ++i) {
        const float stddev = std::exp(0.5F * std::clamp(moments[i + count], -30.0F, 20.0F));
        latent[i] = (moments[i] + stddev * posterior_noise[i]) * kLatentScale;
    }
    return latent;
}

std::vector<float> LatentSyncSdVaeRuntime::decode(const std::vector<float> & scaled_latent) {
    const auto start = std::chrono::steady_clock::now();
    if (impl_->share_graph_workspace) impl_->ensure_shared_graphs();
    else if (!impl_->decoder) impl_->decoder = std::make_unique<LatentSyncSdVaeGraph>(impl_->execution,
        impl_->decoder_weights, impl_->resolution, false, impl_->cuda_cudnn, impl_->cuda_f16_activations);
    auto latent = scaled_latent;
    for (auto & value : latent) value /= kLatentScale;
    auto output = impl_->decoder->run(latent);
    debug::timing_log_scalar("latentsync.vae_decoder.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

}  // namespace engine::models::latentsync
