#include "engine/models/controlfoley/flow_denoiser.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::controlfoley {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_config(const ControlFoleyFlowConfig & config) {
    if (config.latent_dim <= 0 || config.clip_dim <= 0 || config.visual_dim <= 0 ||
        config.sync_dim <= 0 || config.text_dim <= 0 || config.audio_dim <= 0 ||
        config.timbre_dim <= 0 || config.hidden_dim <= 0 || config.depth <= 0 ||
        config.fused_depth <= 0 || config.depth <= config.fused_depth || config.num_heads <= 0 ||
        config.latent_seq_len <= 0 || config.clip_seq_len <= 0 || config.visual_seq_len <= 0 ||
        config.sync_seq_len <= 0 || config.text_seq_len <= 0 || config.audio_seq_len <= 0 ||
        config.timbre_seq_len <= 0) {
        throw std::runtime_error("ControlFoley flow config dimensions must be positive");
    }
    if (config.hidden_dim % config.num_heads != 0 || (config.hidden_dim / config.num_heads) % 2 != 0) {
        throw std::runtime_error("ControlFoley hidden dimension must divide into even attention head dimensions");
    }
    if (config.sync_seq_len % 8 != 0) {
        throw std::runtime_error("ControlFoley sync sequence length must be divisible by 8");
    }
}

bool same_weight_shape_config(
    const ControlFoleyFlowConfig & lhs,
    const ControlFoleyFlowConfig & rhs) {
    return lhs.v2 == rhs.v2 &&
           lhs.latent_dim == rhs.latent_dim &&
           lhs.clip_dim == rhs.clip_dim &&
           lhs.visual_dim == rhs.visual_dim &&
           lhs.sync_dim == rhs.sync_dim &&
           lhs.text_dim == rhs.text_dim &&
           lhs.audio_dim == rhs.audio_dim &&
           lhs.timbre_dim == rhs.timbre_dim &&
           lhs.hidden_dim == rhs.hidden_dim &&
           lhs.depth == rhs.depth &&
           lhs.fused_depth == rhs.fused_depth &&
           lhs.num_heads == rhs.num_heads &&
           lhs.mlp_ratio == rhs.mlp_ratio &&
           lhs.layer_norm_eps == rhs.layer_norm_eps &&
           lhs.rms_norm_eps == rhs.rms_norm_eps &&
           lhs.rope_theta == rhs.rope_theta;
}

int64_t hidden_multiple(int64_t hidden, float ratio) {
    const int64_t raw = static_cast<int64_t>(2.0F * static_cast<float>(hidden) * ratio / 3.0F);
    return 256 * ((raw + 255) / 256);
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    if (value.shape.rank == like.shape.rank) {
        bool same = true;
        for (size_t axis = 0; axis < value.shape.rank; ++axis) {
            same = same && value.shape.dims[axis] == like.shape.dims[axis];
        }
        if (same) {
            return value;
        }
        for (size_t axis = 0; axis < value.shape.rank; ++axis) {
            if (like.shape.dims[axis] % value.shape.dims[axis] != 0) {
                throw std::runtime_error(
                    "ControlFoley flow broadcast shape mismatch: value " + value.shape.to_string() +
                    " cannot repeat to " + like.shape.to_string());
            }
        }
    }
    return modules::RepeatModule({like.shape}).build(ctx, value);
}

core::TensorValue add_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto rhs_expanded = repeat_like(ctx, rhs, lhs);
    return core::wrap_tensor(ggml_add(ctx.ggml, lhs.tensor, rhs_expanded.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue mul_tensors(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto rhs_expanded = repeat_like(ctx, rhs, lhs);
    return core::wrap_tensor(ggml_mul(ctx.ggml, lhs.tensor, rhs_expanded.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & shift,
    const core::TensorValue & scale) {
    auto scaled_delta = mul_tensors(ctx, input, scale);
    auto scaled = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, scaled_delta.tensor), input.shape, GGML_TYPE_F32);
    return add_tensors(ctx, scaled, shift);
}

core::TensorValue unsqueeze_sequence(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    core::validate_rank_between(value, 2, 2, "value");
    return core::reshape_tensor(ctx, value, core::TensorShape::from_dims({value.shape.dims[0], 1, value.shape.dims[1]}));
}

core::TensorValue squeeze_sequence(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    core::validate_rank_between(value, 3, 3, "value");
    if (value.shape.dims[1] != 1) {
        throw std::runtime_error("ControlFoley squeeze_sequence expects a singleton sequence axis");
    }
    return core::reshape_tensor(ctx, value, core::TensorShape::from_dims({value.shape.dims[0], value.shape.dims[2]}));
}

core::TensorValue channel_last_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = modules::Conv1dModule({in_channels, out_channels, kernel_size, 1, static_cast<int>(kernel_size / 2), 1, use_bias})
            .build(ctx, x, weights);
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

struct ProjectionWeights {
    modules::LinearWeights linear;
    modules::Conv1dWeights conv;
    modules::GatedFeedForwardWeights mlp;
    modules::GatedConvFeedForwardWeights conv_mlp;
};

struct AttentionWeights {
    modules::LinearWeights qkv;
    modules::NormWeights q_norm;
    modules::NormWeights k_norm;
};

struct SingleBlockWeights {
    AttentionWeights attention;
    modules::LinearWeights modulation;
    modules::LinearWeights linear_out;
    modules::Conv1dWeights conv_out;
    modules::GatedFeedForwardWeights linear_ffn;
    modules::GatedConvFeedForwardWeights conv_ffn;
    bool pre_only = false;
    bool conv_path = false;
};

struct JointBlockWeights {
    SingleBlockWeights latent;
    SingleBlockWeights clip;
    SingleBlockWeights text;
    SingleBlockWeights audio;
    bool pre_only = false;
};

struct RepaWeights {
    modules::LinearWeights linear;
    modules::NormWeights norm;
};

struct FinalWeights {
    modules::LinearWeights modulation;
    modules::Conv1dWeights conv;
};

struct FlowWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    ProjectionWeights audio_input;
    ProjectionWeights clip_input;
    ProjectionWeights visual_input;
    ProjectionWeights sync_input;
    ProjectionWeights text_input;
    ProjectionWeights clap_input;
    ProjectionWeights timbre_input;
    modules::LinearWeights clip_cond;
    modules::LinearWeights text_cond;
    modules::LinearWeights timbre_cond;
    modules::GatedFeedForwardWeights global_cond;
    modules::LinearWeights time_fc1;
    modules::LinearWeights time_fc2;
    RepaWeights repa;
    FinalWeights final_layer;
    std::vector<JointBlockWeights> joint_blocks;
    std::vector<SingleBlockWeights> fused_blocks;
    core::TensorValue latent_mean;
    core::TensorValue latent_std;
    core::TensorValue sync_pos_emb;
};

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t out_features,
    int64_t in_features,
    bool use_bias = true) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

modules::Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    bool use_bias = true) {
    return binding::conv1d_from_source(store, source, prefix, storage, out_channels, in_channels, kernel_size, use_bias);
}

modules::GatedFeedForwardWeights load_gated_ffn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t hidden,
    int64_t intermediate) {
    return {
        load_linear(store, source, prefix + ".w1", storage, intermediate, hidden, false),
        load_linear(store, source, prefix + ".w3", storage, intermediate, hidden, false),
        load_linear(store, source, prefix + ".w2", storage, hidden, intermediate, false),
    };
}

modules::GatedConvFeedForwardWeights load_gated_conv_ffn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t hidden,
    int64_t intermediate,
    int64_t kernel_size) {
    return {
        load_conv1d(store, source, prefix + ".w1", storage, intermediate, hidden, kernel_size, false),
        load_conv1d(store, source, prefix + ".w3", storage, intermediate, hidden, kernel_size, false),
        load_conv1d(store, source, prefix + ".w2", storage, hidden, intermediate, kernel_size, false),
    };
}

ProjectionWeights load_projection(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t input_dim,
    int64_t hidden,
    int64_t kernel_size,
    bool first_conv,
    bool conv_mlp,
    int64_t mlp_index,
    int64_t mlp_kernel_size,
    int64_t intermediate) {
    ProjectionWeights weights;
    if (first_conv) {
        weights.conv = load_conv1d(store, source, prefix + ".0", storage, hidden, input_dim, kernel_size, true);
    } else {
        weights.linear = load_linear(store, source, prefix + ".0", storage, hidden, input_dim, true);
    }
    const std::string mlp_prefix = prefix + "." + std::to_string(mlp_index);
    if (conv_mlp) {
        weights.conv_mlp = load_gated_conv_ffn(store, source, mlp_prefix, storage, hidden, intermediate, mlp_kernel_size);
    } else {
        weights.mlp = load_gated_ffn(store, source, mlp_prefix, storage, hidden, intermediate);
    }
    return weights;
}

AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options) {
    const int64_t head_dim = config.hidden_dim / config.num_heads;
    return {
        load_linear(store, source, prefix + ".qkv", options.weight_storage_type, config.hidden_dim * 3, config.hidden_dim, true),
        binding::norm_weight_from_source(store, source, prefix + ".q_norm", head_dim),
        binding::norm_weight_from_source(store, source, prefix + ".k_norm", head_dim),
    };
}

SingleBlockWeights load_single_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    bool pre_only,
    bool conv_path) {
    const int64_t intermediate = hidden_multiple(config.hidden_dim, config.mlp_ratio);
    SingleBlockWeights weights;
    weights.pre_only = pre_only;
    weights.conv_path = conv_path;
    weights.attention = load_attention(store, source, prefix + ".attn", config, options);
    weights.modulation = load_linear(
        store,
        source,
        prefix + ".adaLN_modulation.1",
        options.weight_storage_type,
        (pre_only ? 2 : 6) * config.hidden_dim,
        config.hidden_dim,
        true);
    if (!pre_only) {
        if (conv_path) {
            weights.conv_out = load_conv1d(
                store,
                source,
                prefix + ".linear1",
                options.weight_storage_type,
                config.hidden_dim,
                config.hidden_dim,
                3,
                true);
            weights.conv_ffn = load_gated_conv_ffn(
                store,
                source,
                prefix + ".ffn",
                options.weight_storage_type,
                config.hidden_dim,
                intermediate,
                3);
        } else {
            weights.linear_out = load_linear(
                store,
                source,
                prefix + ".linear1",
                options.weight_storage_type,
                config.hidden_dim,
                config.hidden_dim,
                true);
            weights.linear_ffn = load_gated_ffn(
                store,
                source,
                prefix + ".ffn",
                options.weight_storage_type,
                config.hidden_dim,
                intermediate);
        }
    }
    return weights;
}

FlowWeights load_flow_weights(
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options) {
    FlowWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "framework.controlfoley_flow.weights",
        options.weight_context_bytes);
    auto & store = *weights.store;
    const int64_t intermediate = hidden_multiple(config.hidden_dim, config.mlp_ratio);
    weights.audio_input = load_projection(
        store, source, "audio_input_proj", options.weight_storage_type, config.latent_dim, config.hidden_dim, 7, true, true, 2, 7, intermediate);
    weights.clip_input = load_projection(
        store, source, "clip_input_proj", options.weight_storage_type, config.clip_dim, config.hidden_dim, 3, false, !config.v2, config.v2 ? 2 : 1, 3, intermediate);
    weights.visual_input = load_projection(
        store, source, "visual_input_proj", options.weight_storage_type, config.visual_dim, config.hidden_dim, 3, false, !config.v2, config.v2 ? 2 : 1, 3, intermediate);
    weights.sync_input = load_projection(
        store, source, "sync_input_proj", options.weight_storage_type, config.sync_dim, config.hidden_dim, 7, true, true, 2, 3, intermediate);
    weights.text_input = load_projection(
        store, source, "text_input_proj", options.weight_storage_type, config.text_dim, config.hidden_dim, 1, false, false, config.v2 ? 2 : 1, 1, intermediate);
    weights.clap_input = load_projection(
        store, source, "clap_input_proj", options.weight_storage_type, config.audio_dim, config.hidden_dim, 1, false, false, config.v2 ? 2 : 1, 1, intermediate);
    weights.timbre_input = load_projection(
        store, source, "timbre_input_proj", options.weight_storage_type, config.timbre_dim, config.hidden_dim, 1, false, false, config.v2 ? 2 : 1, 1, intermediate);
    weights.clip_cond = load_linear(store, source, "clip_cond_proj", options.weight_storage_type, config.hidden_dim, config.hidden_dim, true);
    weights.text_cond = load_linear(store, source, "text_cond_proj", options.weight_storage_type, config.hidden_dim, config.hidden_dim, true);
    weights.timbre_cond = load_linear(store, source, "timbre_cond_proj", options.weight_storage_type, config.hidden_dim, config.hidden_dim, true);
    weights.global_cond = load_gated_ffn(store, source, "global_cond_mlp", options.weight_storage_type, config.hidden_dim, intermediate);
    weights.time_fc1 = load_linear(store, source, "t_embed.mlp.0", options.weight_storage_type, config.hidden_dim, config.v2 ? config.hidden_dim : 256, true);
    weights.time_fc2 = load_linear(store, source, "t_embed.mlp.2", options.weight_storage_type, config.hidden_dim, config.hidden_dim, true);
    weights.repa = {
        load_linear(store, source, "repa_mlp.feature_mlp.0", options.weight_storage_type, config.hidden_dim, config.hidden_dim, true),
        binding::norm_from_source(store, source, "repa_mlp.feature_mlp.1", config.hidden_dim),
    };
    weights.final_layer = {
        load_linear(store, source, "final_layer.adaLN_modulation.1", options.weight_storage_type, 2 * config.hidden_dim, config.hidden_dim, true),
        load_conv1d(store, source, "final_layer.conv", options.weight_storage_type, config.latent_dim, config.hidden_dim, 7, true),
    };
    const int64_t joint_depth = config.depth - config.fused_depth;
    weights.joint_blocks.reserve(static_cast<size_t>(joint_depth));
    for (int64_t i = 0; i < joint_depth; ++i) {
        const bool pre_only = i == joint_depth - 1;
        const std::string prefix = "joint_blocks." + std::to_string(i);
        weights.joint_blocks.push_back({
            load_single_block(store, source, prefix + ".latent_block", config, options, false, true),
            load_single_block(store, source, prefix + ".clip_block", config, options, pre_only, true),
            load_single_block(store, source, prefix + ".text_block", config, options, pre_only, false),
            load_single_block(store, source, prefix + ".audio_block", config, options, pre_only, false),
            pre_only,
        });
    }
    weights.fused_blocks.reserve(static_cast<size_t>(config.fused_depth));
    for (int64_t i = 0; i < config.fused_depth; ++i) {
        weights.fused_blocks.push_back(load_single_block(
            store,
            source,
            "fused_blocks." + std::to_string(i),
            config,
            options,
            false,
            true));
    }
    weights.latent_mean = store.load_f32_tensor(source, "latent_mean", {1, 1, config.latent_dim});
    weights.latent_std = store.load_f32_tensor(source, "latent_std", {1, 1, config.latent_dim});
    weights.sync_pos_emb = store.load_f32_tensor(source, "sync_pos_emb", {1, 1, 8, config.sync_dim});
    store.upload();
    return weights;
}

core::TensorValue build_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ProjectionWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    int64_t input_dim,
    bool first_conv,
    int64_t first_kernel,
    bool conv_mlp,
    int64_t mlp_kernel,
    bool use_activation) {
    const int64_t intermediate = hidden_multiple(config.hidden_dim, config.mlp_ratio);
    core::TensorValue x = first_conv
        ? channel_last_conv1d(ctx, input, weights.conv, input_dim, config.hidden_dim, first_kernel, true)
        : modules::LinearModule({input_dim, config.hidden_dim, true, options.projection_precision}).build(ctx, input, weights.linear);
    if (use_activation) {
        x = config.v2 ? modules::SiluModule{}.build(ctx, x) : modules::SeluModule{}.build(ctx, x);
    }
    if (conv_mlp) {
        return modules::GatedConvFeedForwardModule({
            config.hidden_dim,
            intermediate,
            mlp_kernel,
            false,
            true,
            false,
            modules::GatedFeedForwardActivation::Silu,
        }).build(ctx, x, weights.conv_mlp);
    }
    return modules::GatedFeedForwardModule({
        config.hidden_dim,
        intermediate,
        false,
        modules::GatedFeedForwardActivation::Silu,
        modules::GeluApproximation::Tanh,
        options.projection_precision,
    }).build(ctx, x, weights.mlp);
}

core::TensorValue mean_sequence(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return squeeze_sequence(ctx, modules::ReduceMeanModule({1}).build(ctx, input));
}

struct Modulation {
    core::TensorValue shift_msa;
    core::TensorValue scale_msa;
    core::TensorValue gate_msa;
    core::TensorValue shift_mlp;
    core::TensorValue scale_mlp;
    core::TensorValue gate_mlp;
};

Modulation build_modulation(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & condition,
    const SingleBlockWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options) {
    auto x = modules::SiluModule{}.build(ctx, condition);
    x = modules::LinearModule({
        config.hidden_dim,
        (weights.pre_only ? 2 : 6) * config.hidden_dim,
        true,
        options.projection_precision,
    }).build(ctx, x, weights.modulation);
    Modulation out;
    out.shift_msa = modules::SliceModule({2, 0, config.hidden_dim}).build(ctx, x);
    out.scale_msa = modules::SliceModule({2, config.hidden_dim, config.hidden_dim}).build(ctx, x);
    if (!weights.pre_only) {
        out.gate_msa = modules::SliceModule({2, 2 * config.hidden_dim, config.hidden_dim}).build(ctx, x);
        out.shift_mlp = modules::SliceModule({2, 3 * config.hidden_dim, config.hidden_dim}).build(ctx, x);
        out.scale_mlp = modules::SliceModule({2, 4 * config.hidden_dim, config.hidden_dim}).build(ctx, x);
        out.gate_mlp = modules::SliceModule({2, 5 * config.hidden_dim, config.hidden_dim}).build(ctx, x);
    }
    return out;
}

struct Qkv {
    core::TensorValue q;
    core::TensorValue k;
    core::TensorValue v;
};

Qkv build_pre_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const SingleBlockWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    const core::TensorValue * positions,
    float rope_freq_scale,
    Modulation & modulation) {
    modulation = build_modulation(ctx, condition, weights, config, options);
    auto x = modules::LayerNormModule({config.hidden_dim, config.layer_norm_eps, false, false}).build(ctx, input, {});
    x = modulate(ctx, x, modulation.shift_msa, modulation.scale_msa);
    auto qkv = modules::LinearModule({
        config.hidden_dim,
        3 * config.hidden_dim,
        true,
        options.projection_precision,
    }).build(ctx, x, weights.attention.qkv);
    const int64_t head_dim = config.hidden_dim / config.num_heads;
    qkv = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, qkv),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_dim, 3}));
    auto q = modules::SliceModule({3, 0, 1}).build(ctx, qkv);
    auto k = modules::SliceModule({3, 1, 1}).build(ctx, qkv);
    auto v = modules::SliceModule({3, 2, 1}).build(ctx, qkv);
    q = core::reshape_tensor(ctx, ensure_contiguous(ctx, q), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads, head_dim}));
    k = core::reshape_tensor(ctx, ensure_contiguous(ctx, k), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads, head_dim}));
    v = core::reshape_tensor(ctx, ensure_contiguous(ctx, v), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_heads, head_dim}));
    q = modules::RMSNormModule({head_dim, config.rms_norm_eps, true, false}).build(ctx, q, weights.attention.q_norm);
    k = modules::RMSNormModule({head_dim, config.rms_norm_eps, true, false}).build(ctx, k, weights.attention.k_norm);
    if (positions != nullptr) {
        q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta, rope_freq_scale}).build(ctx, q, *positions);
        k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta, rope_freq_scale}).build(ctx, k, *positions);
    }
    return {
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q),
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k),
        modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v),
    };
}

core::TensorValue scaled_attention(
    core::ModuleBuildContext & ctx,
    const Qkv & qkv,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options) {
    const int64_t head_dim = config.hidden_dim / config.num_heads;
    auto kt = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, qkv.k);
    auto scores = modules::MatMulModule{}.build(ctx, qkv.q, kt);
    if (options.attention_precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(scores.tensor, options.attention_precision);
    }
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))), scores.shape, GGML_TYPE_F32);
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, ensure_contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto out = modules::MatMulModule{}.build(ctx, attn, qkv.v);
    if (options.attention_precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(out.tensor, options.attention_precision);
    }
    out = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, out);
    return core::reshape_tensor(ctx, ensure_contiguous(ctx, out), core::TensorShape::from_dims({qkv.q.shape.dims[0], qkv.q.shape.dims[2], config.hidden_dim}));
}

core::TensorValue apply_block_post_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & attn_out,
    const Modulation & modulation,
    const SingleBlockWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options) {
    if (weights.pre_only) {
        return input;
    }
    core::TensorValue projected = weights.conv_path
        ? channel_last_conv1d(ctx, attn_out, weights.conv_out, config.hidden_dim, config.hidden_dim, 3, true)
        : modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision}).build(ctx, attn_out, weights.linear_out);
    auto x = core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, mul_tensors(ctx, projected, modulation.gate_msa).tensor), input.shape, GGML_TYPE_F32);
    auto normalized = modules::LayerNormModule({config.hidden_dim, config.layer_norm_eps, false, false}).build(ctx, x, {});
    normalized = modulate(ctx, normalized, modulation.shift_mlp, modulation.scale_mlp);
    core::TensorValue ffn = weights.conv_path
        ? modules::GatedConvFeedForwardModule({
              config.hidden_dim,
              hidden_multiple(config.hidden_dim, config.mlp_ratio),
              3,
              false,
              true,
              false,
              modules::GatedFeedForwardActivation::Silu,
          }).build(ctx, normalized, weights.conv_ffn)
        : modules::GatedFeedForwardModule({
              config.hidden_dim,
              hidden_multiple(config.hidden_dim, config.mlp_ratio),
              false,
              modules::GatedFeedForwardActivation::Silu,
              modules::GeluApproximation::Tanh,
              options.projection_precision,
          }).build(ctx, normalized, weights.linear_ffn);
    return core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, mul_tensors(ctx, ffn, modulation.gate_mlp).tensor), x.shape, GGML_TYPE_F32);
}

core::TensorValue apply_single_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const SingleBlockWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    const core::TensorValue & positions) {
    Modulation modulation;
    auto qkv = build_pre_attention(ctx, input, condition, weights, config, options, &positions, 1.0F, modulation);
    auto attn_out = scaled_attention(ctx, qkv, config, options);
    return apply_block_post_attention(ctx, input, attn_out, modulation, weights, config, options);
}

struct JointOutputs {
    core::TensorValue latent;
    core::TensorValue clip;
    core::TensorValue text;
    core::TensorValue audio;
};

JointOutputs apply_joint_block(
    core::ModuleBuildContext & ctx,
    const JointBlockWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    const core::TensorValue & latent,
    const core::TensorValue & clip,
    const core::TensorValue & text,
    const core::TensorValue & audio,
    const core::TensorValue & global,
    const core::TensorValue & extended,
    const core::TensorValue & latent_positions,
    const core::TensorValue & clip_positions) {
    Modulation latent_mod;
    auto latent_qkv = build_pre_attention(ctx, latent, extended, weights.latent, config, options, &latent_positions, 1.0F, latent_mod);
    Modulation clip_mod;
    auto clip_qkv = build_pre_attention(
        ctx,
        clip,
        global,
        weights.clip,
        config,
        options,
        &clip_positions,
        static_cast<float>(config.latent_seq_len) / static_cast<float>(config.clip_seq_len),
        clip_mod);
    Modulation text_mod;
    auto text_qkv = build_pre_attention(ctx, text, global, weights.text, config, options, nullptr, 1.0F, text_mod);
    Modulation audio_mod;
    auto audio_qkv = build_pre_attention(ctx, audio, global, weights.audio, config, options, nullptr, 1.0F, audio_mod);

    auto joint_q = modules::ConcatModule({2}).build(
        ctx,
        modules::ConcatModule({2}).build(ctx, latent_qkv.q, audio_qkv.q),
        modules::ConcatModule({2}).build(ctx, clip_qkv.q, text_qkv.q));
    auto joint_k = modules::ConcatModule({2}).build(
        ctx,
        modules::ConcatModule({2}).build(ctx, latent_qkv.k, audio_qkv.k),
        modules::ConcatModule({2}).build(ctx, clip_qkv.k, text_qkv.k));
    auto joint_v = modules::ConcatModule({2}).build(
        ctx,
        modules::ConcatModule({2}).build(ctx, latent_qkv.v, audio_qkv.v),
        modules::ConcatModule({2}).build(ctx, clip_qkv.v, text_qkv.v));
    auto attn_out = scaled_attention(ctx, {joint_q, joint_k, joint_v}, config, options);
    const int64_t latent_len = latent.shape.dims[1];
    const int64_t audio_len = audio.shape.dims[1];
    const int64_t clip_len = clip.shape.dims[1];
    const int64_t text_len = text.shape.dims[1];
    auto latent_attn = modules::SliceModule({1, 0, latent_len}).build(ctx, attn_out);
    auto audio_attn = modules::SliceModule({1, latent_len, audio_len}).build(ctx, attn_out);
    auto clip_attn = modules::SliceModule({1, latent_len + audio_len, clip_len}).build(ctx, attn_out);
    auto text_attn = modules::SliceModule({1, latent_len + audio_len + clip_len, text_len}).build(ctx, attn_out);
    return {
        apply_block_post_attention(ctx, latent, latent_attn, latent_mod, weights.latent, config, options),
        apply_block_post_attention(ctx, clip, clip_attn, clip_mod, weights.clip, config, options),
        apply_block_post_attention(ctx, text, text_attn, text_mod, weights.text, config, options),
        apply_block_post_attention(ctx, audio, audio_attn, audio_mod, weights.audio, config, options),
    };
}

std::vector<int32_t> positions_vector(int64_t count) {
    std::vector<int32_t> out(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

std::vector<float> timestep_embedding(
    const std::vector<float> & timesteps,
    int64_t frequency_size,
    int64_t max_period) {
    const int64_t batch = static_cast<int64_t>(timesteps.size());
    std::vector<float> out(static_cast<size_t>(batch * frequency_size), 0.0F);
    const int64_t half = frequency_size / 2;
    const float freq_scale = 10000.0F / static_cast<float>(max_period);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < half; ++i) {
            const float freq = freq_scale /
                std::pow(10000.0F, static_cast<float>(2 * i) / static_cast<float>(frequency_size));
            const float phase = timesteps[static_cast<size_t>(b)] * freq;
            out[static_cast<size_t>(b * frequency_size + i)] = std::cos(phase);
            out[static_cast<size_t>(b * frequency_size + half + i)] = std::sin(phase);
        }
    }
    return out;
}

std::vector<float> linear_align_corners_false_matrix(int64_t input_frames, int64_t output_frames) {
    std::vector<float> out(static_cast<size_t>(output_frames * input_frames), 0.0F);
    const float scale = static_cast<float>(input_frames) / static_cast<float>(output_frames);
    for (int64_t output = 0; output < output_frames; ++output) {
        const float source = (static_cast<float>(output) + 0.5F) * scale - 0.5F;
        int64_t left = static_cast<int64_t>(std::floor(source));
        int64_t right = left + 1;
        float right_weight = source - static_cast<float>(left);
        if (left < 0) {
            left = 0;
            right = 0;
            right_weight = 0.0F;
        } else if (right >= input_frames) {
            left = input_frames - 1;
            right = input_frames - 1;
            right_weight = 0.0F;
        }
        out[static_cast<size_t>(output * input_frames + left)] += 1.0F - right_weight;
        out[static_cast<size_t>(output * input_frames + right)] += right_weight;
    }
    return out;
}

std::vector<float> nearest_exact_matrix(int64_t input_frames, int64_t output_frames) {
    std::vector<float> out(static_cast<size_t>(output_frames * input_frames), 0.0F);
    const float scale = static_cast<float>(input_frames) / static_cast<float>(output_frames);
    for (int64_t output = 0; output < output_frames; ++output) {
        int64_t source = static_cast<int64_t>(std::floor((static_cast<float>(output) + 0.5F) * scale));
        source = std::max<int64_t>(0, std::min<int64_t>(source, input_frames - 1));
        out[static_cast<size_t>(output * input_frames + source)] = 1.0F;
    }
    return out;
}

core::TensorValue interpolate_time_with_matrix(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & matrix,
    int64_t input_frames,
    int64_t output_frames) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = modules::LinearModule({input_frames, output_frames, false}).build(ctx, x, {matrix, std::nullopt});
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

struct ConditionGraph {
    explicit ConditionGraph(ggml_backend_t owner_backend_in)
        : owner_backend(owner_backend_in) {}

    std::unique_ptr<ggml_context, GgmlContextDeleter> context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_t owner_backend = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue clip_in;
    core::TensorValue visual_in;
    core::TensorValue sync_in;
    core::TensorValue text_in;
    core::TensorValue audio_in;
    core::TensorValue timbre_in;
    core::TensorValue visual_interp;
    core::TensorValue sync_interp;
    core::TensorValue clip_out;
    core::TensorValue sync_out;
    core::TensorValue text_out;
    core::TensorValue audio_out;
    core::TensorValue timbre_out;
    core::TensorValue clip_cond;
    core::TensorValue text_cond;
    std::vector<float> visual_interp_values;
    std::vector<float> sync_interp_values;
    int64_t batch = 0;

    ~ConditionGraph() {
        if (owner_backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(owner_backend, graph);
            graph = nullptr;
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }
};

struct FlowGraph {
    explicit FlowGraph(ggml_backend_t owner_backend_in)
        : owner_backend(owner_backend_in) {}

    std::unique_ptr<ggml_context, GgmlContextDeleter> context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_t owner_backend = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue latent;
    core::TensorValue t_freq;
    core::TensorValue clip;
    core::TensorValue sync;
    core::TensorValue text;
    core::TensorValue audio;
    core::TensorValue timbre;
    core::TensorValue clip_cond;
    core::TensorValue text_cond;
    core::TensorValue latent_positions;
    core::TensorValue clip_positions;
    core::TensorValue flow;
    core::TensorValue multimodal;
    core::TensorValue hidden;
    int64_t batch = 0;

    ~FlowGraph() {
        if (owner_backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(owner_backend, graph);
            graph = nullptr;
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
    }
};

std::unique_ptr<ConditionGraph> build_condition_graph(
    core::ExecutionContext & execution,
    const FlowWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    int64_t batch) {
    auto out = std::make_unique<ConditionGraph>(execution.backend());
    ggml_init_params params{options.condition_graph_arena_bytes, nullptr, true};
    out->context.reset(ggml_init(params));
    if (out->context == nullptr) {
        throw std::runtime_error("failed to initialize ControlFoley condition graph context");
    }
    core::ModuleBuildContext ctx{out->context.get(), "framework.controlfoley_flow.conditions", execution.backend_type()};
    out->batch = batch;
    out->clip_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.clip_seq_len, config.clip_dim}));
    out->visual_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.visual_seq_len, config.visual_dim}));
    out->sync_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.sync_seq_len, config.sync_dim}));
    out->text_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.text_seq_len, config.text_dim}));
    out->audio_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.audio_seq_len, config.audio_dim}));
    out->timbre_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.timbre_seq_len, config.timbre_dim}));
    out->visual_interp = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.clip_seq_len, config.visual_seq_len}));
    out->sync_interp = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({config.latent_seq_len, config.sync_seq_len}));

    auto sync = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, out->sync_in),
        core::TensorShape::from_dims({batch, config.sync_seq_len / 8, 8, config.sync_dim}));
    sync = add_tensors(ctx, sync, weights.sync_pos_emb);
    sync = core::reshape_tensor(ctx, ensure_contiguous(ctx, sync), core::TensorShape::from_dims({batch, config.sync_seq_len, config.sync_dim}));

    out->clip_out = build_projection(
        ctx,
        out->clip_in,
        weights.clip_input,
        config,
        options,
        config.clip_dim,
        false,
        3,
        !config.v2,
        3,
        config.v2);
    auto visual = build_projection(
        ctx,
        out->visual_in,
        weights.visual_input,
        config,
        options,
        config.visual_dim,
        false,
        3,
        !config.v2,
        3,
        config.v2);
    if (config.visual_seq_len != config.clip_seq_len) {
        visual = interpolate_time_with_matrix(ctx, visual, out->visual_interp, config.visual_seq_len, config.clip_seq_len);
    }
    out->clip_out = add_tensors(ctx, out->clip_out, visual);

    out->sync_out = build_projection(
        ctx,
        sync,
        weights.sync_input,
        config,
        options,
        config.sync_dim,
        true,
        7,
        true,
        3,
        true);
    out->sync_out = interpolate_time_with_matrix(ctx, out->sync_out, out->sync_interp, config.sync_seq_len, config.latent_seq_len);

    out->text_out = build_projection(
        ctx,
        out->text_in,
        weights.text_input,
        config,
        options,
        config.text_dim,
        false,
        1,
        false,
        1,
        config.v2);
    out->audio_out = build_projection(
        ctx,
        out->audio_in,
        weights.clap_input,
        config,
        options,
        config.audio_dim,
        false,
        1,
        false,
        1,
        config.v2);
    auto timbre = build_projection(
        ctx,
        out->timbre_in,
        weights.timbre_input,
        config,
        options,
        config.timbre_dim,
        false,
        1,
        false,
        1,
        config.v2);

    out->clip_cond = modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision})
                         .build(ctx, mean_sequence(ctx, out->clip_out), weights.clip_cond);
    out->text_cond = modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision})
                         .build(ctx, mean_sequence(ctx, out->text_out), weights.text_cond);
    out->timbre_out = modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision})
                          .build(ctx, mean_sequence(ctx, timbre), weights.timbre_cond);

    out->graph = ggml_new_graph_custom(out->context.get(), options.graph_nodes, false);
    ggml_build_forward_expand(out->graph, out->clip_out.tensor);
    ggml_build_forward_expand(out->graph, out->sync_out.tensor);
    ggml_build_forward_expand(out->graph, out->text_out.tensor);
    ggml_build_forward_expand(out->graph, out->audio_out.tensor);
    ggml_build_forward_expand(out->graph, out->timbre_out.tensor);
    ggml_build_forward_expand(out->graph, out->clip_cond.tensor);
    ggml_build_forward_expand(out->graph, out->text_cond.tensor);
    out->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    if (out->gallocr == nullptr || !ggml_gallocr_reserve(out->gallocr, out->graph) ||
        !ggml_gallocr_alloc_graph(out->gallocr, out->graph)) {
        throw std::runtime_error("failed to allocate ControlFoley condition graph");
    }
    out->visual_interp_values = linear_align_corners_false_matrix(config.visual_seq_len, config.clip_seq_len);
    out->sync_interp_values = nearest_exact_matrix(config.sync_seq_len, config.latent_seq_len);
    core::write_tensor_f32(out->visual_interp, out->visual_interp_values);
    core::write_tensor_f32(out->sync_interp, out->sync_interp_values);
    core::prepare_host_graph_plan(execution, out->graph, out->plan);
    return out;
}

std::unique_ptr<FlowGraph> build_flow_graph(
    core::ExecutionContext & execution,
    const FlowWeights & weights,
    const ControlFoleyFlowConfig & config,
    const ControlFoleyFlowRuntimeOptions & options,
    int64_t batch) {
    auto out = std::make_unique<FlowGraph>(execution.backend());
    ggml_init_params params{options.flow_graph_arena_bytes, nullptr, true};
    out->context.reset(ggml_init(params));
    if (out->context == nullptr) {
        throw std::runtime_error("failed to initialize ControlFoley flow graph context");
    }
    core::ModuleBuildContext ctx{out->context.get(), "framework.controlfoley_flow", execution.backend_type()};
    out->batch = batch;
    out->latent = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.latent_seq_len, config.latent_dim}));
    out->t_freq = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.v2 ? config.hidden_dim : 256}));
    out->clip = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.clip_seq_len, config.hidden_dim}));
    out->sync = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.latent_seq_len, config.hidden_dim}));
    out->text = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.text_seq_len, config.hidden_dim}));
    out->audio = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.audio_seq_len, config.hidden_dim}));
    out->timbre = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_dim}));
    out->clip_cond = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_dim}));
    out->text_cond = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_dim}));
    out->latent_positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config.latent_seq_len}));
    out->clip_positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({config.clip_seq_len}));

    auto latent = build_projection(
        ctx,
        out->latent,
        weights.audio_input,
        config,
        options,
        config.latent_dim,
        true,
        7,
        true,
        7,
        true);
    auto global_seed = add_tensors(ctx, add_tensors(ctx, out->clip_cond, out->text_cond), out->timbre);
    auto global_c = modules::GatedFeedForwardModule({
        config.hidden_dim,
        hidden_multiple(config.hidden_dim, config.mlp_ratio),
        false,
        modules::GatedFeedForwardActivation::Silu,
        modules::GeluApproximation::Tanh,
        options.projection_precision,
    }).build(ctx, global_seed, weights.global_cond);
    out->multimodal = core::wrap_tensor(
        ggml_cpy(ctx.ggml, global_c.tensor, ggml_dup_tensor(ctx.ggml, global_c.tensor)),
        global_c.shape,
        GGML_TYPE_F32);
    auto time = modules::LinearModule({config.v2 ? config.hidden_dim : 256, config.hidden_dim, true, options.projection_precision})
                    .build(ctx, out->t_freq, weights.time_fc1);
    time = modules::SiluModule{}.build(ctx, time);
    time = modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision})
               .build(ctx, time, weights.time_fc2);
    global_c = add_tensors(ctx, unsqueeze_sequence(ctx, global_c), unsqueeze_sequence(ctx, time));
    auto extended_c = add_tensors(ctx, out->sync, global_c);

    auto clip = out->clip;
    auto text = out->text;
    auto audio = out->audio;
    for (const auto & block : weights.joint_blocks) {
        auto next = apply_joint_block(
            ctx,
            block,
            config,
            options,
            latent,
            clip,
            text,
            audio,
            global_c,
            extended_c,
            out->latent_positions,
            out->clip_positions);
        latent = next.latent;
        clip = next.clip;
        text = next.text;
        audio = next.audio;
    }
    out->hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config.hidden_dim}));
    for (int64_t i = 0; i < static_cast<int64_t>(weights.fused_blocks.size()); ++i) {
        latent = apply_single_block(ctx, latent, extended_c, weights.fused_blocks[static_cast<size_t>(i)], config, options, out->latent_positions);
        if (i == 7) {
            auto hidden = mean_sequence(ctx, latent);
            hidden = modules::LinearModule({config.hidden_dim, config.hidden_dim, true, options.projection_precision})
                         .build(ctx, hidden, weights.repa.linear);
            hidden = modules::LayerNormModule({config.hidden_dim, config.layer_norm_eps, true, true})
                         .build(ctx, hidden, weights.repa.norm);
            hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
            out->hidden = hidden;
        }
    }
    auto final_mod = modules::SiluModule{}.build(ctx, extended_c);
    final_mod = modules::LinearModule({config.hidden_dim, 2 * config.hidden_dim, true, options.projection_precision})
                    .build(ctx, final_mod, weights.final_layer.modulation);
    auto final_shift = modules::SliceModule({2, 0, config.hidden_dim}).build(ctx, final_mod);
    auto final_scale = modules::SliceModule({2, config.hidden_dim, config.hidden_dim}).build(ctx, final_mod);
    latent = modules::LayerNormModule({config.hidden_dim, config.layer_norm_eps, false, false}).build(ctx, latent, {});
    latent = modulate(ctx, latent, final_shift, final_scale);
    out->flow = channel_last_conv1d(ctx, latent, weights.final_layer.conv, config.hidden_dim, config.latent_dim, 7, true);

    out->graph = ggml_new_graph_custom(out->context.get(), options.graph_nodes, false);
    ggml_build_forward_expand(out->graph, out->flow.tensor);
    ggml_build_forward_expand(out->graph, out->multimodal.tensor);
    ggml_build_forward_expand(out->graph, out->hidden.tensor);
    out->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    if (out->gallocr == nullptr || !ggml_gallocr_reserve(out->gallocr, out->graph) ||
        !ggml_gallocr_alloc_graph(out->gallocr, out->graph)) {
        throw std::runtime_error("failed to allocate ControlFoley flow graph");
    }
    core::write_tensor_i32(out->latent_positions, positions_vector(config.latent_seq_len));
    core::write_tensor_i32(out->clip_positions, positions_vector(config.clip_seq_len));
    core::prepare_host_graph_plan(execution, out->graph, out->plan);
    return out;
}

}  // namespace

struct ControlFoleyFlowDenoiserRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution_ref,
        ControlFoleyFlowConfig config_value,
        ControlFoleyFlowRuntimeOptions options_value)
        : execution(execution_ref),
          config(config_value),
          options(options_value),
          weights(load_flow_weights(*source, execution_ref, config, options)) {
        validate_config(config);
    }

    core::ExecutionContext & execution;
    ControlFoleyFlowConfig config;
    ControlFoleyFlowRuntimeOptions options;
    FlowWeights weights;
    std::unique_ptr<ConditionGraph> condition_graph;
    std::unique_ptr<FlowGraph> flow_graph;
};

ControlFoleyFlowDenoiserRuntime::ControlFoleyFlowDenoiserRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    ControlFoleyFlowConfig config,
    ControlFoleyFlowRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, config, options)) {}

ControlFoleyFlowDenoiserRuntime::~ControlFoleyFlowDenoiserRuntime() = default;

void ControlFoleyFlowDenoiserRuntime::update_config(ControlFoleyFlowConfig config) {
    if (impl_ == nullptr) {
        throw std::runtime_error("ControlFoley flow runtime is not initialized");
    }
    validate_config(config);
    if (!same_weight_shape_config(impl_->config, config)) {
        throw std::runtime_error("ControlFoley flow runtime cannot change weight-bearing config after initialization");
    }
    if (impl_->config.latent_seq_len == config.latent_seq_len &&
        impl_->config.clip_seq_len == config.clip_seq_len &&
        impl_->config.visual_seq_len == config.visual_seq_len &&
        impl_->config.sync_seq_len == config.sync_seq_len &&
        impl_->config.text_seq_len == config.text_seq_len &&
        impl_->config.audio_seq_len == config.audio_seq_len &&
        impl_->config.timbre_seq_len == config.timbre_seq_len) {
        return;
    }
    impl_->config = config;
    impl_->condition_graph.reset();
    impl_->flow_graph.reset();
}

ControlFoleyFlowConditions ControlFoleyFlowDenoiserRuntime::preprocess_conditions(
    const ControlFoleyFlowConditionInput & input) {
    if (impl_ == nullptr) {
        throw std::runtime_error("ControlFoley flow runtime is not initialized");
    }
    const auto & config = impl_->config;
    const int64_t batch = input.batch;
    if (batch <= 0) {
        throw std::runtime_error("ControlFoley condition batch must be positive");
    }
    if (impl_->condition_graph == nullptr || impl_->condition_graph->batch != batch) {
        impl_->condition_graph.reset();
        impl_->condition_graph = build_condition_graph(impl_->execution, impl_->weights, config, impl_->options, batch);
    }
    auto & graph = *impl_->condition_graph;
    core::write_tensor_f32(graph.clip_in, input.clip);
    core::write_tensor_f32(graph.visual_in, input.visual);
    core::write_tensor_f32(graph.sync_in, input.sync);
    core::write_tensor_f32(graph.text_in, input.text);
    core::write_tensor_f32(graph.audio_in, input.audio);
    core::write_tensor_f32(graph.timbre_in, input.timbre);
    core::write_tensor_f32(graph.visual_interp, graph.visual_interp_values);
    core::write_tensor_f32(graph.sync_interp, graph.sync_interp_values);
    const ggml_status status = core::compute_graph(
        impl_->execution,
        graph.graph,
        graph.plan,
        "framework.controlfoley_flow.conditions");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ControlFoley condition graph compute failed");
    }
    return {
        batch,
        core::read_tensor_f32(graph.clip_out.tensor),
        core::read_tensor_f32(graph.sync_out.tensor),
        core::read_tensor_f32(graph.text_out.tensor),
        core::read_tensor_f32(graph.audio_out.tensor),
        core::read_tensor_f32(graph.timbre_out.tensor),
        core::read_tensor_f32(graph.clip_cond.tensor),
        core::read_tensor_f32(graph.text_cond.tensor),
    };
}

ControlFoleyFlowPrediction ControlFoleyFlowDenoiserRuntime::predict_flow(
    const std::vector<float> & latent,
    const std::vector<float> & timesteps,
    const ControlFoleyFlowConditions & conditions) {
    if (impl_ == nullptr) {
        throw std::runtime_error("ControlFoley flow runtime is not initialized");
    }
    const auto & config = impl_->config;
    const int64_t batch = conditions.batch;
    if (batch <= 0 || static_cast<int64_t>(timesteps.size()) != batch) {
        throw std::runtime_error("ControlFoley flow batch mismatch");
    }
    if (impl_->flow_graph == nullptr || impl_->flow_graph->batch != batch) {
        impl_->flow_graph.reset();
        impl_->flow_graph = build_flow_graph(impl_->execution, impl_->weights, config, impl_->options, batch);
    }
    auto & graph = *impl_->flow_graph;
    core::write_tensor_f32(graph.latent, latent);
    core::write_tensor_f32(graph.t_freq, timestep_embedding(timesteps, config.v2 ? config.hidden_dim : 256, config.v2 ? 1 : 10000));
    core::write_tensor_f32(graph.clip, conditions.clip);
    core::write_tensor_f32(graph.sync, conditions.sync);
    core::write_tensor_f32(graph.text, conditions.text);
    core::write_tensor_f32(graph.audio, conditions.audio);
    core::write_tensor_f32(graph.timbre, conditions.timbre);
    core::write_tensor_f32(graph.clip_cond, conditions.clip_cond);
    core::write_tensor_f32(graph.text_cond, conditions.text_cond);
    const ggml_status status = core::compute_graph(
        impl_->execution,
        graph.graph,
        graph.plan,
        "framework.controlfoley_flow");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ControlFoley flow graph compute failed");
    }
    return {
        batch,
        config.latent_seq_len,
        config.latent_dim,
        core::read_tensor_f32(graph.flow.tensor),
        core::read_tensor_f32(graph.multimodal.tensor),
        core::read_tensor_f32(graph.hidden.tensor),
    };
}

void ControlFoleyFlowDenoiserRuntime::release_runtime_graphs() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->condition_graph.reset();
    impl_->flow_graph.reset();
}

}  // namespace engine::models::controlfoley
