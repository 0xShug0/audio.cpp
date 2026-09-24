#include "pipeline_internal.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::liveavatar {

constexpr size_t kDenoiserWeightContextBytes = 18000ull * 1024ull * 1024ull;
constexpr size_t kDenoiserConditionGraphContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDenoiserStaticCacheGraphContextBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDenoiserGraphContextBytes = 4096ull * 1024ull * 1024ull;
constexpr size_t kDenoiserLayerwiseGraphContextBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDenoiserLayerwiseInputContextBytes = 256ull * 1024ull * 1024ull;

constexpr int32_t kT5PadTokenId = 0;
constexpr int32_t kT5EosTokenId = 1;

engine::modules::Conv3dWeights load_denoiser_conv3d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::Native) {
    return engine::modules::binding::conv3d_from_source(
        store,
        source,
        prefix,
        storage_type,
        out_channels,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        true);
}

LiveAvatarDenoiserAttentionWeights load_denoiser_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden,
    engine::assets::TensorStorageType storage_type) {
    LiveAvatarDenoiserAttentionWeights weights;
    weights.q = engine::modules::binding::linear_from_source(store, source, prefix + ".q", storage_type, hidden, hidden, true);
    weights.k = engine::modules::binding::linear_from_source(store, source, prefix + ".k", storage_type, hidden, hidden, true);
    weights.v = engine::modules::binding::linear_from_source(store, source, prefix + ".v", storage_type, hidden, hidden, true);
    weights.o = engine::modules::binding::linear_from_source(store, source, prefix + ".o", storage_type, hidden, hidden, true);
    weights.norm_q = store.load_f32_tensor(source, prefix + ".norm_q.weight", {hidden});
    weights.norm_k = store.load_f32_tensor(source, prefix + ".norm_k.weight", {hidden});
    return weights;
}

LiveAvatarDenoiserWeights load_denoiser_weights(
    const engine::assets::TensorSource & source,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const LiveAvatarConfig & config,
    bool stream_block_weights) {
    LiveAvatarDenoiserWeights weights;
    weights.store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "liveavatar.denoiser.weights",
        kDenoiserWeightContextBytes);
    auto & store = *weights.store;
    engine::core::BackendWeightStore * block_store = &store;
    if (stream_block_weights) {
        auto * host_buffer_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (host_buffer_type == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming requires a host buffer type for the selected backend");
        }
        weights.block_store = std::make_shared<engine::core::BackendWeightStore>(
            backend,
            backend_type,
            "liveavatar.denoiser.block_weights",
            kDenoiserWeightContextBytes,
            host_buffer_type);
        block_store = weights.block_store.get();
        weights.block_weights_host_resident = true;
    }
    const auto native = engine::assets::TensorStorageType::Native;
    const int64_t hidden = config.hidden_size;
    const int64_t head_dim = hidden / config.num_heads;
    if (hidden % config.num_heads != 0 || head_dim <= 0) {
        throw std::runtime_error("LiveAvatar denoiser head configuration is invalid");
    }

    weights.patch_embedding = load_denoiser_conv3d(store, source, "patch_embedding", config.latent_channels, hidden, 1, 2, 2, native);
    weights.cond_encoder = load_denoiser_conv3d(store, source, "cond_encoder", config.latent_channels, hidden, 1, 2, 2, native);
    weights.text_embedding_0 =
        engine::modules::binding::linear_from_source(store, source, "text_embedding.0", native, hidden, config.text_dim, true);
    weights.text_embedding_2 =
        engine::modules::binding::linear_from_source(store, source, "text_embedding.2", native, hidden, hidden, true);
    weights.time_embedding_0 =
        engine::modules::binding::linear_from_source(store, source, "time_embedding.0", native, hidden, 256, true);
    weights.time_embedding_2 =
        engine::modules::binding::linear_from_source(store, source, "time_embedding.2", native, hidden, hidden, true);
    weights.time_projection =
        engine::modules::binding::linear_from_source(store, source, "time_projection.1", native, 6 * hidden, hidden, true);

    weights.blocks.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "blocks." + std::to_string(layer);
        LiveAvatarDenoiserBlockWeights block;
        block.modulation = block_store->load_f32_tensor(source, prefix + ".modulation", {1, 6, hidden});
        block.self_attention = load_denoiser_attention(*block_store, source, prefix + ".self_attn", hidden, native);
        block.cross_attention = load_denoiser_attention(*block_store, source, prefix + ".cross_attn", hidden, native);
        block.norm3.weight = block_store->load_f32_tensor(source, prefix + ".norm3.weight", {hidden});
        block.norm3.bias = block_store->load_f32_tensor(source, prefix + ".norm3.bias", {hidden});
        block.ffn_in =
            engine::modules::binding::linear_from_source(*block_store, source, prefix + ".ffn.0", native, config.ffn_dim, hidden, true);
        block.ffn_out =
            engine::modules::binding::linear_from_source(*block_store, source, prefix + ".ffn.2", native, hidden, config.ffn_dim, true);
        weights.blocks.push_back(std::move(block));
    }

    weights.audio_encoder.layer_weights = store.load_f32_tensor(source, "casual_audio_encoder.weights", {1, config.audio_layers, 1, 1});
    weights.audio_encoder.conv1_local = engine::modules::binding::conv1d_from_source(
        store,
        source,
        "casual_audio_encoder.encoder.conv1_local.conv",
        native,
        hidden / 4 * config.audio_tokens,
        config.audio_dim,
        3,
        true);
    weights.audio_encoder.conv1_global = engine::modules::binding::conv1d_from_source(
        store,
        source,
        "casual_audio_encoder.encoder.conv1_global.conv",
        native,
        hidden / 4,
        config.audio_dim,
        3,
        true);
    weights.audio_encoder.conv2 = engine::modules::binding::conv1d_from_source(
        store,
        source,
        "casual_audio_encoder.encoder.conv2.conv",
        native,
        hidden / 2,
        hidden / 4,
        3,
        true);
    weights.audio_encoder.conv3 = engine::modules::binding::conv1d_from_source(
        store,
        source,
        "casual_audio_encoder.encoder.conv3.conv",
        native,
        hidden,
        hidden / 2,
        3,
        true);
    weights.audio_encoder.final_linear = engine::modules::binding::linear_from_source(
        store, source, "casual_audio_encoder.encoder.final_linear", native, hidden, hidden, true);
    weights.audio_encoder.padding_tokens =
        store.load_f32_tensor(source, "casual_audio_encoder.encoder.padding_tokens", {1, 1, 1, hidden});

    weights.audio_injectors.reserve(12);
    for (int64_t index = 0; index < 12; ++index) {
        LiveAvatarDenoiserAudioInjectorWeights injector;
        injector.attention = load_denoiser_attention(
            store,
            source,
            "audio_injector.injector." + std::to_string(index),
            hidden,
            native);
        injector.adain_linear = engine::modules::binding::linear_from_source(
            store,
            source,
            "audio_injector.injector_adain_layers." + std::to_string(index) + ".linear",
            native,
            2 * hidden,
            hidden,
            true);
        weights.audio_injectors.push_back(std::move(injector));
    }

    weights.frame_packer.proj = load_denoiser_conv3d(store, source, "frame_packer.proj", config.latent_channels, hidden, 1, 2, 2, native);
    weights.frame_packer.proj_2x = load_denoiser_conv3d(store, source, "frame_packer.proj_2x", config.latent_channels, hidden, 2, 4, 4, native);
    weights.frame_packer.proj_4x = load_denoiser_conv3d(store, source, "frame_packer.proj_4x", config.latent_channels, hidden, 4, 8, 8, native);
    weights.trainable_cond_mask = store.load_f32_tensor(source, "trainable_cond_mask.weight", {3, hidden});
    weights.head.modulation = store.load_f32_tensor(source, "head.modulation", {1, 2, hidden});
    weights.head.projection =
        engine::modules::binding::linear_from_source(store, source, "head.head", native, config.latent_channels * 4, hidden, true);

    weights.store->upload();
    if (weights.block_store != nullptr) {
        weights.block_store->upload();
    }
    return weights;
}


std::vector<float> sinusoidal_embedding_1d(int64_t dim, float position) {
    if (dim <= 0 || dim % 2 != 0) {
        throw std::runtime_error("LiveAvatar sinusoidal embedding dimension must be positive and even");
    }
    std::vector<float> out(static_cast<size_t>(dim));
    const int64_t half = dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const double freq = std::pow(10000.0, -static_cast<double>(i) / static_cast<double>(half));
        const double value = static_cast<double>(position) * freq;
        out[static_cast<size_t>(i)] = static_cast<float>(std::cos(value));
        out[static_cast<size_t>(half + i)] = static_cast<float>(std::sin(value));
    }
    return out;
}

std::vector<int64_t> linspace_indices(int64_t start, int64_t end, int64_t count) {
    if (count <= 0) {
        return {};
    }
    if (count == 1) {
        return {start};
    }
    std::vector<int64_t> out(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(start) +
            (static_cast<double>(end - start) * static_cast<double>(i)) /
                static_cast<double>(count - 1);
        out[static_cast<size_t>(i)] = static_cast<int64_t>(value);
    }
    return out;
}

std::vector<float> make_liveavatar_rope_table(
    int64_t heads,
    int64_t head_dim,
    const std::vector<LiveAvatarRopeRange> & ranges,
    bool sine) {
    if (heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("LiveAvatar RoPE head configuration is invalid");
    }
    const int64_t complex_dim = head_dim / 2;
    const int64_t spatial = complex_dim / 3;
    const int64_t time_dim = complex_dim - 2 * spatial;
    const int64_t height_dim = spatial;
    const int64_t width_dim = spatial;
    int64_t tokens = 0;
    for (const auto & range : ranges) {
        const int64_t ft = range.end_t - range.start_t;
        const int64_t fh = range.end_h - range.start_h;
        const int64_t fw = range.end_w - range.start_w;
        if (ft < 0 || fh < 0 || fw < 0) {
            throw std::runtime_error("LiveAvatar RoPE range is invalid");
        }
        tokens += ft * fh * fw;
    }
    std::vector<float> out(static_cast<size_t>(tokens * heads * complex_dim), 0.0F);
    int64_t token = 0;
    auto phase = [](int64_t index, int64_t dim_index, int64_t dim_count) {
        return static_cast<double>(index) *
            std::pow(10000.0, -static_cast<double>(2 * dim_index) / static_cast<double>(dim_count * 2));
    };
    for (const auto & range : ranges) {
        const int64_t seq_t = range.end_t - range.start_t;
        const int64_t seq_h = range.end_h - range.start_h;
        const int64_t seq_w = range.end_w - range.start_w;
        const auto time_indices = range.start_t >= 0
            ? linspace_indices(range.start_t, range.total_t + range.start_t - 1, seq_t)
            : linspace_indices(-range.start_t, -range.total_t - range.start_t + 1, seq_t);
        const auto height_indices = linspace_indices(range.start_h, range.total_h + range.start_h - 1, seq_h);
        const auto width_indices = linspace_indices(range.start_w, range.total_w + range.start_w - 1, seq_w);
        for (int64_t t = 0; t < seq_t; ++t) {
            for (int64_t h = 0; h < seq_h; ++h) {
                for (int64_t w = 0; w < seq_w; ++w) {
                    for (int64_t head = 0; head < heads; ++head) {
                        int64_t offset = 0;
                        for (int64_t d = 0; d < time_dim; ++d, ++offset) {
                            const double value = phase(time_indices[static_cast<size_t>(t)], d, time_dim);
                            const double signed_value = range.start_t >= 0 ? value : -value;
                            out[static_cast<size_t>((token * heads + head) * complex_dim + offset)] =
                                sine ? static_cast<float>(std::sin(signed_value)) : static_cast<float>(std::cos(signed_value));
                        }
                        for (int64_t d = 0; d < height_dim; ++d, ++offset) {
                            const double value = phase(height_indices[static_cast<size_t>(h)], d, height_dim);
                            out[static_cast<size_t>((token * heads + head) * complex_dim + offset)] =
                                sine ? static_cast<float>(std::sin(value)) : static_cast<float>(std::cos(value));
                        }
                        for (int64_t d = 0; d < width_dim; ++d, ++offset) {
                            const double value = phase(width_indices[static_cast<size_t>(w)], d, width_dim);
                            out[static_cast<size_t>((token * heads + head) * complex_dim + offset)] =
                                sine ? static_cast<float>(std::sin(value)) : static_cast<float>(std::cos(value));
                        }
                    }
                    ++token;
                }
            }
        }
    }
    return out;
}

engine::core::TensorValue div_tensor(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    return engine::core::wrap_tensor(ggml_div(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

engine::core::TensorValue add_one(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    const auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, value);
    const auto flat = engine::core::reshape_tensor(
        ctx,
        contiguous,
        engine::core::TensorShape::from_dims({contiguous.shape.num_elements()}));
    const auto shifted = engine::core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, flat.tensor, 1.0F, 1.0F),
        flat.shape,
        GGML_TYPE_F32);
    return engine::core::reshape_tensor(ctx, shifted, value.shape);
}

engine::core::TensorValue linear_native(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features) {
    const bool use_nvfp4_activation_path = weights.weight.type == GGML_TYPE_NVFP4;
    return engine::modules::LinearModule({
        in_features,
        out_features,
        weights.bias.has_value(),
        GGML_PREC_DEFAULT,
        use_nvfp4_activation_path,
    }).build(ctx, input, weights);
}

engine::core::TensorValue conv3d_tokens(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::Conv3dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t stride_depth,
    int64_t stride_height,
    int64_t stride_width) {
    auto patched = engine::modules::Conv3dModule({
        in_channels,
        out_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        static_cast<int>(stride_depth),
        static_cast<int>(stride_height),
        static_cast<int>(stride_width),
        0,
        0,
        0,
        1,
        1,
        1,
        false,
    }).build(ctx, input, weights);
    if (weights.bias.has_value()) {
        const auto bias = engine::core::reshape_tensor(
            ctx,
            *weights.bias,
            engine::core::TensorShape::from_dims({out_channels, 1, 1, 1}));
        patched = engine::modules::AddModule().build(
            ctx,
            patched,
            engine::modules::RepeatModule({patched.shape}).build(ctx, bias));
    }
    const int64_t tokens = patched.shape.dims[1] * patched.shape.dims[2] * patched.shape.dims[3];
    auto flat = engine::core::reshape_tensor(ctx, patched, engine::core::TensorShape::from_dims({out_channels, tokens}));
    auto transposed = engine::modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, flat);
    return engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, transposed),
        engine::core::TensorShape::from_dims({1, tokens, out_channels}));
}

std::vector<LiveAvatarRopeRange> denoiser_rope_ranges(
    int64_t target_frames,
    int64_t latent_height,
    int64_t latent_width) {
    const int64_t grid_h = latent_height / 2;
    const int64_t grid_w = latent_width / 2;
    return {
        {0, 0, 0, target_frames, grid_h, grid_w, target_frames, grid_h, grid_w},
        {30, 0, 0, 31, grid_h, grid_w, 1, grid_h, grid_w},
    };
}

engine::core::TensorValue apply_shift_scale(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & shift,
    const engine::core::TensorValue & scale) {
    auto one_plus_scale = engine::modules::RepeatModule({input.shape}).build(ctx, add_one(ctx, scale));
    auto shift_expanded = engine::modules::RepeatModule({input.shape}).build(ctx, shift);
    return engine::modules::AddModule().build(ctx, engine::modules::MulModule().build(ctx, input, one_plus_scale), shift_expanded);
}

engine::core::TensorValue split_modulated_tokens(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & values,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t coefficient,
    int64_t segment_tokens,
    bool scale_add_one) {
    const int64_t total_tokens = values.shape.dims[1];
    if (segment_tokens < 0 || segment_tokens > total_tokens) {
        throw std::runtime_error("LiveAvatar segmented modulation token count is invalid");
    }
    auto actual_coeff = engine::modules::SliceModule({1, coefficient, 1}).build(ctx, actual);
    auto zero_coeff = engine::modules::SliceModule({1, coefficient, 1}).build(ctx, zero);
    if (scale_add_one) {
        actual_coeff = add_one(ctx, actual_coeff);
        zero_coeff = add_one(ctx, zero_coeff);
    }
    auto target = engine::modules::SliceModule({1, 0, segment_tokens}).build(ctx, values);
    target = engine::modules::MulModule().build(
        ctx,
        target,
        engine::modules::RepeatModule({target.shape}).build(ctx, actual_coeff));
    if (segment_tokens == total_tokens) {
        return target;
    }
    auto suffix = engine::modules::SliceModule({1, segment_tokens, total_tokens - segment_tokens}).build(ctx, values);
    suffix = engine::modules::MulModule().build(
        ctx,
        suffix,
        engine::modules::RepeatModule({suffix.shape}).build(ctx, zero_coeff));
    return engine::modules::ConcatModule({1, true}).build(ctx, target, suffix);
}

engine::core::TensorValue segmented_adaln_input(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & normalized,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t shift_index,
    int64_t scale_index,
    int64_t segment_tokens) {
    auto scale_applied = split_modulated_tokens(ctx, normalized, actual, zero, scale_index, segment_tokens, true);
    auto target_shift = engine::modules::SliceModule({1, shift_index, 1}).build(ctx, actual);
    auto zero_shift = engine::modules::SliceModule({1, shift_index, 1}).build(ctx, zero);
    auto target = engine::modules::SliceModule({1, 0, segment_tokens}).build(ctx, scale_applied);
    target = engine::modules::AddModule().build(
        ctx,
        target,
        engine::modules::RepeatModule({target.shape}).build(ctx, target_shift));
    if (segment_tokens == scale_applied.shape.dims[1]) {
        return target;
    }
    auto suffix = engine::modules::SliceModule({1, segment_tokens, scale_applied.shape.dims[1] - segment_tokens}).build(ctx, scale_applied);
    suffix = engine::modules::AddModule().build(
        ctx,
        suffix,
        engine::modules::RepeatModule({suffix.shape}).build(ctx, zero_shift));
    return engine::modules::ConcatModule({1, true}).build(ctx, target, suffix);
}

engine::core::TensorValue segmented_adaln_input_for_token_range(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & normalized,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t shift_index,
    int64_t scale_index,
    int64_t token_offset,
    int64_t original_tokens) {
    if (token_offset >= original_tokens) {
        auto scale = add_one(ctx, engine::modules::SliceModule({1, scale_index, 1}).build(ctx, zero));
        auto shift = engine::modules::SliceModule({1, shift_index, 1}).build(ctx, zero);
        auto scale_expanded = engine::modules::RepeatModule({normalized.shape}).build(ctx, scale);
        auto shift_expanded = engine::modules::RepeatModule({normalized.shape}).build(ctx, shift);
        return engine::modules::AddModule().build(
            ctx,
            engine::modules::MulModule().build(ctx, normalized, scale_expanded),
            shift_expanded);
    }
    if (token_offset + normalized.shape.dims[1] <= original_tokens) {
        return segmented_adaln_input(ctx, normalized, actual, zero, shift_index, scale_index, normalized.shape.dims[1]);
    }
    return segmented_adaln_input(ctx, normalized, actual, zero, shift_index, scale_index, original_tokens - token_offset);
}

engine::core::TensorValue split_modulated_tokens_for_token_range(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & values,
    const engine::core::TensorValue & actual,
    const engine::core::TensorValue & zero,
    int64_t coefficient,
    int64_t token_offset,
    int64_t original_tokens,
    bool scale_add_one) {
    if (token_offset >= original_tokens) {
        auto coeff = engine::modules::SliceModule({1, coefficient, 1}).build(ctx, zero);
        if (scale_add_one) {
            coeff = add_one(ctx, coeff);
        }
        return engine::modules::MulModule().build(
            ctx,
            values,
            engine::modules::RepeatModule({values.shape}).build(ctx, coeff));
    }
    if (token_offset + values.shape.dims[1] <= original_tokens) {
        return split_modulated_tokens(ctx, values, actual, zero, coefficient, values.shape.dims[1], scale_add_one);
    }
    return split_modulated_tokens(ctx, values, actual, zero, coefficient, original_tokens - token_offset, scale_add_one);
}

engine::core::TensorValue attention_heads(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    auto x = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
    return engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
}

engine::core::TensorValue merge_attention_heads(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    auto merged = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads * head_dim}));
    return merged;
}

engine::core::TensorValue apply_wan_rope(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cos,
    const engine::core::TensorValue & sin,
    int64_t head_dim) {
    if (input.shape.rank != 4 || head_dim <= 0 || head_dim % 2 != 0 ||
        input.shape.dims[3] != head_dim) {
        throw std::runtime_error("LiveAvatar RoPE input shape mismatch");
    }
    const int64_t half_dim = head_dim / 2;
    const int64_t pair_count = input.shape.dims[1] * input.shape.dims[2] * half_dim;
    const auto pairs = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], pair_count, 2}));
    const auto even = engine::modules::SliceModule({2, 0, 1}).build(ctx, pairs);
    const auto odd = engine::modules::SliceModule({2, 1, 1}).build(ctx, pairs);
    const auto cos_pairs = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, cos),
        engine::core::TensorShape::from_dims({cos.shape.dims[0], pair_count, 1}));
    const auto sin_pairs = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, sin),
        engine::core::TensorShape::from_dims({sin.shape.dims[0], pair_count, 1}));
    const auto rotated = engine::modules::RopeInterleavedPairsModule().build(ctx, even, odd, cos_pairs, sin_pairs);
    return engine::core::reshape_tensor(ctx, rotated, input.shape);
}

engine::core::TensorValue scaled_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & q_heads,
    const engine::core::TensorValue & k_heads,
    const engine::core::TensorValue & v_heads,
    int64_t head_dim,
    const std::optional<engine::core::TensorValue> & attention_mask) {
    if (!attention_mask.has_value() &&
        use_sage_attention &&
        ctx.backend_type == engine::core::BackendType::Cuda &&
        (head_dim == 64 || head_dim == 128)) {
        auto q_f16 = engine::core::wrap_tensor(
            ggml_cont_4d(
                ctx.ggml,
                ggml_cast(ctx.ggml, q_heads.tensor, GGML_TYPE_F16),
                head_dim,
                q_heads.shape.dims[2],
                q_heads.shape.dims[1],
                q_heads.shape.dims[0]),
            q_heads.shape,
            GGML_TYPE_F16);
        auto k_f16 = engine::core::wrap_tensor(
            ggml_cont_4d(
                ctx.ggml,
                ggml_cast(ctx.ggml, k_heads.tensor, GGML_TYPE_F16),
                head_dim,
                k_heads.shape.dims[2],
                k_heads.shape.dims[1],
                k_heads.shape.dims[0]),
            k_heads.shape,
            GGML_TYPE_F16);
        auto v_f16 = engine::core::wrap_tensor(
            ggml_cont_4d(
                ctx.ggml,
                ggml_cast(ctx.ggml, v_heads.tensor, GGML_TYPE_F16),
                head_dim,
                v_heads.shape.dims[2],
                v_heads.shape.dims[1],
                v_heads.shape.dims[0]),
            v_heads.shape,
            GGML_TYPE_F16);
        auto * sage = ggml_sage_attn2(
            ctx.ggml,
            q_f16.tensor,
            k_f16.tensor,
            v_f16.tensor,
            1.0F / std::sqrt(static_cast<float>(head_dim)),
            false);
        if (ggml_backend_supports_op(backend, sage)) {
            return merge_attention_heads(
                ctx,
                engine::core::wrap_tensor(
                    sage,
                    engine::core::TensorShape::from_dims({
                        q_heads.shape.dims[0],
                        q_heads.shape.dims[2],
                        q_heads.shape.dims[1],
                        q_heads.shape.dims[3],
                    }),
                    GGML_TYPE_F32),
                q_heads.shape.dims[1],
                head_dim);
        }
    }
    return merge_attention_heads(
        ctx,
        engine::modules::ScaledDotProductAttentionModule({
            head_dim,
            engine::modules::ScaledDotProductAttentionLowering::FlashPreserveViews,
            GGML_PREC_DEFAULT,
            engine::modules::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads, attention_mask),
        q_heads.shape.dims[1],
        head_dim);
}

engine::core::TensorValue build_self_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & rope_cos,
    const engine::core::TensorValue & rope_sin,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config) {
    const int64_t hidden = config.hidden_size;
    const int64_t heads = config.num_heads;
    const int64_t head_dim = hidden / heads;
    auto q = linear_native(ctx, input, weights.q, hidden, hidden);
    auto k = linear_native(ctx, input, weights.k, hidden, hidden);
    auto v = linear_native(ctx, input, weights.v, hidden, hidden);
    q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
            .build(ctx, q, {weights.norm_q, std::nullopt});
    k = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
            .build(ctx, k, {weights.norm_k, std::nullopt});
    q = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, q),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
    k = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, k),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4})
                       .build(ctx, apply_wan_rope(ctx, q, rope_cos, rope_sin, head_dim));
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4})
                       .build(ctx, apply_wan_rope(ctx, k, rope_cos, rope_sin, head_dim));
    auto v_heads = attention_heads(ctx, v, heads, head_dim);
    auto attended = scaled_attention(ctx, backend, use_sage_attention, q_heads, k_heads, v_heads, head_dim);
    return linear_native(ctx, attended, weights.o, hidden, hidden);
}

engine::core::TensorValue build_cross_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & context,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config) {
    const int64_t hidden = config.hidden_size;
    const int64_t heads = config.num_heads;
    const int64_t head_dim = hidden / heads;
    auto q = linear_native(ctx, input, weights.q, hidden, hidden);
    auto k = linear_native(ctx, context, weights.k, hidden, hidden);
    auto v = linear_native(ctx, context, weights.v, hidden, hidden);
    q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
            .build(ctx, q, {weights.norm_q, std::nullopt});
    k = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
            .build(ctx, k, {weights.norm_k, std::nullopt});
    auto attended = scaled_attention(
        ctx,
        backend,
        use_sage_attention,
        attention_heads(ctx, q, heads, head_dim),
        attention_heads(ctx, k, heads, head_dim),
        attention_heads(ctx, v, heads, head_dim),
        head_dim);
    return linear_native(ctx, attended, weights.o, hidden, hidden);
}

LiveAvatarAttentionKV build_cross_attention_kv(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & context,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config) {
    const int64_t hidden = config.hidden_size;
    const int64_t heads = config.num_heads;
    const int64_t head_dim = hidden / heads;
    auto key = linear_native(ctx, context, weights.k, hidden, hidden);
    key = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
              .build(ctx, key, {weights.norm_k, std::nullopt});
    auto value = linear_native(ctx, context, weights.v, hidden, hidden);
    return {
        attention_heads(ctx, key, heads, head_dim),
        attention_heads(ctx, value, heads, head_dim),
    };
}

engine::core::TensorValue build_cross_attention_with_cached_kv(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & kv,
    const LiveAvatarDenoiserAttentionWeights & weights,
    const LiveAvatarConfig & config) {
    const int64_t hidden = config.hidden_size;
    const int64_t heads = config.num_heads;
    const int64_t head_dim = hidden / heads;
    auto q = linear_native(ctx, input, weights.q, hidden, hidden);
    q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
            .build(ctx, q, {weights.norm_q, std::nullopt});
    if (use_sage_attention &&
        ctx.backend_type == engine::core::BackendType::Cuda &&
        (head_dim == 64 || head_dim == 128) &&
        kv.key.type == GGML_TYPE_F16 &&
        kv.value.type == GGML_TYPE_F16) {
        auto q_heads = attention_heads(ctx, q, heads, head_dim);
        auto q_f16 = engine::core::wrap_tensor(
            ggml_cont_4d(
                ctx.ggml,
                ggml_cast(ctx.ggml, q_heads.tensor, GGML_TYPE_F16),
                head_dim,
                q_heads.shape.dims[2],
                q_heads.shape.dims[1],
                q_heads.shape.dims[0]),
            q_heads.shape,
            GGML_TYPE_F16);
        auto * sage = ggml_sage_attn2(
            ctx.ggml,
            q_f16.tensor,
            kv.key.tensor,
            kv.value.tensor,
            1.0F / std::sqrt(static_cast<float>(head_dim)),
            false);
        if (ggml_backend_supports_op(backend, sage)) {
            return linear_native(
                ctx,
                merge_attention_heads(
                    ctx,
                    engine::core::wrap_tensor(
                        sage,
                        engine::core::TensorShape::from_dims({
                            q_heads.shape.dims[0],
                            q_heads.shape.dims[2],
                            q_heads.shape.dims[1],
                            q_heads.shape.dims[3],
                        }),
                        GGML_TYPE_F32),
                    heads,
                    head_dim),
                weights.o,
                hidden,
                hidden);
        }
    }
    auto attended = scaled_attention(
        ctx,
        backend,
        use_sage_attention,
        attention_heads(ctx, q, heads, head_dim),
        kv.key,
        kv.value,
        head_dim);
    return linear_native(ctx, attended, weights.o, hidden, hidden);
}

engine::core::TensorValue build_denoiser_block(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & context,
    const engine::core::TensorValue & actual_e0,
    const engine::core::TensorValue & zero_e0,
    const engine::core::TensorValue & rope_cos,
    const engine::core::TensorValue & rope_sin,
    const LiveAvatarDenoiserBlockWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens) {
    const int64_t hidden = config.hidden_size;
    auto block_actual = engine::modules::AddModule().build(
        ctx,
        actual_e0,
        engine::modules::RepeatModule({actual_e0.shape}).build(ctx, weights.modulation));
    auto block_zero = engine::modules::AddModule().build(
        ctx,
        zero_e0,
        engine::modules::RepeatModule({zero_e0.shape}).build(ctx, weights.modulation));
    auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                     .build(ctx, input, {});
    auto self_input = segmented_adaln_input(ctx, norm1, block_actual, block_zero, 0, 1, original_tokens);
    auto self_out = build_self_attention(ctx, backend, use_sage_attention, self_input, rope_cos, rope_sin, weights.self_attention, config);
    self_out = split_modulated_tokens(ctx, self_out, block_actual, block_zero, 2, original_tokens, false);
    auto x = engine::modules::AddModule().build(ctx, input, self_out);

    auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true})
                     .build(ctx, x, weights.norm3);
    x = engine::modules::AddModule().build(ctx, x, build_cross_attention(ctx, backend, use_sage_attention, norm3, context, weights.cross_attention, config));

    auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                     .build(ctx, x, {});
    auto ffn_input = segmented_adaln_input(ctx, norm2, block_actual, block_zero, 3, 4, original_tokens);
    auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config.ffn_dim);
    ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
    ffn = linear_native(ctx, ffn, weights.ffn_out, config.ffn_dim, hidden);
    ffn = split_modulated_tokens(ctx, ffn, block_actual, block_zero, 5, original_tokens, false);
    return engine::modules::AddModule().build(ctx, x, ffn);
}

engine::core::TensorValue build_denoiser_block_with_cached_cross_attention(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & cross_kv,
    const engine::core::TensorValue & actual_e0,
    const engine::core::TensorValue & zero_e0,
    const engine::core::TensorValue & rope_cos,
    const engine::core::TensorValue & rope_sin,
    const LiveAvatarDenoiserBlockWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens) {
    const int64_t hidden = config.hidden_size;
    auto block_actual = engine::modules::AddModule().build(
        ctx,
        actual_e0,
        engine::modules::RepeatModule({actual_e0.shape}).build(ctx, weights.modulation));
    auto block_zero = engine::modules::AddModule().build(
        ctx,
        zero_e0,
        engine::modules::RepeatModule({zero_e0.shape}).build(ctx, weights.modulation));
    auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                     .build(ctx, input, {});
    auto self_input = segmented_adaln_input(ctx, norm1, block_actual, block_zero, 0, 1, original_tokens);
    auto self_out = build_self_attention(ctx, backend, use_sage_attention, self_input, rope_cos, rope_sin, weights.self_attention, config);
    self_out = split_modulated_tokens(ctx, self_out, block_actual, block_zero, 2, original_tokens, false);
    auto x = engine::modules::AddModule().build(ctx, input, self_out);

    auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true})
                     .build(ctx, x, weights.norm3);
    x = engine::modules::AddModule().build(ctx, x, build_cross_attention_with_cached_kv(ctx, backend, use_sage_attention, norm3, cross_kv, weights.cross_attention, config));

    auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                     .build(ctx, x, {});
    auto ffn_input = segmented_adaln_input(ctx, norm2, block_actual, block_zero, 3, 4, original_tokens);
    auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config.ffn_dim);
    ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
    ffn = linear_native(ctx, ffn, weights.ffn_out, config.ffn_dim, hidden);
    ffn = split_modulated_tokens(ctx, ffn, block_actual, block_zero, 5, original_tokens, false);
    return engine::modules::AddModule().build(ctx, x, ffn);
}

engine::core::TensorValue build_denoiser_block_with_cached_cross_attention_chunked_mlp(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & cross_kv,
    const engine::core::TensorValue & actual_e0,
    const engine::core::TensorValue & zero_e0,
    const engine::core::TensorValue & rope_cos,
    const engine::core::TensorValue & rope_sin,
    const LiveAvatarDenoiserBlockWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens,
    int64_t chunk_tokens) {
    const int64_t hidden = config.hidden_size;
    auto block_actual = engine::modules::AddModule().build(
        ctx,
        actual_e0,
        engine::modules::RepeatModule({actual_e0.shape}).build(ctx, weights.modulation));
    auto block_zero = engine::modules::AddModule().build(
        ctx,
        zero_e0,
        engine::modules::RepeatModule({zero_e0.shape}).build(ctx, weights.modulation));
    auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                     .build(ctx, input, {});
    auto self_input = segmented_adaln_input(ctx, norm1, block_actual, block_zero, 0, 1, original_tokens);
    auto self_out = build_self_attention(ctx, backend, use_sage_attention, self_input, rope_cos, rope_sin, weights.self_attention, config);
    self_out = split_modulated_tokens(ctx, self_out, block_actual, block_zero, 2, original_tokens, false);
    auto x = engine::modules::AddModule().build(ctx, input, self_out);

    auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true})
                     .build(ctx, x, weights.norm3);
    x = engine::modules::AddModule().build(ctx, x, build_cross_attention_with_cached_kv(ctx, backend, use_sage_attention, norm3, cross_kv, weights.cross_attention, config));

    const int64_t total_tokens = x.shape.dims[1];
    engine::core::TensorValue output;
    for (int64_t start = 0; start < total_tokens; start += chunk_tokens) {
        const int64_t rows = std::min<int64_t>(chunk_tokens, total_tokens - start);
        auto chunk = engine::modules::SliceModule({1, start, rows}).build(ctx, x);
        auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false})
                         .build(ctx, chunk, {});
        auto ffn_input = segmented_adaln_input_for_token_range(
            ctx, norm2, block_actual, block_zero, 3, 4, start, original_tokens);
        auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config.ffn_dim);
        ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
        ffn = linear_native(ctx, ffn, weights.ffn_out, config.ffn_dim, hidden);
        ffn = split_modulated_tokens_for_token_range(
            ctx, ffn, block_actual, block_zero, 5, start, original_tokens, false);
        auto chunk_out = engine::modules::AddModule().build(ctx, chunk, ffn);
        output = output.valid() ? engine::modules::ConcatModule({1, true}).build(ctx, output, chunk_out) : chunk_out;
    }
    return output;
}

struct LiveAvatarDenoiserPreludeOutput {
    std::vector<float> hidden;
    std::vector<float> actual_e0;
    std::vector<float> zero_e0;
    std::vector<float> actual_e;
};

std::vector<float> make_timestep_features(float timestep) {
    auto actual = sinusoidal_embedding_1d(256, timestep);
    auto zero = sinusoidal_embedding_1d(256, 0.0F);
    actual.insert(actual.end(), zero.begin(), zero.end());
    return actual;
}

engine::core::TensorValue build_condition_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & embedding,
    const engine::core::TensorValue & target_like,
    const engine::core::TensorValue & ref_like) {
    const int64_t hidden = target_like.shape.last_dim();
    auto target = engine::modules::SliceModule({0, 0, 1}).build(ctx, embedding);
    engine::core::TensorShape target_broadcast_shape;
    target_broadcast_shape.rank = target_like.shape.rank;
    target_broadcast_shape.dims.fill(1);
    target_broadcast_shape.dims[target_broadcast_shape.rank - 1] = hidden;
    target = engine::core::reshape_tensor(ctx, target, target_broadcast_shape);
    target = engine::modules::RepeatModule({target_like.shape}).build(ctx, target);
    auto ref = engine::modules::SliceModule({0, 1, 1}).build(ctx, embedding);
    engine::core::TensorShape ref_broadcast_shape;
    ref_broadcast_shape.rank = ref_like.shape.rank;
    ref_broadcast_shape.dims.fill(1);
    ref_broadcast_shape.dims[ref_broadcast_shape.rank - 1] = hidden;
    ref = engine::core::reshape_tensor(ctx, ref, ref_broadcast_shape);
    ref = engine::modules::RepeatModule({ref_like.shape}).build(ctx, ref);
    return engine::modules::ConcatModule({1, true}).build(ctx, target, ref);
}

std::vector<float> unpatchify_head_tokens(
    const std::vector<float> & tokens,
    int64_t frames,
    int64_t latent_height,
    int64_t latent_width) {
    const int64_t patch_h = 2;
    const int64_t patch_w = 2;
    const int64_t channels = 16;
    const int64_t grid_h = latent_height / patch_h;
    const int64_t grid_w = latent_width / patch_w;
    const int64_t token_count = frames * grid_h * grid_w;
    const int64_t token_dim = channels * patch_h * patch_w;
    if (latent_height % patch_h != 0 || latent_width % patch_w != 0 ||
        static_cast<int64_t>(tokens.size()) != token_count * token_dim) {
        throw std::runtime_error("LiveAvatar denoiser head token shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * frames * latent_height * latent_width));
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t h = 0; h < grid_h; ++h) {
            for (int64_t w = 0; w < grid_w; ++w) {
                const int64_t token = (f * grid_h + h) * grid_w + w;
                for (int64_t ph = 0; ph < patch_h; ++ph) {
                    for (int64_t pw = 0; pw < patch_w; ++pw) {
                        for (int64_t c = 0; c < channels; ++c) {
                            const int64_t token_index = ((ph * patch_w + pw) * channels) + c;
                            const int64_t src = token * token_dim + token_index;
                            const int64_t dst = ((c * frames + f) * latent_height + h * patch_h + ph) * latent_width + w * patch_w + pw;
                            out[static_cast<size_t>(dst)] = tokens[static_cast<size_t>(src)];
                        }
                    }
                }
            }
        }
    }
    return out;
}

engine::core::TensorValue causal_conv1d_replicate(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    int64_t stride) {
    auto padded = input;
    for (int64_t i = 0; i < kernel_size - 1; ++i) {
        auto first = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
        padded = engine::modules::ConcatModule({2, true}).build(ctx, first, padded);
    }
    return engine::modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel_size,
        static_cast<int>(stride),
        0,
        1,
        true,
    }).build(ctx, padded, weights);
}

LiveAvatarEncodedAudio build_causal_audio_encoder(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & audio,
    const LiveAvatarDenoiserAudioEncoderWeights & weights,
    const LiveAvatarConfig & config,
    int64_t target_latent_frames,
    int64_t latent_start_frame) {
    const int64_t hidden = config.hidden_size;
    engine::core::TensorValue weighted_sum;
    engine::core::TensorValue weight_sum;
    for (int64_t layer = 0; layer < config.audio_layers; ++layer) {
        auto layer_value = engine::modules::SliceModule({1, layer, 1}).build(ctx, audio);
        layer_value = engine::core::reshape_tensor(
            ctx,
            layer_value,
            engine::core::TensorShape::from_dims({1, config.audio_dim, audio.shape.dims[3]}));
        auto weight = engine::modules::SliceModule({1, layer, 1}).build(ctx, weights.layer_weights);
        weight = engine::modules::SiluModule{}.build(ctx, weight);
        weight = engine::core::reshape_tensor(
            ctx,
            weight,
            engine::core::TensorShape::from_dims({1, 1, 1}));
        const auto scaled = engine::modules::MulModule().build(
            ctx,
            layer_value,
            engine::modules::RepeatModule({layer_value.shape}).build(ctx, weight));
        weighted_sum = weighted_sum.valid() ? engine::modules::AddModule().build(ctx, weighted_sum, scaled) : scaled;
        weight_sum = weight_sum.valid() ? engine::modules::AddModule().build(ctx, weight_sum, weight) : weight;
    }
    weight_sum = engine::core::reshape_tensor(
        ctx,
        weight_sum,
        engine::core::TensorShape::from_dims({1, 1, 1}));
    auto feature = div_tensor(
        ctx,
        weighted_sum,
        engine::modules::RepeatModule({weighted_sum.shape}).build(ctx, weight_sum));

    auto local = causal_conv1d_replicate(
        ctx,
        feature,
        weights.conv1_local,
        config.audio_dim,
        hidden / 4 * config.audio_tokens,
        3,
        1);
    local = engine::core::reshape_tensor(ctx, local, engine::core::TensorShape::from_dims({config.audio_tokens, hidden / 4, local.shape.dims[2]}));
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, local);
    local = engine::modules::LayerNormModule({hidden / 4, 1.0e-6F, false, false}).build(ctx, local, {});
    local = engine::modules::SiluModule{}.build(ctx, local);
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, local);
    local = causal_conv1d_replicate(ctx, local, weights.conv2, hidden / 4, hidden / 2, 3, 2);
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, local);
    local = engine::modules::LayerNormModule({hidden / 2, 1.0e-6F, false, false}).build(ctx, local, {});
    local = engine::modules::SiluModule{}.build(ctx, local);
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, local);
    local = causal_conv1d_replicate(ctx, local, weights.conv3, hidden / 2, hidden, 3, 2);
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, local);
    local = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, local, {});
    local = engine::modules::SiluModule{}.build(ctx, local);
    local = engine::core::reshape_tensor(ctx, local, engine::core::TensorShape::from_dims({1, config.audio_tokens, local.shape.dims[1], hidden}));
    local = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, local);
    const auto local_first_frame = engine::modules::SliceModule({2, 0, 1}).build(ctx, local);
    const auto padding = engine::modules::RepeatModule({local_first_frame.shape}).build(ctx, weights.padding_tokens);
    local = engine::modules::ConcatModule({2, true}).build(ctx, local, padding);
    local = engine::modules::SliceModule({1, latent_start_frame, target_latent_frames}).build(ctx, local);

    auto global = causal_conv1d_replicate(ctx, feature, weights.conv1_global, config.audio_dim, hidden / 4, 3, 1);
    global = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, global);
    global = engine::modules::LayerNormModule({hidden / 4, 1.0e-6F, false, false}).build(ctx, global, {});
    global = engine::modules::SiluModule{}.build(ctx, global);
    global = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, global);
    global = causal_conv1d_replicate(ctx, global, weights.conv2, hidden / 4, hidden / 2, 3, 2);
    global = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, global);
    global = engine::modules::LayerNormModule({hidden / 2, 1.0e-6F, false, false}).build(ctx, global, {});
    global = engine::modules::SiluModule{}.build(ctx, global);
    global = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, global);
    global = causal_conv1d_replicate(ctx, global, weights.conv3, hidden / 2, hidden, 3, 2);
    global = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, global);
    global = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, global, {});
    global = engine::modules::SiluModule{}.build(ctx, global);
    global = linear_native(ctx, global, weights.final_linear, hidden, hidden);
    global = engine::core::reshape_tensor(ctx, global, engine::core::TensorShape::from_dims({1, global.shape.dims[1], 1, hidden}));
    global = engine::modules::SliceModule({1, latent_start_frame, target_latent_frames}).build(ctx, global);
    return {local, global};
}

engine::core::TensorValue build_audio_injection(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarEncodedAudio & audio,
    const LiveAvatarDenoiserAudioInjectorWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens,
    int64_t target_latent_frames) {
    const int64_t hidden = config.hidden_size;
    const int64_t spatial_tokens = original_tokens / target_latent_frames;
    if (target_latent_frames <= 0 || original_tokens % target_latent_frames != 0) {
        throw std::runtime_error("LiveAvatar audio injection token shape mismatch");
    }
    auto target = engine::modules::SliceModule({1, 0, original_tokens}).build(ctx, input);
    const int64_t suffix_tokens = input.shape.dims[1] - original_tokens;
    engine::core::TensorValue suffix;
    if (suffix_tokens > 0) {
        suffix = engine::modules::SliceModule({1, original_tokens, suffix_tokens}).build(ctx, input);
    }
    auto attn_hidden = engine::core::reshape_tensor(ctx, target, engine::core::TensorShape::from_dims({target_latent_frames, spatial_tokens, hidden}));
    auto temb = engine::core::reshape_tensor(
        ctx,
        engine::modules::SliceModule({2, 0, 1}).build(ctx, audio.global),
        engine::core::TensorShape::from_dims({target_latent_frames, hidden}));
    auto scale_shift = linear_native(
        ctx,
        engine::modules::SiluModule{}.build(ctx, temb),
        weights.adain_linear,
        hidden,
        2 * hidden);
    auto shift = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(
            ctx,
            engine::modules::SliceModule({1, 0, hidden}).build(ctx, scale_shift)),
        engine::core::TensorShape::from_dims({target_latent_frames, 1, hidden}));
    auto scale = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(
            ctx,
            engine::modules::SliceModule({1, hidden, hidden}).build(ctx, scale_shift)),
        engine::core::TensorShape::from_dims({target_latent_frames, 1, hidden}));
    attn_hidden = engine::modules::LayerNormModule({hidden, 1.0e-5F, false, false}).build(ctx, attn_hidden, {});
    attn_hidden = apply_shift_scale(ctx, attn_hidden, shift, scale);
    auto audio_context = engine::core::reshape_tensor(
        ctx,
        audio.local,
        engine::core::TensorShape::from_dims({target_latent_frames, audio.local.shape.dims[2], hidden}));
    auto residual = build_cross_attention(ctx, backend, use_sage_attention, attn_hidden, audio_context, weights.attention, config);
    residual = engine::core::reshape_tensor(ctx, residual, engine::core::TensorShape::from_dims({1, original_tokens, hidden}));
    target = engine::modules::AddModule().build(ctx, target, residual);
    return suffix.valid() ? engine::modules::ConcatModule({1, true}).build(ctx, target, suffix) : target;
}

engine::core::TensorValue build_audio_injection_with_cached_condition(
    engine::core::ModuleBuildContext & ctx,
    ggml_backend_t backend,
    bool use_sage_attention,
    const engine::core::TensorValue & input,
    const LiveAvatarAttentionKV & audio_kv,
    const engine::core::TensorValue & shift,
    const engine::core::TensorValue & scale,
    const LiveAvatarDenoiserAudioInjectorWeights & weights,
    const LiveAvatarConfig & config,
    int64_t original_tokens,
    int64_t target_latent_frames) {
    const int64_t hidden = config.hidden_size;
    const int64_t spatial_tokens = original_tokens / target_latent_frames;
    if (target_latent_frames <= 0 || original_tokens % target_latent_frames != 0) {
        throw std::runtime_error("LiveAvatar audio injection token shape mismatch");
    }
    auto target = engine::modules::SliceModule({1, 0, original_tokens}).build(ctx, input);
    const int64_t suffix_tokens = input.shape.dims[1] - original_tokens;
    engine::core::TensorValue suffix;
    if (suffix_tokens > 0) {
        suffix = engine::modules::SliceModule({1, original_tokens, suffix_tokens}).build(ctx, input);
    }
    auto attn_hidden = engine::core::reshape_tensor(ctx, target, engine::core::TensorShape::from_dims({target_latent_frames, spatial_tokens, hidden}));
    attn_hidden = engine::modules::LayerNormModule({hidden, 1.0e-5F, false, false}).build(ctx, attn_hidden, {});
    attn_hidden = apply_shift_scale(ctx, attn_hidden, shift, scale);
    auto residual = build_cross_attention_with_cached_kv(ctx, backend, use_sage_attention, attn_hidden, audio_kv, weights.attention, config);
    residual = engine::core::reshape_tensor(ctx, residual, engine::core::TensorShape::from_dims({1, original_tokens, hidden}));
    target = engine::modules::AddModule().build(ctx, target, residual);
    return suffix.valid() ? engine::modules::ConcatModule({1, true}).build(ctx, target, suffix) : target;
}

class LiveAvatarDenoiserConditionGraph {
public:
    LiveAvatarDenoiserConditionGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        int64_t target_frames,
        int64_t audio_frames,
        int64_t latent_audio_start_frame)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          target_frames_(target_frames),
          audio_frames_(audio_frames),
          latent_audio_start_frame_(latent_audio_start_frame) {
        build();
    }

    ~LiveAvatarDenoiserConditionGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarDenoiserConditionGraph(const LiveAvatarDenoiserConditionGraph &) = delete;
    LiveAvatarDenoiserConditionGraph & operator=(const LiveAvatarDenoiserConditionGraph &) = delete;

    bool matches(int64_t target_frames, int64_t audio_frames, int64_t latent_audio_start_frame) const noexcept {
        return target_frames_ == target_frames &&
            audio_frames_ == audio_frames &&
            latent_audio_start_frame_ == latent_audio_start_frame;
    }

    LiveAvatarDenoiserPreparedCondition run(const LiveAvatarDenoiserConditionRunInput & input) const {
        const auto text_shape = engine::core::TensorShape::from_dims({1, config_.text_len, config_.text_dim});
        const auto audio_shape = engine::core::TensorShape::from_dims({1, config_.audio_layers, config_.audio_dim, audio_frames_});
        if (input.text_context == nullptr ||
            input.audio_input == nullptr ||
            static_cast<int64_t>(input.text_context->size()) != text_shape.num_elements() ||
            static_cast<int64_t>(input.audio_input->size()) != audio_shape.num_elements()) {
            throw std::runtime_error("LiveAvatar denoiser condition input payload size mismatch");
        }
        engine::core::write_tensor_f32(text_context_, *input.text_context);
        engine::core::write_tensor_f32(audio_, *input.audio_input);
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser_condition");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser condition graph compute failed");
        }
        return {
            engine::core::read_tensor_f32(context_output_),
            engine::core::read_tensor_f32(audio_local_output_),
            engine::core::read_tensor_f32(audio_global_output_),
        };
    }

private:
    void build() {
        if (target_frames_ <= 0 || audio_frames_ <= 0 || latent_audio_start_frame_ < 0) {
            throw std::runtime_error("LiveAvatar denoiser condition graph shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        ggml_init_params params{kDenoiserConditionGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser condition ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.denoiser_condition", execution_.backend_type()};
        text_context_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.text_len, config_.text_dim}));
        audio_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.audio_layers, config_.audio_dim, audio_frames_}));
        ggml_set_input(text_context_.tensor);
        ggml_set_input(audio_.tensor);

        auto context = linear_native(build_ctx, text_context_, weights_->text_embedding_0, config_.text_dim, hidden);
        context = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(build_ctx, context);
        context = linear_native(build_ctx, context, weights_->text_embedding_2, hidden, hidden);
        const auto encoded_audio =
            build_causal_audio_encoder(build_ctx, audio_, weights_->audio_encoder, config_, target_frames_, latent_audio_start_frame_);
        context = engine::core::ensure_backend_addressable_layout(build_ctx, context);
        const auto audio_local = engine::core::ensure_backend_addressable_layout(build_ctx, encoded_audio.local);
        const auto audio_global = engine::core::ensure_backend_addressable_layout(build_ctx, encoded_audio.global);
        context_output_ = context.tensor;
        audio_local_output_ = audio_local.tensor;
        audio_global_output_ = audio_global.tensor;
        ggml_set_output(context_output_);
        ggml_set_output(audio_local_output_);
        ggml_set_output(audio_global_output_);

        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, context_output_);
        ggml_build_forward_expand(graph_, audio_local_output_);
        ggml_build_forward_expand(graph_, audio_global_output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser condition backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    int64_t target_frames_ = 0;
    int64_t audio_frames_ = 0;
    int64_t latent_audio_start_frame_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue text_context_;
    engine::core::TensorValue audio_;
    ggml_tensor * context_output_ = nullptr;
    ggml_tensor * audio_local_output_ = nullptr;
    ggml_tensor * audio_global_output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarDenoiserStaticCache {
public:
    LiveAvatarDenoiserStaticCache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        int64_t lanes,
        bool use_sage_attention)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          lanes_(lanes),
          use_sage_attention_(use_sage_attention) {
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.denoiser_static_cache_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarDenoiserStaticCache() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
        }
    }

    LiveAvatarDenoiserStaticCache(const LiveAvatarDenoiserStaticCache &) = delete;
    LiveAvatarDenoiserStaticCache & operator=(const LiveAvatarDenoiserStaticCache &) = delete;

    bool matches(const engine::core::TensorShape & latent_shape, int64_t lanes, bool use_sage_attention) const noexcept {
        return same_shape(latent_shape_, latent_shape) && lanes_ == lanes && use_sage_attention_ == use_sage_attention;
    }

    void populate(const LiveAvatarDenoiserPreparedCondition & condition) {
        const int64_t target_frames = latent_shape_.dims[1];
        const size_t text_values = static_cast<size_t>(lanes_ * config_.text_len * config_.hidden_size);
        const size_t audio_local_values =
            static_cast<size_t>(lanes_ * target_frames * (config_.audio_tokens + 1) * config_.hidden_size);
        const size_t audio_global_values = static_cast<size_t>(lanes_ * target_frames * config_.hidden_size);
        if (condition.projected_text_context.size() != text_values ||
            condition.encoded_audio_local.size() != audio_local_values ||
            condition.encoded_audio_global.size() != audio_global_values) {
            throw std::runtime_error("LiveAvatar denoiser static cache input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(context_, condition.projected_text_context);
        engine::core::write_tensor_f32(audio_local_, condition.encoded_audio_local);
        engine::core::write_tensor_f32(audio_global_, condition.encoded_audio_global);
        engine::debug::timing_log_scalar("liveavatar.denoiser_static_cache_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser_static_cache");
        engine::debug::timing_log_scalar("liveavatar.denoiser_static_cache_compute_ms", engine::debug::elapsed_ms(compute_start));
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser static cache graph compute failed");
        }
        release_populate_graph();
    }

    LiveAvatarAttentionKV text_kv(engine::core::ModuleBuildContext & ctx, int64_t layer, int64_t lane) const {
        const auto index = static_cast<size_t>(layer);
        return {slice_text_lane(ctx, text_keys_.at(index), lane), slice_text_lane(ctx, text_values_.at(index), lane)};
    }

    LiveAvatarAttentionKV audio_kv(int64_t injector, int64_t lane) const {
        const auto index = audio_index(injector, lane);
        return {audio_keys_.at(index), audio_values_.at(index)};
    }

    engine::core::TensorValue audio_shift(int64_t injector, int64_t lane) const {
        return audio_shifts_.at(audio_index(injector, lane));
    }

    engine::core::TensorValue audio_scale(int64_t injector, int64_t lane) const {
        return audio_scales_.at(audio_index(injector, lane));
    }

private:
    static engine::core::TensorValue contiguous(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & value) {
        return engine::core::ensure_backend_addressable_layout(ctx, value);
    }

    engine::core::TensorValue slice_text_lane(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & value,
        int64_t lane) const {
        if (lane < 0 || lane >= lanes_) {
            throw std::runtime_error("LiveAvatar denoiser static cache lane is invalid");
        }
        return engine::modules::SliceModule({0, lane, 1}).build(ctx, value);
    }

    size_t audio_index(int64_t injector, int64_t lane) const {
        if (injector < 0 || injector >= static_cast<int64_t>(weights_->audio_injectors.size()) ||
            lane < 0 || lane >= lanes_) {
            throw std::runtime_error("LiveAvatar denoiser static audio cache index is invalid");
        }
        return static_cast<size_t>(lane * static_cast<int64_t>(weights_->audio_injectors.size()) + injector);
    }

    void build() {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] <= 0 ||
            lanes_ <= 0) {
            throw std::runtime_error("LiveAvatar denoiser static cache shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        const int64_t target_frames = latent_shape_.dims[1];
        const size_t audio_injectors = weights_->audio_injectors.size();
        const ggml_type kv_cache_type =
            use_sage_attention_ &&
                    execution_.backend_type() == engine::core::BackendType::Cuda &&
                    (head_dim == 64 || head_dim == 128)
                ? GGML_TYPE_F16
                : GGML_TYPE_F32;

        ggml_init_params cache_params{
            ggml_tensor_overhead() * (weights_->blocks.size() * 2 + audio_injectors * static_cast<size_t>(lanes_) * 4 + 16),
            nullptr,
            true};
        cache_ctx_.reset(ggml_init(cache_params));
        if (cache_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser static cache context initialization failed");
        }
        engine::core::ModuleBuildContext cache_ctx{cache_ctx_.get(), "liveavatar.denoiser_static_cache.data", execution_.backend_type()};
        auto make_text_cache = [&]() {
            return engine::core::make_tensor(
                cache_ctx,
                kv_cache_type,
                engine::core::TensorShape::from_dims({lanes_, heads, config_.text_len, head_dim}));
        };
        text_keys_.reserve(weights_->blocks.size());
        text_values_.reserve(weights_->blocks.size());
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            text_keys_.push_back(make_text_cache());
            text_values_.push_back(make_text_cache());
        }
        auto make_audio_cache = [&](int64_t tokens) {
            return engine::core::make_tensor(
                cache_ctx,
                kv_cache_type,
                engine::core::TensorShape::from_dims({target_frames, heads, tokens, head_dim}));
        };
        auto make_audio_modulation = [&]() {
            return engine::core::make_tensor(
                cache_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({target_frames, 1, hidden}));
        };
        audio_keys_.reserve(audio_injectors * static_cast<size_t>(lanes_));
        audio_values_.reserve(audio_injectors * static_cast<size_t>(lanes_));
        audio_shifts_.reserve(audio_injectors * static_cast<size_t>(lanes_));
        audio_scales_.reserve(audio_injectors * static_cast<size_t>(lanes_));
        for (int64_t lane = 0; lane < lanes_; ++lane) {
            for (size_t injector = 0; injector < audio_injectors; ++injector) {
                audio_keys_.push_back(make_audio_cache(config_.audio_tokens + 1));
                audio_values_.push_back(make_audio_cache(config_.audio_tokens + 1));
                audio_shifts_.push_back(make_audio_modulation());
                audio_scales_.push_back(make_audio_modulation());
            }
        }
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), execution_.backend());
        if (cache_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser static cache backend buffer allocation failed");
        }

        ggml_init_params graph_params{kDenoiserStaticCacheGraphContextBytes, nullptr, true};
        graph_ctx_.reset(ggml_init(graph_params));
        if (graph_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser static cache graph context initialization failed");
        }
        engine::core::ModuleBuildContext graph_ctx{graph_ctx_.get(), "liveavatar.denoiser_static_cache", execution_.backend_type()};
        context_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({lanes_, config_.text_len, hidden}));
        audio_local_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({lanes_, target_frames, config_.audio_tokens + 1, hidden}));
        audio_global_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({lanes_, target_frames, 1, hidden}));
        ggml_set_input(context_.tensor);
        ggml_set_input(audio_local_.tensor);
        ggml_set_input(audio_global_.tensor);

        graph_ = ggml_new_graph_custom(graph_ctx_.get(), 1048576, false);
        for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
            const auto kv = build_cross_attention_kv(
                graph_ctx,
                context_,
                weights_->blocks[layer].cross_attention,
                config_);
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.key).tensor, text_keys_[layer].tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.value).tensor, text_values_[layer].tensor));
        }
        for (int64_t lane = 0; lane < lanes_; ++lane) {
            auto local_lane = engine::modules::SliceModule({0, lane, 1}).build(graph_ctx, audio_local_);
            local_lane = engine::core::reshape_tensor(
                graph_ctx,
                local_lane,
                engine::core::TensorShape::from_dims({target_frames, config_.audio_tokens + 1, hidden}));
            auto global_lane = engine::modules::SliceModule({0, lane, 1}).build(graph_ctx, audio_global_);
            auto temb = engine::core::reshape_tensor(
                graph_ctx,
                engine::modules::SliceModule({2, 0, 1}).build(graph_ctx, global_lane),
                engine::core::TensorShape::from_dims({target_frames, hidden}));
            for (size_t injector = 0; injector < audio_injectors; ++injector) {
                const auto & weights = weights_->audio_injectors[injector];
                auto scale_shift = linear_native(
                    graph_ctx,
                    engine::modules::SiluModule{}.build(graph_ctx, temb),
                    weights.adain_linear,
                    hidden,
                    2 * hidden);
                auto shift = engine::core::reshape_tensor(
                    graph_ctx,
                    contiguous(graph_ctx, engine::modules::SliceModule({1, 0, hidden}).build(graph_ctx, scale_shift)),
                    engine::core::TensorShape::from_dims({target_frames, 1, hidden}));
                auto scale = engine::core::reshape_tensor(
                    graph_ctx,
                    contiguous(graph_ctx, engine::modules::SliceModule({1, hidden, hidden}).build(graph_ctx, scale_shift)),
                    engine::core::TensorShape::from_dims({target_frames, 1, hidden}));
                const auto kv = build_cross_attention_kv(graph_ctx, local_lane, weights.attention, config_);
                const auto index = audio_index(static_cast<int64_t>(injector), lane);
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.key).tensor, audio_keys_[index].tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.value).tensor, audio_values_[index].tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, shift).tensor, audio_shifts_[index].tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, scale).tensor, audio_scales_[index].tensor));
            }
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser static cache graph allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    void release_populate_graph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        graph_ctx_.reset();
        context_ = {};
        audio_local_ = {};
        audio_global_ = {};
        engine::core::trim_backend_pools(execution_.backend());
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    int64_t lanes_ = 1;
    bool use_sage_attention_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::vector<engine::core::TensorValue> text_keys_;
    std::vector<engine::core::TensorValue> text_values_;
    std::vector<engine::core::TensorValue> audio_keys_;
    std::vector<engine::core::TensorValue> audio_values_;
    std::vector<engine::core::TensorValue> audio_shifts_;
    std::vector<engine::core::TensorValue> audio_scales_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    engine::core::TensorValue context_;
    engine::core::TensorValue audio_local_;
    engine::core::TensorValue audio_global_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarDenoiserRopeCache {
public:
    LiveAvatarDenoiserRopeCache(
        engine::core::ExecutionContext & execution,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape)
        : execution_(execution),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)) {
        build();
    }

    ~LiveAvatarDenoiserRopeCache() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarDenoiserRopeCache(const LiveAvatarDenoiserRopeCache &) = delete;
    LiveAvatarDenoiserRopeCache & operator=(const LiveAvatarDenoiserRopeCache &) = delete;

    const engine::core::TensorValue & cos() const noexcept { return rope_cos_; }
    const engine::core::TensorValue & sin() const noexcept { return rope_sin_; }

private:
    void build() {
        const int64_t hidden = config_.hidden_size;
        const int64_t head_dim = hidden / config_.num_heads;
        const int64_t target_frames = latent_shape_.dims[1];
        const int64_t grid_h = latent_shape_.dims[2] / 2;
        const int64_t grid_w = latent_shape_.dims[3] / 2;
        const int64_t original_tokens = target_frames * grid_h * grid_w;
        const int64_t ref_tokens = grid_h * grid_w;
        const int64_t total_tokens = original_tokens + ref_tokens;
        const std::vector<LiveAvatarRopeRange> ranges{
            {0, 0, 0, target_frames, grid_h, grid_w, target_frames, grid_h, grid_w},
            {30, 0, 0, 31, grid_h, grid_w, 1, grid_h, grid_w},
        };
        const auto rope_cos_values = make_liveavatar_rope_table(config_.num_heads, head_dim, ranges, false);
        const auto rope_sin_values = make_liveavatar_rope_table(config_.num_heads, head_dim, ranges, true);

        ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser layerwise RoPE context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.denoiser.layerwise.rope", execution_.backend_type()};
        rope_cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, total_tokens, config_.num_heads, head_dim / 2}));
        rope_sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, total_tokens, config_.num_heads, head_dim / 2}));
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser layerwise RoPE buffer allocation failed");
        }
        engine::core::write_tensor_f32(rope_cos_, rope_cos_values);
        engine::core::write_tensor_f32(rope_sin_, rope_sin_values);
    }

    engine::core::ExecutionContext & execution_;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue rope_cos_;
    engine::core::TensorValue rope_sin_;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class LiveAvatarDenoiserPreludeGraph {
public:
    LiveAvatarDenoiserPreludeGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)) {
        build();
    }

    ~LiveAvatarDenoiserPreludeGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarDenoiserPreludeGraph(const LiveAvatarDenoiserPreludeGraph &) = delete;
    LiveAvatarDenoiserPreludeGraph & operator=(const LiveAvatarDenoiserPreludeGraph &) = delete;

    LiveAvatarDenoiserPreludeOutput run(const LiveAvatarDenoiserRunInput & input) const {
        const auto ref_shape = engine::core::TensorShape::from_dims({
            config_.latent_channels,
            1,
            latent_shape_.dims[2],
            latent_shape_.dims[3],
        });
        if (input.latent == nullptr ||
            input.ref_latents == nullptr ||
            input.cond_latents == nullptr ||
            static_cast<int64_t>(input.latent->size()) != latent_shape_.num_elements() ||
            static_cast<int64_t>(input.ref_latents->size()) != ref_shape.num_elements() ||
            static_cast<int64_t>(input.cond_latents->size()) != latent_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar denoiser layerwise prelude input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(latent_, *input.latent);
        engine::core::write_tensor_f32(ref_latents_, *input.ref_latents);
        engine::core::write_tensor_f32(cond_latents_, *input.cond_latents);
        engine::core::write_tensor_f32(timestep_, make_timestep_features(input.timestep));
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_prelude_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser.layerwise.prelude");
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_prelude_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser layerwise prelude graph compute failed");
        }
        const auto read_start = Clock::now();
        LiveAvatarDenoiserPreludeOutput out;
        out.hidden = engine::core::read_tensor_f32(hidden_output_);
        out.actual_e0 = engine::core::read_tensor_f32(actual_e0_output_);
        out.zero_e0 = engine::core::read_tensor_f32(zero_e0_output_);
        out.actual_e = engine::core::read_tensor_f32(actual_e_output_);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_prelude_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        const int64_t hidden = config_.hidden_size;
        ggml_init_params params{kDenoiserLayerwiseGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser layerwise prelude context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.denoiser.layerwise.prelude", execution_.backend_type()};
        latent_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, latent_shape_);
        ref_latents_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({config_.latent_channels, 1, latent_shape_.dims[2], latent_shape_.dims[3]}));
        cond_latents_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, latent_shape_);
        timestep_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({2, 256}));
        ggml_set_input(latent_.tensor);
        ggml_set_input(ref_latents_.tensor);
        ggml_set_input(cond_latents_.tensor);
        ggml_set_input(timestep_.tensor);

        auto time_emb = linear_native(build_ctx, timestep_, weights_->time_embedding_0, 256, hidden);
        time_emb = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_emb = linear_native(build_ctx, time_emb, weights_->time_embedding_2, hidden, hidden);
        auto time_proj = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_proj = linear_native(build_ctx, time_proj, weights_->time_projection, hidden, 6 * hidden);
        time_proj = engine::core::reshape_tensor(build_ctx, time_proj, engine::core::TensorShape::from_dims({2, 6, hidden}));
        auto actual_e0 = engine::core::ensure_backend_addressable_layout(
            build_ctx,
            engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_proj));
        auto zero_e0 = engine::core::ensure_backend_addressable_layout(
            build_ctx,
            engine::modules::SliceModule({0, 1, 1}).build(build_ctx, time_proj));
        auto actual_e = engine::core::reshape_tensor(
            build_ctx,
            engine::core::ensure_backend_addressable_layout(
                build_ctx,
                engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_emb)),
            engine::core::TensorShape::from_dims({1, 1, hidden}));

        auto static_prefix = conv3d_tokens(build_ctx, latent_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        auto cond = conv3d_tokens(build_ctx, cond_latents_, weights_->cond_encoder, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        static_prefix = engine::modules::AddModule().build(build_ctx, static_prefix, cond);
        auto ref = conv3d_tokens(build_ctx, ref_latents_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        const auto mask_embedding = build_condition_mask(build_ctx, weights_->trainable_cond_mask, static_prefix, ref);
        static_prefix = engine::modules::ConcatModule({1, true}).build(build_ctx, static_prefix, ref);
        static_prefix = engine::modules::AddModule().build(build_ctx, static_prefix, mask_embedding);
        hidden_output_ = engine::core::ensure_backend_addressable_layout(build_ctx, static_prefix).tensor;
        actual_e0_output_ = actual_e0.tensor;
        zero_e0_output_ = zero_e0.tensor;
        actual_e_output_ = actual_e.tensor;
        for (auto * output : {hidden_output_, actual_e0_output_, zero_e0_output_, actual_e_output_}) {
            ggml_set_output(output);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (auto * output : {hidden_output_, actual_e0_output_, zero_e0_output_, actual_e_output_}) {
            ggml_build_forward_expand(graph_, output);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser layerwise prelude backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue latent_;
    engine::core::TensorValue ref_latents_;
    engine::core::TensorValue cond_latents_;
    engine::core::TensorValue timestep_;
    ggml_tensor * hidden_output_ = nullptr;
    ggml_tensor * actual_e0_output_ = nullptr;
    ggml_tensor * zero_e0_output_ = nullptr;
    ggml_tensor * actual_e_output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarDenoiserLayerGraph {
public:
    LiveAvatarDenoiserLayerGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        const LiveAvatarDenoiserRopeCache & rope_cache,
        const LiveAvatarDenoiserStaticCache & static_cache,
        int64_t layer_start,
        int64_t layer_end,
        int64_t lane,
        bool use_sage_attention)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          rope_cache_(&rope_cache),
          static_cache_(&static_cache),
          layer_start_(layer_start),
          layer_end_(layer_end),
          lane_(lane),
          use_sage_attention_(use_sage_attention) {
        build();
    }

    ~LiveAvatarDenoiserLayerGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    LiveAvatarDenoiserLayerGraph(const LiveAvatarDenoiserLayerGraph &) = delete;
    LiveAvatarDenoiserLayerGraph & operator=(const LiveAvatarDenoiserLayerGraph &) = delete;

    std::vector<float> run(
        const std::vector<float> & hidden,
        const std::vector<float> & actual_e0,
        const std::vector<float> & zero_e0) const {
        const int64_t total_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2) +
                                     (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);
        const size_t hidden_values = static_cast<size_t>(total_tokens * config_.hidden_size);
        const size_t modulation_values = static_cast<size_t>(6 * config_.hidden_size);
        if (hidden.size() != hidden_values || actual_e0.size() != modulation_values || zero_e0.size() != modulation_values) {
            throw std::runtime_error("LiveAvatar denoiser layerwise block input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(hidden_, hidden);
        engine::core::write_tensor_f32(actual_e0_, actual_e0);
        engine::core::write_tensor_f32(zero_e0_, zero_e0);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_block_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser.layerwise.block");
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_block_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser layerwise block graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = engine::core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_block_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        if (layer_start_ < 0 || layer_end_ <= layer_start_ || layer_end_ > config_.num_layers || lane_ < 0 || lane_ > 1) {
            throw std::runtime_error("LiveAvatar denoiser layerwise block index is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t target_frames = latent_shape_.dims[1];
        const int64_t grid_h = latent_shape_.dims[2] / 2;
        const int64_t grid_w = latent_shape_.dims[3] / 2;
        const int64_t original_tokens = target_frames * grid_h * grid_w;
        const int64_t total_tokens = original_tokens + grid_h * grid_w;
        input_ctx_.reset(ggml_init({kDenoiserLayerwiseInputContextBytes, nullptr, true}));
        ctx_.reset(ggml_init({kDenoiserLayerwiseGraphContextBytes, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser layerwise block context initialization failed");
        }
        engine::core::ModuleBuildContext input_ctx{input_ctx_.get(), "liveavatar.denoiser.layerwise.block.inputs", execution_.backend_type()};
        hidden_ = engine::core::make_tensor(input_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, total_tokens, hidden}));
        actual_e0_ = engine::core::make_tensor(input_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 6, hidden}));
        zero_e0_ = engine::core::make_tensor(input_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 6, hidden}));
        ggml_set_input(hidden_.tensor);
        ggml_set_input(actual_e0_.tensor);
        ggml_set_input(zero_e0_.tensor);

        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.denoiser.layerwise.block", execution_.backend_type()};
        static constexpr std::array<int64_t, 12> kAudioInjectLayers{0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39};
        auto output = hidden_;
        for (int64_t layer = layer_start_; layer < layer_end_; ++layer) {
            output = build_denoiser_block_with_cached_cross_attention_chunked_mlp(
                build_ctx,
                execution_.backend(),
                use_sage_attention_,
                output,
                static_cache_->text_kv(build_ctx, layer, lane_),
                actual_e0_,
                zero_e0_,
                rope_cache_->cos(),
                rope_cache_->sin(),
                weights_->blocks.at(static_cast<size_t>(layer)),
                config_,
                original_tokens,
                4096);
            const auto audio_it = std::find(kAudioInjectLayers.begin(), kAudioInjectLayers.end(), layer);
            if (audio_it != kAudioInjectLayers.end()) {
                const int64_t audio_injector = static_cast<int64_t>(std::distance(kAudioInjectLayers.begin(), audio_it));
                output = build_audio_injection_with_cached_condition(
                    build_ctx,
                    execution_.backend(),
                    use_sage_attention_,
                    output,
                    static_cache_->audio_kv(audio_injector, lane_),
                    static_cache_->audio_shift(audio_injector, lane_),
                    static_cache_->audio_scale(audio_injector, lane_),
                    weights_->audio_injectors.at(static_cast<size_t>(audio_injector)),
                    config_,
                    original_tokens,
                    target_frames);
            }
        }
        output_ = engine::core::ensure_backend_addressable_layout(build_ctx, output).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser layerwise block backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    const LiveAvatarDenoiserRopeCache * rope_cache_ = nullptr;
    const LiveAvatarDenoiserStaticCache * static_cache_ = nullptr;
    int64_t layer_start_ = 0;
    int64_t layer_end_ = 0;
    int64_t lane_ = 0;
    bool use_sage_attention_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    engine::core::TensorValue hidden_;
    engine::core::TensorValue actual_e0_;
    engine::core::TensorValue zero_e0_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarDenoiserFinalGraph {
public:
    LiveAvatarDenoiserFinalGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)) {
        build();
    }

    ~LiveAvatarDenoiserFinalGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    LiveAvatarDenoiserFinalGraph(const LiveAvatarDenoiserFinalGraph &) = delete;
    LiveAvatarDenoiserFinalGraph & operator=(const LiveAvatarDenoiserFinalGraph &) = delete;

    std::vector<float> run(const std::vector<float> & hidden, const std::vector<float> & actual_e) const {
        const int64_t total_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2) +
                                     (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);
        const size_t hidden_values = static_cast<size_t>(total_tokens * config_.hidden_size);
        if (hidden.size() != hidden_values || actual_e.size() != static_cast<size_t>(config_.hidden_size)) {
            throw std::runtime_error("LiveAvatar denoiser layerwise final input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(hidden_, hidden);
        engine::core::write_tensor_f32(actual_e_, actual_e);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_final_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser.layerwise.final");
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_final_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser layerwise final graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = engine::core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_final_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        const int64_t hidden = config_.hidden_size;
        const int64_t target_frames = latent_shape_.dims[1];
        const int64_t grid_h = latent_shape_.dims[2] / 2;
        const int64_t grid_w = latent_shape_.dims[3] / 2;
        const int64_t original_tokens = target_frames * grid_h * grid_w;
        const int64_t total_tokens = original_tokens + grid_h * grid_w;
        input_ctx_.reset(ggml_init({kDenoiserLayerwiseInputContextBytes, nullptr, true}));
        ctx_.reset(ggml_init({kDenoiserLayerwiseGraphContextBytes, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser layerwise final context initialization failed");
        }
        engine::core::ModuleBuildContext input_ctx{input_ctx_.get(), "liveavatar.denoiser.layerwise.final.inputs", execution_.backend_type()};
        hidden_ = engine::core::make_tensor(input_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, total_tokens, hidden}));
        actual_e_ = engine::core::make_tensor(input_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, hidden}));
        ggml_set_input(hidden_.tensor);
        ggml_set_input(actual_e_.tensor);

        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.denoiser.layerwise.final", execution_.backend_type()};
        auto path_x = engine::modules::SliceModule({1, 0, original_tokens}).build(build_ctx, hidden_);
        path_x = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(build_ctx, path_x, {});
        auto head_e = engine::modules::AddModule().build(
            build_ctx,
            weights_->head.modulation,
            engine::modules::RepeatModule({weights_->head.modulation.shape}).build(build_ctx, actual_e_));
        const auto head_shift = engine::modules::SliceModule({1, 0, 1}).build(build_ctx, head_e);
        const auto head_scale = engine::modules::SliceModule({1, 1, 1}).build(build_ctx, head_e);
        path_x = apply_shift_scale(build_ctx, path_x, head_shift, head_scale);
        output_ = linear_native(build_ctx, path_x, weights_->head.projection, hidden, config_.latent_channels * 4).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser layerwise final backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    engine::core::TensorValue hidden_;
    engine::core::TensorValue actual_e_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarDenoiserGraph {
public:
    LiveAvatarDenoiserGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        bool cfg_enabled,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache * static_cache)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(latent_shape),
          cfg_enabled_(cfg_enabled),
          use_sage_attention_(use_sage_attention),
          static_cache_(static_cache) {
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.denoiser_graph_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarDenoiserGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (const_buffer_ != nullptr) {
            ggml_backend_buffer_free(const_buffer_);
        }
    }

    LiveAvatarDenoiserGraph(const LiveAvatarDenoiserGraph &) = delete;
    LiveAvatarDenoiserGraph & operator=(const LiveAvatarDenoiserGraph &) = delete;

    bool matches(
        const engine::core::TensorShape & latent_shape,
        bool cfg_enabled,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache * static_cache) const noexcept {
        return same_shape(latent_shape_, latent_shape) &&
               cfg_enabled_ == cfg_enabled &&
               use_sage_attention_ == use_sage_attention &&
               static_cache_ == static_cache;
    }

    std::vector<float> run(const LiveAvatarDenoiserRunInput & input) const {
        const auto ref_shape = engine::core::TensorShape::from_dims({
            config_.latent_channels,
            1,
            latent_shape_.dims[2],
            latent_shape_.dims[3],
        });
        const auto cond_shape = latent_shape_;
        const int64_t target_frames = latent_shape_.dims[1];
        const int64_t condition_batch = cfg_enabled_ ? 2 : 1;
        const auto context_shape = engine::core::TensorShape::from_dims({condition_batch, config_.text_len, config_.hidden_size});
        const auto audio_local_shape =
            engine::core::TensorShape::from_dims({condition_batch, target_frames, config_.audio_tokens + 1, config_.hidden_size});
        const auto audio_global_shape =
            engine::core::TensorShape::from_dims({condition_batch, target_frames, 1, config_.hidden_size});
        if (cfg_enabled_ && static_cache_ == nullptr) {
            throw std::runtime_error("LiveAvatar fused denoiser requires a static cache");
        }
        if (input.latent == nullptr ||
             input.ref_latents == nullptr ||
             input.cond_latents == nullptr ||
             static_cast<int64_t>(input.latent->size()) != latent_shape_.num_elements() ||
             static_cast<int64_t>(input.ref_latents->size()) != ref_shape.num_elements() ||
             static_cast<int64_t>(input.cond_latents->size()) != cond_shape.num_elements()) {
            throw std::runtime_error("LiveAvatar denoiser input payload size mismatch");
        }
        if (!cfg_enabled_ &&
            (input.projected_text_context == nullptr ||
             input.encoded_audio_local == nullptr ||
             input.encoded_audio_global == nullptr ||
             static_cast<int64_t>(input.projected_text_context->size()) != context_shape.num_elements() ||
             static_cast<int64_t>(input.encoded_audio_local->size()) != audio_local_shape.num_elements() ||
             static_cast<int64_t>(input.encoded_audio_global->size()) != audio_global_shape.num_elements())) {
            throw std::runtime_error("LiveAvatar denoiser input payload size mismatch");
        }
        engine::debug::trace_log_f32("liveavatar.dit.input.latent", {latent_shape_.dims[0], latent_shape_.dims[1], latent_shape_.dims[2], latent_shape_.dims[3]}, *input.latent);
        engine::debug::trace_log_f32("liveavatar.dit.input.reference_latent", {ref_shape.dims[0], ref_shape.dims[1], ref_shape.dims[2], ref_shape.dims[3]}, *input.ref_latents);
        engine::debug::trace_log_f32("liveavatar.dit.input.control_video", {cond_shape.dims[0], cond_shape.dims[1], cond_shape.dims[2], cond_shape.dims[3]}, *input.cond_latents);
        if (!cfg_enabled_) {
            engine::debug::trace_log_f32("liveavatar.dit.input.text_context", {context_shape.dims[0], context_shape.dims[1], context_shape.dims[2]}, *input.projected_text_context);
            engine::debug::trace_log_f32("liveavatar.dit.input.audio_local", {audio_local_shape.dims[0], audio_local_shape.dims[1], audio_local_shape.dims[2], audio_local_shape.dims[3]}, *input.encoded_audio_local);
            engine::debug::trace_log_f32("liveavatar.dit.input.audio_global", {audio_global_shape.dims[0], audio_global_shape.dims[1], audio_global_shape.dims[2], audio_global_shape.dims[3]}, *input.encoded_audio_global);
        }
        engine::debug::trace_log_scalar("liveavatar.dit.input.timestep", input.timestep);
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(latent_, *input.latent);
        engine::core::write_tensor_f32(ref_latents_, *input.ref_latents);
        engine::core::write_tensor_f32(cond_latents_, *input.cond_latents);
        const size_t context_values = static_cast<size_t>(config_.text_len * config_.hidden_size);
        const size_t audio_local_values =
            static_cast<size_t>(target_frames * (config_.audio_tokens + 1) * config_.hidden_size);
        const size_t audio_global_values = static_cast<size_t>(target_frames * config_.hidden_size);
        if (!cfg_enabled_) {
            engine::core::write_tensor_f32(context_, input.projected_text_context->data(), context_values);
            engine::core::write_tensor_f32(audio_local_, input.encoded_audio_local->data(), audio_local_values);
            engine::core::write_tensor_f32(audio_global_, input.encoded_audio_global->data(), audio_global_values);
        } else {
            engine::core::write_tensor_f32(guidance_scale_, &input.guidance_scale, 1);
        }
        engine::core::write_tensor_f32(timestep_, make_timestep_features(input.timestep));
        engine::debug::timing_log_scalar("liveavatar.denoiser_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.denoiser");
        engine::debug::timing_log_scalar("liveavatar.denoiser_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar denoiser graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = unpatchify_head_tokens(
            engine::core::read_tensor_f32(output_),
            latent_shape_.dims[1],
            latent_shape_.dims[2],
            latent_shape_.dims[3]);
        engine::debug::timing_log_scalar("liveavatar.denoiser_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] <= 0 ||
            latent_shape_.dims[2] <= 0 ||
            latent_shape_.dims[3] <= 0 ||
            latent_shape_.dims[2] % 2 != 0 ||
            latent_shape_.dims[3] % 2 != 0) {
            throw std::runtime_error("LiveAvatar denoiser graph shape is invalid");
        }
        if (cfg_enabled_ && static_cache_ == nullptr) {
            throw std::runtime_error("LiveAvatar fused denoiser graph requires a static cache");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t head_dim = hidden / config_.num_heads;
        const int64_t target_frames = latent_shape_.dims[1];
        const int64_t grid_h = latent_shape_.dims[2] / 2;
        const int64_t grid_w = latent_shape_.dims[3] / 2;
        const int64_t original_tokens = target_frames * grid_h * grid_w;
        const int64_t ref_tokens = grid_h * grid_w;
        const int64_t total_tokens = original_tokens + ref_tokens;
        const auto ranges = denoiser_rope_ranges(target_frames, latent_shape_.dims[2], latent_shape_.dims[3]);
        rope_cos_values_ = make_liveavatar_rope_table(config_.num_heads, head_dim, ranges, false);
        rope_sin_values_ = make_liveavatar_rope_table(config_.num_heads, head_dim, ranges, true);

        ggml_init_params const_params{ggml_tensor_overhead() * 6, nullptr, true};
        const_ctx_.reset(ggml_init(const_params));
        if (const_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser constant context initialization failed");
        }
        engine::core::ModuleBuildContext const_ctx{
            const_ctx_.get(), "liveavatar.denoiser.const", execution_.backend_type()};
        rope_cos_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, total_tokens, config_.num_heads, head_dim / 2}));
        rope_sin_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, total_tokens, config_.num_heads, head_dim / 2}));
        const_buffer_ = ggml_backend_alloc_ctx_tensors(const_ctx_.get(), execution_.backend());
        if (const_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser constant buffer allocation failed");
        }
        engine::core::write_tensor_f32(rope_cos_, rope_cos_values_);
        engine::core::write_tensor_f32(rope_sin_, rope_sin_values_);

        ggml_init_params params{kDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.denoiser", execution_.backend_type()};
        latent_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, latent_shape_);
        ref_latents_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({config_.latent_channels, 1, latent_shape_.dims[2], latent_shape_.dims[3]}));
        cond_latents_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, latent_shape_);
        if (cfg_enabled_) {
            guidance_scale_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1}));
        } else {
            context_ = engine::core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, config_.text_len, hidden}));
            audio_local_ = engine::core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, target_frames, config_.audio_tokens + 1, hidden}));
            audio_global_ = engine::core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, target_frames, 1, hidden}));
        }
        timestep_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({2, 256}));
        ggml_set_input(latent_.tensor);
        ggml_set_input(ref_latents_.tensor);
        ggml_set_input(cond_latents_.tensor);
        if (cfg_enabled_) {
            ggml_set_input(guidance_scale_.tensor);
        } else {
            ggml_set_input(context_.tensor);
            ggml_set_input(audio_local_.tensor);
            ggml_set_input(audio_global_.tensor);
        }
        ggml_set_input(timestep_.tensor);

        auto time_emb = linear_native(build_ctx, timestep_, weights_->time_embedding_0, 256, hidden);
        time_emb = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_emb = linear_native(build_ctx, time_emb, weights_->time_embedding_2, hidden, hidden);
        auto time_proj = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_proj = linear_native(build_ctx, time_proj, weights_->time_projection, hidden, 6 * hidden);
        time_proj = engine::core::reshape_tensor(build_ctx, time_proj, engine::core::TensorShape::from_dims({2, 6, hidden}));
        const auto actual_e0 = engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_proj);
        const auto zero_e0 = engine::modules::SliceModule({0, 1, 1}).build(build_ctx, time_proj);
        const auto actual_e = engine::core::reshape_tensor(
            build_ctx,
            engine::core::ensure_backend_addressable_layout(
                build_ctx,
                engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_emb)),
            engine::core::TensorShape::from_dims({1, 1, hidden}));

        auto static_prefix = conv3d_tokens(build_ctx, latent_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        auto cond = conv3d_tokens(build_ctx, cond_latents_, weights_->cond_encoder, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        static_prefix = engine::modules::AddModule().build(build_ctx, static_prefix, cond);
        auto ref = conv3d_tokens(build_ctx, ref_latents_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        const auto mask_embedding = build_condition_mask(build_ctx, weights_->trainable_cond_mask, static_prefix, ref);
        static_prefix = engine::modules::ConcatModule({1, true}).build(build_ctx, static_prefix, ref);
        static_prefix = engine::modules::AddModule().build(build_ctx, static_prefix, mask_embedding);

        const auto build_path = [&](const engine::core::TensorValue * path_context,
                                    const LiveAvatarEncodedAudio * encoded_audio,
                                    int64_t lane) {
            auto path_x = static_prefix;
            constexpr std::array<int64_t, 12> kAudioInjectLayers{0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39};
            int64_t audio_injector_index = 0;
            for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
                if (cfg_enabled_) {
                    path_x = build_denoiser_block_with_cached_cross_attention(
                        build_ctx,
                        execution_.backend(),
                        use_sage_attention_,
                        path_x,
                        static_cache_->text_kv(build_ctx, layer, lane),
                        actual_e0,
                        zero_e0,
                        rope_cos_,
                        rope_sin_,
                        weights_->blocks.at(static_cast<size_t>(layer)),
                        config_,
                        original_tokens);
                } else {
                    path_x = build_denoiser_block(
                        build_ctx,
                        execution_.backend(),
                        use_sage_attention_,
                        path_x,
                        *path_context,
                        actual_e0,
                        zero_e0,
                        rope_cos_,
                        rope_sin_,
                        weights_->blocks.at(static_cast<size_t>(layer)),
                        config_,
                        original_tokens);
                }
                if (audio_injector_index < static_cast<int64_t>(kAudioInjectLayers.size()) &&
                    layer == kAudioInjectLayers[static_cast<size_t>(audio_injector_index)]) {
                    if (cfg_enabled_) {
                        path_x = build_audio_injection_with_cached_condition(
                            build_ctx,
                            execution_.backend(),
                            use_sage_attention_,
                            path_x,
                            static_cache_->audio_kv(audio_injector_index, lane),
                            static_cache_->audio_shift(audio_injector_index, lane),
                            static_cache_->audio_scale(audio_injector_index, lane),
                            weights_->audio_injectors.at(static_cast<size_t>(audio_injector_index)),
                            config_,
                            original_tokens,
                            target_frames);
                    } else {
                        path_x = build_audio_injection(
                            build_ctx,
                            execution_.backend(),
                            use_sage_attention_,
                            path_x,
                            *encoded_audio,
                            weights_->audio_injectors.at(static_cast<size_t>(audio_injector_index)),
                            config_,
                            original_tokens,
                            target_frames);
                    }
                    ++audio_injector_index;
                }
            }

            path_x = engine::modules::SliceModule({1, 0, original_tokens}).build(build_ctx, path_x);
            path_x = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(build_ctx, path_x, {});
            auto head_e = engine::modules::AddModule().build(
                build_ctx,
                weights_->head.modulation,
                engine::modules::RepeatModule({weights_->head.modulation.shape}).build(build_ctx, actual_e));
            const auto head_shift = engine::modules::SliceModule({1, 0, 1}).build(build_ctx, head_e);
            const auto head_scale = engine::modules::SliceModule({1, 1, 1}).build(build_ctx, head_e);
            path_x = apply_shift_scale(build_ctx, path_x, head_shift, head_scale);
            return linear_native(build_ctx, path_x, weights_->head.projection, hidden, config_.latent_channels * 4);
        };

        const LiveAvatarEncodedAudio encoded_audio{audio_local_, audio_global_};
        engine::core::TensorValue x;
        if (cfg_enabled_) {
            auto cond_output = build_path(nullptr, nullptr, 0);
            auto uncond_output = build_path(nullptr, nullptr, 1);
            auto diff = engine::core::wrap_tensor(
                ggml_sub(build_ctx.ggml, cond_output.tensor, uncond_output.tensor),
                cond_output.shape,
                GGML_TYPE_F32);
            auto scaled_diff = engine::core::wrap_tensor(
                ggml_mul(build_ctx.ggml, diff.tensor, guidance_scale_.tensor),
                diff.shape,
                GGML_TYPE_F32);
            x = engine::modules::AddModule().build(build_ctx, uncond_output, scaled_diff);
        } else {
            x = build_path(&context_, &encoded_audio, 0);
        }
        output_ = x.tensor;
        ggml_set_output(output_);

        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        engine::debug::trace_log_scalar("liveavatar.denoiser.cfg_enabled", cfg_enabled_);
        engine::debug::trace_log_scalar("liveavatar.denoiser.sage_attention", use_sage_attention_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar denoiser backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);

    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    bool cfg_enabled_ = false;
    bool use_sage_attention_ = false;
    const LiveAvatarDenoiserStaticCache * static_cache_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> const_ctx_;
    ggml_backend_buffer_t const_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue latent_;
    engine::core::TensorValue ref_latents_;
    engine::core::TensorValue cond_latents_;
    engine::core::TensorValue context_;
    engine::core::TensorValue audio_local_;
    engine::core::TensorValue audio_global_;
    engine::core::TensorValue timestep_;
    engine::core::TensorValue guidance_scale_;
    engine::core::TensorValue rope_cos_;
    engine::core::TensorValue rope_sin_;
    std::vector<float> rope_cos_values_;
    std::vector<float> rope_sin_values_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};



struct LiveAvatarDenoiserRuntime::Data {
    Data(std::shared_ptr<const LiveAvatarAssets> assets, bool denoiser_weight_streaming)
        : assets_(require_assets(std::move(assets))),
          denoiser_weight_streaming_(denoiser_weight_streaming) {}

    LiveAvatarDenoiserWeights & ensure_weights(engine::core::ExecutionContext & execution) {
        if (weights == nullptr) {
            weights = std::make_unique<LiveAvatarDenoiserWeights>(load_denoiser_weights(
                *assets_->denoiser_weights,
                execution.backend(),
                execution.backend_type(),
                assets_->config,
                denoiser_weight_streaming_));
            assets_->denoiser_weights->release_storage();
        }
        return *weights;
    }

    void release(engine::core::ExecutionContext & execution) {
        graphs.clear();
        static_cache.reset();
        condition_graph.reset();
        weights.reset();
        engine::core::trim_backend_pools(execution.backend());
    }

    void release_condition_graph() {
        condition_graph.reset();
    }

    LiveAvatarDenoiserConditionGraph & ensure_condition_graph(
        engine::core::ExecutionContext & execution,
        int64_t target_frames,
        int64_t audio_frames,
        int64_t latent_audio_start_frame) {
        if (condition_graph == nullptr ||
            !condition_graph->matches(target_frames, audio_frames, latent_audio_start_frame)) {
            condition_graph = std::make_unique<LiveAvatarDenoiserConditionGraph>(
                execution,
                ensure_weights(execution),
                assets_->config,
                target_frames,
                audio_frames,
                latent_audio_start_frame);
        }
        return *condition_graph;
    }

    LiveAvatarDenoiserPreparedCondition prepare_condition(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserConditionRunInput & input,
        int64_t target_frames,
        int64_t audio_frames,
        int64_t latent_audio_start_frame) {
        return ensure_condition_graph(execution, target_frames, audio_frames, latent_audio_start_frame).run(input);
    }

    LiveAvatarDenoiserStaticCache & prepare_static_cache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserPreparedCondition & condition,
        const engine::core::TensorShape & latent_shape,
        int64_t lanes,
        bool use_sage_attention) {
        if (static_cache == nullptr || !static_cache->matches(latent_shape, lanes, use_sage_attention)) {
            graphs.clear();
            static_cache = std::make_unique<LiveAvatarDenoiserStaticCache>(
                execution,
                ensure_weights(execution),
                assets_->config,
                latent_shape,
                lanes,
                use_sage_attention);
        }
        static_cache->populate(condition);
        return *static_cache;
    }

    LiveAvatarDenoiserGraph & graph(
        engine::core::ExecutionContext & execution,
        const engine::core::TensorShape & latent_shape,
        bool cfg_enabled,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache * static_cache_ptr) {
        for (const auto & candidate : graphs) {
            if (candidate->matches(latent_shape, cfg_enabled, use_sage_attention, static_cache_ptr)) {
                return *candidate;
            }
        }
        auto next = std::make_unique<LiveAvatarDenoiserGraph>(
            execution,
            ensure_weights(execution),
            assets_->config,
            latent_shape,
            cfg_enabled,
            use_sage_attention,
            static_cache_ptr);
        auto & out = *next;
        graphs.push_back(std::move(next));
        return out;
    }

    std::vector<float> denoise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool cfg_enabled,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache * static_cache_ptr) {
        return graph(execution, latent_shape, cfg_enabled, use_sage_attention, static_cache_ptr).run(input);
    }

    std::vector<float> denoise_layerwise(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserRunInput & input,
        const engine::core::TensorShape & latent_shape,
        bool use_sage_attention,
        const LiveAvatarDenoiserStaticCache & static_cache_ref,
        int64_t layer_batch) {
        const auto layerwise_start = Clock::now();
        if (layer_batch <= 0) {
            throw std::runtime_error("LiveAvatar denoiser layerwise batch must be positive");
        }
        const auto & denoiser_weights = ensure_weights(execution);
        LiveAvatarDenoiserRopeCache rope_cache(execution, assets_->config, latent_shape);
        LiveAvatarDenoiserPreludeOutput prelude;
        {
            LiveAvatarDenoiserPreludeGraph prelude_graph(execution, denoiser_weights, assets_->config, latent_shape);
            prelude = prelude_graph.run(input);
        }
        engine::core::trim_backend_pools(execution.backend());
        auto run_lane = [&](int64_t lane) {
            std::vector<float> hidden = prelude.hidden;
            for (int64_t layer = 0; layer < assets_->config.num_layers; layer += layer_batch) {
                const int64_t layer_end = std::min<int64_t>(layer + layer_batch, assets_->config.num_layers);
                LiveAvatarDenoiserLayerGraph layer_graph(
                    execution,
                    denoiser_weights,
                    assets_->config,
                    latent_shape,
                    rope_cache,
                    static_cache_ref,
                    layer,
                    layer_end,
                    lane,
                    use_sage_attention);
                hidden = layer_graph.run(hidden, prelude.actual_e0, prelude.zero_e0);
            }
            LiveAvatarDenoiserFinalGraph final_graph(execution, denoiser_weights, assets_->config, latent_shape);
            return final_graph.run(hidden, prelude.actual_e);
        };
        auto cond_tokens = run_lane(0);
        auto uncond_tokens = run_lane(1);
        auto guided_tokens = combine_cfg(cond_tokens, uncond_tokens, input.guidance_scale);
        engine::debug::timing_log_scalar("liveavatar.denoiser_layerwise_total_ms", engine::debug::elapsed_ms(layerwise_start));
        return unpatchify_head_tokens(
            guided_tokens,
            latent_shape.dims[1],
            latent_shape.dims[2],
            latent_shape.dims[3]);
    }

    std::shared_ptr<const LiveAvatarAssets> assets_;
    bool denoiser_weight_streaming_ = false;
    std::unique_ptr<LiveAvatarDenoiserWeights> weights;
    std::unique_ptr<LiveAvatarDenoiserConditionGraph> condition_graph;
    std::unique_ptr<LiveAvatarDenoiserStaticCache> static_cache;
    std::vector<std::unique_ptr<LiveAvatarDenoiserGraph>> graphs;
};

LiveAvatarDenoiserRuntime::LiveAvatarDenoiserRuntime(
    std::shared_ptr<const LiveAvatarAssets> assets,
    bool denoiser_weight_streaming)
    : data_(std::make_unique<Data>(std::move(assets), denoiser_weight_streaming)) {}

LiveAvatarDenoiserRuntime::~LiveAvatarDenoiserRuntime() = default;

LiveAvatarDenoiserWeights & LiveAvatarDenoiserRuntime::ensure_weights(engine::core::ExecutionContext & execution) {
    return data_->ensure_weights(execution);
}

void LiveAvatarDenoiserRuntime::release(engine::core::ExecutionContext & execution) {
    data_->release(execution);
}

void LiveAvatarDenoiserRuntime::release_condition_graph() {
    data_->release_condition_graph();
}

LiveAvatarDenoiserPreparedCondition LiveAvatarDenoiserRuntime::prepare_condition(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserConditionRunInput & input,
    int64_t target_frames,
    int64_t audio_frames,
    int64_t latent_audio_start_frame) {
    return data_->prepare_condition(execution, input, target_frames, audio_frames, latent_audio_start_frame);
}

LiveAvatarDenoiserStaticCache & LiveAvatarDenoiserRuntime::prepare_static_cache(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserPreparedCondition & condition,
    const engine::core::TensorShape & latent_shape,
    int64_t lanes,
    bool use_sage_attention) {
    return data_->prepare_static_cache(execution, condition, latent_shape, lanes, use_sage_attention);
}

std::vector<float> LiveAvatarDenoiserRuntime::denoise(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserRunInput & input,
    const engine::core::TensorShape & latent_shape,
    bool cfg_enabled,
    bool use_sage_attention,
    const LiveAvatarDenoiserStaticCache * static_cache) {
    return data_->denoise(
        execution,
        input,
        latent_shape,
        cfg_enabled,
        use_sage_attention,
        static_cache);
}

std::vector<float> LiveAvatarDenoiserRuntime::denoise_layerwise(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserRunInput & input,
    const engine::core::TensorShape & latent_shape,
    bool use_sage_attention,
    const LiveAvatarDenoiserStaticCache & static_cache,
    int64_t layer_batch) {
    return data_->denoise_layerwise(execution, input, latent_shape, use_sage_attention, static_cache, layer_batch);
}

}  // namespace engine::community_models::liveavatar
