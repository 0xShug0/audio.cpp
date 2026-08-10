#include "engine/models/dots_tts/patch_encoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/core/backend.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::dots_tts {
namespace {

using Clock = std::chrono::steady_clock;
using engine::modules::AddModule;
using engine::modules::ConcatModule;
using engine::modules::FastKVSetRowsMode;
using engine::modules::FastKVSetRowsModule;
using engine::modules::LinearModule;
using engine::modules::LinearWeights;
using engine::modules::RMSNormModule;
using engine::modules::ScaledDotProductAttentionLowering;
using engine::modules::ScaledDotProductAttentionModule;
using engine::modules::SiluModule;
using engine::modules::SliceModule;
using engine::modules::SplitRoPEModule;
using engine::modules::TransposeModule;

constexpr int64_t kDownsampleRate = 2;
constexpr size_t kWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 512ull * 1024ull * 1024ull;
constexpr float kPatchEncoderNormEps = std::numeric_limits<float>::epsilon();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct PatchLayerWeights {
    core::TensorValue attn_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    std::optional<core::TensorValue> q_norm;
    std::optional<core::TensorValue> k_norm;
    core::TensorValue o_proj;
    core::TensorValue o_bias;
    core::TensorValue ffn_norm;
    core::TensorValue fc1;
    core::TensorValue fc1_bias;
    core::TensorValue fc2;
    core::TensorValue fc2_bias;
};

struct DownsampleWeights {
    std::vector<float> weight;
    std::vector<float> bias;
};

struct HostLinearWeights {
    std::vector<float> weight;
    std::vector<float> bias;
};

struct PatchEncoderWeights {
    DotsConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    DownsampleWeights ds_proj;
    HostLinearWeights in_proj;
    std::vector<PatchLayerWeights> layers;
    LinearWeights out_proj;
};

struct PatchLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct PatchPrefillGraphOutputs {
    core::TensorValue embeddings;
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

struct PatchPrefillRunOutput {
    DotsPatchEmbeddings embeddings;
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
};

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("DotTTS patch encoder tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("DotTTS patch encoder tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

int64_t head_dim(const DotsTransformerConfig & config) {
    if (config.hidden_size <= 0 || config.num_heads <= 0 || config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("DotTTS patch encoder attention config is invalid");
    }
    return config.hidden_size / config.num_heads;
}

int64_t out_downsample_rate(const DotsConfig & config) {
    if (config.patch_size % kDownsampleRate != 0) {
        throw std::runtime_error("DotTTS patch encoder patch_size must be divisible by downsample rate");
    }
    return config.patch_size / kDownsampleRate;
}

void fill_split_rope_tables(
    std::vector<float> & cos,
    std::vector<float> & sin,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    float theta) {
    if (steps <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 || !(theta > 0.0F)) {
        throw std::runtime_error("DotTTS split RoPE table config is invalid");
    }
    const int64_t half_dim = head_dim / 2;
    cos.assign(static_cast<size_t>(heads * steps * half_dim), 0.0F);
    sin.assign(cos.size(), 0.0F);
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t step = 0; step < steps; ++step) {
            const double position = static_cast<double>(start + step);
            for (int64_t dim = 0; dim < half_dim; ++dim) {
                const double inv_freq = 1.0 / std::pow(
                    static_cast<double>(theta),
                    static_cast<double>(2 * dim) / static_cast<double>(head_dim));
                const double angle = position * inv_freq;
                const size_t index = static_cast<size_t>((head * steps + step) * half_dim + dim);
                cos[index] = static_cast<float>(std::cos(angle));
                sin[index] = static_cast<float>(std::sin(angle));
            }
        }
    }
}

assets::TensorStorageType derived_qkv_storage_type(
    const assets::TensorSource & source,
    std::string_view q_name,
    assets::TensorStorageType requested,
    core::BackendType backend_type) {
    if (requested != assets::TensorStorageType::Native) {
        return requested;
    }
    auto storage = assets::tensor_storage_type_for_dtype(source.require_metadata(q_name).dtype);
    if ((backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) &&
        storage == assets::TensorStorageType::BF16) {
        return assets::TensorStorageType::F16;
    }
    return storage;
}

DownsampleWeights load_stride2_downsample_weights(
    const assets::TensorSource & source,
    int64_t channels) {
    return {
        source.require_f32("patch_encoder.ds_proj.weight", {channels, channels, kDownsampleRate}),
        source.require_f32("patch_encoder.ds_proj.bias", {channels}),
    };
}

HostLinearWeights load_host_linear_weights(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features) {
    return {
        source.require_f32(prefix + ".weight", {out_features, in_features}),
        source.require_f32(prefix + ".bias", {out_features}),
    };
}

std::vector<float> downsample_latents(
    const std::vector<float> & latents,
    int64_t frames,
    int64_t channels,
    const DownsampleWeights & weights,
    const float * prefix_frame) {
    if (frames <= 0 || channels <= 0 || static_cast<int64_t>(latents.size()) != frames * channels) {
        throw std::runtime_error("DotTTS patch encoder downsample latent size mismatch");
    }
    if (weights.weight.size() != static_cast<size_t>(channels * channels * kDownsampleRate) ||
        weights.bias.size() != static_cast<size_t>(channels)) {
        throw std::runtime_error("DotTTS patch encoder downsample weight shape mismatch");
    }
    const int64_t stream_frames = frames + 1;
    const int64_t output_frames = (stream_frames - kDownsampleRate) / kDownsampleRate + 1;
    std::vector<float> output(static_cast<size_t>(output_frames * channels), 0.0F);
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        for (int64_t out = 0; out < channels; ++out) {
            float sum = weights.bias[static_cast<size_t>(out)];
            for (int64_t in = 0; in < channels; ++in) {
                for (int64_t kernel = 0; kernel < kDownsampleRate; ++kernel) {
                    const int64_t stream_index = frame * kDownsampleRate + kernel;
                    float value = 0.0F;
                    if (stream_index == 0) {
                        value = prefix_frame == nullptr ? 0.0F : prefix_frame[static_cast<size_t>(in)];
                    } else {
                        value = latents[static_cast<size_t>((stream_index - 1) * channels + in)];
                    }
                    sum += weights.weight[static_cast<size_t>((out * channels + in) * kDownsampleRate + kernel)] * value;
                }
            }
            output[static_cast<size_t>(frame * channels + out)] = sum;
        }
    }
    return output;
}

std::vector<float> host_linear(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    const HostLinearWeights & weights) {
    if (rows <= 0 || in_features <= 0 || out_features <= 0 ||
        static_cast<int64_t>(input.size()) != rows * in_features ||
        static_cast<int64_t>(weights.weight.size()) != out_features * in_features ||
        static_cast<int64_t>(weights.bias.size()) != out_features) {
        throw std::runtime_error("DotTTS patch encoder host linear shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(rows * out_features), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t out_col = 0; out_col < out_features; ++out_col) {
            float sum = weights.bias[static_cast<size_t>(out_col)];
            const float * weight_row = weights.weight.data() + static_cast<std::ptrdiff_t>(out_col * in_features);
            const float * input_row = input.data() + static_cast<std::ptrdiff_t>(row * in_features);
            for (int64_t in_col = 0; in_col < in_features; ++in_col) {
                sum += input_row[static_cast<size_t>(in_col)] * weight_row[static_cast<size_t>(in_col)];
            }
            out[static_cast<size_t>(row * out_features + out_col)] = sum;
        }
    }
    engine::core::round_f32_to_bf16_in_place(out);
    return out;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto addressable = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        addressable,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

PatchLayerOutputs patch_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const PatchLayerWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask) {
    const int64_t dim = head_dim(config);
    auto h = RMSNormModule({config.hidden_size, kPatchEncoderNormEps, true, false})
                 .build(ctx, input, {weights.attn_norm, std::nullopt});
    auto q = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.q_proj, std::nullopt});
    auto k = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.k_proj, std::nullopt});
    auto v = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.v_proj, std::nullopt});
    q = reshape_heads(ctx, q, config.num_heads, dim);
    k = reshape_heads(ctx, k, config.num_heads, dim);
    v = reshape_heads(ctx, v, config.num_heads, dim);
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    const bool use_qk_norm = weights.q_norm.has_value() || weights.k_norm.has_value();
    if (use_qk_norm) {
        q = core::ensure_backend_addressable_layout(ctx, q);
        k = core::ensure_backend_addressable_layout(ctx, k);
        q = RMSNormModule({dim, kPatchEncoderNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
        k = RMSNormModule({dim, kPatchEncoderNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
    }
    if (config.rotary_bias) {
        q = SplitRoPEModule({dim}).build(ctx, q, rope_cos, rope_sin);
        k = SplitRoPEModule({dim}).build(ctx, k, rope_cos, rope_sin);
    }
    auto key = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    auto value = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto context = ScaledDotProductAttentionModule({
        dim,
        ScaledDotProductAttentionLowering::FlashPreserveViews,
        GGML_PREC_DEFAULT,
        engine::modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k, v, attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    auto output = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_DEFAULT})
                      .build(ctx, context, {weights.o_proj, weights.o_bias});
    return {
        output,
        key,
        value,
    };
}

PatchLayerOutputs patch_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const PatchLayerWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    auto attn = patch_attention(ctx, input, rope_cos, rope_sin, weights, config, attention_mask);
    auto x = AddModule().build(ctx, input, attn.output);
    auto h = RMSNormModule({config.hidden_size, kPatchEncoderNormEps, true, false})
                 .build(ctx, x, {weights.ffn_norm, std::nullopt});
    h = LinearModule({config.hidden_size, config.ffn_hidden_size, true, GGML_PREC_DEFAULT})
            .build(ctx, h, {weights.fc1, weights.fc1_bias});
    h = SiluModule().build(ctx, h);
    h = LinearModule({config.ffn_hidden_size, config.hidden_size, true, GGML_PREC_DEFAULT})
            .build(ctx, h, {weights.fc2, weights.fc2_bias});
    return {
        AddModule().build(ctx, x, h),
        attn.key,
        attn.value,
    };
}

PatchPrefillGraphOutputs build_prefill_graph(
    core::ModuleBuildContext & ctx,
    const PatchEncoderWeights & weights,
    const core::TensorValue & downsampled_latents,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const core::TensorValue & attention_mask) {
    const auto & config = weights.config;
    PatchPrefillGraphOutputs outputs;
    outputs.keys.reserve(weights.layers.size());
    outputs.values.reserve(weights.layers.size());
    auto x = downsampled_latents;
    for (const auto & layer : weights.layers) {
        auto out = patch_layer(ctx, x, rope_cos, rope_sin, layer, config.patch_encoder, attention_mask);
        x = out.output;
        outputs.keys.push_back(out.key);
        outputs.values.push_back(out.value);
    }
    const int64_t rate = out_downsample_rate(config);
    const int64_t patches = x.shape.dims[1] / rate;
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({x.shape.dims[0], patches, rate * config.patch_encoder.hidden_size}));
    outputs.embeddings = LinearModule({rate * config.patch_encoder.hidden_size, config.llm.hidden_size, true, GGML_PREC_DEFAULT})
                             .build(ctx, x, weights.out_proj);
    return outputs;
}

PatchLayerOutputs patch_decode_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const PatchLayerWeights & weights,
    const DotsTransformerConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask) {
    const int64_t dim = head_dim(config);
    auto h = RMSNormModule({config.hidden_size, kPatchEncoderNormEps, true, false})
                 .build(ctx, input, {weights.attn_norm, std::nullopt});
    auto q = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.q_proj, std::nullopt});
    auto k = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.k_proj, std::nullopt});
    auto v = LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_DEFAULT})
                 .build(ctx, h, {weights.v_proj, std::nullopt});
    q = reshape_heads(ctx, q, config.num_heads, dim);
    k = reshape_heads(ctx, k, config.num_heads, dim);
    v = reshape_heads(ctx, v, config.num_heads, dim);
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    const bool use_qk_norm = weights.q_norm.has_value() || weights.k_norm.has_value();
    if (use_qk_norm) {
        q = core::ensure_backend_addressable_layout(ctx, q);
        k = core::ensure_backend_addressable_layout(ctx, k);
        q = RMSNormModule({dim, kPatchEncoderNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
        k = RMSNormModule({dim, kPatchEncoderNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
    }
    if (config.rotary_bias) {
        q = SplitRoPEModule({dim}).build(ctx, q, rope_cos, rope_sin);
        k = SplitRoPEModule({dim}).build(ctx, k, rope_cos, rope_sin);
    }
    auto k_rows = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    auto v_rows = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto k_heads = ConcatModule({2}).build(ctx, TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, cache_key), k);
    auto v_heads = ConcatModule({2}).build(ctx, TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, cache_value), v);
    auto context = ScaledDotProductAttentionModule({
        dim,
        ScaledDotProductAttentionLowering::FlashPreserveViews,
        GGML_PREC_DEFAULT,
        engine::modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k_heads, v_heads, attention_mask);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    auto attn_out = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_DEFAULT})
                        .build(ctx, context, {weights.o_proj, weights.o_bias});
    auto x = AddModule().build(ctx, input, attn_out);
    h = RMSNormModule({config.hidden_size, kPatchEncoderNormEps, true, false})
            .build(ctx, x, {weights.ffn_norm, std::nullopt});
    h = LinearModule({config.hidden_size, config.ffn_hidden_size, true, GGML_PREC_DEFAULT})
            .build(ctx, h, {weights.fc1, weights.fc1_bias});
    h = SiluModule().build(ctx, h);
    h = LinearModule({config.ffn_hidden_size, config.hidden_size, true, GGML_PREC_DEFAULT})
            .build(ctx, h, {weights.fc2, weights.fc2_bias});
    return {
        AddModule().build(ctx, x, h),
        k_rows,
        v_rows,
    };
}

struct PatchDecodeGraphOutputs {
    core::TensorValue embeddings;
    std::vector<core::TensorValue> keys;
    std::vector<core::TensorValue> values;
};

PatchDecodeGraphOutputs build_decode_graph(
    core::ModuleBuildContext & ctx,
    const PatchEncoderWeights & weights,
    const core::TensorValue & downsampled_patch,
    const core::TensorValue & rope_cos,
    const core::TensorValue & rope_sin,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_slots,
    const std::vector<core::TensorValue> & cache_keys,
    const std::vector<core::TensorValue> & cache_values) {
    const auto & config = weights.config;
    PatchDecodeGraphOutputs outputs;
    outputs.keys.reserve(weights.layers.size());
    outputs.values.reserve(weights.layers.size());
    auto x = downsampled_patch;
    const FastKVSetRowsModule set_rows({FastKVSetRowsMode::BackendViewOptimized});
    for (size_t layer_index = 0; layer_index < weights.layers.size(); ++layer_index) {
        auto out = patch_decode_layer(
            ctx,
            x,
            rope_cos,
            rope_sin,
            weights.layers[layer_index],
            config.patch_encoder,
            cache_keys[layer_index],
            cache_values[layer_index],
            attention_mask);
        auto updated_key = cache_keys[layer_index];
        auto updated_value = cache_values[layer_index];
        for (int64_t row = 0; row < out.key.shape.dims[1]; ++row) {
            auto slot = SliceModule({0, row, 1}).build(ctx, cache_slots);
            auto key_row = SliceModule({1, row, 1}).build(ctx, out.key);
            auto value_row = SliceModule({1, row, 1}).build(ctx, out.value);
            updated_key = set_rows.build(ctx, updated_key, key_row, slot);
            updated_value = set_rows.build(ctx, updated_value, value_row, slot);
        }
        x = out.output;
        outputs.keys.push_back(updated_key);
        outputs.values.push_back(updated_value);
    }
    const int64_t rate = out_downsample_rate(config);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({1, 1, rate * config.patch_encoder.hidden_size}));
    outputs.embeddings = LinearModule({rate * config.patch_encoder.hidden_size, config.llm.hidden_size, true, GGML_PREC_DEFAULT})
                             .build(ctx, x, weights.out_proj);
    return outputs;
}

PatchLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const DotsTransformerConfig & config,
    int64_t index,
    assets::TensorStorageType storage_type,
    core::BackendType backend_type) {
    const std::string prefix = "patch_encoder.encoder.layers." + std::to_string(index);
    PatchLayerWeights layer;
    layer.attn_norm = store.load_f32_tensor(source, prefix + ".attn_norm.weight", {config.hidden_size});
    layer.q_proj = store.load_tensor(
        source,
        prefix + ".attn.q_proj.weight",
        derived_qkv_storage_type(source, prefix + ".attn.q_proj.weight", storage_type, backend_type),
        {config.hidden_size, config.hidden_size});
    layer.k_proj = store.load_tensor(
        source,
        prefix + ".attn.k_proj.weight",
        derived_qkv_storage_type(source, prefix + ".attn.k_proj.weight", storage_type, backend_type),
        {config.hidden_size, config.hidden_size});
    layer.v_proj = store.load_tensor(
        source,
        prefix + ".attn.v_proj.weight",
        derived_qkv_storage_type(source, prefix + ".attn.v_proj.weight", storage_type, backend_type),
        {config.hidden_size, config.hidden_size});
    const int64_t dim = head_dim(config);
    if (source.has_tensor(prefix + ".attn.q_norm.weight")) {
        layer.q_norm = store.load_f32_tensor(source, prefix + ".attn.q_norm.weight", {dim});
    }
    if (source.has_tensor(prefix + ".attn.k_norm.weight")) {
        layer.k_norm = store.load_f32_tensor(source, prefix + ".attn.k_norm.weight", {dim});
    }
    if (layer.q_norm.has_value() != layer.k_norm.has_value()) {
        throw std::runtime_error("DotTTS patch encoder requires q_norm and k_norm tensors to be present together");
    }
    layer.o_proj = store.load_tensor(source, prefix + ".attn.o_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    layer.o_bias = store.load_f32_tensor(source, prefix + ".attn.o_proj.bias", {config.hidden_size});
    layer.ffn_norm = store.load_f32_tensor(source, prefix + ".ffn_norm.weight", {config.hidden_size});
    layer.fc1 = store.load_tensor(source, prefix + ".ffn.fc1.weight", storage_type, {config.ffn_hidden_size, config.hidden_size});
    layer.fc1_bias = store.load_f32_tensor(source, prefix + ".ffn.fc1.bias", {config.ffn_hidden_size});
    layer.fc2 = store.load_tensor(source, prefix + ".ffn.fc2.weight", storage_type, {config.hidden_size, config.ffn_hidden_size});
    layer.fc2_bias = store.load_f32_tensor(source, prefix + ".ffn.fc2.bias", {config.hidden_size});
    return layer;
}

void validate_config(const DotsConfig & config, const assets::TensorSource & source) {
    if (config.latent_dim <= 0 || config.patch_size <= 0 || config.llm.hidden_size <= 0) {
        throw std::runtime_error("DotTTS patch encoder config contains non-positive dimensions");
    }
    if (config.patch_encoder.norm_layer != "RMSNorm") {
        throw std::runtime_error("DotTTS patch encoder currently requires RMSNorm checkpoints");
    }
    if (!config.patch_encoder.causal) {
        throw std::runtime_error("DotTTS patch encoder currently requires causal attention");
    }
    if (config.patch_encoder.hidden_size % config.patch_encoder.num_heads != 0) {
        throw std::runtime_error("DotTTS patch encoder hidden_size must be divisible by heads");
    }
    if (config.patch_encoder.input_dim != config.latent_dim || config.patch_encoder.ffn_hidden_size <= 0) {
        throw std::runtime_error("DotTTS patch encoder dimensions do not match model config");
    }
    if (source.has_tensor("patch_encoder.encoder.layers.0.attn.q_proj.bias") ||
        source.has_tensor("patch_encoder.encoder.layers.0.attn.k_proj.bias") ||
        source.has_tensor("patch_encoder.encoder.layers.0.attn.v_proj.bias")) {
        throw std::runtime_error("DotTTS patch encoder qkv-bias checkpoint is not implemented yet");
    }
}

std::shared_ptr<PatchEncoderWeights> load_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType weight_storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("DotTTS patch encoder requires tensor source");
    }
    validate_config(config, *source);
    auto weights = std::make_shared<PatchEncoderWeights>();
    weights->config = std::move(config);
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "dots_tts.patch_encoder.weights",
        kWeightContextBytes);
    weights->ds_proj = load_stride2_downsample_weights(*source, weights->config.latent_dim);
    weights->in_proj = load_host_linear_weights(
        *source,
        "patch_encoder.in_proj",
        weights->config.patch_encoder.hidden_size,
        weights->config.latent_dim);
    weights->layers.reserve(static_cast<size_t>(weights->config.patch_encoder.num_layers));
    for (int64_t layer = 0; layer < weights->config.patch_encoder.num_layers; ++layer) {
        weights->layers.push_back(load_layer(
            *weights->store,
            *source,
            weights->config.patch_encoder,
            layer,
            weight_storage_type,
            weights->execution_context->backend_type()));
    }
    const int64_t rate = out_downsample_rate(weights->config);
    weights->out_proj = {
        weights->store->load_tensor(*source, "patch_encoder.out_proj.weight", weight_storage_type, {weights->config.llm.hidden_size, weights->config.patch_encoder.hidden_size * rate}),
        weights->store->load_f32_tensor(*source, "patch_encoder.out_proj.bias", {weights->config.llm.hidden_size}),
    };
    int64_t parameter_count = 0;
    for (const auto & tensor : source->tensors()) {
        if (tensor.name.rfind("patch_encoder.", 0) == 0) {
            parameter_count += tensor_elements(tensor.shape);
        }
    }
    (void)parameter_count;
    weights->store->upload();
    return weights;
}

class PrefillRunner {
public:
    explicit PrefillRunner(std::shared_ptr<const PatchEncoderWeights> weights)
        : weights_(std::move(weights)) {}

    ~PrefillRunner() { release_graph(); }

    PatchPrefillRunOutput run(const std::vector<float> & latents, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        if (frames <= 0 || frames % config.patch_size != 0) {
            throw std::runtime_error("DotTTS patch encoder prefill frame count must be positive and divisible by patch_size");
        }
        if (static_cast<int64_t>(latents.size()) != frames * config.latent_dim) {
            throw std::runtime_error("DotTTS patch encoder prefill latent size mismatch");
        }
        auto downsampled = downsample_latents(latents, frames, config.latent_dim, weights_->ds_proj, nullptr);
        auto projected = host_linear(
            downsampled,
            static_cast<int64_t>(downsampled.size()) / config.latent_dim,
            config.latent_dim,
            config.patch_encoder.hidden_size,
            weights_->in_proj);
        ensure_graph(frames);
        ggml_backend_tensor_set(latents_, projected.data(), 0, projected.size() * sizeof(float));
        if (config.patch_encoder.rotary_bias) {
            fill_split_rope_tables(rope_cos_values_, rope_sin_values_, 0, tokens_, config.patch_encoder.num_heads, head_dim(config.patch_encoder), config.patch_encoder.rotary_theta);
            ggml_backend_tensor_set(rope_cos_, rope_cos_values_.data(), 0, rope_cos_values_.size() * sizeof(float));
            ggml_backend_tensor_set(rope_sin_, rope_sin_values_.data(), 0, rope_sin_values_.size() * sizeof(float));
        }
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(ggml_fp16_t));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.patch_encoder.prefill") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS patch encoder prefill graph compute failed");
        }
        PatchPrefillRunOutput out;
        out.embeddings.patches = frames / config.patch_size;
        out.embeddings.hidden_size = config.llm.hidden_size;
        out.embeddings.values = core::read_tensor_f32(output_);
        out.keys.reserve(keys_.size());
        out.values.reserve(values_.size());
        for (ggml_tensor * key : keys_) {
            out.keys.push_back(core::read_tensor_f32(key));
        }
        for (ggml_tensor * value : values_) {
            out.values.push_back(core::read_tensor_f32(value));
        }
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        latents_ = nullptr;
        rope_cos_ = nullptr;
        rope_sin_ = nullptr;
        attention_mask_ = nullptr;
        output_ = nullptr;
        keys_.clear();
        values_.clear();
        frames_ = 0;
        tokens_ = 0;
    }

private:
    void ensure_graph(int64_t frames) {
        if (ggml_ != nullptr && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS patch encoder prefill graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.patch_encoder.prefill", weights_->execution_context->backend_type()};
        const int64_t tokens = (frames / weights_->config.patch_size) * out_downsample_rate(weights_->config);
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens, weights_->config.patch_encoder.hidden_size}));
        latents_ = input.tensor;
        auto rope_cos = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.patch_encoder.num_heads, tokens, head_dim(weights_->config.patch_encoder) / 2}));
        auto rope_sin = core::make_tensor(build_ctx, GGML_TYPE_F32, rope_cos.shape);
        rope_cos_ = rope_cos.tensor;
        rope_sin_ = rope_sin.tensor;
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, tokens, tokens}));
        attention_mask_ = mask.tensor;
        auto output = build_prefill_graph(build_ctx, *weights_, input, rope_cos, rope_sin, mask);
        output_ = output.embeddings.tensor;
        keys_.reserve(output.keys.size());
        values_.reserve(output.values.size());
        for (const auto & key : output.keys) {
            keys_.push_back(key.tensor);
        }
        for (const auto & value : output.values) {
            values_.push_back(value.tensor);
        }
        graph_ = ggml_new_graph_custom(ggml_.get(), 262144, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        for (ggml_tensor * key : keys_) {
            ggml_set_output(key);
            ggml_build_forward_expand(graph_, key);
        }
        for (ggml_tensor * value : values_) {
            ggml_set_output(value);
            ggml_build_forward_expand(graph_, value);
        }
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.patch_encoder.prefill");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS patch encoder prefill graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        fill_attention_mask(tokens);
        tokens_ = tokens;
        frames_ = frames;
    }

    void fill_attention_mask(int64_t tokens) {
        const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const auto visible = ggml_fp32_to_fp16(0.0F);
        attention_mask_values_.assign(static_cast<size_t>(tokens * tokens), masked);
        for (int64_t row = 0; row < tokens; ++row) {
            for (int64_t col = 0; col <= row; ++col) {
                attention_mask_values_[static_cast<size_t>(row * tokens + col)] = visible;
            }
        }
    }

    std::shared_ptr<const PatchEncoderWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * latents_ = nullptr;
    ggml_tensor * rope_cos_ = nullptr;
    ggml_tensor * rope_sin_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<float> rope_cos_values_;
    std::vector<float> rope_sin_values_;
    int64_t tokens_ = 0;
    int64_t frames_ = 0;
};

class DecodeRunner {
public:
    explicit DecodeRunner(std::shared_ptr<const PatchEncoderWeights> weights)
        : weights_(std::move(weights)) {}

    ~DecodeRunner() { release_graph(); }

    PatchPrefillRunOutput run(
        const std::vector<float> & latents,
        std::vector<float> & conv_tail,
        std::vector<std::vector<float>> & state_keys,
        std::vector<std::vector<float>> & state_values,
        int64_t & seq_len,
        int64_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        const int64_t patch_frames = config.patch_size;
        const int64_t produced_tokens = out_downsample_rate(config);
        if (static_cast<int64_t>(latents.size()) != patch_frames * config.latent_dim) {
            throw std::runtime_error("DotTTS patch encoder decode latent size mismatch");
        }
        if (conv_tail.size() != static_cast<size_t>(config.latent_dim * (kDownsampleRate - 1))) {
            throw std::runtime_error("DotTTS patch encoder decode state conv tail has invalid size");
        }
        if (seq_len + produced_tokens > capacity) {
            throw std::runtime_error("DotTTS patch encoder decode exceeds state capacity");
        }
        auto downsampled = downsample_latents(latents, patch_frames, config.latent_dim, weights_->ds_proj, conv_tail.data());
        auto projected = host_linear(
            downsampled,
            produced_tokens,
            config.latent_dim,
            config.patch_encoder.hidden_size,
            weights_->in_proj);
        ensure_graph(capacity);
        sync_cache_state(state_keys, state_values, seq_len);
        ggml_backend_tensor_set(latents_, projected.data(), 0, projected.size() * sizeof(float));
        cache_slot_values_.resize(static_cast<size_t>(produced_tokens));
        for (int64_t index = 0; index < produced_tokens; ++index) {
            cache_slot_values_[static_cast<size_t>(index)] = static_cast<int32_t>(seq_len + index);
        }
        ggml_backend_tensor_set(cache_slots_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));
        if (config.patch_encoder.rotary_bias) {
            fill_split_rope_tables(rope_cos_values_, rope_sin_values_, seq_len, produced_tokens, config.patch_encoder.num_heads, head_dim(config.patch_encoder), config.patch_encoder.rotary_theta);
            ggml_backend_tensor_set(rope_cos_, rope_cos_values_.data(), 0, rope_cos_values_.size() * sizeof(float));
            ggml_backend_tensor_set(rope_sin_, rope_sin_values_.data(), 0, rope_sin_values_.size() * sizeof(float));
        }
        fill_attention_mask(seq_len, produced_tokens, capacity_);
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(ggml_fp16_t));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.patch_encoder.decode") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS patch encoder decode graph compute failed");
        }
        PatchPrefillRunOutput out;
        out.embeddings.patches = 1;
        out.embeddings.hidden_size = config.llm.hidden_size;
        out.embeddings.values = core::read_tensor_f32(output_);
        conv_tail.assign(
            latents.end() - static_cast<std::ptrdiff_t>(config.latent_dim * (kDownsampleRate - 1)),
            latents.end());
        seq_len += produced_tokens;
        cache_synced_seq_len_ = seq_len;
        cache_synced_state_ = static_cast<const void *>(&state_keys);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        latents_ = nullptr;
        cache_slots_ = nullptr;
        rope_cos_ = nullptr;
        rope_sin_ = nullptr;
        attention_mask_ = nullptr;
        output_ = nullptr;
        cache_keys_.clear();
        cache_values_.clear();
        keys_.clear();
        values_.clear();
        capacity_ = 0;
        cache_synced_seq_len_ = -1;
        cache_synced_state_ = nullptr;
    }

private:
    void ensure_graph(int64_t capacity) {
        if (ggml_ != nullptr && capacity_ >= capacity) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS patch encoder decode graph context");
        }
        const auto & config = weights_->config;
        const int64_t dim = head_dim(config.patch_encoder);
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.patch_encoder.decode", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, out_downsample_rate(config), config.patch_encoder.hidden_size}));
        latents_ = input.tensor;
        const int64_t block = out_downsample_rate(config);
        auto cache_slots = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({block}));
        cache_slots_ = cache_slots.tensor;
        auto rope_cos = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.patch_encoder.num_heads, out_downsample_rate(config), dim / 2}));
        auto rope_sin = core::make_tensor(build_ctx, GGML_TYPE_F32, rope_cos.shape);
        rope_cos_ = rope_cos.tensor;
        rope_sin_ = rope_sin.tensor;
        auto mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, block, capacity + block}));
        attention_mask_ = mask.tensor;
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(weights_->layers.size());
        cache_values.reserve(weights_->layers.size());
        for (int64_t layer = 0; layer < config.patch_encoder.num_layers; ++layer) {
            auto key = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, capacity, config.patch_encoder.num_heads, dim}));
            auto value = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, capacity, config.patch_encoder.num_heads, dim}));
            cache_keys_.push_back(key.tensor);
            cache_values_.push_back(value.tensor);
            cache_keys.push_back(key);
            cache_values.push_back(value);
        }
        auto output = build_decode_graph(build_ctx, *weights_, input, rope_cos, rope_sin, mask, cache_slots, cache_keys, cache_values);
        output_ = output.embeddings.tensor;
        keys_.reserve(output.keys.size());
        values_.reserve(output.values.size());
        for (const auto & key : output.keys) {
            keys_.push_back(key.tensor);
        }
        for (const auto & value : output.values) {
            values_.push_back(value.tensor);
        }
        graph_ = ggml_new_graph_custom(ggml_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        for (ggml_tensor * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (ggml_tensor * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.patch_encoder.decode");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS patch encoder decode graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        attention_mask_values_.assign(
            static_cast<size_t>(block * (capacity + block)),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        capacity_ = capacity;
    }

    void sync_cache_state(
        const std::vector<std::vector<float>> & state_keys,
        const std::vector<std::vector<float>> & state_values,
        int64_t seq_len) {
        if (cache_synced_state_ == static_cast<const void *>(&state_keys) && cache_synced_seq_len_ == seq_len) {
            return;
        }
        const auto & config = weights_->config.patch_encoder;
        const int64_t dim = head_dim(config);
        const size_t row_values = static_cast<size_t>(config.num_heads * dim);
        const size_t total_values = static_cast<size_t>(capacity_) * row_values;
        std::vector<float> padded(total_values, 0.0F);
        for (size_t layer = 0; layer < cache_keys_.size(); ++layer) {
            std::fill(padded.begin(), padded.end(), 0.0F);
            if (layer < state_keys.size() && !state_keys[layer].empty()) {
                std::copy(state_keys[layer].begin(), state_keys[layer].end(), padded.begin());
            }
            ggml_backend_tensor_set(cache_keys_[layer], padded.data(), 0, padded.size() * sizeof(float));
            std::fill(padded.begin(), padded.end(), 0.0F);
            if (layer < state_values.size() && !state_values[layer].empty()) {
                std::copy(state_values[layer].begin(), state_values[layer].end(), padded.begin());
            }
            ggml_backend_tensor_set(cache_values_[layer], padded.data(), 0, padded.size() * sizeof(float));
        }
        cache_synced_seq_len_ = seq_len;
        cache_synced_state_ = static_cast<const void *>(&state_keys);
    }

    void fill_attention_mask(int64_t start, int64_t block, int64_t capacity) {
        const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        const int64_t kv_len = capacity + block;
        for (int64_t row = 0; row < block; ++row) {
            for (int64_t col = 0; col < start && col < capacity; ++col) {
                attention_mask_values_[static_cast<size_t>(row * kv_len + col)] = visible;
            }
            for (int64_t tail = 0; tail <= row; ++tail) {
                attention_mask_values_[static_cast<size_t>(row * kv_len + capacity + tail)] = visible;
            }
        }
    }

    std::shared_ptr<const PatchEncoderWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * latents_ = nullptr;
    ggml_tensor * cache_slots_ = nullptr;
    ggml_tensor * rope_cos_ = nullptr;
    ggml_tensor * rope_sin_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_tensor *> cache_keys_;
    std::vector<ggml_tensor *> cache_values_;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<int32_t> cache_slot_values_;
    std::vector<float> rope_cos_values_;
    std::vector<float> rope_sin_values_;
    int64_t capacity_ = 0;
    int64_t cache_synced_seq_len_ = -1;
    const void * cache_synced_state_ = nullptr;
};

}  // namespace

struct DotsPatchEncoderState::Impl {
    std::vector<float> conv_tail;
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    int64_t seq_len = 0;
    int64_t capacity = 0;
};

DotsPatchEncoderState::DotsPatchEncoderState() : impl_(std::make_unique<Impl>()) {}
DotsPatchEncoderState::~DotsPatchEncoderState() = default;
DotsPatchEncoderState::DotsPatchEncoderState(DotsPatchEncoderState &&) noexcept = default;
DotsPatchEncoderState & DotsPatchEncoderState::operator=(DotsPatchEncoderState &&) noexcept = default;

int64_t DotsPatchEncoderState::seq_len() const noexcept {
    return impl_ == nullptr ? 0 : impl_->seq_len;
}

int64_t DotsPatchEncoderState::capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->capacity;
}

struct DotsPatchEncoderComponent::Impl {
    explicit Impl(std::shared_ptr<const PatchEncoderWeights> weights)
        : weights(std::move(weights)),
          prefill_runner(std::make_unique<PrefillRunner>(this->weights)),
          decode_runner(std::make_unique<DecodeRunner>(this->weights)) {}

    std::shared_ptr<const PatchEncoderWeights> weights;
    std::unique_ptr<PrefillRunner> prefill_runner;
    std::unique_ptr<DecodeRunner> decode_runner;
};

DotsPatchEncoderComponent DotsPatchEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType weight_storage_type) {
    DotsPatchEncoderComponent component;
    component.impl_ = std::make_unique<Impl>(load_weights(
        std::move(source),
        backend,
        std::move(config),
        weight_storage_type));
    return component;
}

DotsPatchEncoderComponent::DotsPatchEncoderComponent() = default;
DotsPatchEncoderComponent::~DotsPatchEncoderComponent() = default;
DotsPatchEncoderComponent::DotsPatchEncoderComponent(DotsPatchEncoderComponent &&) noexcept = default;
DotsPatchEncoderComponent & DotsPatchEncoderComponent::operator=(DotsPatchEncoderComponent &&) noexcept = default;

bool DotsPatchEncoderComponent::is_loaded() const noexcept {
    return impl_ != nullptr && impl_->weights != nullptr;
}

DotsPatchEncoderState DotsPatchEncoderComponent::create_state(int64_t max_audio_patch_count) const {
    if (impl_ == nullptr || impl_->weights == nullptr) {
        throw std::runtime_error("DotTTS patch encoder is not initialized");
    }
    if (max_audio_patch_count <= 0) {
        throw std::runtime_error("DotTTS patch encoder state capacity must be positive");
    }
    DotsPatchEncoderState state;
    state.impl_->capacity = max_audio_patch_count * out_downsample_rate(impl_->weights->config);
    state.impl_->seq_len = 0;
    state.impl_->conv_tail.assign(static_cast<size_t>(impl_->weights->config.latent_dim * (kDownsampleRate - 1)), 0.0F);
    state.impl_->keys.assign(static_cast<size_t>(impl_->weights->config.patch_encoder.num_layers), {});
    state.impl_->values.assign(static_cast<size_t>(impl_->weights->config.patch_encoder.num_layers), {});
    return state;
}

DotsPatchEmbeddings DotsPatchEncoderComponent::prefill(
    const std::vector<float> & normalized_latents,
    int64_t frames,
    DotsPatchEncoderState & state) const {
    if (impl_ == nullptr || impl_->prefill_runner == nullptr) {
        throw std::runtime_error("DotTTS patch encoder is not initialized");
    }
    const auto & config = impl_->weights->config;
    const int64_t produced_tokens = (frames / config.patch_size) * out_downsample_rate(config);
    if (state.impl_ == nullptr || state.impl_->seq_len + produced_tokens > state.impl_->capacity) {
        throw std::runtime_error("DotTTS patch encoder prefill exceeds state capacity");
    }
    auto run = impl_->prefill_runner->run(normalized_latents, frames);
    state.impl_->seq_len += produced_tokens;
    state.impl_->keys = std::move(run.keys);
    state.impl_->values = std::move(run.values);
    state.impl_->conv_tail.assign(
        normalized_latents.end() - static_cast<std::ptrdiff_t>(config.latent_dim * (kDownsampleRate - 1)),
        normalized_latents.end());
    return std::move(run.embeddings);
}

DotsPatchEmbeddings DotsPatchEncoderComponent::decode_patch(
    const std::vector<float> & normalized_latents,
    DotsPatchEncoderState & state) const {
    if (impl_ == nullptr || impl_->decode_runner == nullptr) {
        throw std::runtime_error("DotTTS patch encoder is not initialized");
    }
    return impl_->decode_runner->run(
               normalized_latents,
               state.impl_->conv_tail,
               state.impl_->keys,
               state.impl_->values,
               state.impl_->seq_len,
               state.impl_->capacity)
        .embeddings;
}

void DotsPatchEncoderComponent::release_runtime_graphs() {
    if (impl_ != nullptr && impl_->prefill_runner != nullptr) {
        impl_->prefill_runner->release_graph();
    }
    if (impl_ != nullptr && impl_->decode_runner != nullptr) {
        impl_->decode_runner->release_graph();
    }
}

}  // namespace engine::models::dots_tts
