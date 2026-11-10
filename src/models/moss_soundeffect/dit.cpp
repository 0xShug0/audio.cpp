#include "engine/models/moss_soundeffect/dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/sampling/device_denoiser_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_soundeffect {
namespace {
using core::TensorShape;
using core::TensorValue;
constexpr size_t kGraphMetadataBytes = 64 * 1024 * 1024;
constexpr size_t kGraphNodes = 32768;

struct AttentionWeights {
    modules::LinearWeights q, k, v, o;
    modules::NormWeights norm_q, norm_k;
};

struct SelfAttentionWeights {
    modules::LinearWeights qkv, o;
    modules::NormWeights norm_q, norm_k;
};

struct BlockWeights {
    SelfAttentionWeights self;
    AttentionWeights cross;
    modules::NormWeights norm3;
    modules::FeedForwardWeights ffn;
    TensorValue modulation;
};

// Materialize autocast parameters once. Packing concatenates native matrix rows
// without dequantization; bias values retain F32 storage after BF16 rounding.
modules::LinearWeights load_autocast_projection(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    core::BackendType backend, assets::TensorStorageType storage, int64_t in_features,
    const std::vector<modules::PackedLinearSource> & sources) {
    // Keep Vulkan's BF16 autocast operands in their native representation;
    // the generic single-tensor loader otherwise lowers native BF16 to F16.
    if (backend == core::BackendType::Vulkan && storage == assets::TensorStorageType::Native)
        storage = assets::resolve_tensor_storage_type(source, sources.front().weight, storage);
    modules::LinearWeights weights;
    if (sources.size() == 1) {
        weights.weight = store.load_tensor(source, sources.front().weight, storage,
            {sources.front().out_features, in_features});
    } else {
        weights = modules::PackedLinearWeightsBuilder({in_features, sources, false, true})
            .build(store, source, storage);
    }
    std::vector<float> bias;
    for (const auto & item : sources) {
        const auto values = source.require_f32(item.bias.value(), {item.out_features});
        bias.insert(bias.end(), values.begin(), values.end());
    }
    const auto shape = TensorShape::from_dims({static_cast<int64_t>(bias.size())});
    core::round_f32_to_bf16_in_place(bias);
    weights.bias = store.make_f32(shape, std::move(bias));
    return weights;
}

// Keep native BF16/quantized matrices in their packaged storage and retain
// F32 accumulation through modulation and residual paths. The expanded FFN
// activation uses the existing native FP16 projection path on CUDA.
TensorValue build_projection(core::ModuleBuildContext & ctx, TensorValue input,
                             modules::LinearWeights weights, bool autocast) {
    modules::LinearConfig config{weights.weight.shape.dims[1], weights.weight.shape.dims[0], true};
    if (autocast) {
        if (!ggml_is_quantized(weights.weight.type) && weights.weight.type != GGML_TYPE_BF16) {
            weights.weight = core::wrap_tensor(ggml_cast(ctx.ggml, weights.weight.tensor, GGML_TYPE_BF16),
                                               weights.weight.shape, GGML_TYPE_BF16);
        }
        // Quantized and Metal kernels consume rounded F32 activations.
        // Vulkan can consume BF16 directly, selecting its existing matrix cores
        // when available without narrowing the BF16 exponent range to F16.
        if (ggml_is_quantized(weights.weight.type) || ctx.backend_type == core::BackendType::Metal) {
            input = modules::Bf16RoundModule().build(ctx, input);
            config.vulkan_f32_activation_lowering = ctx.backend_type == core::BackendType::Vulkan;
        } else if (ctx.backend_type != core::BackendType::Cuda && input.type != GGML_TYPE_BF16) {
            input = core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_BF16),
                                      input.shape, GGML_TYPE_BF16);
        }
    }
    auto result = modules::LinearModule(config).build(ctx, input, weights);
    return result;
}

// Q/K/V enter in token-major layout; SDPA consumes head-major and returns token-major.
TensorValue build_attention(core::ModuleBuildContext & ctx, TensorValue q, TensorValue k,
                            TensorValue v, const DitConfig & c, modules::ScaledDotProductAttentionLowering lowering,
                            bool head_major_kv = false) {
    const auto head_dim = c.dim / c.heads;
    const auto batch = q.shape.dims[0];
    const auto queries = q.shape.dims[1];
    const auto keys = k.shape.dims[head_major_kv ? 2 : 1];
    // Split the contiguous feature axis without copying strided packed-QKV
    // rows. Flash attention already accepts these views and converts K/V once.
    const auto split_heads = [&](TensorValue value, int64_t length) {
        if (value.shape.rank == 4) return value;
        auto * view = ggml_view_4d(ctx.ggml, value.tensor, head_dim, c.heads, length, batch,
            head_dim * value.tensor->nb[0], value.tensor->nb[1], value.tensor->nb[2], 0);
        return core::wrap_tensor(view, TensorShape::from_dims({batch, length, c.heads, head_dim}), value.type);
    };
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, split_heads(q, queries));
    if (!head_major_kv) {
        k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, split_heads(k, keys));
        v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, split_heads(v, keys));
    }
    auto result = modules::ScaledDotProductAttentionModule({
        head_dim, lowering, GGML_PREC_F32,
        modules::AttentionCausality::NonCausal}).build(ctx, q, k, v);
    // The sole consumer is an autocast output projection, which supplies the
    // BF16 boundary before its GEMM; do not materialize a redundant F32 round.
    return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, result),
                                TensorShape::from_dims({batch, queries, c.dim}));
}
}  // namespace

struct DitRuntime::Impl {
    core::ExecutionContext & execution;
    std::shared_ptr<const Assets> assets;
    core::BackendWeightStore store;
    core::ConstantTensorCache constants;
    modules::Conv1dWeights patch;
    modules::LinearWeights text_in, text_out, time_in, time_out, time_projection, head;
    TensorValue head_modulation;
    std::vector<BlockWeights> blocks;
    // Persistent [2, text_length * dim] leaves: positive then negative.
    // Flash caches use F16 head-major rows; explicit attention keeps F32 token-major.
    // The denoiser does not traverse conditioning producers.
    std::vector<std::pair<TensorValue, TensorValue>> cross_kv;
    TensorValue rope_positions;
    modules::DeviceDenoiserOptimizations runtime_optimizations;
    struct DenoiserState {
        int64_t batch = 0;
        std::unique_ptr<modules::DeviceDenoiserRuntime> runtime;
        TensorValue latent_input, time_input, branch_input, step_input, guidance_input;
    };
    std::unique_ptr<core::PreparedGraph> condition_graph;
    std::unique_ptr<DenoiserState> denoiser;
    TensorValue text_input;
    int64_t frames = 0;
    bool prepared = false;
    modules::ScaledDotProductAttentionLowering attention;

    Impl(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
         assets::TensorStorageType storage_type, core::AttentionPreference preference, bool optimize)
        : execution(execution), assets(std::move(assets)),
          store(execution.backend(), execution.backend_type(), "moss_soundeffect.dit.weights", 4 * 1024 * 1024),
          constants(execution.backend(), execution.config().threads, "moss_soundeffect.dit.cache", 1024 * 1024) {
        const auto started = std::chrono::steady_clock::now();
        runtime_optimizations = {optimize, optimize, optimize};
        if (!this->assets || !this->assets->dit_weights)
            throw std::runtime_error("MOSS SoundEffect DiT requires assets and DiT weights");
        const auto & a = *this->assets;
        const auto & c = a.dit;
        if (c.dim <= 0 || c.ffn_dim <= 0 || c.heads <= 0 || c.layers <= 0 ||
            c.latent_dim <= 0 || c.text_dim <= 0 || c.freq_dim <= 0 || c.freq_dim % 2 ||
            c.patch_size <= 0 || c.patch_size > std::numeric_limits<int>::max() ||
            c.dim % c.heads || (c.dim / c.heads) % 2 || c.eps <= 0 ||
            a.text_length <= 0 || a.sample_rate <= 0 || a.max_duration_sec <= 0 || a.vae.hop_length <= 0)
            throw std::runtime_error("Invalid MOSS SoundEffect DiT configuration");
        attention = core::resolve_flash_attention(execution.backend(), c.dim / c.heads, preference)
            ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
            : modules::ScaledDotProductAttentionLowering::Explicit;
        frames = static_cast<int64_t>(a.max_duration_sec) * a.sample_rate / a.vae.hop_length;
        if (frames <= 0 || frames % c.patch_size || frames / c.patch_size > 16384)
            throw std::runtime_error("MOSS SoundEffect latent frames must fit temporal patches and the RoPE table");

        const auto & source = *a.dit_weights;
        patch = modules::binding::conv1d_from_source(store, source, "patch_embedding",
            assets::TensorStorageType::Native, c.dim, c.latent_dim, c.patch_size, false);
        auto patch_bias = source.require_f32("patch_embedding.bias", {c.dim});
        core::round_f32_to_bf16_in_place(patch_bias);
        patch.bias = store.make_f32(TensorShape::from_dims({c.dim}), std::move(patch_bias));
        text_in = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.text_dim,
            {{"text_embedding.0.weight", "text_embedding.0.bias", c.dim}});
        text_out = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
            {{"text_embedding.2.weight", "text_embedding.2.bias", c.dim}});
        time_in = modules::binding::linear_from_source(store, source, "time_embedding.0", assets::TensorStorageType::F32, c.dim, c.freq_dim, true);
        time_out = modules::binding::linear_from_source(store, source, "time_embedding.2", assets::TensorStorageType::F32, c.dim, c.dim, true);
        time_projection = modules::binding::linear_from_source(store, source, "time_projection.1", assets::TensorStorageType::F32, 6 * c.dim, c.dim, true);
        head = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
            {{"head.head.weight", "head.head.bias", c.patch_size * c.latent_dim}});
        head_modulation = store.load_f32_tensor(source, "head.modulation", {1, 2, c.dim});
        blocks.resize(static_cast<size_t>(c.layers));
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto prefix = "blocks." + std::to_string(i);
            auto & b = blocks[static_cast<size_t>(i)];
            const auto self = prefix + ".self_attn";
            b.self.qkv = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim, {
                {self + ".q.weight", self + ".q.bias", c.dim},
                {self + ".k.weight", self + ".k.bias", c.dim},
                {self + ".v.weight", self + ".v.bias", c.dim},
            });
            b.self.o = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
                {{self + ".o.weight", self + ".o.bias", c.dim}});
            b.self.norm_q = modules::binding::norm_weight_from_source(store, source, self + ".norm_q", c.dim);
            b.self.norm_k = modules::binding::norm_weight_from_source(store, source, self + ".norm_k", c.dim);
            const auto cross = prefix + ".cross_attn";
            // Fold the fixed LayerNorm affine into its sole query-projection
            // consumer. Keep normalization itself and avoid a per-step affine pass.
            if (storage_type == assets::TensorStorageType::Native &&
                assets::resolve_tensor_storage_type(source, cross + ".q.weight", storage_type) == assets::TensorStorageType::BF16) {
                auto matrix = source.require_f32(cross + ".q.weight", {c.dim, c.dim});
                auto bias = source.require_f32(cross + ".q.bias", {c.dim});
                const auto scale = source.require_f32(prefix + ".norm3.weight", {c.dim});
                const auto shift = source.require_f32(prefix + ".norm3.bias", {c.dim});
                core::round_f32_to_bf16_in_place(matrix);
                core::round_f32_to_bf16_in_place(bias);
                for (int64_t row = 0; row < c.dim; ++row) {
                    float offset = bias[static_cast<size_t>(row)];
                    for (int64_t col = 0; col < c.dim; ++col) {
                        auto & value = matrix[static_cast<size_t>(row * c.dim + col)];
                        offset += value * shift[static_cast<size_t>(col)];
                        value *= scale[static_cast<size_t>(col)];
                    }
                    bias[static_cast<size_t>(row)] = offset;
                }
                std::vector<ggml_bf16_t> folded(matrix.size());
                ggml_fp32_to_bf16_row(matrix.data(), folded.data(), static_cast<int64_t>(matrix.size()));
                b.cross.q = {store.make_tensor(TensorShape::from_dims({c.dim, c.dim}), GGML_TYPE_BF16,
                    folded.data(), folded.size() * sizeof(ggml_bf16_t)),
                    store.make_f32(TensorShape::from_dims({c.dim}), std::move(bias))};
            } else {
                b.cross.q = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
                    {{cross + ".q.weight", cross + ".q.bias", c.dim}});
                b.norm3 = modules::binding::norm_from_source(store, source, prefix + ".norm3", c.dim);
            }
            b.cross.k = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
                {{cross + ".k.weight", cross + ".k.bias", c.dim}});
            b.cross.v = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
                {{cross + ".v.weight", cross + ".v.bias", c.dim}});
            b.cross.o = load_autocast_projection(store, source, execution.backend_type(), storage_type, c.dim,
                {{cross + ".o.weight", cross + ".o.bias", c.dim}});
            b.cross.norm_q = modules::binding::norm_weight_from_source(store, source, cross + ".norm_q", c.dim);
            b.cross.norm_k = modules::binding::norm_weight_from_source(store, source, cross + ".norm_k", c.dim);
            auto ffn_storage = storage_type;
            if (storage_type == assets::TensorStorageType::Native &&
                execution.backend_type() == core::BackendType::Cuda &&
                assets::resolve_tensor_storage_type(source, prefix + ".ffn.0.weight", storage_type) == assets::TensorStorageType::BF16)
                ffn_storage = assets::TensorStorageType::F16;
            const auto fc1 = load_autocast_projection(store, source, execution.backend_type(), ffn_storage, c.dim,
                {{prefix + ".ffn.0.weight", prefix + ".ffn.0.bias", c.ffn_dim}});
            const auto fc2 = load_autocast_projection(store, source, execution.backend_type(), ffn_storage, c.ffn_dim,
                {{prefix + ".ffn.2.weight", prefix + ".ffn.2.bias", c.dim}});
            b.ffn = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
            b.modulation = store.load_f32_tensor(source, prefix + ".modulation", {1, 6, c.dim});
        }
        store.upload();

        const auto steps = frames / c.patch_size;
        std::vector<int32_t> positions(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        rope_positions = constants.make_tensor(TensorShape::from_dims({steps}), GGML_TYPE_I32,
            positions.data(), positions.size() * sizeof(int32_t));
        const auto cache_shape = TensorShape::from_dims({2, a.text_length * c.dim});
        const auto cache_type = attention == modules::ScaledDotProductAttentionLowering::FlashPreserveViews
            ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const std::vector<uint8_t> zeros(static_cast<size_t>(cache_shape.num_elements()) * ggml_type_size(cache_type), 0);
        for (int64_t i = 0; i < c.layers; ++i) {
            auto k = constants.make_tensor(cache_shape, cache_type, zeros.data(), zeros.size());
            auto v = constants.make_tensor(cache_shape, cache_type, zeros.data(), zeros.size());
            cross_kv.emplace_back(k, v);
        }
        constants.ensure_uploaded();
        build_condition_graph();
        debug::timing_log_scalar("moss_soundeffect.dit.build_ms", debug::elapsed_ms(started));
    }

    void build_condition_graph() {
        const auto & c = assets->dit;
        condition_graph = std::make_unique<core::PreparedGraph>(execution,
            core::PreparedGraphConfig{"moss_soundeffect.dit.condition", kGraphMetadataBytes, kGraphNodes});
        auto & g = *condition_graph;
        auto & ctx = g.build_context();
        ctx.elide_identity_reshapes = true;
        // Batching here only combines independent text rows, never the denoising branches.
        text_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({2, assets->text_length, c.text_dim}));
        ggml_set_name(text_input.tensor, "moss_soundeffect.dit.context");
        ggml_set_input(text_input.tensor);
        auto text = build_projection(ctx, text_input, text_in, true);
        text = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, text);
        text = build_projection(ctx, text, text_out, true);
        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto & w = blocks[i].cross;
            auto k = build_projection(ctx, text, w.k, true);
            k = modules::RMSNormModule({c.dim, c.eps, true, false}).build(ctx, k, w.norm_k);
            // torch.nn.RMSNorm multiplies its F32 parameter before casting back to input dtype.
            auto v = build_projection(ctx, text, w.v, true);
            if (attention == modules::ScaledDotProductAttentionLowering::FlashPreserveViews) {
                const auto shape = TensorShape::from_dims({2, assets->text_length, c.heads, c.dim / c.heads});
                k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, core::reshape_tensor(ctx, k, shape));
                v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, core::reshape_tensor(ctx, v, shape));
            }
            k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), cross_kv[i].first.shape);
            v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), cross_kv[i].second.shape);
            ggml_build_forward_expand(g.graph(), ggml_cpy(ctx.ggml, k.tensor, cross_kv[i].first.tensor));
            ggml_build_forward_expand(g.graph(), ggml_cpy(ctx.ggml, v.tensor, cross_kv[i].second.tensor));
        }
        g.allocate();
    }

    void build_denoiser_graph(int64_t batch) {
        const auto started = std::chrono::steady_clock::now();
        denoiser.reset();
        auto state = std::make_unique<DenoiserState>();
        state->batch = batch;
        auto & latent_input = state->latent_input;
        auto & time_input = state->time_input;
        auto & branch_input = state->branch_input;
        const auto & c = assets->dit;
        const auto steps = frames / c.patch_size, hd = c.dim / c.heads;
        state->runtime = std::make_unique<modules::DeviceDenoiserRuntime>(execution,
            core::PreparedGraphConfig{"moss_soundeffect.dit", kGraphMetadataBytes, kGraphNodes},
            [&](core::ModuleBuildContext & ctx) -> modules::DeviceDenoiserGraph {
        ctx.elide_identity_reshapes = true;
        latent_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, c.latent_dim, frames}));
        time_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, c.freq_dim}));
        if (batch == 1) {
            branch_input = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
            ggml_set_name(branch_input.tensor, "moss_soundeffect.dit.branch");
            ggml_set_input(branch_input.tensor);
        }
        ggml_set_name(latent_input.tensor, "moss_soundeffect.dit.latents");
        ggml_set_name(time_input.tensor, "moss_soundeffect.dit.time_embedding_input");
        ggml_set_input(latent_input.tensor);
        ggml_set_input(time_input.tensor);
        if (c.patch_size != 1)
            throw std::runtime_error("MOSS BF16 patch embedding requires the upstream kernel-one configuration");
        // Kernel-one Conv1d is a tokenwise affine projection. This retains
        // the bias in the accumulator before the single BF16 result round.
        auto x = latent_input;
        if (batch == 2)
            x = modules::RepeatModule({TensorShape::from_dims({batch, c.latent_dim, frames})}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        const auto weight = core::reshape_tensor(ctx, patch.weight,
            TensorShape::from_dims({c.dim, c.latent_dim}));
        x = build_projection(ctx, x, {weight, patch.bias}, true);
        // The public pipeline disables autocast for time embedding and modulation.
        auto time = build_projection(ctx, time_input, time_in, false);
        time = modules::SiluModule().build(ctx, time);
        time = build_projection(ctx, time, time_out, false);
        auto t_mod = modules::SiluModule().build(ctx, time);
        t_mod = build_projection(ctx, t_mod, time_projection, false);
        t_mod = core::reshape_tensor(ctx, t_mod, TensorShape::from_dims({1, 6, c.dim}));
        const modules::LayerNormModule plain_norm({c.dim, c.eps, false, false});
        const modules::AdaptiveLayerNormModule adaptive_norm(
            {c.dim, c.eps, modules::AdaptiveLayerNormMode::LayerAffine});
        const modules::LayerNormModule affine_norm({c.dim, c.eps, true, true});
        const modules::RMSNormModule qk_norm({c.dim, c.eps, true, false, true});
        const modules::RoPEModule rope({hd, GGML_ROPE_TYPE_NORMAL, 10000.0F});
        const auto rotary = [&](TensorValue value) {
            return rope.build(ctx, value, rope_positions);
        };
        for (size_t i = 0; i < blocks.size(); ++i) {
            const auto & b = blocks[i];
            const auto modulation = modules::AddModule().build(ctx, b.modulation, t_mod);
            // Slices are [1,1,D], broadcasting across tokens and both CFG branches.
            const auto shift_sa = modules::SliceModule({1, 0, 1}).build(ctx, modulation);
            const auto scale_sa = modules::SliceModule({1, 1, 1}).build(ctx, modulation);
            const auto gate_sa = modules::SliceModule({1, 2, 1}).build(ctx, modulation);
            const auto shift_ff = modules::SliceModule({1, 3, 1}).build(ctx, modulation);
            const auto scale_ff = modules::SliceModule({1, 4, 1}).build(ctx, modulation);
            const auto gate_ff = modules::SliceModule({1, 5, 1}).build(ctx, modulation);
            // Time modulation and LayerNorm remain F32; projections autocast to BF16.
            auto y = adaptive_norm.build(ctx, x, {shift_sa, scale_sa});
            // Wan normalizes the entire D-wide projection before head splitting.
            const auto qkv = build_projection(ctx, y, b.self.qkv, true);
            auto q = modules::SliceModule({2, 0, c.dim}).build(ctx, qkv);
            auto k = modules::SliceModule({2, c.dim, c.dim}).build(ctx, qkv);
            auto v = modules::SliceModule({2, 2 * c.dim, c.dim}).build(ctx, qkv);
            q = qk_norm.build(ctx, q, b.self.norm_q);
            k = qk_norm.build(ctx, k, b.self.norm_k);
            q = core::reshape_tensor(ctx, q, TensorShape::from_dims({batch, steps, c.heads, hd}));
            k = core::reshape_tensor(ctx, k, TensorShape::from_dims({batch, steps, c.heads, hd}));
            q = rotary(q);
            k = rotary(k);
            y = build_projection(ctx, build_attention(ctx, q, k, v, c, attention), b.self.o, true);
            y = modules::MulModule(true).build(ctx, y, gate_sa);
            x = modules::AddModule().build(ctx, x, y);

            y = b.norm3.weight ? affine_norm.build(ctx, x, b.norm3) : plain_norm.build(ctx, x, {});
            q = qk_norm.build(ctx, build_projection(ctx, y, b.cross.q, true), b.cross.norm_q);
            // B2 directly views both persistent rows in positive/negative order.
            k = cross_kv[i].first;
            v = cross_kv[i].second;
            if (batch == 1) {
                const modules::EmbeddingModule select({2, assets->text_length * c.dim});
                k = select.build(ctx, branch_input, k);
                v = select.build(ctx, branch_input, v);
            }
            const bool head_major = attention == modules::ScaledDotProductAttentionLowering::FlashPreserveViews;
            const auto cache_shape = head_major
                ? TensorShape::from_dims({batch, c.heads, assets->text_length, hd})
                : TensorShape::from_dims({batch, assets->text_length, c.dim});
            k = core::reshape_tensor(ctx, k, cache_shape);
            v = core::reshape_tensor(ctx, v, cache_shape);
            y = build_projection(ctx, build_attention(ctx, q, k, v, c, attention, head_major), b.cross.o, true);
            x = modules::AddModule().build(ctx, x, y);
            y = adaptive_norm.build(ctx, x, {shift_ff, scale_ff});
            // Keep the pointwise FFN in matrix layout until both projections
            // finish, so metadata-only batch reshapes do not split its chain.
            y = core::reshape_tensor(ctx, y, TensorShape::from_dims({batch * steps, c.dim}));
            if (ctx.backend_type == core::BackendType::Cuda && b.ffn.fc1_weight.type == GGML_TYPE_F16) {
                y = core::wrap_tensor(ggml_cast(ctx.ggml, y.tensor, GGML_TYPE_F16), y.shape, GGML_TYPE_F16);
                modules::LinearConfig fc1{c.dim, c.ffn_dim, true};
                fc1.cuda_f16_output = true;
                y = modules::LinearModule(fc1).build(ctx, y, {b.ffn.fc1_weight, b.ffn.fc1_bias});
                y = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, y);
                modules::LinearConfig fc2{c.ffn_dim, c.dim, true};
                fc2.cuda_f16_output = true;
                y = modules::LinearModule(fc2).build(ctx, y, {b.ffn.fc2_weight, b.ffn.fc2_bias});
                // Modulation and the residual stream retain F32 arithmetic.
                y = core::wrap_tensor(ggml_cast(ctx.ggml, y.tensor, GGML_TYPE_F32), y.shape, GGML_TYPE_F32);
            } else {
                y = build_projection(ctx, y, {b.ffn.fc1_weight, b.ffn.fc1_bias}, true);
                y = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, y);
                y = build_projection(ctx, y, {b.ffn.fc2_weight, b.ffn.fc2_bias}, true);
            }
            y = core::reshape_tensor(ctx, y, TensorShape::from_dims({batch, steps, c.dim}));
            y = modules::MulModule(true).build(ctx, y, gate_ff);
            x = modules::AddModule().build(ctx, x, y);
        }
        // The head adds the time embedding itself, not the six-way time projection.
        const auto head_time = core::reshape_tensor(ctx, time, TensorShape::from_dims({1, 1, c.dim}));
        const auto modulation = modules::AddModule(false, true).build(ctx, head_modulation, head_time);
        const auto shift = modules::SliceModule({1, 0, 1}).build(ctx, modulation);
        const auto scale = modules::SliceModule({1, 1, 1}).build(ctx, modulation);
        x = adaptive_norm.build(ctx, x, {shift, scale});
        x = build_projection(ctx, x, head, true);
        // [B,S,p*C] -> [B,S*p,C] -> [B,C,S*p], preserving (patch,channel) order.
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
                                TensorShape::from_dims({batch, frames, c.latent_dim}));
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        const auto output = core::ensure_backend_addressable_layout(ctx, x);
        // Euler state remains in the persistent input buffer between submissions.
        state->step_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1}));
        ggml_set_input(state->step_input.tensor);
        auto velocity = output;
        if (batch == 2) {
            state->guidance_input = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1}));
            ggml_set_input(state->guidance_input.tensor);
            const auto positive = modules::SliceModule({0, 0, 1}).build(ctx, output);
            const auto negative = modules::SliceModule({0, 1, 1}).build(ctx, output);
            velocity = modules::FlowSamplerCfgGraphModule().build(ctx, positive, negative, state->guidance_input);
        }
        const auto rounding = batch == 1 ? modules::FlowSamplerGraphRounding::Bf16
                                         : modules::FlowSamplerGraphRounding::FullPrecision;
        const auto updated = modules::FlowSamplerEulerGraphModule({rounding, rounding})
            .build(ctx, latent_input, velocity, state->step_input);
        return {latent_input, output, updated};
        }, runtime_optimizations);
        denoiser = std::move(state);
        debug::timing_log_scalar("moss_soundeffect.dit.build_ms", debug::elapsed_ms(started));
    }

    std::vector<float> predict(const std::vector<float> & latents, float timestep, int64_t batch, bool negative) {
        const auto started = std::chrono::steady_clock::now();
        if (!prepared) throw std::runtime_error("MOSS SoundEffect DiT requires prepare_context before predict");
        if (latents.size() != static_cast<size_t>(assets->dit.latent_dim * frames))
            throw std::runtime_error("MOSS SoundEffect DiT latent shape mismatch");
        if (!denoiser || denoiser->batch != batch) build_denoiser_graph(batch);
        auto & state = *denoiser;
        const auto half = assets->dit.freq_dim / 2;
        std::vector<float> time(static_cast<size_t>(2 * half));
        for (int64_t i = 0; i < half; ++i) {
            const double angle = static_cast<double>(timestep) * std::pow(10000.0, -static_cast<double>(i) / half);
            time[static_cast<size_t>(i)] = static_cast<float>(std::cos(angle));
            time[static_cast<size_t>(i + half)] = static_cast<float>(std::sin(angle));
        }
        const auto upload_started = std::chrono::steady_clock::now();
        core::write_tensor_f32(state.latent_input, latents);
        core::write_tensor_f32(state.step_input, std::vector<float>{0.0F});
        if (batch == 2) core::write_tensor_f32(state.guidance_input, std::vector<float>{1.0F});
        core::write_tensor_f32(state.time_input, time);
        if (batch == 1) {
            const int32_t branch = negative ? 1 : 0;
            core::write_tensor_i32(state.branch_input, &branch, 1);
        }
        debug::timing_log_scalar("moss_soundeffect.dit.input_upload_ms", debug::elapsed_ms(upload_started));
        auto result = state.runtime->predict();
        debug::timing_log_scalar("moss_soundeffect.dit.wall_ms", debug::elapsed_ms(started));
        return result;
    }
};

DitRuntime::DitRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
                       assets::TensorStorageType storage_type, core::AttentionPreference preference, bool optimize)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), storage_type, preference, optimize)) {}

DitRuntime::~DitRuntime() = default;

void DitRuntime::prepare_context(const std::vector<float> & positive, const std::vector<float> & negative) {
    const auto started = std::chrono::steady_clock::now();
    auto & s = *impl_;
    s.prepared = false;
    const auto count = static_cast<size_t>(s.assets->text_length * s.assets->dit.text_dim);
    if (positive.size() != count || negative.size() != count)
        throw std::runtime_error("MOSS SoundEffect positive/negative context shape mismatch");
    const auto upload_started = std::chrono::steady_clock::now();
    core::write_tensor_f32_slice(s.text_input, 0, positive.data(), positive.size());
    core::write_tensor_f32_slice(s.text_input, count, negative.data(), negative.size());
    debug::timing_log_scalar("moss_soundeffect.dit.condition.input_upload_ms", debug::elapsed_ms(upload_started));
    s.condition_graph->compute();
    s.prepared = true;
    debug::timing_log_scalar("moss_soundeffect.dit.condition.wall_ms", debug::elapsed_ms(started));
}

std::vector<float> DitRuntime::predict(const std::vector<float> & latents, float timestep, bool negative) {
    return impl_->predict(latents, timestep, 1, negative);
}

std::vector<float> DitRuntime::sample(const std::vector<float> & noise, int steps, float guidance, float shift) {
    auto & s = *impl_;
    if (!s.prepared) throw std::runtime_error("MOSS SoundEffect DiT requires prepared conditioning");
    if (steps < 1 || !std::isfinite(guidance) || !std::isfinite(shift) || shift <= 0 ||
        noise.size() != static_cast<size_t>(s.assets->dit.latent_dim * s.frames))
        throw std::runtime_error("Invalid MOSS SoundEffect sampling inputs");
    const int batch = guidance == 1.0F ? 1 : 2;
    if (!s.denoiser || s.denoiser->batch != batch) s.build_denoiser_graph(batch);
    auto & state = *s.denoiser;
    if (batch == 2) core::write_tensor_f32(state.guidance_input, std::vector<float>{guidance});
    if (batch == 1) {
        const int32_t branch = 0;
        core::write_tensor_i32(state.branch_input, &branch, 1);
    }
    const auto half = s.assets->dit.freq_dim / 2;
    const size_t width = static_cast<size_t>(2 * half);
    // Keep upload sources alive until the queued sequence completes.
    std::vector<float> times(static_cast<size_t>(steps) * width);
    std::vector<float> deltas(static_cast<size_t>(steps));
    for (int step = 0; step < steps; ++step) {
        const float r = 1.0F - float(step) / float(steps);
        const float next_r = 1.0F - float(step + 1) / float(steps);
        const float sigma = shift * r / (1.0F + (shift - 1.0F) * r);
        const float next = shift * next_r / (1.0F + (shift - 1.0F) * next_r);
        for (int64_t i = 0; i < half; ++i) {
            const double angle = static_cast<double>(sigma * 1000.0F) *
                                 std::pow(10000.0, -static_cast<double>(i) / half);
            times[static_cast<size_t>(step) * width + static_cast<size_t>(i)] = static_cast<float>(std::cos(angle));
            times[static_cast<size_t>(step) * width + static_cast<size_t>(i + half)] = static_cast<float>(std::sin(angle));
        }
        deltas[static_cast<size_t>(step)] = next - sigma;
    }
    modules::FlowSamplerGraphSequence sequence;
    sequence.steps = static_cast<size_t>(steps);
    sequence.step_bindings = {
        {state.time_input, 0, width * sizeof(float), times.data(), times.size() * sizeof(float)},
        {state.step_input, 0, sizeof(float), deltas.data(), deltas.size() * sizeof(float)},
    };
    return state.runtime->sample(noise, sequence);
}

std::vector<float> DitRuntime::predict_cfg(const std::vector<float> & latents, float timestep) {
    return impl_->predict(latents, timestep, 2, false);
}

}  // namespace engine::models::moss_soundeffect
