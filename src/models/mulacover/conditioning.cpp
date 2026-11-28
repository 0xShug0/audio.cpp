#include "engine/models/mulacover/conditioning.h"

#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::mulacover {
using core::TensorShape;
using core::TensorValue;
using namespace modules;

namespace {
TensorValue round_activation(core::ModuleBuildContext & ctx, const TensorValue & value, ggml_type type) {
    if (type == GGML_TYPE_BF16) return Bf16RoundModule{}.build(ctx, value);
    if (type == GGML_TYPE_F16) return core::wrap_tensor(
        ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, value.tensor, type), GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
    return value;
}
}  // namespace

ConditioningWeights ConditioningWeights::bind(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    int64_t first_peft_layer, int64_t embed_dim, assets::TensorStorageType storage_type,
    const std::string & prefix) {
    if (embed_dim <= 0 || embed_dim % 16 != 0 || first_peft_layer < 0) {
        throw std::invalid_argument("MuLaCover requires an even head dimension and valid PEFT index");
    }
    ConditioningWeights result;
    // Llama3ScaledRoPE defaults: old context 8192, low/high factors 1/4,
    // scale 32. ggml expects divisors, not inverse frequencies. Each path
    // needs its own table, especially the non-power-of-two gate head (12).
    const std::array<int64_t, 3> dimensions{64, embed_dim / 8, 12};
    std::array<TensorValue *, 3> tables{&result.adaptor_rope_factors,
        &result.attention_rope_factors, &result.gate_rope_factors};
    for (size_t table = 0; table < tables.size(); ++table) {
        const auto dim = dimensions[table];
        std::vector<float> factors(static_cast<size_t>(dim / 2));
        for (int64_t i = 0; i < dim / 2; ++i) {
            const float frequency = 1.0f / std::pow(500000.0f, static_cast<float>(2 * i) / dim);
            const float wavelength = 2.0f * 3.14159265358979323846f / frequency;
            const float smooth = (8192.0f / wavelength - 1.0f) / 3.0f;
            factors[static_cast<size_t>(i)] = wavelength < 2048.0f ? 1.0f
                : wavelength > 8192.0f ? 32.0f : 1.0f / ((1.0f - smooth) / 32.0f + smooth);
        }
        *tables[table] = store.make_f32(TensorShape::from_dims({dim / 2}), std::move(factors));
    }
    for (size_t i = 0; i < result.layers.size(); ++i) {
        auto & layer = result.layers[i];
        const auto index = std::to_string(first_peft_layer + static_cast<int64_t>(i));
        const auto base = prefix + "peft_layers." + index;
        auto & adaptor = layer.adaptor;
        adaptor.pianoroll_proj = binding::linear_from_named_source(store, source,
            base + ".adaptor.pianoroll_proj.weight",
            base + ".adaptor.pianoroll_proj.bias", storage_type);
        adaptor.drum_pianoroll_proj = binding::linear_from_named_source(store, source,
            base + ".adaptor.drum_pianoroll_proj.weight",
            base + ".adaptor.drum_pianoroll_proj.bias", storage_type);
        adaptor.chord_proj = binding::linear_from_named_source(store, source,
            base + ".adaptor.chord_proj.weight",
            base + ".adaptor.chord_proj.bias", storage_type);
        adaptor.fusion_proj = binding::linear_from_named_source(store, source,
            base + ".adaptor.fusion_proj.weight",
            base + ".adaptor.fusion_proj.bias", storage_type);
        adaptor.out_proj = binding::linear_from_named_source(store, source,
            base + ".adaptor.out_proj.weight",
            base + ".adaptor.out_proj.bias", storage_type);
        for (size_t j = 0; j < adaptor.decoder.size(); ++j) {
            auto & decoder = adaptor.decoder[j];
            const auto path = base + ".adaptor.decoder.layers." + std::to_string(j);
            decoder.input_norm = binding::norm_from_named_source(store, source, path + ".sa_norm.scale", std::nullopt);
            decoder.post_norm = binding::norm_from_named_source(store, source, path + ".mlp_norm.scale", std::nullopt);
            decoder.self_attention.q_weight = binding::tensor_from_named_source(store, source,
                path + ".attn.q_proj.weight", storage_type);
            decoder.self_attention.k_weight = binding::tensor_from_named_source(store, source,
                path + ".attn.k_proj.weight", storage_type);
            decoder.self_attention.v_weight = binding::tensor_from_named_source(store, source,
                path + ".attn.v_proj.weight", storage_type);
            decoder.self_attention.out_weight = binding::tensor_from_named_source(store, source,
                path + ".attn.output_proj.weight", storage_type);
            decoder.mlp.gate_proj = binding::linear_from_named_source(store, source,
                path + ".mlp.w1.weight", std::nullopt, storage_type);
            decoder.mlp.up_proj = binding::linear_from_named_source(store, source,
                path + ".mlp.w3.weight", std::nullopt, storage_type);
            decoder.mlp.down_proj = binding::linear_from_named_source(store, source,
                path + ".mlp.w2.weight", std::nullopt, storage_type);
            decoder.rope_frequency_factors = result.adaptor_rope_factors;
        }
        adaptor.decoder_norm = binding::norm_from_named_source(store, source,
            base + ".adaptor.decoder.norm.scale", std::nullopt);
        auto & cross = layer.cross_attn;
        cross.norm = binding::norm_from_named_source(store, source, base + ".cross_attn.norm.scale", std::nullopt);
        cross.attn.q_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.attn.q_proj.weight", std::nullopt, storage_type);
        cross.attn.k_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.attn.k_proj.weight", std::nullopt, storage_type);
        cross.attn.v_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.attn.v_proj.weight", std::nullopt, storage_type);
        cross.attn.output_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.attn.output_proj.weight", std::nullopt, storage_type);
        cross.gate.q_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.gate.q_proj.weight", std::nullopt, storage_type);
        cross.gate.k_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.gate.k_proj.weight", std::nullopt, storage_type);
        cross.gate.v_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.gate.v_proj.weight", std::nullopt, storage_type);
        cross.gate.output_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.gate.output_proj.weight", std::nullopt, storage_type);
        cross.x_gate_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.x_gate_proj.weight",
            base + ".cross_attn.x_gate_proj.bias", storage_type);
        cross.context_gate_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.context_gate_proj.weight",
            base + ".cross_attn.context_gate_proj.bias", storage_type);
        cross.gate_proj = binding::linear_from_named_source(store, source,
            base + ".cross_attn.gate_proj.weight",
            base + ".cross_attn.gate_proj.bias", storage_type);
        layer.style_mlp = binding::linear_from_named_source(store, source,
            prefix + "tag_mlp_layers." + index + ".mlp.weight",
            prefix + "tag_mlp_layers." + index + ".mlp.bias", storage_type);
    }
    return result;
}

TensorValue SymbolicAdaptorModule::build(core::ModuleBuildContext & ctx,
    const TensorValue & pianoroll, const TensorValue & drum_pianoroll,
    const TensorValue & chord, const TensorValue & positions,
    const TensorValue & attention_mask, const SymbolicAdaptorWeights & weights) const {
    core::validate_rank_between(pianoroll, 3, 3, "MuLaCover pianoroll");
    const auto batch = pianoroll.shape.dims[0], steps = pianoroll.shape.dims[1];
    core::validate_rank_between(attention_mask, 4, 4, "adaptor mask");
    core::validate_shape(attention_mask, TensorShape::from_dims({
        attention_mask.shape.dims[0] == 1 ? 1 : batch, 1, steps, steps}), "adaptor mask");
    if (attention_mask.type != GGML_TYPE_F16 && attention_mask.type != GGML_TYPE_F32) {
        throw std::invalid_argument("MuLaCover adaptor mask must be F16 or F32");
    }
    auto flash_mask = core::ensure_backend_addressable_layout(ctx, attention_mask);
    if (flash_mask.type != GGML_TYPE_F16) {
        flash_mask = core::wrap_tensor(ggml_cast(ctx.ggml, flash_mask.tensor, GGML_TYPE_F16),
            flash_mask.shape, GGML_TYPE_F16);
    }
    auto piano = LinearModule({256, 24, true}).build(ctx, pianoroll, weights.pianoroll_proj);
    auto drums = LinearModule({128, 24, true}).build(ctx, drum_pianoroll, weights.drum_pianoroll_proj);
    auto chords = LinearModule({12, 24, true}).build(ctx, chord, weights.chord_proj);
    piano = round_activation(ctx, piano, compute_type_);
    drums = round_activation(ctx, drums, compute_type_);
    chords = round_activation(ctx, chords, compute_type_);
    auto fused = ConcatModule({2}).build(ctx, piano, drums);
    fused = ConcatModule({2}).build(ctx, fused, chords);
    auto hidden = LinearModule({72, 512, true}).build(ctx, fused, weights.fusion_proj);
    hidden = round_activation(ctx, hidden, compute_type_);
    DecoderLayerConfig config;
    config.hidden_size = 512;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 8;
    config.head_dim = 64;
    config.intermediate_size = 2048;
    config.rms_norm_eps = 1e-5f;
    config.rope_theta = 500000.0f;
    config.rope_type = GGML_ROPE_TYPE_NORMAL;
    config.use_qk_norm = false;
    config.runtime.attention.prefill_mode = DecoderAttentionMode::FlashGrouped;
    config.runtime.attention.allow_flash_attention = true;
    auto & cast = config.activation_cast;
    cast.enabled = compute_type_ != GGML_TYPE_F32;
    cast.type = compute_type_;
    cast.fused_round = compute_type_ == GGML_TYPE_BF16;
    cast.after_input_norm = cast.after_qkv_projection = cast.after_rope = true;
    cast.after_attention = cast.after_attention_output = cast.after_residual = true;
    cast.after_ffn_norm = cast.after_mlp_projection = cast.after_mlp_silu = true;
    cast.after_mlp_mul = cast.after_output = true;
    if (cast.enabled) cast.norm_intermediate_type = compute_type_;
    for (const auto & layer : weights.decoder) {
        hidden = DecoderLayerModule(config).build(ctx, hidden, positions, layer,
            std::nullopt, std::nullopt, flash_mask).output;
    }
    hidden = RMSNormModule({512, 1e-5f, true, false, false, false,
        compute_type_ == GGML_TYPE_F32 ? std::nullopt : std::optional<ggml_type>(compute_type_)})
        .build(ctx, hidden, weights.decoder_norm);
    hidden = round_activation(ctx, hidden, compute_type_);
    return round_activation(ctx, LinearModule({512, embed_dim_, true}).build(ctx, hidden, weights.out_proj), compute_type_);
}

CrossAttentionRoPEModule::CrossAttentionRoPEModule(int64_t embed_dim, int64_t heads, ggml_type compute_type)
    : embed_dim_(embed_dim), heads_(heads), compute_type_(compute_type) {
    if (heads <= 0 || embed_dim <= 0 || embed_dim % heads != 0 || (embed_dim / heads) % 2 != 0) {
        throw std::invalid_argument("MuLaCover cross attention requires an even rotary head dimension");
    }
}

ProjectedCrossAttention CrossAttentionRoPEModule::build_context(core::ModuleBuildContext & ctx,
    const TensorValue & context, const TensorValue & positions,
    const TensorValue & rope_factors, const CrossAttentionRoPEWeights & weights) const {
    core::validate_rank_between(context, 3, 3, "MuLaCover context");
    const auto batch = context.shape.dims[0], steps = context.shape.dims[1];
    const auto head_dim = embed_dim_ / heads_;
    auto key = LinearModule({embed_dim_, embed_dim_, false}).build(ctx, context, weights.k_proj);
    auto value = LinearModule({embed_dim_, embed_dim_, false}).build(ctx, context, weights.v_proj);
    key = round_activation(ctx, key, compute_type_);
    value = round_activation(ctx, value, compute_type_);
    key = ReshapeModule({TensorShape::from_dims({batch, steps, heads_, head_dim})}).build(ctx, key);
    value = ReshapeModule({TensorShape::from_dims({batch, steps, heads_, head_dim})}).build(ctx, value);
    key = RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 500000.0f}).build(ctx, key, positions, &rope_factors);
    key = round_activation(ctx, key, compute_type_);
    return {TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, key),
        TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value)};
}

TensorValue CrossAttentionRoPEModule::build(core::ModuleBuildContext & ctx,
    const TensorValue & query, const TensorValue & positions,
    const std::optional<TensorValue> & attention_mask, const ProjectedCrossAttention & context,
    const TensorValue & rope_factors, const CrossAttentionRoPEWeights & weights) const {
    core::validate_rank_between(query, 3, 3, "MuLaCover query");
    const auto batch = query.shape.dims[0], steps = query.shape.dims[1];
    const auto head_dim = embed_dim_ / heads_;
    auto mask = attention_mask;
    if (mask) {
        core::validate_rank_between(*mask, 4, 4, "cross attention mask");
        core::validate_shape(*mask, TensorShape::from_dims({
            mask->shape.dims[0] == 1 ? 1 : batch, 1, steps, context.key.shape.dims[2]}),
            "cross attention mask");
        if (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
            throw std::invalid_argument("MuLaCover cross attention mask must be F16 or F32");
        }
        *mask = core::ensure_backend_addressable_layout(ctx, *mask);

    }
    auto q = LinearModule({embed_dim_, embed_dim_, false}).build(ctx, query, weights.q_proj);
    q = round_activation(ctx, q, compute_type_);
    q = ReshapeModule({TensorShape::from_dims({batch, steps, heads_, head_dim})}).build(ctx, q);
    q = RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 500000.0f}).build(ctx, q, positions, &rope_factors);
    q = round_activation(ctx, q, compute_type_);
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    auto output = ScaledDotProductAttentionModule({head_dim,
        ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_DEFAULT, AttentionCausality::NonCausal})
        .build(ctx, q, context.key, context.value, mask);
    output = round_activation(ctx, output, compute_type_);
    output = core::ensure_backend_addressable_layout(ctx, output);
    output = ReshapeModule({TensorShape::from_dims({batch, steps, embed_dim_})}).build(ctx, output);
    return round_activation(ctx, LinearModule({embed_dim_, embed_dim_, false}).build(ctx, output, weights.output_proj), compute_type_);
}

ConditioningModule::ConditioningModule(int64_t embed_dim, int64_t style_dim, ggml_type compute_type)
    : embed_dim_(embed_dim), style_dim_(style_dim), compute_type_(compute_type) {
    if (embed_dim <= 0 || embed_dim % 16 != 0 || style_dim <= 0) {
        throw std::invalid_argument("MuLaCover conditioning dimensions are invalid");
    }
}

PreparedConditioning ConditioningModule::build_context(core::ModuleBuildContext & ctx,
    const TensorValue & pianoroll, const TensorValue & drum_pianoroll,
    const TensorValue & chord, const TensorValue & context_positions,
    const TensorValue & adaptor_mask, const TensorValue & style,
    const ConditioningWeights & weights) const {
    core::validate_rank_between(pianoroll, 3, 3, "MuLaCover pianoroll");
    core::validate_shape(style, TensorShape::from_dims({pianoroll.shape.dims[0], 1, style_dim_}), "MuLaCover style");
    PreparedConditioning result;
    TensorValue context;
    const auto activated_style = SiluModule().build(ctx, style);
    for (size_t i = 0; i < result.layers.size(); ++i) {
        auto & prepared = result.layers[i];
        const auto & layer = weights.layers[i];
        auto current = SymbolicAdaptorModule(embed_dim_, compute_type_).build(ctx, pianoroll, drum_pianoroll,
            chord, context_positions, adaptor_mask, layer.adaptor);
        context = i == 0 ? current : AddModule().build(ctx, context, current);
        context = round_activation(ctx, context, compute_type_);
        auto gate_context = LinearModule({embed_dim_, 12, true})
            .build(ctx, context, layer.cross_attn.context_gate_proj);
        gate_context = round_activation(ctx, gate_context, compute_type_);
        prepared.attention = CrossAttentionRoPEModule(embed_dim_, 8, compute_type_).build_context(ctx,
            context, context_positions, weights.attention_rope_factors, layer.cross_attn.attn);
        prepared.gate = CrossAttentionRoPEModule(12, 1, compute_type_).build_context(ctx,
            gate_context, context_positions, weights.gate_rope_factors, layer.cross_attn.gate);
        auto affine = LinearModule({style_dim_, 2 * embed_dim_, true}).build(ctx, round_activation(ctx, activated_style, compute_type_), layer.style_mlp);
        affine = round_activation(ctx, affine, compute_type_);
        prepared.style_scale = SliceModule({2, 0, embed_dim_}).build(ctx, affine);
        prepared.style_shift = SliceModule({2, embed_dim_, embed_dim_}).build(ctx, affine);
    }
    return result;
}

TensorValue ConditioningModule::build_cross_attention(core::ModuleBuildContext & ctx,
    const TensorValue & query, const TensorValue & positions,
    const std::optional<TensorValue> & attention_mask, const PreparedConditioningLayer & prepared,
    const CrossAttentionBlockWeights & weights, const TensorValue & attention_rope_factors,
    const TensorValue & gate_rope_factors) const {
    auto normalized = RMSNormModule({embed_dim_, 1e-5f, true, false, false, false,
        compute_type_ == GGML_TYPE_F32 ? std::nullopt : std::optional<ggml_type>(compute_type_)})
        .build(ctx, query, weights.norm);
    normalized = round_activation(ctx, normalized, compute_type_);
    auto attention = CrossAttentionRoPEModule(embed_dim_, 8, compute_type_).build(ctx, normalized, positions,
        attention_mask, prepared.attention, attention_rope_factors, weights.attn);
    auto gate_query = LinearModule({embed_dim_, 12, true}).build(ctx, normalized, weights.x_gate_proj);
    gate_query = round_activation(ctx, gate_query, compute_type_);
    auto gate = CrossAttentionRoPEModule(12, 1, compute_type_).build(ctx, gate_query, positions,
        attention_mask, prepared.gate, gate_rope_factors, weights.gate);
    gate = LinearModule({12, 1, true}).build(ctx, gate, weights.gate_proj);
    gate = round_activation(ctx, gate, compute_type_);
    gate = round_activation(ctx, TanhModule().build(ctx, gate), compute_type_);
    auto delta = round_activation(ctx, MulModule(true).build(ctx, attention, gate), compute_type_);
    return round_activation(ctx, AddModule().build(ctx, query, delta), compute_type_);
}

struct ConditioningRuntime::Impl {
    struct RequestGraph {
        core::ExecutionContext & execution;
        int64_t steps, batch;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> output_context{nullptr, ggml_free};
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
            output_buffer{nullptr, ggml_backend_buffer_free};
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        core::HostGraphPlan host_plan;
        core::ConstantTensorCache constants;
        ggml_cgraph * graph = nullptr;
        TensorValue pianoroll, drums, chord, style;
        PreparedConditioning detached;
        bool ready = false;

        RequestGraph(Impl & runtime, int64_t length, int64_t batch_size)
            : execution(runtime.execution), steps(length), batch(batch_size),
              constants(execution.backend(), execution.config().threads,
                  "mulacover.conditioning.constants", 1024ull * 1024ull) {
            context.reset(ggml_init({64ull * 1024ull * 1024ull, nullptr, true}));
            output_context.reset(ggml_init({1024ull * 1024ull, nullptr, true}));
            if (!context || !output_context) {
                throw std::runtime_error("failed to initialize MuLaCover conditioning graph contexts");
            }
            core::ModuleBuildContext ctx{context.get(), "mulacover.conditioning.prepare", execution.backend_type()};
            pianoroll = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, steps, 256}));
            drums = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, steps, 128}));
            chord = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, steps, 12}));
            style = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 1, runtime.style_dim}));
            for (const auto & input : {pianoroll, drums, chord, style}) {
                ggml_set_input(input.tensor);
            }
            std::vector<int32_t> position_values(static_cast<size_t>(steps));
            for (int64_t i = 0; i < steps; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            constants.begin_graph();
            auto positions = constants.make_tensor(TensorShape::from_dims({steps}), GGML_TYPE_I32,
                position_values.data(), position_values.size() * sizeof(int32_t));
            // DecoderLayerModule needs a mask to select its noncausal flash
            // prefill path. One immutable F16 mask broadcasts across all batches
            // and heads and is shared by every adaptor layer.
            std::vector<ggml_fp16_t> zero_mask(static_cast<size_t>(steps * steps), 0);
            auto mask = constants.make_tensor(TensorShape::from_dims({1, 1, steps, steps}), GGML_TYPE_F16,
                zero_mask.data(), zero_mask.size() * sizeof(ggml_fp16_t));
            auto prepared = runtime.module.build_context(ctx, pianoroll, drums, chord, positions,
                mask, style, runtime.weights);
            constants.finish_graph();
            constants.ensure_uploaded();

            // Output descriptors belong to a different context and have no op,
            // view_src, or src dependencies. The copy nodes belong ONLY to this
            // precompute graph; backbone graphs see independently allocated leaves.
            graph = ggml_new_graph_custom(context.get(), 65536, false);
            core::ModuleBuildContext output_ctx{output_context.get(), "mulacover.conditioning.cache", execution.backend_type()};
            for (size_t i = 0; i < detached.layers.size(); ++i) {
                auto & target = detached.layers[i];
                const auto & source = prepared.layers[i];
                const std::array<TensorValue, 6> sources{
                    source.style_scale, source.style_shift, source.attention.key, source.attention.value,
                    source.gate.key, source.gate.value};
                const std::array<TensorValue *, 6> targets{
                    &target.style_scale, &target.style_shift, &target.attention.key, &target.attention.value,
                    &target.gate.key, &target.gate.value};
                for (size_t j = 0; j < sources.size(); ++j) {
                    *targets[j] = core::make_tensor(output_ctx,
                        j < 2 ? sources[j].type : runtime.cache_type, sources[j].shape);
                    auto * copy = ggml_cpy(context.get(), sources[j].tensor, targets[j]->tensor);
                    ggml_set_output(copy);
                    ggml_build_forward_expand(graph, copy);
                }
            }
            output_buffer.reset(ggml_backend_alloc_ctx_tensors(output_context.get(), execution.backend()));
            if (!output_buffer) {
                throw std::runtime_error("failed to allocate MuLaCover detached conditioning outputs");
            }
            core::validate_backend_graph_supported(execution.backend(), graph, "mulacover.conditioning.prepare");
            debug::timing_log_scalar("mulacover.conditioning.cache.buffer_mb",
                static_cast<double>(ggml_backend_buffer_get_size(output_buffer.get())) / (1024.0 * 1024.0));
        }

        void allocate_workspace() {
            if (allocator) return;
            allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            if (!allocator || !ggml_gallocr_reserve(allocator.get(), graph) ||
                !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                throw std::runtime_error("failed to allocate MuLaCover conditioning graph workspace");
            }
            core::prepare_host_graph_plan(execution, graph, host_plan);
            debug::timing_log_scalar("mulacover.conditioning.prepare.graph_buffer_mb",
                static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024.0 * 1024.0));
        }

        void release_workspace() {
            host_plan.reset();
            if (!allocator) return;
            core::release_backend_graph_resources(execution.backend(), graph, true);
            allocator.reset();
            for (auto * tensor = ggml_get_first_tensor(context.get()); tensor;
                 tensor = ggml_get_next_tensor(context.get(), tensor)) {
                tensor->data = nullptr;
                tensor->buffer = nullptr;
                tensor->extra = nullptr;
            }
            debug::timing_log_scalar("mulacover.conditioning.prepare.graph_buffer_mb", 0.0);
        }

        ~RequestGraph() { release_workspace(); }
    };

    core::ExecutionContext & execution;
    std::shared_ptr<const assets::TensorSource> source;
    int64_t first_layer, style_dim;
    bool train_tag;
    core::BackendWeightStore store;
    ConditioningWeights weights;
    ConditioningModule module;
    ggml_type cache_type;
    std::unique_ptr<RequestGraph> request;
    uint64_t revision = 0;

    Impl(core::ExecutionContext & execution_context, std::shared_ptr<const assets::TensorSource> tensors,
        int64_t first_peft_layer, int64_t style_dimension, bool enable_style,
        assets::TensorStorageType storage_type, ggml_type projected_cache_type)
        : execution(execution_context), source(std::move(tensors)), first_layer(first_peft_layer),
          style_dim(style_dimension), train_tag(enable_style),
          store(execution.backend(), execution.backend_type(), "mulacover.conditioning.weights",
              16ull * 1024ull * 1024ull), module(3072, style_dim, projected_cache_type),
          cache_type(projected_cache_type) {
        if (cache_type != GGML_TYPE_F32 && cache_type != GGML_TYPE_F16 && cache_type != GGML_TYPE_BF16) {
            throw std::invalid_argument("MuLaCover conditioning cache requires F32, F16 or BF16 storage");
        }
        if (!source || first_layer < 0) {
            throw std::invalid_argument("MuLaCover conditioning requires a tensor source and valid first layer");
        }
        weights = ConditioningWeights::bind(store, *source, first_layer, 3072, storage_type);
        store.upload();
    }
};

ConditioningRuntime::ConditioningRuntime(core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source, int64_t first_peft_layer,
    int64_t style_dim, bool train_tag, assets::TensorStorageType storage_type, ggml_type cache_type)
    : impl_(std::make_unique<Impl>(execution, std::move(source), first_peft_layer,
          style_dim, train_tag, storage_type, cache_type)) {}

ConditioningRuntime::~ConditioningRuntime() = default;

bool ConditioningRuntime::prepare(const SymbolicInputs & symbolic,
    const std::vector<float> & qwen_embedding, bool cfg_enabled) {
    const auto started = std::chrono::steady_clock::now();
    if (symbolic.length <= 0 || symbolic.length > 5000 ||
        symbolic.pianoroll.size() != static_cast<size_t>(symbolic.length * 256) ||
        symbolic.drum_pianoroll.size() != static_cast<size_t>(symbolic.length * 128) ||
        symbolic.chord.size() != static_cast<size_t>(symbolic.length * 12) ||
        qwen_embedding.size() != static_cast<size_t>(impl_->style_dim)) {
        throw std::invalid_argument("MuLaCover conditioning input lengths or style dimension do not match");
    }
    const int64_t batch = cfg_enabled ? 2 : 1;
    const bool changed = !impl_->request || impl_->request->steps != symbolic.length || impl_->request->batch != batch;
    std::unique_ptr<Impl::RequestGraph> replacement;
    if (changed) {
        replacement = std::make_unique<Impl::RequestGraph>(*impl_, symbolic.length, batch);
    }
    auto & request = changed ? *replacement : *impl_->request;
    request.ready = false;
    request.allocate_workspace();
    const std::array<const std::vector<float> *, 4> payloads{
        &symbolic.pianoroll, &symbolic.drum_pianoroll, &symbolic.chord, &qwen_embedding};
    const std::array<TensorValue, 4> inputs{request.pianoroll, request.drums, request.chord, request.style};
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Upstream orders CFG rows [conditional, unconditional]. Mask the raw
        // features before projections/adaptor, including the raw Qwen style.
        std::vector<float> values(payloads[i]->size() * static_cast<size_t>(batch), -1.0f);
        std::copy(payloads[i]->begin(), payloads[i]->end(), values.begin());
        core::write_tensor_f32(inputs[i], values);
    }
    core::set_backend_threads(impl_->execution.backend(), impl_->execution.config().threads);
    const auto status = core::compute_graph(impl_->execution, request.graph, request.host_plan,
        "mulacover.conditioning.prepare");
    ggml_backend_synchronize(impl_->execution.backend());
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MuLaCover conditioning precompute failed");
    }
    request.release_workspace();
    request.ready = true;
    if (changed) {
        impl_->request = std::move(replacement);
        ++impl_->revision;
    }
    debug::timing_log_scalar("mulacover.conditioning.prepare_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return changed;
}

uint64_t ConditioningRuntime::revision() const noexcept { return impl_->revision; }
int64_t ConditioningRuntime::context_length() const noexcept { return impl_->request ? impl_->request->steps : 0; }
int64_t ConditioningRuntime::batch_size() const noexcept { return impl_->request ? impl_->request->batch : 0; }

const PreparedConditioning & ConditioningRuntime::prepared() const {
    if (!impl_->request || !impl_->request->ready) {
        throw std::runtime_error("MuLaCover conditioning must be prepared before backbone graph construction");
    }
    return impl_->request->detached;
}

TensorValue ConditioningRuntime::before_layer(core::ModuleBuildContext & ctx,
    const TensorValue & input, const TensorValue & positions, size_t layer_index) {
    (void)positions;
    if (layer_index < static_cast<size_t>(impl_->first_layer) ||
        layer_index >= static_cast<size_t>(impl_->first_layer) + 8 || !impl_->train_tag) {
        return input;
    }
    const auto & layer = prepared().layers[layer_index - static_cast<size_t>(impl_->first_layer)];
    core::validate_rank_between(input, 3, 3, "MuLaCover backbone input");
    if (input.shape.dims[0] != batch_size()) {
        throw std::invalid_argument("MuLaCover conditioning and backbone batch sizes differ");
    }
    auto scale = ScaleBiasModule({1.0f, 1.0f}).build(ctx, layer.style_scale);
    scale = round_activation(ctx, scale, impl_->cache_type);
    auto styled = round_activation(ctx, MulModule(true).build(ctx, input, scale), impl_->cache_type);
    return round_activation(ctx, AddModule(false, true).build(ctx, styled, layer.style_shift), impl_->cache_type);
}

TensorValue ConditioningRuntime::after_layer(core::ModuleBuildContext & ctx,
    const TensorValue & output, const TensorValue & positions, size_t layer_index) {
    if (layer_index < static_cast<size_t>(impl_->first_layer) ||
        layer_index >= static_cast<size_t>(impl_->first_layer) + 8) {
        return output;
    }
    const auto index = layer_index - static_cast<size_t>(impl_->first_layer);
    const auto & layer = prepared().layers[index];
    core::validate_rank_between(output, 3, 3, "MuLaCover backbone output");
    if (output.shape.dims[0] != batch_size()) {
        throw std::invalid_argument("MuLaCover conditioning and backbone batch sizes differ");
    }
    // Both paths are explicitly noncausal. No all-zero query mask tensor or
    // mask cache is needed, and no precompute dependency enters this graph.
    return impl_->module.build_cross_attention(ctx, output, positions, std::nullopt,
        layer, impl_->weights.layers[index].cross_attn, impl_->weights.attention_rope_factors,
        impl_->weights.gate_rope_factors);
}

}  // namespace engine::models::mulacover
