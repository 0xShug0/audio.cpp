#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/errors.h"
#include <ggml-cpp.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

using core::TensorShape;
using core::TensorValue;

// Backing tensors belong to the runtime, outside every graph/scratch context.
struct LayerState {
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    ggml_tensor * conv_tail = nullptr;
};

struct CachedBuild {
    const std::vector<LayerState> & layers;
    ggml_tensor * slots;
    int64_t capacity;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> tail_updates;
};

// The upstream zero-tail RoPE rotates pairs (i, i + head_dim/2), with
// theta^(-2*i/head_dim) even when only a prefix of those pairs is rotated.
// RoPEModule supplies that progression. Restore each half's unrotated tail
// rather than changing its dimensions (which would change the frequencies).
TensorValue rotary(
    core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & positions,
    const LfmDecoderRopeConfig & config,
    const TensorValue * frequency_factors) {
    const int64_t dim = input.shape.last_dim();
    const RoPEModule rope({dim, GGML_ROPE_TYPE_NEOX, config.theta});
    if (frequency_factors != nullptr) {
        if (frequency_factors->type != GGML_TYPE_F32) {
            throw std::invalid_argument("LfmDecoder RoPE frequency factors must be F32");
        }
        return rope.build(ctx, input, positions, frequency_factors);
    }
    const int64_t rotated = config.rotary_dims < 0 ? dim : config.rotary_dims;
    if (rotated == 0) {
        return input;
    }
    auto output = rope.build(ctx, input, positions);
    if (rotated == dim) {
        return output;
    }
    const int64_t half = dim / 2;
    const int64_t prefix = rotated / 2;
    const auto piece = [&](ggml_tensor * source, int64_t offset, int64_t width) {
        return ggml_view_4d(ctx.ggml, source, width, source->ne[1], source->ne[2], source->ne[3],
                            source->nb[1], source->nb[2], source->nb[3], offset * source->nb[0]);
    };
    auto * first = ggml_concat(ctx.ggml, piece(output.tensor, 0, prefix), piece(input.tensor, prefix, half - prefix), 0);
    auto * second = ggml_concat(ctx.ggml, piece(output.tensor, half, prefix), piece(input.tensor, half + prefix, half - prefix), 0);
    return core::wrap_tensor(ggml_concat(ctx.ggml, first, second, 0), input.shape);
}

TensorValue attention(
    core::ModuleBuildContext & ctx,
    const TensorValue & hidden,
    const LfmDecoderInputs & inputs,
    const LfmDecoderAttentionWeights & weights,
    LfmDecoderLayerKind kind,
    const LfmDecoderConfig & config,
    const TensorValue * frequency_factors,
    const LayerState * state = nullptr, ggml_tensor * slots = nullptr) {
    const int64_t batch = hidden.shape.dims[0];
    const int64_t steps = hidden.shape.dims[1];
    const int64_t heads = config.num_attention_heads;
    const int64_t kv_heads = config.num_key_value_heads;
    const int64_t dim = config.head_dim;
    const auto project = [&](const LinearWeights & weight, int64_t count) {
        auto result = LinearModule({config.hidden_size, count * dim, false, config.projection_precision}).build(ctx, hidden, weight);
        return core::reshape_tensor(ctx, result, TensorShape::from_dims({batch, steps, count, dim}));
    };
    NormConfig norm_config{dim, config.norm_eps, true, false};
    norm_config.normalized_round_type = config.normalized_round_type;
    const RMSNormModule norm(norm_config);
    auto q = norm.build(ctx, project(weights.q_proj, heads), weights.q_norm);
    auto k = norm.build(ctx, project(weights.k_proj, kv_heads), weights.k_norm);
    auto v = project(weights.v_proj, kv_heads);
    const bool full = kind == LfmDecoderLayerKind::FullAttention;
    const auto & rope = full ? config.full_rope : config.sliding_rope;
    q = rotary(ctx, q, inputs.positions, rope, frequency_factors);
    k = rotary(ctx, k, inputs.positions, rope, frequency_factors);

    if (state) {
        // Cache RoPE-applied K and projected V. SET_ROWS retains backend residency
        // and makes the attention read depend on the append, without a host copy.
        const int64_t capacity = state->key->ne[1];
        const auto cache_rows = [&](const TensorValue & rows, ggml_tensor * cache) {
            auto * flat = ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, rows.tensor), kv_heads * dim, steps);
            auto * updated = ggml_set_rows(ctx.ggml, cache, flat, slots);
            return core::wrap_tensor(ggml_reshape_4d(ctx.ggml, updated, dim, kv_heads, capacity, 1),
                                    TensorShape::from_dims({1, capacity, kv_heads, dim}));
        };
        k = cache_rows(k, state->key);
        v = cache_rows(v, state->value);
    }

    const TransposeModule heads_first({{0, 2, 1, 3}, 4});
    auto context = GroupedQueryAttentionModule({dim, config.attention_lowering, config.attention_precision,
                                               AttentionCausality::NonCausal})
                       .build(ctx, heads_first.build(ctx, q), heads_first.build(ctx, k), heads_first.build(ctx, v),
                              full ? inputs.full_mask : inputs.sliding_mask);
    if (config.attention_gate) {
        auto gate = LinearModule({config.hidden_size, heads, false, config.projection_precision})
                        .build(ctx, hidden, *weights.gate_proj);
        gate = SigmoidModule{}.build(ctx, gate);
        gate = core::reshape_tensor(ctx, gate, TensorShape::from_dims({batch, steps, heads, 1}));
        // Broadcast one scalar per head without allocating a repeated gate.
        context = core::wrap_tensor(ggml_mul(ctx.ggml, context.tensor, gate.tensor), context.shape);
    }
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, TensorShape::from_dims({batch, steps, heads * dim}));
    return LinearModule({heads * dim, config.hidden_size, false, config.projection_precision})
        .build(ctx, context, weights.out_proj);
}

// The shared depthwise module consumes channel-first activations. Explicit left
// padding makes its cross-correlation causal without adding a right-hand tail.
TensorValue short_conv(
    core::ModuleBuildContext & ctx,
    TensorValue hidden,
    const LfmDecoderInputs & inputs,
    const LfmDecoderConvWeights & weights,
    const LfmDecoderConfig & config,
    const LayerState * state = nullptr, CachedBuild * cached = nullptr) {
    const int64_t batch = hidden.shape.dims[0];
    const int64_t steps = hidden.shape.dims[1];
    const int64_t width = config.hidden_size;
    const int64_t kernel = config.conv_kernel_size;
    std::optional<TensorValue> keep;
    if (inputs.conv_padding_mask) {
        keep = core::reshape_tensor(ctx, *inputs.conv_padding_mask, TensorShape::from_dims({batch, steps, 1}));
        hidden = core::wrap_tensor(ggml_mul(ctx.ggml, hidden.tensor, keep->tensor), hidden.shape);
    }
    auto packed = LinearModule({width, 3 * width, config.conv_bias, config.projection_precision})
                      .build(ctx, hidden, weights.in_proj);
    const auto b = SliceModule({2, 0, width}).build(ctx, packed);
    const auto c = SliceModule({2, width, width}).build(ctx, packed);
    const auto x = SliceModule({2, 2 * width, width}).build(ctx, packed);
    auto bx = MulModule{}.build(ctx, b, x);
    if (keep) {
        bx = core::wrap_tensor(ggml_mul(ctx.ggml, bx.tensor, keep->tensor), bx.shape);
    }
    TensorValue padded;
    if (state && state->conv_tail) {
        auto * joined = ggml_concat(ctx.ggml, state->conv_tail, bx.tensor, 1);
        padded = core::wrap_tensor(joined, TensorShape::from_dims({batch, steps + kernel - 1, width}));
        auto * tail = ggml_cont(ctx.ggml, ggml_view_2d(ctx.ggml, joined, width, kernel - 1,
                                                     joined->nb[1], steps * joined->nb[1]));
        cached->tail_updates.emplace_back(tail, state->conv_tail);
    } else {
        padded = core::wrap_tensor(
            ggml_pad_ext(ctx.ggml, bx.tensor, 0, 0, static_cast<int>(kernel - 1), 0, 0, 0, 0, 0),
            TensorShape::from_dims({batch, steps + kernel - 1, width}));
    }
    const TransposeModule channels_first({{0, 2, 1}, 3});
    auto output = DepthwiseConv1dModule({width, kernel, 1, 0, 1, config.conv_bias})
        .build(ctx, channels_first.build(ctx, padded), {weights.kernel, weights.bias});
    output = channels_first.build(ctx, output);
    output = MulModule{}.build(ctx, c, output);
    return LinearModule({width, width, config.conv_bias, config.projection_precision}).build(ctx, output, weights.out_proj);
}

}  // namespace

static void validate_config(const LfmDecoderConfig & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.layers.empty() ||
        !std::isfinite(config.norm_eps) || config.norm_eps <= 0) {
        throw std::invalid_argument("LfmDecoder requires positive hidden/intermediate dimensions, layers and norm epsilon");
    }
    for (const auto kind : config.layers) {
        if (kind == LfmDecoderLayerKind::ShortConv) {
            if (config.conv_kernel_size <= 0) {
                throw std::invalid_argument("LfmDecoder short convolution requires a positive kernel size");
            }
        } else if (kind == LfmDecoderLayerKind::FullAttention || kind == LfmDecoderLayerKind::SlidingAttention) {
            if (config.head_dim <= 0 || config.head_dim % 2 != 0 || config.num_attention_heads <= 0 ||
                config.num_key_value_heads <= 0 || config.num_attention_heads % config.num_key_value_heads != 0) {
                throw std::invalid_argument("LfmDecoder attention requires an even head dimension and query heads divisible by KV heads");
            }
            const auto & rope = kind == LfmDecoderLayerKind::FullAttention ? config.full_rope : config.sliding_rope;
            if (!std::isfinite(rope.theta) || rope.theta <= 0 || rope.rotary_dims < -1 ||
                rope.rotary_dims > config.head_dim || (rope.rotary_dims >= 0 && rope.rotary_dims % 2 != 0)) {
                throw std::invalid_argument("LfmDecoder RoPE requires positive theta and even rotary dimensions within the head");
            }
        } else {
            throw std::invalid_argument("Unknown LfmDecoder layer kind");
        }
    }
}

static LfmDecoderOutputs build_stack(
    core::ModuleBuildContext & ctx,
    const LfmDecoderInputs & inputs,
    const LfmDecoderWeights & weights, const LfmDecoderConfig & config, CachedBuild * cached = nullptr) {
    validate_config(config);
    if (ctx.ggml == nullptr) {
        throw std::invalid_argument("LfmDecoder requires a graph context");
    }
    core::validate_rank_between(inputs.hidden, 3, 3, "hidden");
    core::validate_last_dim(inputs.hidden, config.hidden_size, "hidden");
    if (inputs.hidden.type != GGML_TYPE_F32 || inputs.hidden.shape.dims[0] <= 0 || inputs.hidden.shape.dims[1] <= 0) {
        throw std::invalid_argument("LfmDecoder hidden must be nonempty F32 [batch, steps, hidden_size]");
    }
    if (weights.layers.size() != config.layers.size()) {
        throw std::invalid_argument("LfmDecoder requires one set of weights per configured layer");
    }
    const int64_t batch = inputs.hidden.shape.dims[0];
    const int64_t steps = inputs.hidden.shape.dims[1];
    core::validate_shape(inputs.positions, TensorShape::from_dims({steps}), "positions");
    if (inputs.positions.type != GGML_TYPE_I32) { throw std::invalid_argument("LFM positions must be I32"); }
    for (const auto index : inputs.capture_layers) {
        if (index < 0 || static_cast<size_t>(index) >= config.layers.size()) {
            throw std::invalid_argument("LfmDecoder capture layer index is out of range");
        }
    }
    for (const auto * mask : {&inputs.full_mask, &inputs.sliding_mask}) {
        if (*mask) {
            core::validate_shape(**mask, TensorShape::from_dims({steps, cached ? cached->capacity : steps}), "attention mask");
            if ((*mask)->type != GGML_TYPE_F32 && (*mask)->type != GGML_TYPE_F16) {
                throw std::invalid_argument("LfmDecoder attention masks must be additive F32 or F16 tensors");
            }
        }
    }
    if (inputs.conv_padding_mask) {
        core::validate_shape(*inputs.conv_padding_mask, TensorShape::from_dims({batch, steps}), "conv padding mask");
        if (inputs.conv_padding_mask->type != GGML_TYPE_F32) {
            throw std::invalid_argument("LfmDecoder convolution padding mask must be F32");
        }
    }
    NormConfig norm_config{config.hidden_size, config.norm_eps, true, false};
    norm_config.normalized_round_type = config.normalized_round_type;
    const RMSNormModule norm(norm_config);
    GatedFeedForwardConfig ff_config;
    ff_config.hidden_size = config.hidden_size;
    ff_config.intermediate_size = config.intermediate_size;
    ff_config.activation = GatedFeedForwardActivation::Silu;
    ff_config.projection_precision = config.projection_precision;
    const GatedFeedForwardModule feed_forward(ff_config);
    LfmDecoderOutputs result;
    result.captured_hidden.resize(inputs.capture_layers.size());
    auto hidden = inputs.hidden;
    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const auto & w = weights.layers[layer];
        if (w.post_operator_norm.has_value() != config.sandwich_norm || w.post_ffn_norm.has_value() != config.sandwich_norm) {
            throw std::invalid_argument("LfmDecoder sandwich norm weights must match the architecture");
        }
        auto normalized = norm.build(ctx, hidden, w.operator_norm);
        TensorValue operated;
        if (config.layers[layer] == LfmDecoderLayerKind::ShortConv) {
            core::validate_shape(w.conv.kernel, TensorShape::from_dims({config.hidden_size, 1, config.conv_kernel_size}), "conv kernel");
            if (w.conv.kernel.type != GGML_TYPE_F32) {
                throw std::invalid_argument("LfmDecoder resolved depthwise kernel must be F32");
            }
            if (w.conv.bias.has_value() != config.conv_bias || w.conv.in_proj.bias.has_value() != config.conv_bias ||
                w.conv.out_proj.bias.has_value() != config.conv_bias) {
                throw std::invalid_argument("LfmDecoder convolution biases must match the architecture");
            }
            if (w.conv.bias) {
                core::validate_shape(*w.conv.bias, TensorShape::from_dims({config.hidden_size}), "conv bias");
                if (w.conv.bias->type != GGML_TYPE_F32) {
                    throw std::invalid_argument("LfmDecoder resolved depthwise bias must be F32");
                }
            }
            operated = short_conv(ctx, normalized, inputs, w.conv, config,
                                  cached ? &cached->layers[layer] : nullptr, cached);
        } else {
            if (w.attention.gate_proj.has_value() != config.attention_gate) {
                throw std::invalid_argument("LfmDecoder attention gate weights must match the architecture");
            }
            const auto & factors = config.layers[layer] == LfmDecoderLayerKind::FullAttention
                                      ? weights.full_freq_factors : weights.sliding_freq_factors;
            operated = attention(ctx, normalized, inputs, w.attention, config.layers[layer], config, factors ? &*factors : nullptr,
                                 cached ? &cached->layers[layer] : nullptr, cached ? cached->slots : nullptr);
        }
        if (config.sandwich_norm) {
            operated = norm.build(ctx, operated, *w.post_operator_norm);
        }
        hidden = AddModule{}.build(ctx, hidden, operated);
        auto ffn = feed_forward.build(ctx, norm.build(ctx, hidden, w.ffn_norm), w.feed_forward);
        if (config.sandwich_norm) {
            ffn = norm.build(ctx, ffn, *w.post_ffn_norm);
        }
        hidden = AddModule{}.build(ctx, hidden, ffn);
        for (size_t capture = 0; capture < inputs.capture_layers.size(); ++capture) {
            if (inputs.capture_layers[capture] == static_cast<int64_t>(layer)) {
                result.captured_hidden[capture] = hidden;
            }
        }
    }
    result.hidden = config.final_norm ? norm.build(ctx, hidden, weights.final_norm) : hidden;
    return result;
}


LfmDecoderOutputs build_lfm_decoder(core::ModuleBuildContext & ctx, const LfmDecoderInputs & inputs,
                                    const LfmDecoderWeights & weights, const LfmDecoderConfig & config) {
    return build_stack(ctx, inputs, weights, config);
}

namespace {

constexpr size_t kNodes = 32768;
enum class InputKind { Tokens, Embeddings };

struct GraphKey {
    int64_t steps;
    size_t replacements, readout_rows, output_tokens;
    InputKind input_kind;
    bool full_mask, sliding_mask;
    LfmDecoderReadout readout;

    bool operator==(const GraphKey & other) const noexcept {
        return std::tie(steps, replacements, readout_rows, output_tokens, input_kind, full_mask, sliding_mask, readout) ==
               std::tie(other.steps, other.replacements, other.readout_rows, other.output_tokens,
                        other.input_kind, other.full_mask, other.sliding_mask, other.readout);
    }
};

struct DecoderState {
    DecoderState(core::ExecutionContext & execution, const LfmDecoderConfig & config, int64_t count)
        : capacity(count), context(ggml_init({(3 * config.layers.size() + 1) * ggml_tensor_overhead(), nullptr, true})) {
        if (!context) { throw runtime::CapacityError("LFM decoder state metadata allocation failed"); }
        for (auto kind : config.layers) {
            LayerState layer;
            if (kind == LfmDecoderLayerKind::ShortConv) {
                if (config.conv_kernel_size > 1) {
                    layer.conv_tail = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, config.hidden_size, config.conv_kernel_size - 1);
                }
            } else {
                layer.key = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, config.head_dim * config.num_key_value_heads, capacity);
                layer.value = ggml_dup_tensor(context.get(), layer.key);
            }
            layers.push_back(layer);
        }
        if (ggml_get_first_tensor(context.get())) {
            buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), execution.backend()));
            if (!buffer) { throw runtime::CapacityError("LFM decoder persistent-state allocation failed"); }
            ggml_backend_buffer_clear(buffer.get(), 0);
        }
    }

    int64_t capacity;
    int64_t valid = 0;
    bool usable = true;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    std::vector<LayerState> layers;
};

struct DecoderGraph {
    DecoderGraph(const LfmDecoderConfig & config, const LfmDecoderRuntimeWeights & weights,
                 core::ExecutionContext & execution_in, const GraphKey & key, DecoderState * state = nullptr)
        : execution(execution_in), context(ggml_init({kNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kNodes, false), nullptr, true})) {
        if (!context) { throw runtime::CapacityError("LFM decoder graph metadata allocation failed"); }
        const int64_t steps = key.steps;
        const auto patches = key.replacements, readouts = key.readout_rows, tokens = key.output_tokens;
        const bool token_input = key.input_kind == InputKind::Tokens;
        const bool full_mask = key.full_mask, sliding_mask = key.sliding_mask;
        const auto kind = key.readout;
        replacement_indices.resize(patches);
        if (state) {
            position_values.resize(static_cast<size_t>(steps));
            slot_values.resize(static_cast<size_t>(steps));
            if (full_mask || sliding_mask) { mask_values.resize(static_cast<size_t>(steps * state->capacity)); }
        }
        auto * g = context.get();
        core::ModuleBuildContext ctx{g, "framework.lfm.decoder", execution.backend_type()};
        const int64_t keys = state ? state->capacity : steps;
        const auto hidden_shape = core::TensorShape::from_dims({1, steps, config.hidden_size});
        core::TensorValue hidden;
        if (token_input) {
            ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps);
            ggml_set_input(ids);
            hidden = EmbeddingModule({weights.token_embedding->shape.dims[0], config.hidden_size}).build(
                ctx, core::wrap_tensor(ids, core::TensorShape::from_dims({steps}), GGML_TYPE_I32), *weights.token_embedding);
            if (patches) {
                patch_positions = ggml_new_tensor_1d(g, GGML_TYPE_I64, patches);
                patch_values = ggml_new_tensor_2d(g, GGML_TYPE_F32, config.hidden_size, patches);
                ggml_set_input(patch_positions);
                ggml_set_input(patch_values);
                hidden = core::wrap_tensor(ggml_set_rows(g, hidden.tensor, patch_values, patch_positions), hidden.shape);
            }
            hidden = core::reshape_tensor(ctx, hidden, hidden_shape);
        } else {
            hidden_values = ggml_new_tensor_3d(g, GGML_TYPE_F32, config.hidden_size, steps, 1);
            ggml_set_input(hidden_values);
            hidden = core::wrap_tensor(hidden_values, hidden_shape);
        }
        positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps);
        ggml_set_input(positions);
        LfmDecoderInputs input;
        input.hidden = hidden;
        input.positions = core::wrap_tensor(positions, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
        if (full_mask) {
            full = ggml_new_tensor_2d(g, GGML_TYPE_F16, keys, steps);
            ggml_set_input(full);
            input.full_mask = core::wrap_tensor(full, core::TensorShape::from_dims({steps, keys}), GGML_TYPE_F16);
        }
        if (sliding_mask) {
            sliding = ggml_new_tensor_2d(g, GGML_TYPE_F16, keys, steps);
            ggml_set_input(sliding);
            input.sliding_mask = core::wrap_tensor(sliding, core::TensorShape::from_dims({steps, keys}), GGML_TYPE_F16);
        }
        if (state) {
            slots = ggml_new_tensor_1d(g, GGML_TYPE_I64, steps);
            ggml_set_input(slots);
        }
        CachedBuild cached{state ? state->layers : empty_layers, slots, keys, {}};
        auto out = build_stack(ctx, input, weights.decoder, config, state ? &cached : nullptr).hidden;
        if (readouts) {
            row_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, readouts);
            ggml_set_input(row_ids);
            out = core::wrap_tensor(ggml_get_rows(g, ggml_reshape_2d(g, out.tensor, config.hidden_size, steps), row_ids),
                                   core::TensorShape::from_dims({static_cast<int64_t>(readouts), config.hidden_size}));
        }
        rows = readouts ? static_cast<int64_t>(readouts) : steps;
        columns = config.hidden_size;
        if (kind == LfmDecoderReadout::Logits) {
            auto projection = *weights.output_projection;
            columns = projection.weight.shape.dims[0];
            if (tokens) {
                output_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, tokens);
                ggml_set_input(output_ids);
                projection.weight = core::wrap_tensor(ggml_get_rows(g, projection.weight.tensor, output_ids),
                    core::TensorShape::from_dims({static_cast<int64_t>(tokens), config.hidden_size}));
                if (projection.bias) {
                    auto * bias = ggml_get_rows(g, ggml_reshape_2d(g, projection.bias->tensor, 1, columns), output_ids);
                    projection.bias = core::wrap_tensor(ggml_reshape_1d(g, bias, tokens), core::TensorShape::from_dims({static_cast<int64_t>(tokens)}));
                }
                columns = static_cast<int64_t>(tokens);
            }
            out = LinearModule({config.hidden_size, columns, projection.bias.has_value(), config.projection_precision}).build(ctx, out, projection);
        }
        output = out.tensor;
        ggml_set_output(output);
        graph = ggml_new_graph_custom(g, kNodes, false);
        // Inputs are part of the execution contract even for a recipe which does
        // not consume one (e.g. positions when every layer is convolutional).
        for (auto * tensor : {ids, hidden_values, positions, patch_positions, patch_values, full, sliding, row_ids, output_ids, slots}) {
            if (tensor) { ggml_build_forward_expand(graph, tensor); }
        }
        ggml_build_forward_expand(graph, output);
        // Copies run after all consumers of the old convolution history. Scratch
        // release never touches their destinations in the persistent-state context.
        for (const auto & update : cached.tail_updates) {
            ggml_build_forward_expand(graph, ggml_cpy(g, update.first, update.second));
        }
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph, "LFM decoder forward");
    }

    ~DecoderGraph() { core::release_backend_graph_resources(execution.backend(), graph, true); }
    void unbind() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
        for (auto * t = ggml_get_first_tensor(context.get()); t; t = ggml_get_next_tensor(context.get(), t)) {
            t->data = nullptr;
            t->buffer = nullptr;
            t->extra = nullptr;
        }
    }

    core::ExecutionContext & execution;
    ggml_context_ptr context;
    ggml_cgraph * graph = nullptr;
    const std::vector<LayerState> empty_layers;
    ggml_tensor * slots = nullptr;
    ggml_tensor * ids = nullptr, * hidden_values = nullptr, * positions = nullptr;
    ggml_tensor * patch_positions = nullptr, * patch_values = nullptr;
    ggml_tensor * full = nullptr, * sliding = nullptr, * row_ids = nullptr, * output_ids = nullptr, * output = nullptr;
    int64_t rows = 0, columns = 0;
    std::vector<int32_t> position_values;
    std::vector<int64_t> slot_values, replacement_indices;
    std::vector<ggml_fp16_t> mask_values;
};

} // namespace

struct LfmDecoderRuntime::Impl {
    Impl(LfmDecoderConfig config_in, LfmDecoderRuntimeWeights weights_in, core::ExecutionContext & execution_in, size_t slots)
        : config(std::move(config_in)), weights(std::move(weights_in)), execution(execution_in), graphs(slots) {
        validate_config(config);
        if (!slots) { throw std::invalid_argument("LFM decoder graph cache requires at least one slot"); }
        has_full_attention = std::find(config.layers.begin(), config.layers.end(), LfmDecoderLayerKind::FullAttention) != config.layers.end();
        has_sliding_attention = std::find(config.layers.begin(), config.layers.end(), LfmDecoderLayerKind::SlidingAttention) != config.layers.end();
    }

    void release_workspace() {
        if (bound) { bound->unbind(); bound.reset(); }
        allocator.reset();
    }

    GraphKey resolve_request(const LfmDecoderForward & input) const;
    LfmDecoderDeviceResult run(const LfmDecoderForward & input, int64_t required_steps = 0);
    void reserve(int64_t cache_steps);

    LfmDecoderConfig config;
    LfmDecoderRuntimeWeights weights;
    core::ExecutionContext & execution;
    std::unique_ptr<DecoderState> state;
    int64_t sliding_window = 0;
    bool has_full_attention = false, has_sliding_attention = false;
    ggml_gallocr_ptr allocator;
    runtime::CacheSlots<GraphKey, std::shared_ptr<DecoderGraph>> graphs;
    std::shared_ptr<DecoderGraph> bound;
    size_t built = 0;
};

LfmDecoderRuntime::LfmDecoderRuntime(LfmDecoderConfig config, LfmDecoderRuntimeWeights weights,
                                   core::ExecutionContext & execution, size_t graph_cache_slots)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(weights), execution, graph_cache_slots)) {}
LfmDecoderRuntime::~LfmDecoderRuntime() = default;

GraphKey LfmDecoderRuntime::Impl::resolve_request(const LfmDecoderForward & input) const {
    const bool append = state != nullptr;
    if (append && !state->usable) {
        throw std::logic_error("LFM decoder requires usable state; reset the decode session before continuing");
    }
    const int64_t d = config.hidden_size;
    const bool tokens = !input.token_ids.empty();
    const int input_count = static_cast<int>(tokens) + static_cast<int>(!input.hidden.empty()) +
                            static_cast<int>(input.hidden_tensor != nullptr);
    if (input_count != 1) { throw std::invalid_argument("LFM decoder requires exactly one token, host embedding or device embedding input"); }
    if (tokens && !weights.token_embedding) { throw std::invalid_argument("LFM token input requires embedding weights"); }
    if (!input.hidden.empty() && input.hidden.size() % static_cast<size_t>(d) != 0) {
        throw std::invalid_argument("LFM host embeddings must contain whole hidden_size rows");
    }
    if ((input.hidden_tensor && !input.hidden_tensor->tensor) ||
        (input.replacement_tensor && !input.replacement_tensor->tensor)) {
        throw std::invalid_argument("LFM decoder borrowed input has no tensor");
    }
    const int64_t steps = tokens ? static_cast<int64_t>(input.token_ids.size()) : input.hidden_tensor ?
        ggml_nelements(input.hidden_tensor->tensor) / d : static_cast<int64_t>(input.hidden.size()) / d;
    if (append && (steps > state->capacity - state->valid)) {
        throw runtime::CapacityError("LFM decoder append exceeds reserved cache capacity");
    }
    if (steps <= 0 || (!append && input.positions.size() != static_cast<size_t>(steps)) ||
        (append && !input.positions.empty())) {
        throw std::invalid_argument("LFM decoder requires one position per input row");
    }
    if (input.hidden_tensor && (input.hidden_tensor->type != GGML_TYPE_F32 ||
        input.hidden_tensor->tensor->ne[0] != d || ggml_nelements(input.hidden_tensor->tensor) != steps * d)) {
        throw std::invalid_argument("LFM decoder device hidden input must be F32 rows of hidden_size");
    }
    if (tokens) {
        for (const auto id : input.token_ids) {
            if (id < 0 || id >= weights.token_embedding->shape.dims[0]) { throw std::out_of_range("LFM decoder token is outside embedding vocabulary"); }
        }
    }
    const auto patches = input.replacement_positions.size();
    if ((!patches && (input.replacement_tensor || !input.replacement_values.empty())) ||
        (!tokens && patches) || (patches && (input.replacement_tensor ? !input.replacement_values.empty() :
        input.replacement_values.size() != patches * static_cast<size_t>(d)))) {
        throw std::invalid_argument("LFM decoder embedding replacement rows do not match their positions");
    }
    const auto check_positions = [steps](const std::vector<int32_t> & positions) {
        for (const auto p : positions) { if (p < 0 || p >= steps) { throw std::out_of_range("LFM decoder row position is outside input"); } }
    };
    check_positions(input.replacement_positions);
    check_positions(input.readout_positions);
    if (append && (!input.full_mask.empty() || !input.sliding_mask.empty())) {
        throw std::invalid_argument("LFM cached decoding uses causal masks; arbitrary masks require forward()");
    }
    for (const auto * mask : {&input.full_mask, &input.sliding_mask}) {
        if (!mask->empty() && mask->size() != static_cast<size_t>(steps * steps)) {
            throw std::invalid_argument("LFM decoder mask must be square over the input sequence");
        }
    }
    if (input.readout == LfmDecoderReadout::Logits && !weights.output_projection) {
        throw std::invalid_argument("LFM decoder logits require output-projection weights");
    }
    if (input.replacement_tensor && (input.replacement_tensor->type != GGML_TYPE_F32 ||
        input.replacement_tensor->tensor->ne[0] != d || ggml_nelements(input.replacement_tensor->tensor) != static_cast<int64_t>(patches) * d)) {
        throw std::invalid_argument("LFM decoder device replacement input must match its positions");
    }
    if (!input.output_tokens.empty()) {
        if (input.readout != LfmDecoderReadout::Logits) { throw std::invalid_argument("LFM decoder output token selection requires logits"); }
        for (const auto id : input.output_tokens) {
            if (id < 0 || id >= weights.output_projection->weight.shape.dims[0]) { throw std::out_of_range("LFM decoder output token is outside projection vocabulary"); }
        }
    }
    return {steps, patches, input.readout_positions.size(), input.output_tokens.size(),
            tokens ? InputKind::Tokens : InputKind::Embeddings,
            append ? has_full_attention : !input.full_mask.empty(),
            append ? has_sliding_attention : !input.sliding_mask.empty(), input.readout};
}

LfmDecoderDeviceResult LfmDecoderRuntime::Impl::run(const LfmDecoderForward & input, int64_t required_steps) {
    const auto key = resolve_request(input);
    if (required_steps && key.steps != required_steps) { throw std::invalid_argument("LFM decode_step requires one row"); }
    const int64_t steps = key.steps;
    const auto patches = key.replacements;
    const bool append = state != nullptr;
    auto * entry = graphs.find(key);
    if (!entry) {
        auto graph = std::make_shared<DecoderGraph>(config, weights, execution, key, state.get());
        graphs.put(key, std::move(graph));
        entry = graphs.find(key);
        ++built;
    }
    auto & graph = **entry;
    if (bound.get() != &graph) {
        if (bound) { bound->unbind(); }
        bound = *entry;
        if (!allocator) { allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()))); }
        if (!allocator || !ggml_gallocr_reserve(allocator.get(), graph.graph) || !ggml_gallocr_alloc_graph(allocator.get(), graph.graph)) {
            release_workspace();
            throw runtime::CapacityError("LFM decoder forward scratch allocation failed");
        }
    }
    if (graph.ids) { ggml_backend_tensor_set(graph.ids, input.token_ids.data(), 0, input.token_ids.size() * sizeof(int32_t)); }
    if (graph.hidden_values) {
        if (input.hidden_tensor) {
            ggml_backend_tensor_copy(input.hidden_tensor->tensor, graph.hidden_values);
        }
        else { ggml_backend_tensor_set(graph.hidden_values, input.hidden.data(), 0, input.hidden.size() * sizeof(float)); }
    }
    if (append) {
        for (int64_t i = 0; i < steps; ++i) {
            graph.position_values[i] = static_cast<int32_t>(state->valid + i);
            graph.slot_values[i] = state->valid + i;
        }
        ggml_backend_tensor_set(graph.slots, graph.slot_values.data(), 0, graph.slot_values.size() * sizeof(int64_t));
    }
    const auto & position_values = append ? graph.position_values : input.positions;
    ggml_backend_tensor_set(graph.positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (patches) {
        std::copy(input.replacement_positions.begin(), input.replacement_positions.end(), graph.replacement_indices.begin());
        ggml_backend_tensor_set(graph.patch_positions, graph.replacement_indices.data(), 0, patches * sizeof(int64_t));
        if (input.replacement_tensor) {
            ggml_backend_tensor_copy(input.replacement_tensor->tensor, graph.patch_values);
        }
        else { ggml_backend_tensor_set(graph.patch_values, input.replacement_values.data(), 0, input.replacement_values.size() * sizeof(float)); }
    }
    if (append && (graph.full || graph.sliding)) {
        auto & mask = graph.mask_values;
        std::fill(mask.begin(), mask.end(), ggml_fp32_to_fp16(-INFINITY));
        for (int64_t q = 0; q < steps; ++q) {
            std::fill(mask.begin() + q * state->capacity, mask.begin() + q * state->capacity + state->valid + q + 1,
                      ggml_fp32_to_fp16(0.f));
        }
        if (graph.full) { ggml_backend_tensor_set(graph.full, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t)); }
        if (sliding_window > 0) {
            for (int64_t q = 0; q < steps; ++q) {
                const auto end = std::max<int64_t>(0, state->valid + q - sliding_window + 1);
                std::fill_n(mask.begin() + q * state->capacity, end, ggml_fp32_to_fp16(-INFINITY));
            }
        }
        if (graph.sliding) { ggml_backend_tensor_set(graph.sliding, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t)); }
    } else {
        if (graph.full) { ggml_backend_tensor_set(graph.full, input.full_mask.data(), 0, input.full_mask.size() * sizeof(ggml_fp16_t)); }
        if (graph.sliding) { ggml_backend_tensor_set(graph.sliding, input.sliding_mask.data(), 0, input.sliding_mask.size() * sizeof(ggml_fp16_t)); }
    }
    if (graph.row_ids) { ggml_backend_tensor_set(graph.row_ids, input.readout_positions.data(), 0, input.readout_positions.size() * sizeof(int32_t)); }
    if (graph.output_ids) { ggml_backend_tensor_set(graph.output_ids, input.output_tokens.data(), 0, input.output_tokens.size() * sizeof(int32_t)); }
    core::set_backend_threads(execution.backend(), execution.config().threads);
    // A failed compute may have partially written K/V or convolution history.
    // Invalidate before submission; only successful completion makes it usable.
    if (append) { state->usable = false; }
    if (core::compute_backend_graph(execution.backend(), graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LFM decoder forward graph execution failed");
    }
    ggml_backend_synchronize(execution.backend());
    if (append) { state->valid += steps; state->usable = true; }
    return {core::wrap_tensor(graph.output, core::TensorShape::from_dims({graph.rows, graph.columns})), graph.rows, graph.columns};
}

LfmDecoderDeviceResult LfmDecoderRuntime::forward_device(const LfmDecoderForward & input) { return impl_->run(input); }
LfmDecoderForwardResult LfmDecoderRuntime::forward(const LfmDecoderForward & input) {
    auto result = forward_device(input);
    return {core::read_tensor_f32(result.output.tensor), result.rows, result.columns};
}

void LfmDecoderRuntime::Impl::reserve(int64_t cache_steps) {
    if (cache_steps <= 0 || cache_steps > std::numeric_limits<int32_t>::max() || sliding_window < 0) {
        throw std::invalid_argument("LFM decode requires positive I32 cache capacity and nonnegative sliding window");
    }
    if (has_sliding_attention && sliding_window == 0) {
        throw std::invalid_argument("LFM sliding-attention decoding requires its window length");
    }
    if (state && !state->usable) { throw std::logic_error("LFM decoder state is invalid; reset before resizing"); }
    if (state && cache_steps <= state->capacity) { return; }
    release_workspace();
    graphs.clear();
    auto next = std::make_unique<DecoderState>(execution, config, cache_steps);
    if (state) {
        // As in TransformerKVCache::import_prefix, temporary descriptors bound
        // the backend copy to valid rows without graph nodes or host staging.
        for (size_t i = 0; i < state->layers.size(); ++i) {
            const auto & old = state->layers[i];
            const auto & target = next->layers[i];
            for (const auto & pair : {std::make_pair(old.key, target.key), std::make_pair(old.value, target.value)}) {
                if (pair.first && state->valid) {
                    ggml_tensor from = *pair.first;
                    from.ne[1] = state->valid;
                    from.nb[2] = from.nb[1] * from.ne[1];
                    from.nb[3] = from.nb[2];
                    ggml_tensor to = *pair.second;
                    std::copy(std::begin(from.ne), std::end(from.ne), std::begin(to.ne));
                    std::copy(std::begin(from.nb), std::end(from.nb), std::begin(to.nb));
                    ggml_backend_tensor_copy(&from, &to);
                }
            }
            if (old.conv_tail) { ggml_backend_tensor_copy(old.conv_tail, target.conv_tail); }
        }
        next->valid = state->valid;
    }
    state = std::move(next);
}

void LfmDecoderRuntime::DecodeSession::reset() {
    auto & state = *impl_->state;
    if (state.buffer) { ggml_backend_buffer_clear(state.buffer.get(), 0); }
    state.valid = 0;
    state.usable = true;
}

LfmDecoderDeviceResult LfmDecoderRuntime::DecodeSession::prefill_device(const LfmDecoderForward & input) { return impl_->run(input); }
LfmDecoderForwardResult LfmDecoderRuntime::DecodeSession::prefill(const LfmDecoderForward & input) {
    auto result = prefill_device(input);
    return {core::read_tensor_f32(result.output.tensor), result.rows, result.columns};
}
LfmDecoderForwardResult LfmDecoderRuntime::DecodeSession::decode_step(const LfmDecoderForward & input) {
    auto result = impl_->run(input, 1);
    return {core::read_tensor_f32(result.output.tensor), result.rows, result.columns};
}

int64_t LfmDecoderRuntime::DecodeSession::valid_steps() const noexcept { return impl_->state->valid; }
int64_t LfmDecoderRuntime::DecodeSession::cache_steps() const noexcept { return impl_->state->capacity; }

LfmDecoderRuntime::DecodeSession::DecodeSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LfmDecoderRuntime::DecodeSession::~DecodeSession() = default;
void LfmDecoderRuntime::DecodeSession::reserve(int64_t cache_steps) { impl_->reserve(cache_steps); }
void LfmDecoderRuntime::DecodeSession::release_workspace() { impl_->release_workspace(); }
size_t LfmDecoderRuntime::DecodeSession::graphs_built() const noexcept { return impl_->built; }

std::unique_ptr<LfmDecoderRuntime::DecodeSession> LfmDecoderRuntime::create_decode_session(int64_t cache_steps, int64_t sliding_window) const {
    auto stream = std::make_unique<Impl>(impl_->config, impl_->weights, impl_->execution, impl_->graphs.capacity());
    stream->sliding_window = sliding_window;
    stream->reserve(cache_steps);
    return std::unique_ptr<DecodeSession>(new DecodeSession(std::move(stream)));
}

void LfmDecoderRuntime::release_workspace() { impl_->release_workspace(); }
size_t LfmDecoderRuntime::graphs_built() const noexcept { return impl_->built; }

} // namespace engine::modules
