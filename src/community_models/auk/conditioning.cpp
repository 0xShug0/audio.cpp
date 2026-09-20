#include "engine/community_models/auk/conditioning.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/debug/trace.h"
#include <ggml-alloc.h>

#include <limits>
#include <numeric>
#include <string_view>

namespace engine::models::auk {

ConditioningInput prepare_conditioning(
    const tokenizers::LlamaBpeTokenizer & tokenizer,
    const std::string & instruction, int64_t audio_tokens) {
    if (audio_tokens < 0) throw std::runtime_error("AuK audio token count must be nonnegative");
    constexpr std::string_view marker = "|<no_prompt_audio>|";
    ConditioningInput input;
    input.formatted_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                           "<|im_start|>user\n";
    input.formatted_text += instruction;
    if (audio_tokens > 0) {
        input.formatted_text += "<|audio_bos|>";
        for (int64_t i = 0; i < audio_tokens; ++i) input.formatted_text += "<|AUDIO|>";
        input.formatted_text += "<|audio_eos|>";
    } else if (instruction.size() < marker.size() ||
        instruction.compare(instruction.size() - marker.size(), marker.size(), marker) != 0) {
        input.formatted_text += marker;
    }
    input.formatted_text += "<|im_end|>\n<|im_start|>assistant\n";
    input.token_ids = tokenizer.encode(input.formatted_text, true);
    input.attention_mask.assign(input.token_ids.size(), 1);
    input.positions.resize(input.token_ids.size());
    // Text and audio-only Qwen mRoPE repeat these positions on all three axes.
    std::iota(input.positions.begin(), input.positions.end(), 0);
    return input;
}

ConditioningWeights load_conditioning_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & qwen,
    const assets::TensorSource & auk) {
    using namespace modules::binding;
    ConditioningWeights weights;
    constexpr auto storage = assets::TensorStorageType::Native;
    // Lookup converts selected rows to F32; expanding the whole source table is unnecessary.
    weights.embedding = store.load_tensor(
        qwen, "thinker.model.embed_tokens.weight", assets::TensorStorageType::Native, {151936, 2048});
    weights.final_norm = norm_weight_from_source(store, qwen, "thinker.model.norm", 2048);
    weights.layers.reserve(36);
    for (int layer = 0; layer < 36; ++layer) {
        const auto prefix = "thinker.model.layers." + std::to_string(layer);
        modules::QwenDecoderLayerWeights block;
        block.input_norm = norm_weight_from_source(store, qwen, prefix + ".input_layernorm", 2048);
        block.post_norm = norm_weight_from_source(store, qwen, prefix + ".post_attention_layernorm", 2048);
        const auto q = linear_from_source(store, qwen, prefix + ".self_attn.q_proj", storage, 2048, 2048, true);
        const auto k = linear_from_source(store, qwen, prefix + ".self_attn.k_proj", storage, 256, 2048, true);
        const auto v = linear_from_source(store, qwen, prefix + ".self_attn.v_proj", storage, 256, 2048, true);
        const auto out = linear_from_source(store, qwen, prefix + ".self_attn.o_proj", storage, 2048, 2048, false);
        block.self_attention.q_weight = q.weight;
        block.self_attention.q_bias = q.bias;
        block.self_attention.k_weight = k.weight;
        block.self_attention.k_bias = k.bias;
        block.self_attention.v_weight = v.weight;
        block.self_attention.v_bias = v.bias;
        block.self_attention.out_weight = out.weight;
        block.mlp.gate_proj = linear_from_source(store, qwen, prefix + ".mlp.gate_proj", storage, 11008, 2048, false);
        block.mlp.up_proj = linear_from_source(store, qwen, prefix + ".mlp.up_proj", storage, 11008, 2048, false);
        block.mlp.down_proj = linear_from_source(store, qwen, prefix + ".mlp.down_proj", storage, 2048, 11008, false);
        weights.layers.push_back(std::move(block));
    }
    weights.layer_weights = store.load_f32_tensor(auk, "layer_weights", {36});
    weights.layer_scale = store.load_f32_tensor(auk, "layer_scale", {1});
    return weights;
}

core::TensorValue build_text_conditioning(
    core::ModuleBuildContext & ctx,
    const ConditioningWeights & weights,
    const core::TensorValue & embeddings,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const modules::QwenDecoderActivationCastPolicy & activation_cast,
    std::vector<core::TensorValue> * captured_layers) {
    modules::QwenDecoderLayerConfig config;
    config.hidden_size = 2048;
    config.num_attention_heads = 16;
    config.num_key_value_heads = 2;
    config.head_dim = 128;
    config.intermediate_size = 11008;
    config.rms_norm_eps = 1e-6F;
    config.rope_theta = 1000000.0F;
    config.use_qk_norm = false;
    config.projection_precision = GGML_PREC_F32;
    config.activation_cast = activation_cast;
    config.runtime.attention.prefill_mode = activation_cast.enabled
        ? modules::QwenDecoderAttentionMode::FlashGrouped : modules::QwenDecoderAttentionMode::ManualRepeat;
    const modules::QwenDecoderLayerModule decoder(config);
    const modules::LayerNormModule layer_norm({2048, 1e-5F, false, false});
    const auto fusion_weights = modules::SoftmaxModule().build(ctx, weights.layer_weights);
    auto hidden = embeddings;
    core::TensorValue fused;
    for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
        hidden = decoder.build(ctx, hidden, positions, weights.layers[layer], std::nullopt, std::nullopt, attention_mask).output;
        if (captured_layers) {
            ggml_set_output(hidden.tensor);
            captured_layers->push_back(hidden);
        }
        // Transformers hidden_states[-1] is the final normalized state, not
        // the last decoder block's unnormalized residual stream.
        if (layer + 1 == weights.layers.size()) {
            hidden = modules::RMSNormModule({2048, 1e-6F, true, false}).build(ctx, hidden, weights.final_norm);
        }
        const auto normalized = layer_norm.build(ctx, hidden, {});
        const auto scale = ggml_view_1d(ctx.ggml, fusion_weights.tensor, 1, layer * sizeof(float));
        const auto weighted = core::wrap_tensor(
            ggml_mul(ctx.ggml, normalized.tensor, scale), normalized.shape);
        fused = layer == 0 ? weighted : core::wrap_tensor(ggml_add(ctx.ggml, fused.tensor, weighted.tensor), weighted.shape);
    }
    return core::wrap_tensor(ggml_mul(ctx.ggml, fused.tensor, weights.layer_scale.tensor), fused.shape);
}

struct ConditioningGraph {
    core::ExecutionContext & execution;
    int64_t tokens;
    int64_t audio_tokens;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    core::TensorValue ids;
    core::TensorValue audio;
    core::TensorValue embedding_rows;
    core::TensorValue positions;
    core::TensorValue mask;
    core::TensorValue output;
    std::vector<core::TensorValue> layer_outputs;

    ConditioningGraph(core::ExecutionContext & execution_, const ConditioningWeights & weights,
                      int64_t tokens_, int64_t audio_tokens_, bool capture_layers, bool bf16_autocast)
        : execution(execution_), tokens(tokens_), audio_tokens(audio_tokens_) {
        if (tokens <= 0) {
            throw std::runtime_error("AuK conditioning token count must be positive");
        }
        constexpr size_t nodes = 32768;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) {
            throw std::runtime_error("AuK conditioning graph context allocation failed");
        }
        core::ModuleBuildContext ctx{context.get(), "auk.conditioning", execution.backend_type()};
        ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens}));
        positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens}));
        mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({tokens, tokens}));
        ggml_set_input(ids.tensor);
        ggml_set_input(positions.tensor);
        ggml_set_input(mask.tensor);
        auto embeddings = modules::EmbeddingModule({151936, 2048}).build(ctx, ids, weights.embedding);
        if (audio_tokens > 0) {
            audio = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, audio_tokens, 2048}));
            embedding_rows = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens}));
            ggml_set_input(audio.tensor);
            ggml_set_input(embedding_rows.tensor);
            auto * combined = ggml_concat(ctx.ggml, embeddings.tensor, audio.tensor, 1);
            const auto table = core::wrap_tensor(ggml_reshape_2d(ctx.ggml, combined, 2048, tokens + audio_tokens),
                core::TensorShape::from_dims({tokens + audio_tokens, 2048}));
            embeddings = modules::EmbeddingModule({tokens + audio_tokens, 2048}).build(ctx, embedding_rows, table);
        }
        modules::QwenDecoderActivationCastPolicy casts;
        casts.enabled = bf16_autocast;
        casts.after_input_norm = true;
        casts.after_qkv_projection = true;
        casts.after_rope = true;
        casts.after_attention = true;
        casts.after_attention_output = true;
        casts.after_ffn_norm = true;
        casts.after_mlp_projection = true;
        casts.after_mlp_silu = true;
        casts.after_mlp_mul = true;
        output = build_text_conditioning(ctx, weights, embeddings, positions, mask, casts,
                                         capture_layers ? &layer_outputs : nullptr);
        if (capture_layers) {
            ggml_set_output(embeddings.tensor);
            ggml_set_output(embeddings.tensor->view_src);
            layer_outputs.push_back(embeddings);
        }
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        if (capture_layers) {
            const auto & attention = weights.layers[0].self_attention;
            for (const auto & bias : {attention.q_bias, attention.k_bias, attention.v_bias}) {
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    auto * node = ggml_graph_node(graph, i);
                    if (node->op == GGML_OP_ADD && node->src[1] == bias->tensor) {
                        ggml_set_output(node);
                        layer_outputs.push_back(core::wrap_tensor(node, core::TensorShape::from_dims({1, tokens, node->ne[0]})));
                        break;
                    }
                }
            }
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT && node->src[0] == attention.out_weight.tensor) {
                    auto * values = node->src[1];
                    ggml_set_output(values);
                    if (values->view_src) ggml_set_output(values->view_src);
                    layer_outputs.push_back(core::wrap_tensor(values, core::TensorShape::from_dims({1, tokens, 2048})));
                    break;
                }
            }
        }
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("AuK conditioning graph buffer allocation failed");
        }
    }

    ~ConditioningGraph() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};

struct ConditioningRuntime::State {
    core::ExecutionContext & execution;
    bool capture_layers;
    bool bf16_autocast;
    core::BackendWeightStore store;
    ConditioningWeights weights;
    std::unique_ptr<ConditioningGraph> prepared;

    State(core::ExecutionContext & execution_, const assets::TensorSource & qwen,
          const assets::TensorSource & auk, bool capture_layers_, bool bf16_autocast_)
        : execution(execution_), capture_layers(capture_layers_), bf16_autocast(bf16_autocast_),
          store(execution.backend(), execution.backend_type(), "auk.conditioning", 4 * 1024 * 1024),
          weights(load_conditioning_weights(store, qwen, auk)) {
        store.upload();
    }
};

ConditioningRuntime::ConditioningRuntime(core::ExecutionContext & execution,
                                       const assets::TensorSource & qwen,
                                       const assets::TensorSource & auk,
                                       int64_t tokens, bool capture_layers, bool bf16_autocast, int64_t audio_tokens)
    : state_(std::make_unique<State>(execution, qwen, auk, capture_layers, bf16_autocast)) {
    prepare(tokens, audio_tokens);
}

ConditioningRuntime::~ConditioningRuntime() = default;

void ConditioningRuntime::prepare(int64_t tokens, int64_t audio_tokens) {
    if (tokens <= 0) throw std::runtime_error("AuK conditioning token count must be positive");
    auto & state = *state_;
    if (audio_tokens < 0 || audio_tokens > tokens) throw std::runtime_error("AuK audio token count exceeds sequence length");
    if (state.prepared && state.prepared->tokens == tokens && state.prepared->audio_tokens == audio_tokens) return;
    state.prepared.reset();
    state.prepared = std::make_unique<ConditioningGraph>(
        state.execution, state.weights, tokens, audio_tokens, state.capture_layers, state.bf16_autocast);
}

std::vector<std::vector<float>> ConditioningRuntime::captured_layers() const {
    if (!state_->prepared) throw std::runtime_error("AuK conditioning graph is not prepared");
    std::vector<std::vector<float>> outputs;
    for (const auto & tensor : state_->prepared->layer_outputs) {
        outputs.push_back(core::read_tensor_f32(tensor.tensor));
    }
    return outputs;
}

std::vector<float> ConditioningRuntime::encode(const ConditioningInput & input, const std::vector<float> & audio_embeddings) {
    if (!state_->prepared) throw std::runtime_error("AuK conditioning graph is not prepared");
    auto & state = *state_->prepared;
    const auto tokens = static_cast<size_t>(state.tokens);
    if (input.token_ids.size() != tokens || input.positions.size() != tokens || input.attention_mask.size() != tokens) {
        throw std::runtime_error("AuK conditioning input length differs from prepared graph");
    }
    if (audio_embeddings.size() != static_cast<size_t>(state.audio_tokens * 2048)) {
        throw std::runtime_error("AuK audio embedding count differs from prepared graph");
    }
    if (state.audio_tokens > 0) {
        std::vector<int32_t> rows(tokens);
        int64_t audio_index = 0;
        for (size_t i = 0; i < tokens; ++i) {
            rows[i] = input.token_ids[i] == 151646 ? static_cast<int32_t>(tokens + audio_index++) : static_cast<int32_t>(i);
        }
        if (audio_index != state.audio_tokens) throw std::runtime_error("AuK audio placeholder count differs from embedding count");
        core::write_tensor_i32(state.embedding_rows, rows);
        core::write_tensor_f32(state.audio, audio_embeddings);
    }
    std::vector<float> mask(tokens * tokens, -std::numeric_limits<float>::infinity());
    for (size_t query = 0; query < tokens; ++query) {
        for (size_t key = 0; key <= query; ++key) {
            if (input.attention_mask[key]) {
                mask[query * tokens + key] = 0.0F;
            }
        }
    }
    core::write_tensor_i32(state.ids, input.token_ids);
    core::write_tensor_i32(state.positions, input.positions);
    core::write_tensor_f16(state.mask, mask);
    if (core::compute_backend_graph(state.execution.backend(), state.graph, nullptr, "auk.conditioning") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK conditioning graph computation failed");
    }
    auto output = core::read_tensor_f32(state.output.tensor);
    debug::trace_log_f32("auk.conditioning.output", {1, state.tokens, 2048}, output);
    return output;
}

}  // namespace engine::models::auk
