#include "engine/models/m5_tts/ar.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace engine::models::m5_tts {
namespace {

modules::CausalDecoderRuntimeConfig runtime_config(const DecoderConfig & config,
    bool local, bool flash, bool bf16) {
    modules::CausalDecoderRuntimeConfig result;
    result.trace_name = local ? "m5_tts.local" : "m5_tts.backbone";
    result.prefill_graph_arena_bytes = 4 * 1024 * 1024;
    result.decode_graph_arena_bytes = 4 * 1024 * 1024;
    result.return_hidden = true;
    if (local) result.output_mode = modules::CausalDecoderOutputMode::Hidden;
    result.decoder.logits_size = 2051;
    // FP32 storage retains the rounded activation values without a second
    // conversion at the cache boundary.
    result.decoder.static_cache_type = GGML_TYPE_F32;
    auto & stack = result.decoder.stack;
    stack.hidden_size = config.hidden;
    stack.intermediate_size = config.intermediate;
    stack.layers = config.layers;
    stack.num_attention_heads = config.heads;
    stack.num_key_value_heads = config.kv_heads;
    stack.head_dim = config.head_dim;
    stack.rms_norm_eps = config.norm_eps;
    stack.rope_theta = config.rope_theta;
    stack.use_qk_norm = false;
    stack.qkv_layout = modules::DecoderQKVLayout::PackedQKV;
    stack.runtime.mlp.mode = modules::DecoderMLPMode::PackedGateUp;
    stack.runtime.attention.prefill_mode = flash
        ? modules::DecoderAttentionMode::FlashGroupedViewKV : modules::DecoderAttentionMode::ManualRepeat;
    stack.runtime.attention.static_mode = stack.runtime.attention.prefill_mode;
    stack.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    if (bf16) {
        auto & cast = stack.activation_cast;
        cast.enabled = true;
        cast.after_input_norm = true;
        cast.after_qkv_projection = true;
        cast.after_rope = true;
        cast.after_attention = true;
        cast.after_attention_output = true;
        cast.after_residual = true;
        cast.after_ffn_norm = true;
        cast.after_mlp_projection = true;
        cast.after_mlp_silu = true;
        cast.after_mlp_mul = true;
        cast.after_output = true;
        cast.norm_intermediate_type = GGML_TYPE_BF16;
        result.readback_round_type = GGML_TYPE_BF16;
    }
    return result;
}

modules::CausalDecoderRuntimeWeights bind_decoder(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    const DecoderConfig & config, assets::TensorStorageType storage) {
    namespace binding = modules::binding;
    modules::CausalDecoderRuntimeWeights result;
    const auto factors = store.load_f32_tensor(source, prefix + ".rope_factors", {config.head_dim / 2});
    for (int64_t index = 0; index < config.layers; ++index) {
        const auto name = prefix + ".layers." + std::to_string(index);
        modules::DecoderLayerWeights layer;
        layer.input_norm = binding::norm_weight_from_source(store, source, name + ".input_layernorm", config.hidden);
        layer.post_norm = binding::norm_weight_from_source(store, source, name + ".post_attention_layernorm", config.hidden);
        layer.self_attention.qkv_weight = modules::PackedLinearWeightsBuilder({config.hidden, {
            {name + ".self_attn.q_proj.weight", std::nullopt, config.heads * config.head_dim},
            {name + ".self_attn.k_proj.weight", std::nullopt, config.kv_heads * config.head_dim},
            {name + ".self_attn.v_proj.weight", std::nullopt, config.kv_heads * config.head_dim}}, false})
            .build(store, source, storage).weight;
        layer.self_attention.out_weight = store.load_tensor(source, name + ".self_attn.o_proj.weight", storage,
            {config.hidden, config.hidden});
        layer.mlp.gate_up_proj = modules::PackedLinearWeightsBuilder({config.hidden, {
            {name + ".mlp.gate_proj.weight", std::nullopt, config.intermediate},
            {name + ".mlp.up_proj.weight", std::nullopt, config.intermediate}}, false})
            .build(store, source, storage);
        layer.mlp.down_proj = binding::linear_from_source(store, source, name + ".mlp.down_proj", storage,
            config.hidden, config.intermediate, false);
        layer.rope_frequency_factors = factors;
        result.stack.layers.push_back(std::move(layer));
    }
    result.final_norm = binding::norm_weight_from_source(store, source, prefix + ".norm", config.hidden);
    return result;
}

}  // namespace

class M5LlamaRuntime::Impl {
public:
    Impl(const ModelAssets & assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, core::AttentionPreference preference)
        : execution_(execution), hidden_(assets.backbone.hidden), vocabulary_(assets.audio_vocabulary),
          text_vocabulary_(assets.text_vocabulary), context_(assets.context_length),
          bf16_(storage == assets::TensorStorageType::BF16 ||
              (storage == assets::TensorStorageType::Native &&
               assets::parse_tensor_storage_type(assets.model->require_metadata("text_embeddings.weight").dtype) == assets::TensorStorageType::BF16)),
          store_(execution.backend(), execution.backend_type(), "m5_tts.ar.weights", 2 * 1024 * 1024) {
        auto backbone = bind_decoder(store_, *assets.model, "backbone", assets.backbone, storage);
        auto local = bind_decoder(store_, *assets.model, "decoder", assets.decoder, storage);
        text_weight_ = store_.load_tensor(*assets.model, "text_embeddings.weight", storage, {text_vocabulary_, hidden_});
        audio_weight_ = store_.load_tensor(*assets.model, "audio_embeddings.weight", storage, {16 * vocabulary_, hidden_});
        backbone.lm_head = modules::binding::linear_from_source(store_, *assets.model,
            "codebook0_head", storage, vocabulary_, hidden_, false);
        projection_ = modules::binding::linear_from_source(store_, *assets.model,
            "projection", storage, hidden_, hidden_, false);
        for (int64_t index = 0; index < 15; ++index) {
            heads_.push_back(modules::binding::linear_from_source(store_, *assets.model,
                "audio_heads." + std::to_string(index), storage, vocabulary_, hidden_, false));
        }
        store_.upload();
        flash_ = core::resolve_flash_attention(execution.backend(), assets.backbone.head_dim, preference);
        backbone_ = std::make_unique<modules::CausalDecoderRuntime>(execution,
            runtime_config(assets.backbone, false, flash_, bf16_), std::move(backbone));
        local_ = std::make_unique<modules::CausalDecoderRuntime>(execution,
            runtime_config(assets.decoder, true, flash_, bf16_), std::move(local));
        prepare_auxiliary();
    }

    std::vector<float> text_embeddings(const std::vector<int32_t> & ids) {
        if (ids.empty()) throw std::runtime_error("M5-TTS text prompt is empty");
        int64_t capacity = 32;
        while (capacity < static_cast<int64_t>(ids.size())) capacity *= 2;
        auto & graph = text_graphs_[capacity];
        if (!graph.owner) {
            graph.owner = std::make_unique<core::PreparedGraph>(execution_,
                core::PreparedGraphConfig{"m5_tts.text_embeddings", 256 * 1024, 64});
            auto & ctx = graph.owner->build_context();
            graph.input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({capacity}));
            ggml_set_input(graph.input.tensor);
            graph.output = modules::EmbeddingModule({text_vocabulary_, hidden_}).build(ctx, graph.input, text_weight_);
            ggml_set_output(graph.output.tensor);
            ggml_build_forward_expand(graph.owner->graph(), graph.output.tensor);
            graph.owner->allocate();
        }
        std::vector<int32_t> padded(static_cast<size_t>(capacity), 0);
        std::copy(ids.begin(), ids.end(), padded.begin());
        ggml_backend_tensor_set(graph.input.tensor, padded.data(), 0, padded.size() * sizeof(int32_t));
        graph.owner->compute();
        auto result = core::read_tensor_f32(graph.output.tensor);
        result.resize(ids.size() * static_cast<size_t>(hidden_));
        return result;
    }

    std::vector<float> audio_embedding(const AudioFrame & frame) {
        AudioFrame indices = frame;
        for (size_t i = 0; i < indices.size(); ++i) indices[i] += static_cast<int32_t>(i * vocabulary_);
        ggml_backend_tensor_set(audio_ids_.tensor, indices.data(), 0, sizeof(indices));
        auxiliary_->compute();
        auto result = core::read_tensor_f32(frame_embedding_.tensor);
        if (bf16_) core::round_f32_to_bf16_in_place(result);
        return result;
    }

    void begin(const std::vector<float> & prompt, int64_t max_frames) {
        const auto steps = static_cast<int64_t>(prompt.size() / hidden_);
        if (prompt.empty() || prompt.size() % static_cast<size_t>(hidden_) ||
            max_frames <= 0 || steps + max_frames > context_) {
            throw std::runtime_error("M5-TTS prompt and generation budget exceed the context");
        }
        const int64_t cache_steps = flash_ ? ((steps + max_frames + 255) / 256) * 256 : steps + max_frames;
        auto prefill = backbone_->prefill_embeddings_into_decode_cache(prompt, steps, cache_steps);
        current_.logits = std::move(prefill.logits);
        current_.hidden = std::move(prefill.hidden);
        frame_ = 0;
    }

    AudioFrame next_frame(const TokenSelector & select) {
        AudioFrame result{};
        result[0] = select(frame_, 0, current_.logits, frame_ > 6);
        const auto first = project_audio(0, result[0]);
        core::write_tensor_f32(global_hidden_, current_.hidden);
        auxiliary_->compute(initial_projection_id_);
        auto projected = core::read_tensor_f32(initial_projection_.tensor);
        if (bf16_) core::round_f32_to_bf16_in_place(projected);
        projected.insert(projected.end(), first.begin(), first.end());
        auto prefill = local_->prefill_embeddings_into_decode_cache(projected, 2, flash_ ? 256 : 16);
        modules::CausalDecoderStepResult state;
        state.hidden = std::move(prefill.hidden);
        for (size_t codebook = 1; codebook < result.size(); ++codebook) {
            core::write_tensor_f32(local_hidden_, state.hidden);
            auxiliary_->compute(head_graph_ids_[codebook - 1]);
            auto logits = core::read_tensor_f32(head_outputs_[codebook - 1].tensor);
            if (bf16_) core::round_f32_to_bf16_in_place(logits);
            result[codebook] = select(frame_, static_cast<int64_t>(codebook), logits, true);
            if (codebook + 1 < result.size()) state = local_->decode_embedding(project_audio(codebook, result[codebook]));
        }
        ++frame_;
        if (std::any_of(result.begin(), result.end(), [](int32_t token) { return token != 0; })) {
            current_ = backbone_->decode_embedding(audio_embedding(result));
        }
        return result;
    }

private:
    void prepare_auxiliary() {
        auxiliary_ = std::make_unique<core::PreparedGraph>(execution_,
            core::PreparedGraphConfig{"m5_tts.conditioning_and_heads", 1024 * 1024, 128, 128 * 1024});
        auto & persistent = auxiliary_->persistent_build_context();
        audio_ids_ = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({16}));
        local_id_ = core::make_tensor(persistent, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        global_hidden_ = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));
        local_hidden_ = core::make_tensor(persistent, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));
        for (auto * tensor : {audio_ids_.tensor, local_id_.tensor, global_hidden_.tensor, local_hidden_.tensor}) ggml_set_input(tensor);
        auxiliary_->allocate_persistent();
        auto & ctx = auxiliary_->build_context();
        const auto embedded = modules::EmbeddingModule({16 * vocabulary_, hidden_}).build(ctx, audio_ids_, audio_weight_);
        frame_embedding_ = modules::ReduceSumModule({0}).build(ctx, embedded);
        ggml_set_output(frame_embedding_.tensor);
        ggml_build_forward_expand(auxiliary_->graph(), frame_embedding_.tensor);
        projection_id_ = auxiliary_->add_graph("local_embedding_projection");
        const auto local_embedding = modules::EmbeddingModule({16 * vocabulary_, hidden_}).build(ctx, local_id_, audio_weight_);
        audio_projection_ = modules::LinearModule({hidden_, hidden_, false}).build(ctx, local_embedding, projection_);
        ggml_set_output(audio_projection_.tensor);
        ggml_build_forward_expand(auxiliary_->graph(projection_id_), audio_projection_.tensor);
        initial_projection_id_ = auxiliary_->add_graph("backbone_hidden_projection");
        initial_projection_ = modules::LinearModule({hidden_, hidden_, false}).build(ctx, global_hidden_, projection_);
        ggml_set_output(initial_projection_.tensor);
        ggml_build_forward_expand(auxiliary_->graph(initial_projection_id_), initial_projection_.tensor);
        for (size_t index = 0; index < heads_.size(); ++index) {
            head_graph_ids_.push_back(auxiliary_->add_graph("local_head_" + std::to_string(index)));
            head_outputs_.push_back(modules::LinearModule({hidden_, vocabulary_, false}).build(ctx, local_hidden_, heads_[index]));
            ggml_set_output(head_outputs_.back().tensor);
            ggml_build_forward_expand(auxiliary_->graph(head_graph_ids_.back()), head_outputs_.back().tensor);
        }
        auxiliary_->allocate();
    }

    std::vector<float> project_audio(size_t codebook, int32_t token) {
        const auto index = static_cast<int32_t>(codebook * vocabulary_ + token);
        ggml_backend_tensor_set(local_id_.tensor, &index, 0, sizeof(index));
        auxiliary_->compute(projection_id_);
        auto result = core::read_tensor_f32(audio_projection_.tensor);
        if (bf16_) core::round_f32_to_bf16_in_place(result);
        return result;
    }

    struct TextGraph {
        std::unique_ptr<core::PreparedGraph> owner;
        core::TensorValue input;
        core::TensorValue output;
    };
    core::ExecutionContext & execution_;
    int64_t hidden_, vocabulary_, text_vocabulary_, context_;
    bool bf16_;
    bool flash_ = false;
    core::BackendWeightStore store_;
    core::TensorValue text_weight_, audio_weight_;
    modules::LinearWeights projection_;
    std::vector<modules::LinearWeights> heads_;
    std::map<int64_t, TextGraph> text_graphs_;
    std::unique_ptr<core::PreparedGraph> auxiliary_;
    core::TensorValue audio_ids_, local_id_, global_hidden_, local_hidden_;
    core::TensorValue frame_embedding_, audio_projection_, initial_projection_;
    core::PreparedGraph::GraphId projection_id_ = 0, initial_projection_id_ = 0;
    std::vector<core::PreparedGraph::GraphId> head_graph_ids_;
    std::vector<core::TensorValue> head_outputs_;
    std::unique_ptr<modules::CausalDecoderRuntime> backbone_, local_;
    modules::CausalDecoderStepResult current_;
    int64_t frame_ = 0;
};

M5LlamaRuntime::M5LlamaRuntime(const ModelAssets & assets, core::ExecutionContext & execution,
    assets::TensorStorageType storage, core::AttentionPreference attention)
    : impl_(std::make_unique<Impl>(assets, execution, storage, attention)) {}
M5LlamaRuntime::~M5LlamaRuntime() = default;
std::vector<float> M5LlamaRuntime::text_embeddings(const std::vector<int32_t> & ids) { return impl_->text_embeddings(ids); }
std::vector<float> M5LlamaRuntime::audio_embedding(const AudioFrame & frame) { return impl_->audio_embedding(frame); }
void M5LlamaRuntime::begin(const std::vector<float> & prompt, int64_t max_frames) { impl_->begin(prompt, max_frames); }
AudioFrame M5LlamaRuntime::next_frame(const TokenSelector & select) { return impl_->next_frame(select); }

}  // namespace engine::models::m5_tts
