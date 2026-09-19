#include "engine/models/auk/conditioner.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace engine::models::auk {
namespace {

constexpr size_t kGraphContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kGraphNodeCapacity = 262144;

struct LayerWeights {
    modules::QwenDecoderLayerWeights decoder;
};

struct ConditionerWeights {
    core::TensorValue embed_tokens;
    std::vector<LayerWeights> layers;
    int64_t loaded = 0;
};

modules::QwenDecoderLayerConfig layer_config(const AukConditionerConfig & config) {
    modules::QwenDecoderStackConfig stack;
    stack.hidden_size = config.hidden_size;
    stack.num_attention_heads = config.attention_heads;
    stack.num_key_value_heads = config.key_value_heads;
    stack.head_dim = config.head_dim;
    stack.intermediate_size = config.intermediate_size;
    stack.layers = config.layers;
    stack.rms_norm_eps = config.rms_norm_eps;
    stack.rope_theta = config.rope_theta;
    // ⚠ Qwen2.5, not Qwen3: the attention projections carry BIASES and there are no
    // q_norm/k_norm tensors. The framework's default is the Qwen3 shape, so leaving
    // use_qk_norm alone would look for weights this checkpoint does not have.
    stack.use_qk_norm = false;
    return modules::qwen_decoder_layer_config_from_stack(stack);
}

ConditionerWeights load_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const AukConditionerConfig & config,
    const std::string & prefix) {
    ConditionerWeights out;
    const auto storage = config.weight_storage;
    out.embed_tokens = store.load_tensor(
        source, prefix + ".embed_tokens.weight", storage, {config.vocab_size, config.hidden_size});
    ++out.loaded;

    const int64_t q_dim = config.attention_heads * config.head_dim;
    const int64_t kv_dim = config.key_value_heads * config.head_dim;
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const std::string base = prefix + ".layers." + std::to_string(layer) + ".";
        LayerWeights weights;
        auto & decoder = weights.decoder;
        decoder.input_norm.weight = store.load_tensor(
            source, base + "input_layernorm.weight", assets::TensorStorageType::F32, {config.hidden_size});
        decoder.post_norm.weight = store.load_tensor(
            source, base + "post_attention_layernorm.weight", assets::TensorStorageType::F32, {config.hidden_size});

        auto & attention = decoder.self_attention;
        attention.q_weight = store.load_tensor(source, base + "self_attn.q_proj.weight", storage, {q_dim, config.hidden_size});
        attention.q_bias = store.load_tensor(source, base + "self_attn.q_proj.bias", assets::TensorStorageType::F32, {q_dim});
        attention.k_weight = store.load_tensor(source, base + "self_attn.k_proj.weight", storage, {kv_dim, config.hidden_size});
        attention.k_bias = store.load_tensor(source, base + "self_attn.k_proj.bias", assets::TensorStorageType::F32, {kv_dim});
        attention.v_weight = store.load_tensor(source, base + "self_attn.v_proj.weight", storage, {kv_dim, config.hidden_size});
        attention.v_bias = store.load_tensor(source, base + "self_attn.v_proj.bias", assets::TensorStorageType::F32, {kv_dim});
        attention.out_weight = store.load_tensor(source, base + "self_attn.o_proj.weight", storage, {config.hidden_size, q_dim});

        auto & mlp = decoder.mlp;
        mlp.gate_proj.weight = store.load_tensor(source, base + "mlp.gate_proj.weight", storage, {config.intermediate_size, config.hidden_size});
        mlp.up_proj.weight = store.load_tensor(source, base + "mlp.up_proj.weight", storage, {config.intermediate_size, config.hidden_size});
        mlp.down_proj.weight = store.load_tensor(source, base + "mlp.down_proj.weight", storage, {config.hidden_size, config.intermediate_size});

        out.loaded += 12;
        out.layers.push_back(std::move(weights));
    }
    return out;
}

}  // namespace

void AukConditionerConfig::validate() const {
    if (hidden_size <= 0 || layers <= 0 || attention_heads <= 0 || key_value_heads <= 0) {
        throw std::runtime_error("AuK conditioner config has non-positive dimensions");
    }
    if (attention_heads % key_value_heads != 0) {
        throw std::runtime_error("AuK conditioner needs attention heads divisible by key/value heads");
    }
    if (head_dim * attention_heads != hidden_size) {
        throw std::runtime_error("AuK conditioner head_dim * heads must equal hidden_size");
    }
}

struct AukConditioner::Impl {
    AukConditionerConfig config;
    AukFusionParameters fusion;
    core::ExecutionContext * execution = nullptr;
    std::unique_ptr<core::BackendWeightStore> store;
    ConditionerWeights weights;
    std::vector<float> softmax_weights;

    std::mutex mutex;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * audio_input = nullptr;
    ggml_tensor * audio_positions_input = nullptr;
    int64_t audio_tokens = 0;
    ggml_tensor * positions = nullptr;
    ggml_tensor * output = nullptr;
    int64_t steps = 0;

    void release() {
        if (graph != nullptr) {
            core::release_backend_graph_resources(execution->backend(), graph);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
        plan.reset();
        graph = nullptr;
        tokens = nullptr;
        audio_input = nullptr;
        audio_positions_input = nullptr;
        audio_tokens = 0;
        positions = nullptr;
        output = nullptr;
        steps = 0;
    }

    void ensure_graph(int64_t wanted, int64_t wanted_audio) {
        if (ggml != nullptr && steps == wanted && audio_tokens == wanted_audio) {
            return;
        }
        release();
        audio_tokens = wanted_audio;
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml = ggml_init(params);
        if (ggml == nullptr) {
            throw std::runtime_error("failed to initialize AuK conditioner graph context");
        }
        core::ModuleBuildContext ctx{ggml, "auk.conditioner", execution->backend_type()};

        tokens = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, wanted);
        ggml_set_input(tokens);
        positions = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, wanted);
        ggml_set_input(positions);

        // The decoder layer wants {batch, seq, hidden}; ggml_get_rows hands back a rank-2
        // [hidden, seq], so give it the batch axis it validates for.
        ggml_tensor * embedded = ggml_get_rows(ggml, weights.embed_tokens.tensor, tokens);
        ggml_tensor * sequence = embedded;
        if (audio_tokens > 0) {
            audio_input = ggml_new_tensor_2d(ggml, GGML_TYPE_F32, config.hidden_size, audio_tokens);
            ggml_set_input(audio_input);
            audio_positions_input = ggml_new_tensor_1d(ggml, GGML_TYPE_I64, audio_tokens);
            ggml_set_input(audio_positions_input);
            sequence = ggml_set_rows(ggml, sequence, audio_input, audio_positions_input);
        }
        auto hidden = core::wrap_tensor(
            ggml_reshape_3d(ggml, sequence, config.hidden_size, wanted, 1),
            core::TensorShape::from_dims({1, wanted, config.hidden_size}),
            GGML_TYPE_F32);
        auto position_value = core::wrap_tensor(
            positions, core::TensorShape::from_dims({wanted}), GGML_TYPE_I32);

        const modules::QwenDecoderLayerModule layer(layer_config(config));
        ggml_tensor * fused = nullptr;
        for (int64_t index = 0; index < config.layers; ++index) {
            auto out = layer.build(ctx, hidden, position_value, weights.layers[static_cast<size_t>(index)].decoder);
            hidden = out.output;
            // ⚠ ggml_norm is LayerNorm without affine parameters, which is exactly what
            // F.layer_norm(h, [d]) does -- and it is applied to EVERY layer's output,
            // including this one, before the weighted sum.
            ggml_tensor * normed = ggml_norm(
                ggml, core::ensure_backend_addressable_layout(ctx, hidden).tensor, config.fusion_norm_eps);
            ggml_tensor * scaled = ggml_scale(ggml, normed, softmax_weights[static_cast<size_t>(index)]);
            fused = fused == nullptr ? scaled : ggml_add(ggml, fused, scaled);
        }
        fused = ggml_scale(ggml, fused, fusion.layer_scale);
        output = core::ensure_backend_addressable_layout(
            ctx, core::wrap_tensor(fused, core::TensorShape::from_dims({1, wanted, config.hidden_size}), GGML_TYPE_F32)).tensor;
        ggml_set_output(output);

        graph = ggml_new_graph_custom(ggml, kGraphNodeCapacity, false);
        ggml_build_forward_expand(graph, output);
        core::validate_backend_graph_supported(execution->backend(), graph, "auk.conditioner");
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution->backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release();
            throw std::runtime_error("failed to allocate AuK conditioner graph memory");
        }
        core::prepare_host_graph_plan(*execution, graph, plan);
        steps = wanted;
    }
};

AukConditioner::AukConditioner(
    AukConditionerConfig config,
    AukFusionParameters fusion,
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    std::string prefix)
    : impl_(std::make_unique<Impl>()) {
    config.validate();
    if (static_cast<int64_t>(fusion.layer_weights.size()) != config.layers) {
        throw std::runtime_error(
            "AuK conditioner needs one fusion weight per layer; got " +
            std::to_string(fusion.layer_weights.size()) + " for " + std::to_string(config.layers) + " layers");
    }
    impl_->config = std::move(config);
    impl_->fusion = std::move(fusion);
    impl_->execution = &execution;

    // softmax over the learned per-layer weights, computed here rather than baked into
    // the fixture so the port owns every step the reference takes.
    const auto & raw = impl_->fusion.layer_weights;
    const float peak = *std::max_element(raw.begin(), raw.end());
    double total = 0.0;
    impl_->softmax_weights.resize(raw.size());
    for (size_t index = 0; index < raw.size(); ++index) {
        const double value = std::exp(static_cast<double>(raw[index] - peak));
        impl_->softmax_weights[index] = static_cast<float>(value);
        total += value;
    }
    for (auto & value : impl_->softmax_weights) {
        value = static_cast<float>(static_cast<double>(value) / total);
    }

    impl_->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "auk.conditioner.weights",
        20ull * 1024ull * 1024ull * 1024ull);
    impl_->weights = load_weights(*impl_->store, source, impl_->config, prefix);
    impl_->store->upload();
}

AukConditioner::~AukConditioner() {
    if (impl_ != nullptr) {
        impl_->release();
    }
}

AukConditioning AukConditioner::encode(
    const std::vector<int32_t> & input_ids,
    const std::vector<float> & audio,
    const std::vector<int64_t> & audio_positions) {
    if (input_ids.empty()) {
        throw std::runtime_error("AuK conditioner received no tokens");
    }
    if (audio_positions.size() * static_cast<size_t>(impl_->config.hidden_size) != audio.size()) {
        throw std::runtime_error("AuK conditioner needs one audio position per audio embedding row");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto steps = static_cast<int64_t>(input_ids.size());
    impl_->ensure_graph(steps, static_cast<int64_t>(audio_positions.size()));
    ggml_backend_tensor_set(impl_->tokens, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
    if (!audio_positions.empty()) {
        ggml_backend_tensor_set(impl_->audio_input, audio.data(), 0, audio.size() * sizeof(float));
        ggml_backend_tensor_set(
            impl_->audio_positions_input, audio_positions.data(), 0, audio_positions.size() * sizeof(int64_t));
    }
    std::vector<int32_t> positions(input_ids.size());
    for (size_t index = 0; index < positions.size(); ++index) {
        positions[index] = static_cast<int32_t>(index);
    }
    ggml_backend_tensor_set(impl_->positions, positions.data(), 0, positions.size() * sizeof(int32_t));
    if (core::compute_graph(*impl_->execution, impl_->graph, impl_->plan, "auk.conditioner") != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("AuK conditioner graph compute failed");
    }
    AukConditioning out;
    out.hidden_size = impl_->output->ne[0];
    out.tokens = impl_->output->ne[1];
    out.values = core::read_tensor_f32(impl_->output);
    return out;
}

int64_t AukConditioner::loaded_tensor_count() const noexcept { return impl_->weights.loaded; }

}  // namespace engine::models::auk
