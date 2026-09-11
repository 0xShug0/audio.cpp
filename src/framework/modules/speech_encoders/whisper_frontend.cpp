#include "engine/framework/modules/speech_encoders/whisper_frontend.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace engine::modules {
using Clock = std::chrono::steady_clock;

namespace {

LinearWeights linear_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    return binding::linear_from_source(
        store,
        source,
        prefix,
        storage_type,
        out_features,
        in_features,
        use_bias);
}

void load_openai_embedding_weights(
    WhisperEmbeddingWeights & weights,
    const WhisperEmbeddingConfig & config,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    weights.conv1 = {
        store.load_tensor(
            source,
            "encoder.conv1.weight",
            conv_storage_type,
            {config.n_audio_state, config.n_mels, 3}),
        store.load_f32_tensor(source, "encoder.conv1.bias", {config.n_audio_state}),
    };
    weights.conv2 = {
        store.load_tensor(
            source,
            "encoder.conv2.weight",
            conv_storage_type,
            {config.n_audio_state, config.n_audio_state, 3}),
        store.load_f32_tensor(source, "encoder.conv2.bias", {config.n_audio_state}),
    };
    weights.positional_embedding = store.load_f32_tensor(
        source,
        "encoder.positional_embedding",
        {config.n_audio_ctx, config.n_audio_state});
    weights.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t layer = 0; layer < config.n_audio_layer; ++layer) {
        const std::string prefix = "encoder.blocks." + std::to_string(layer);
        WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = binding::norm_from_source(store, source, prefix + ".attn_ln", config.n_audio_state);
        layer_weights.attention.query = linear_weights(
            store,
            source,
            prefix + ".attn.query",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.key = linear_weights(
            store,
            source,
            prefix + ".attn.key",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            false);
        layer_weights.attention.value = linear_weights(
            store,
            source,
            prefix + ".attn.value",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.out = linear_weights(
            store,
            source,
            prefix + ".attn.out",
            matmul_storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.mlp_norm = binding::norm_from_source(store, source, prefix + ".mlp_ln", config.n_audio_state);
        layer_weights.mlp.fc1_weight = store.load_tensor(
            source,
            prefix + ".mlp.0.weight",
            matmul_storage_type,
            {config.n_audio_state * 4, config.n_audio_state});
        layer_weights.mlp.fc1_bias = store.load_f32_tensor(
            source,
            prefix + ".mlp.0.bias",
            {config.n_audio_state * 4});
        layer_weights.mlp.fc2_weight = store.load_tensor(
            source,
            prefix + ".mlp.2.weight",
            matmul_storage_type,
            {config.n_audio_state, config.n_audio_state * 4});
        layer_weights.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".mlp.2.bias", {config.n_audio_state});
        weights.layers.push_back(std::move(layer_weights));
    }
    weights.final_norm = binding::norm_from_source(store, source, "encoder.ln_post", config.n_audio_state);
}

WhisperEmbeddingConfig infer_hf_encoder_config(const assets::TensorSource & source) {
    const auto positions = source.require_metadata("model.encoder.embed_positions.weight");
    const auto conv1 = source.require_metadata("model.encoder.conv1.weight");
    const auto q_proj = source.require_metadata("model.encoder.layers.0.self_attn.q_proj.weight");
    WhisperEmbeddingConfig config;
    config.n_mels = conv1.shape.at(1);
    config.n_audio_state = conv1.shape.at(0);
    config.n_audio_ctx = positions.shape.at(0);
    config.n_audio_head = 12;
    config.n_audio_layer = 0;
    while (source.has_tensor("model.encoder.layers." + std::to_string(config.n_audio_layer) + ".fc1.weight")) {
        ++config.n_audio_layer;
    }
    if (q_proj.shape.at(0) != config.n_audio_state || config.n_audio_layer <= 0) {
        throw std::runtime_error("Whisper HF encoder config is invalid");
    }
    return config;
}

void load_hf_embedding_weights(
    WhisperEmbeddingWeights & weights,
    const WhisperEmbeddingConfig & config,
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    weights.conv1 = {
        store.load_tensor(
            source,
            "model.encoder.conv1.weight",
            storage_type,
            {config.n_audio_state, config.n_mels, 3}),
        store.load_f32_tensor(source, "model.encoder.conv1.bias", {config.n_audio_state}),
    };
    weights.conv2 = {
        store.load_tensor(
            source,
            "model.encoder.conv2.weight",
            storage_type,
            {config.n_audio_state, config.n_audio_state, 3}),
        store.load_f32_tensor(source, "model.encoder.conv2.bias", {config.n_audio_state}),
    };
    weights.positional_embedding = store.load_f32_tensor(
        source,
        "model.encoder.embed_positions.weight",
        {config.n_audio_ctx, config.n_audio_state});
    weights.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t layer = 0; layer < config.n_audio_layer; ++layer) {
        const std::string prefix = "model.encoder.layers." + std::to_string(layer);
        WhisperEncoderLayerWeights layer_weights;
        layer_weights.attention_norm = binding::norm_from_source(store, source, prefix + ".self_attn_layer_norm", config.n_audio_state);
        layer_weights.attention.query = linear_weights(
            store,
            source,
            prefix + ".self_attn.q_proj",
            storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.key = linear_weights(
            store,
            source,
            prefix + ".self_attn.k_proj",
            storage_type,
            config.n_audio_state,
            config.n_audio_state,
            false);
        layer_weights.attention.value = linear_weights(
            store,
            source,
            prefix + ".self_attn.v_proj",
            storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.attention.out = linear_weights(
            store,
            source,
            prefix + ".self_attn.out_proj",
            storage_type,
            config.n_audio_state,
            config.n_audio_state,
            true);
        layer_weights.mlp_norm = binding::norm_from_source(store, source, prefix + ".final_layer_norm", config.n_audio_state);
        layer_weights.mlp.fc1_weight = store.load_tensor(
            source,
            prefix + ".fc1.weight",
            storage_type,
            {config.n_audio_state * 4, config.n_audio_state});
        layer_weights.mlp.fc1_bias = store.load_f32_tensor(
            source,
            prefix + ".fc1.bias",
            {config.n_audio_state * 4});
        layer_weights.mlp.fc2_weight = store.load_tensor(
            source,
            prefix + ".fc2.weight",
            storage_type,
            {config.n_audio_state, config.n_audio_state * 4});
        layer_weights.mlp.fc2_bias = store.load_f32_tensor(source, prefix + ".fc2.bias", {config.n_audio_state});
        weights.layers.push_back(std::move(layer_weights));
    }
    weights.final_norm = binding::norm_from_source(store, source, "model.encoder.layer_norm", config.n_audio_state);
}

}  // namespace

struct WhisperFrontendComponent::State {
    explicit State(std::shared_ptr<const WhisperFrontendComponentWeights> weights)
        : weights(std::move(weights)) {}

    ~State() {
        release_graph();
    }

    void release_graph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        graph = nullptr;
        input = {};
        output = {};
    }

    void ensure_graph(const WhisperFrontendComponentConfig & component_config) {
        if (ctx != nullptr) {
            return;
        }
        if (weights == nullptr || weights->execution_context == nullptr) {
            throw std::runtime_error("Whisper frontend component requires weights and execution context");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{component_config.graph_context_bytes, nullptr, true};
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Whisper frontend graph context");
        }
        core::ModuleBuildContext build_ctx{
            ctx,
            component_config.name.c_str(),
            weights->execution_context->backend_type()};
        input = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, weights->config.n_mels, weights->config.n_audio_ctx * 2}));
        output = WhisperEmbeddingModule(weights->config).build(build_ctx, input, weights->embedding);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(ctx, 65536, false);
        ggml_build_forward_expand(graph, output.tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights->execution_context->backend()));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release_graph();
            throw std::runtime_error("failed to allocate Whisper frontend graph tensors");
        }
        debug::timing_log_scalar(component_config.name + ".graph.build_ms", debug::elapsed_ms(build_start));
    }

    std::shared_ptr<const WhisperFrontendComponentWeights> weights;
    mutable std::mutex mutex;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::TensorValue input;
    core::TensorValue output;
};

WhisperFrontendComponent WhisperFrontendComponent::load_openai_layout(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    WhisperEmbeddingConfig config,
    WhisperFrontendComponentConfig component_config) {
    if (source == nullptr) {
        throw std::runtime_error("Whisper frontend requires weights");
    }
    auto weights = std::make_shared<WhisperFrontendComponentWeights>();
    weights->config = config;
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        component_config.name + ".weights",
        component_config.weight_context_bytes);
    load_openai_embedding_weights(
        weights->embedding,
        weights->config,
        *weights->store,
        *source,
        component_config.matmul_weight_storage_type,
        component_config.conv_weight_storage_type);
    weights->store->upload();
    source->release_storage();
    return WhisperFrontendComponent(std::move(weights), std::move(component_config));
}

WhisperFrontendComponent WhisperFrontendComponent::load_hf_encoder_layout(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    WhisperFrontendComponentConfig component_config) {
    if (source == nullptr) {
        throw std::runtime_error("Whisper frontend requires weights");
    }
    auto weights = std::make_shared<WhisperFrontendComponentWeights>();
    weights->config = infer_hf_encoder_config(*source);
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        component_config.name + ".weights",
        component_config.weight_context_bytes);
    load_hf_embedding_weights(
        weights->embedding,
        weights->config,
        *weights->store,
        *source,
        component_config.matmul_weight_storage_type);
    weights->store->upload();
    source->release_storage();
    return WhisperFrontendComponent(std::move(weights), std::move(component_config));
}

WhisperFrontendComponent::WhisperFrontendComponent(
    std::shared_ptr<const WhisperFrontendComponentWeights> weights,
    WhisperFrontendComponentConfig component_config)
    : weights_(std::move(weights)),
      component_config_(std::move(component_config)),
      state_(std::make_shared<State>(weights_)) {
    if (weights_ == nullptr) {
        throw std::runtime_error("Whisper frontend component requires weights");
    }
}

const WhisperEmbeddingConfig & WhisperFrontendComponent::config() const {
    if (weights_ == nullptr) {
        throw std::runtime_error("Whisper frontend component is not initialized");
    }
    return weights_->config;
}

int64_t WhisperFrontendComponent::channels() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.n_audio_state;
}

std::vector<float> WhisperFrontendComponent::encode_log_mel(const std::vector<float> & log_mel) const {
    if (state_ == nullptr || weights_ == nullptr || weights_->execution_context == nullptr) {
        throw std::runtime_error("Whisper frontend component is not initialized");
    }
    const size_t input_size = static_cast<size_t>(weights_->config.n_mels * weights_->config.n_audio_ctx * 2);
    if (log_mel.size() != input_size) {
        throw std::runtime_error("Whisper frontend log-mel input shape mismatch");
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->ensure_graph(component_config_);
    const auto upload_start = Clock::now();
    core::write_tensor_f32(state_->input, log_mel);
    debug::timing_log_scalar(component_config_.name + ".input_upload_ms", debug::elapsed_ms(upload_start));
    const auto compute_start = Clock::now();
    if (core::compute_backend_graph(weights_->execution_context->backend(), state_->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed for Whisper frontend");
    }
    debug::timing_log_scalar(component_config_.name + ".graph.compute_ms", debug::elapsed_ms(compute_start));
    const auto readback_start = Clock::now();
    auto output = core::read_tensor_f32(state_->output.tensor);
    debug::timing_log_scalar(component_config_.name + ".output_readback_ms", debug::elapsed_ms(readback_start));
    return output;
}

void WhisperFrontendComponent::release_runtime_graph() const {
    if (state_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->release_graph();
}

}  // namespace engine::modules
