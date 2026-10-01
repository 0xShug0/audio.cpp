#include "engine/models/index_echo/mapper.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_echo {
namespace {

namespace binding = engine::modules::binding;
namespace json = engine::io::json;
namespace modules = engine::modules;

struct MapperConfig {
    int64_t hidden_size = 0;
    int64_t embedding_size = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t intermediate_size = 0;
    int64_t max_tokens = 0;
    int64_t languages = 0;
    std::vector<std::string> language_names;
};

MapperConfig load_config(const IndexEchoAssets & assets) {
    const auto value = assets.resources.parse_json("mapper_config");
    MapperConfig config;
    config.hidden_size = json::require_i64(value, "d_h");
    config.embedding_size = json::optional_i64(value, "d_e", 896);
    config.layers = json::require_i64(value, "n_layer");
    config.heads = json::require_i64(value, "n_head");
    config.intermediate_size = json::require_i64(value, "d_ff");
    config.max_tokens = json::require_i64(value, "max_len");
    config.languages = json::require_i64(value, "n_lang");
    for (const auto & language : value.require("langs").as_array()) {
        config.language_names.push_back(language.as_string());
    }
    if (config.hidden_size != assets.text_hidden_size || config.embedding_size <= 0 ||
        config.layers <= 0 || config.heads <= 0 || config.embedding_size % config.heads != 0 ||
        config.intermediate_size <= 0 || config.max_tokens <= 0 ||
        config.languages != static_cast<int64_t>(config.language_names.size())) {
        throw std::runtime_error("Index-Echo mapper configuration does not match model dimensions");
    }
    return config;
}

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t width) {
    const auto packed_weight = source.require_f32(prefix + ".in_proj_weight", {3 * width, width});
    const auto packed_bias = source.require_f32(prefix + ".in_proj_bias", {3 * width});
    const auto matrix_size = static_cast<size_t>(width * width);
    const auto vector_size = static_cast<size_t>(width);
    modules::AttentionWeights weights;
    auto load_projection = [&](int64_t index, core::TensorValue & weight, std::optional<core::TensorValue> & bias) {
        const size_t matrix_offset = static_cast<size_t>(index) * matrix_size;
        const size_t vector_offset = static_cast<size_t>(index) * vector_size;
        weight = store.make_from_f32(
            core::TensorShape::from_dims({width, width}), storage_type,
            std::vector<float>(packed_weight.begin() + static_cast<ptrdiff_t>(matrix_offset),
                               packed_weight.begin() + static_cast<ptrdiff_t>(matrix_offset + matrix_size)));
        bias = store.make_f32(
            core::TensorShape::from_dims({width}),
            std::vector<float>(packed_bias.begin() + static_cast<ptrdiff_t>(vector_offset),
                               packed_bias.begin() + static_cast<ptrdiff_t>(vector_offset + vector_size)));
    };
    load_projection(0, weights.q_weight, weights.q_bias);
    load_projection(1, weights.k_weight, weights.k_bias);
    load_projection(2, weights.v_weight, weights.v_bias);
    const auto out = binding::linear_from_source(store, source, prefix + ".out_proj", storage_type, width, width, true);
    weights.out_weight = out.weight;
    weights.out_bias = out.bias;
    return weights;
}

struct MapperWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::NormWeights h_norm;
    modules::LinearWeights h_proj;
    modules::NormWeights e_norm;
    modules::LinearWeights e_proj;
    core::TensorValue positions;
    std::vector<float> language_embeddings;
    std::vector<modules::TransformerEncoderBlockWeights> encoder;
    modules::LinearWeights out;
};

MapperWeights load_weights(
    const assets::TensorSource & source,
    const MapperConfig & config,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type) {
    MapperWeights weights;
    weights.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "index_echo.mapper.weights", 4ull * 1024ull * 1024ull);
    auto & store = *weights.store;
    const int64_t d_h = config.hidden_size;
    const int64_t d_e = config.embedding_size;
    weights.h_norm = binding::norm_from_source(store, source, "h_norm", d_h);
    weights.h_proj = binding::linear_from_source(store, source, "h_proj", storage_type, d_e, d_h, true);
    weights.e_norm = binding::norm_from_source(store, source, "e_norm", d_e);
    weights.e_proj = binding::linear_from_source(store, source, "e_proj", storage_type, d_e, d_e, true);
    weights.positions = store.load_f32_tensor(source, "pos", {1, config.max_tokens, d_e});
    weights.language_embeddings = source.require_f32("lang_emb.weight", {config.languages, d_e});
    weights.encoder.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        const auto prefix = "encoder.layers." + std::to_string(layer);
        modules::TransformerEncoderBlockWeights block;
        block.norm1 = binding::norm_from_source(store, source, prefix + ".norm1", d_e);
        block.self_attention = load_attention(store, source, prefix + ".self_attn", storage_type, d_e);
        block.norm2 = binding::norm_from_source(store, source, prefix + ".norm2", d_e);
        const auto fc1 = binding::linear_from_source(
            store, source, prefix + ".linear1", storage_type, config.intermediate_size, d_e, true);
        const auto fc2 = binding::linear_from_source(
            store, source, prefix + ".linear2", storage_type, d_e, config.intermediate_size, true);
        block.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        weights.encoder.push_back(std::move(block));
    }
    weights.out = binding::linear_from_source(store, source, "out", storage_type, d_e, d_e, true);
    store.upload();
    return weights;
}

struct MapperGraph {
    int64_t tokens = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan host_plan;
    core::TensorValue hidden;
    core::TensorValue embeddings;
    core::TensorValue language;
    core::TensorValue output;

    ~MapperGraph() {
        host_plan.reset();
        if (backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

}  // namespace

class IndexEchoHidden2CVRuntime::Impl {
public:
    Impl(std::shared_ptr<const IndexEchoAssets> model_assets,
         core::ExecutionContext & execution,
         assets::TensorStorageType storage_type)
        : assets_(std::move(model_assets)), execution_(&execution), config_(load_config(*assets_)) {
        const auto source = assets_->resources.open_tensor_source("mapper_weights");
        weights_ = load_weights(*source, config_, *execution_, storage_type);
    }

    std::vector<float> map(
        const std::vector<float> & text_hidden,
        const std::vector<float> & cosyvoice_embeddings,
        int64_t tokens,
        std::string_view target_language) {
        if (tokens <= 0 || tokens > config_.max_tokens ||
            text_hidden.size() != static_cast<size_t>(tokens * config_.hidden_size) ||
            cosyvoice_embeddings.size() != static_cast<size_t>(tokens * config_.embedding_size)) {
            throw std::runtime_error("Index-Echo mapper input shape exceeds checkpoint limits");
        }
        const auto language = std::find(config_.language_names.begin(), config_.language_names.end(), target_language);
        if (language == config_.language_names.end()) {
            throw std::runtime_error("Index-Echo mapper target language is unsupported");
        }
        const size_t language_index = static_cast<size_t>(language - config_.language_names.begin());
        const size_t width = static_cast<size_t>(config_.embedding_size);
        ensure_graph(tokens);
        core::write_tensor_f32(graph_->hidden, text_hidden);
        core::write_tensor_f32(graph_->embeddings, cosyvoice_embeddings);
        core::write_tensor_f32(
            graph_->language,
            std::vector<float>(weights_.language_embeddings.begin() + static_cast<ptrdiff_t>(language_index * width),
                               weights_.language_embeddings.begin() + static_cast<ptrdiff_t>((language_index + 1) * width)));
        if (core::compute_graph(*execution_, graph_->graph, graph_->host_plan, "Index-Echo hidden2cv") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Index-Echo mapper execution failed");
        }
        return core::read_tensor_f32(graph_->output.tensor);
    }

private:
    void ensure_graph(int64_t tokens) {
        if (graph_ != nullptr && graph_->tokens == tokens && graph_->backend == execution_->backend()) {
            return;
        }
        auto next = std::make_unique<MapperGraph>();
        next->tokens = tokens;
        next->backend = execution_->backend();
        ggml_init_params params{8ull * 1024ull * 1024ull, nullptr, true};
        next->ggml = ggml_init(params);
        if (next->ggml == nullptr) {
            throw std::runtime_error("Index-Echo mapper graph context allocation failed");
        }
        core::ModuleBuildContext context{next->ggml, "index_echo.mapper", execution_->backend_type()};
        next->hidden = core::make_tensor(
            context, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens, config_.hidden_size}));
        next->embeddings = core::make_tensor(
            context, GGML_TYPE_F32, core::TensorShape::from_dims({1, tokens, config_.embedding_size}));
        next->language = core::make_tensor(
            context, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config_.embedding_size}));
        ggml_set_input(next->hidden.tensor);
        ggml_set_input(next->embeddings.tensor);
        ggml_set_input(next->language.tensor);

        auto hidden = modules::LayerNormModule({config_.hidden_size, 1e-5f}).build(
            context, next->hidden, weights_.h_norm);
        hidden = modules::LinearModule({config_.hidden_size, config_.embedding_size, true}).build(
            context, hidden, weights_.h_proj);
        auto embeddings = modules::LayerNormModule({config_.embedding_size, 1e-5f}).build(
            context, next->embeddings, weights_.e_norm);
        embeddings = modules::LinearModule({config_.embedding_size, config_.embedding_size, true}).build(
            context, embeddings, weights_.e_proj);
        auto combined = modules::AddModule().build(context, hidden, embeddings);
        const auto positions = modules::SliceModule({1, 0, tokens}).build(context, weights_.positions);
        combined = modules::AddModule().build(context, combined, positions);
        const auto language = modules::RepeatModule({combined.shape}).build(context, next->language);
        combined = modules::AddModule().build(context, combined, language);
        for (const auto & layer : weights_.encoder) {
            combined = modules::TransformerEncoderBlockModule({
                config_.embedding_size, config_.heads, config_.intermediate_size, 1e-5f, true,
            }).build(context, combined, layer);
        }
        const auto delta = modules::LinearModule({config_.embedding_size, config_.embedding_size, true}).build(
            context, combined, weights_.out);
        next->output = modules::AddModule().build(context, next->embeddings, delta);
        ggml_set_output(next->output.tensor);
        next->graph = ggml_new_graph_custom(next->ggml, 2048, false);
        ggml_build_forward_expand(next->graph, next->output.tensor);
        core::validate_backend_graph_supported(next->backend, next->graph, "Index-Echo hidden2cv");
        next->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
        if (next->allocator == nullptr ||
            !ggml_gallocr_reserve(next->allocator, next->graph) ||
            !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
            throw std::runtime_error("Index-Echo mapper graph allocation failed");
        }
        core::prepare_host_graph_plan(*execution_, next->graph, next->host_plan);
        graph_ = std::move(next);
    }

    std::shared_ptr<const IndexEchoAssets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    MapperConfig config_;
    MapperWeights weights_;
    std::unique_ptr<MapperGraph> graph_;
};

IndexEchoHidden2CVRuntime::IndexEchoHidden2CVRuntime(
    std::shared_ptr<const IndexEchoAssets> model_assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(model_assets), execution, storage_type)) {}

IndexEchoHidden2CVRuntime::~IndexEchoHidden2CVRuntime() = default;

std::vector<float> IndexEchoHidden2CVRuntime::map(
    const std::vector<float> & text_hidden,
    const std::vector<float> & cosyvoice_embeddings,
    int64_t tokens,
    std::string_view target_language) {
    return impl_->map(text_hidden, cosyvoice_embeddings, tokens, target_language);
}

}  // namespace engine::models::index_echo
