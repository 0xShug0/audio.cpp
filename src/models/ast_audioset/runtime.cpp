#include "engine/models/ast_audioset/runtime.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::ast_audioset {
namespace {

using Clock = std::chrono::steady_clock;

struct ContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct BackendWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights patch_projection;
    core::TensorValue cls_token;
    core::TensorValue distillation_token;
    core::TensorValue position_embeddings;
    std::vector<modules::TransformerEncoderBlockWeights> layers;
    modules::NormWeights final_norm;
    modules::NormWeights classifier_norm;
    modules::LinearWeights classifier;
};

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage,
    int64_t hidden) {
    modules::AttentionWeights result;
    result.q_weight = store.load_tensor(source, prefix + ".attention.query.weight", storage, {hidden, hidden});
    result.q_bias = store.load_f32_tensor(source, prefix + ".attention.query.bias", {hidden});
    result.k_weight = store.load_tensor(source, prefix + ".attention.key.weight", storage, {hidden, hidden});
    result.k_bias = store.load_f32_tensor(source, prefix + ".attention.key.bias", {hidden});
    result.v_weight = store.load_tensor(source, prefix + ".attention.value.weight", storage, {hidden, hidden});
    result.v_bias = store.load_f32_tensor(source, prefix + ".attention.value.bias", {hidden});
    result.out_weight = store.load_tensor(source, prefix + ".output.dense.weight", storage, {hidden, hidden});
    result.out_bias = store.load_f32_tensor(source, prefix + ".output.dense.bias", {hidden});
    return result;
}

std::shared_ptr<BackendWeights> load_weights(
    const assets::TensorSource & source,
    const Config & config,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage) {
    auto result = std::make_shared<BackendWeights>();
    result->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "ast_audioset.weights", 32ULL * 1024ULL * 1024ULL);
    auto & store = *result->store;
    const std::string root = "audio_spectrogram_transformer";
    result->patch_projection = modules::binding::conv2d_from_source(
        store, source, root + ".embeddings.patch_embeddings.projection", storage,
        config.hidden_size, 1, config.patch_size, config.patch_size, true);
    result->cls_token = store.load_f32_tensor(source, root + ".embeddings.cls_token", {1, 1, config.hidden_size});
    result->distillation_token = store.load_f32_tensor(
        source, root + ".embeddings.distillation_token", {1, 1, config.hidden_size});
    const int64_t frequency_patches = (config.num_mel_bins - config.patch_size) / config.frequency_stride + 1;
    const int64_t time_patches = (config.max_length - config.patch_size) / config.time_stride + 1;
    const int64_t tokens = frequency_patches * time_patches + 2;
    result->position_embeddings = store.load_f32_tensor(
        source, root + ".embeddings.position_embeddings", {1, tokens, config.hidden_size});
    result->layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t index = 0; index < config.layers; ++index) {
        const std::string prefix = root + ".encoder.layer." + std::to_string(index);
        modules::TransformerEncoderBlockWeights layer;
        layer.norm1 = modules::binding::norm_from_source(store, source, prefix + ".layernorm_before", config.hidden_size);
        layer.self_attention = load_attention(store, source, prefix + ".attention", storage, config.hidden_size);
        layer.norm2 = modules::binding::norm_from_source(store, source, prefix + ".layernorm_after", config.hidden_size);
        const auto fc1 = modules::binding::linear_from_source(
            store, source, prefix + ".intermediate.dense", storage,
            config.intermediate_size, config.hidden_size, true);
        const auto fc2 = modules::binding::linear_from_source(
            store, source, prefix + ".output.dense", storage,
            config.hidden_size, config.intermediate_size, true);
        layer.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        result->layers.push_back(std::move(layer));
    }
    result->final_norm = modules::binding::norm_from_source(store, source, root + ".layernorm", config.hidden_size);
    result->classifier_norm = modules::binding::norm_from_source(store, source, "classifier.layernorm", config.hidden_size);
    result->classifier = modules::binding::linear_from_source(
        store, source, "classifier.dense", storage,
        static_cast<int64_t>(config.labels.size()), config.hidden_size, true);
    store.upload();
    return result;
}

core::TensorValue build_encoder_layer(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const modules::TransformerEncoderBlockWeights & weights,
    const Config & config,
    bool use_flash_attention) {
    const modules::NormConfig norm_config{config.hidden_size, config.layer_norm_eps, true, true, false};
    auto normalized = modules::LayerNormModule(norm_config).build(context, input, weights.norm1);
    modules::AttentionConfig attention_config;
    attention_config.hidden_size = config.hidden_size;
    attention_config.num_heads = config.heads;
    attention_config.use_bias = true;
    attention_config.use_flash_attention = use_flash_attention;
    auto attention = modules::SelfAttentionModule(attention_config).build(context, normalized, weights.self_attention);
    auto hidden = modules::ResidualAddModule().build(context, input, attention);
    normalized = modules::LayerNormModule(norm_config).build(context, hidden, weights.norm2);
    auto feed_forward = modules::FeedForwardModule({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
        GGML_PREC_DEFAULT,
        modules::FeedForwardActivation::Gelu,
    }).build(context, normalized, weights.feed_forward);
    return modules::ResidualAddModule().build(context, hidden, feed_forward);
}

}  // namespace

struct Runtime::Impl {
    class Graph {
    public:
        Graph(
            const Config & config,
            std::shared_ptr<const BackendWeights> weights,
            core::ExecutionContext & execution,
            bool use_flash_attention)
            : config_(config), weights_(std::move(weights)), backend_(execution.backend()),
              threads_(std::max(1, execution.config().threads)) {
            if (backend_ == nullptr) {
                throw std::runtime_error("invalid AST graph configuration");
            }
            ggml_init_params params{64ULL * 1024ULL * 1024ULL, nullptr, true};
            context_.reset(ggml_init(params));
            if (!context_) throw std::runtime_error("failed to create AST graph context");
            core::ModuleBuildContext build{context_.get(), "ast_audioset", execution.backend_type()};
            auto input = core::make_tensor(
                build, GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, config_.num_mel_bins, config_.max_length}));
            input_ = input.tensor;

            const int64_t frequency_patches =
                (config_.num_mel_bins - config_.patch_size) / config_.frequency_stride + 1;
            const int64_t time_patches =
                (config_.max_length - config_.patch_size) / config_.time_stride + 1;
            const int64_t patch_count = frequency_patches * time_patches;
            const int64_t token_count = patch_count + 2;
            auto patches = modules::Conv2dModule({
                1, config_.hidden_size,
                config_.patch_size, config_.patch_size,
                static_cast<int>(config_.frequency_stride), static_cast<int>(config_.time_stride),
                0, 0, 1, 1, true,
            }).build(build, input, weights_->patch_projection);
            patches = modules::ReshapeModule({core::TensorShape::from_dims({1, config_.hidden_size, patch_count})})
                          .build(build, patches);
            patches = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, patches);
            auto cls = modules::RepeatModule({core::TensorShape::from_dims({1, 1, config_.hidden_size})})
                           .build(build, weights_->cls_token);
            auto distillation = modules::RepeatModule({core::TensorShape::from_dims({1, 1, config_.hidden_size})})
                                    .build(build, weights_->distillation_token);
            auto hidden = modules::ConcatModule({1}).build(build, cls, distillation);
            hidden = modules::ConcatModule({1}).build(build, hidden, patches);
            auto positions = modules::RepeatModule({core::TensorShape::from_dims({1, token_count, config_.hidden_size})})
                                 .build(build, weights_->position_embeddings);
            hidden = modules::AddModule().build(build, hidden, positions);
            for (const auto & layer : weights_->layers) {
                hidden = build_encoder_layer(build, hidden, layer, config_, use_flash_attention);
            }
            hidden = modules::LayerNormModule({config_.hidden_size, config_.layer_norm_eps, true, true, false})
                         .build(build, hidden, weights_->final_norm);
            auto first = modules::SliceModule({1, 0, 1}).build(build, hidden);
            auto second = modules::SliceModule({1, 1, 1}).build(build, hidden);
            auto pooled = modules::ConcatModule({1}).build(build, first, second);
            pooled = modules::ReduceMeanModule({1}).build(build, pooled);
            pooled = modules::LayerNormModule({config_.hidden_size, config_.layer_norm_eps, true, true, false})
                         .build(build, pooled, weights_->classifier_norm);
            auto output = modules::LinearModule({
                config_.hidden_size, static_cast<int64_t>(config_.labels.size()), true,
            }).build(build, pooled, weights_->classifier);
            output_ = output.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(context_.get(), 8192, false);
            ggml_build_forward_expand(graph_, output_);
            allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (allocator_ == nullptr || !ggml_gallocr_reserve(allocator_, graph_) ||
                !ggml_gallocr_alloc_graph(allocator_, graph_)) {
                throw std::runtime_error("failed to allocate AST graph");
            }
        }

        ~Graph() {
            if (allocator_ != nullptr) ggml_gallocr_free(allocator_);
        }

        std::vector<float> run(const std::vector<float> & input) {
            const size_t expected = static_cast<size_t>(config_.num_mel_bins * config_.max_length);
            if (input.size() != expected) throw std::runtime_error("AST graph input size mismatch");
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            core::set_backend_threads(backend_, threads_);
            const auto started = Clock::now();
            const auto status = core::compute_backend_graph(backend_, graph_);
            ggml_backend_synchronize(backend_);
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("AST graph compute failed");
            std::vector<float> output(config_.labels.size());
            ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
            debug::timing_log_scalar("ast_audioset.graph_ms", debug::elapsed_ms(started));
            return output;
        }

    private:
        Config config_;
        std::shared_ptr<const BackendWeights> weights_;
        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::unique_ptr<ggml_context, ContextDeleter> context_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t allocator_ = nullptr;
    };

    Impl(
        std::shared_ptr<const assets::TensorSource> source,
        Config config,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage,
        bool use_flash_attention)
        : config(std::move(config)),
          weights(load_weights(*source, this->config, execution, storage)) {
        graph = std::make_unique<Graph>(this->config, weights, execution, use_flash_attention);
        source->release_storage();
    }

    std::vector<float> run(const std::vector<float> & features) {
        return graph->run(features);
    }

    Config config;
    std::shared_ptr<const BackendWeights> weights;
    std::unique_ptr<Graph> graph;
};

Runtime::Runtime(
    std::shared_ptr<const assets::TensorSource> source,
    Config config,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    bool use_flash_attention)
    : impl_(std::make_unique<Impl>(
          std::move(source), std::move(config), execution, storage_type, use_flash_attention)) {}

Runtime::~Runtime() = default;

std::vector<float> Runtime::classify(const std::vector<float> & features) {
    const int64_t values_per_item = impl_->config.max_length * impl_->config.num_mel_bins;
    if (values_per_item <= 0 || features.size() != static_cast<size_t>(values_per_item)) {
        throw std::runtime_error("AST feature size mismatch");
    }
    return impl_->run(features);
}

}  // namespace engine::models::ast_audioset
