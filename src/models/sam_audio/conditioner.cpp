#include "engine/models/sam_audio/conditioner.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <limits>

namespace engine::models::sam_audio {
namespace {
using core::TensorShape;

class T5TextEncoderGraph {
public:
    T5TextEncoderGraph(core::ExecutionContext & execution, const modules::T5BaseEncoderConfig & config,
              const modules::T5BaseEncoderWeights & weights, int64_t length)
        : backend_(execution.backend()), length_(length), heads_(config.attention_heads), hidden_(config.hidden_size) {
        constexpr size_t nodes = 8192;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio T5 context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.t5", execution.backend_type()};
        tokens_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, length}));
        auto buckets = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({length, length}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, config.attention_heads, length, length}));
        ggml_set_input(tokens_.tensor);
        ggml_set_input(buckets.tensor);
        // These indices are uploaded once and must survive allocator reuse between executions.
        ggml_set_output(buckets.tensor);
        ggml_set_input(mask_.tensor);
        output_ = modules::T5BaseEncoderModule(config).build(ctx, tokens_, buckets, mask_, weights).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_);
        runtime::optimize_graph(*graph_, runtime::GraphOptimizationBackend::Gpu);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("SAM Audio T5 graph allocation failed");
        core::write_tensor_i32(buckets, modules::t5_base_relative_position_buckets(length, length,
            config.relative_attention_num_buckets, config.relative_attention_max_distance));
    }

    ~T5TextEncoderGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    std::vector<float> run(const std::vector<int32_t> & tokens, int32_t pad_id) {
        const auto started = std::chrono::steady_clock::now();
        auto padded = tokens;
        padded.resize(static_cast<size_t>(length_), pad_id);
        core::write_tensor_i32(tokens_, padded);
        std::vector<float> mask(static_cast<size_t>(heads_ * length_ * length_), 0.0f);
        for (int64_t row = 0; row < heads_ * length_; ++row)
            std::fill(mask.begin() + row * length_ + tokens.size(), mask.begin() + (row + 1) * length_,
                      -std::numeric_limits<float>::max());
        core::write_tensor_f32(mask_, mask);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM Audio T5 graph execution failed");
        auto result = core::read_tensor_f32(output_);
        result.resize(tokens.size() * static_cast<size_t>(hidden_));
        debug::timing_log_scalar("sam_audio.t5.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    ggml_backend_t backend_;
    int64_t length_, heads_, hidden_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue tokens_;
    core::TensorValue mask_;
    ggml_tensor * output_ = nullptr;
};
}  // namespace

struct T5TextEncoder::Impl {
    core::ExecutionContext & execution;
    modules::T5BaseEncoderConfig config;
    core::BackendWeightStore store;
    modules::T5BaseEncoderWeights weights;
    std::vector<tokenizers::SentencePiecePiece> pieces;
    std::unique_ptr<T5TextEncoderGraph> graph;
    int64_t max_length;
    int32_t eos;
    int32_t pad;
    size_t graph_length = 0;

    Impl(const assets::TensorSource & source, core::ExecutionContext & execution_,
         const std::filesystem::path & config_path, const std::filesystem::path & tokenizer, int64_t max_length_)
        : execution(execution_),
          store(execution.backend(), execution.backend_type(), "sam_audio.t5.weights", 2 * 1024 * 1024),
          pieces(tokenizers::load_sentencepiece_model(tokenizer)), max_length(max_length_) {
        const auto json = io::json::parse_file(config_path);
        config.hidden_size = io::json::require_i64(json, "d_model");
        config.layers = io::json::require_i64(json, "num_layers");
        config.attention_heads = io::json::require_i64(json, "num_heads");
        config.head_dim = io::json::require_i64(json, "d_kv");
        config.intermediate_size = io::json::require_i64(json, "d_ff");
        config.vocab_size = io::json::require_i64(json, "vocab_size");
        config.relative_attention_num_buckets = io::json::require_i64(json, "relative_attention_num_buckets");
        config.relative_attention_max_distance = io::json::require_i64(json, "relative_attention_max_distance");
        config.rms_norm_eps = io::json::require_f32(json, "layer_norm_epsilon");
        config.flash_attention = true;
        eos = io::json::require_i32(json, "eos_token_id");
        pad = io::json::require_i32(json, "pad_token_id");
        if (io::json::require_string(json, "feed_forward_proj") != "relu")
            throw std::runtime_error("SAM Audio T5 expects the upstream relu feed-forward architecture");
        auto load = [&](const std::string & name) {
            const std::string full = "text_encoder.model." + name;
            return store.load_tensor(source, full, assets::TensorStorageType::Native, source.require_metadata(full).shape);
        };
        weights.embed_tokens = load("shared.weight");
        weights.relative_attention_bias = load("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
        weights.final_layer_norm = load("encoder.final_layer_norm.weight");
        for (int64_t i = 0; i < config.layers; ++i) {
            modules::T5BaseEncoderLayerWeights layer;
            const std::string prefix = "encoder.block." + std::to_string(i);
            layer.self_attention_layer_norm = load(prefix + ".layer.0.layer_norm.weight");
            layer.ffn_layer_norm = load(prefix + ".layer.1.layer_norm.weight");
            layer.q_proj.weight = load(prefix + ".layer.0.SelfAttention.q.weight");
            layer.k_proj.weight = load(prefix + ".layer.0.SelfAttention.k.weight");
            layer.v_proj.weight = load(prefix + ".layer.0.SelfAttention.v.weight");
            layer.o_proj.weight = load(prefix + ".layer.0.SelfAttention.o.weight");
            layer.wi_proj.weight = load(prefix + ".layer.1.DenseReluDense.wi.weight");
            layer.wo_proj.weight = load(prefix + ".layer.1.DenseReluDense.wo.weight");
            weights.layers.push_back(std::move(layer));
        }
        store.upload();
    }
};

T5TextEncoder::T5TextEncoder(std::shared_ptr<const assets::TensorSource> source,
                         core::ExecutionContext & execution, const std::filesystem::path & config,
                         const std::filesystem::path & tokenizer, int64_t max_length)
    : impl_(std::make_unique<Impl>(*source, execution, config, tokenizer, max_length)) {}
T5TextEncoder::~T5TextEncoder() = default;

TextConditioning T5TextEncoder::encode(const std::string & text) {
    TextConditioning result;
    result.tokens = tokenizers::tokenize_sentencepiece(impl_->pieces, text);
    if (result.tokens.size() >= static_cast<size_t>(impl_->max_length))
        result.tokens.resize(static_cast<size_t>(impl_->max_length - 1));
    result.tokens.push_back(impl_->eos);
    // Bucket prompt lengths for graph reuse; masked padding also avoids degenerate one-token matrices.
    const size_t capacity = (result.tokens.size() + 15) / 16 * 16;
    if (!impl_->graph || impl_->graph_length != capacity) {
        impl_->graph.reset();
        impl_->graph = std::make_unique<T5TextEncoderGraph>(impl_->execution, impl_->config, impl_->weights, capacity);
        impl_->graph_length = capacity;
    }
    result.features = impl_->graph->run(result.tokens, impl_->pad);
    return result;
}

}  // namespace engine::models::sam_audio
