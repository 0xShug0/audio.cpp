#include "engine/models/ace_step/text_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;
namespace binding = modules::binding;

using Clock = std::chrono::steady_clock;

struct TextEncoderLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct TextEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue embed_tokens;
    std::vector<TextEncoderLayerWeights> layers;
    core::TensorValue norm;
};

int64_t head_dim(const AceStepTextEncoderConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("ACE-Step text encoder attention config is invalid");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("ACE-Step text encoder num_attention_heads must be divisible by num_key_value_heads");
    }
    return config.head_dim;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const std::optional<core::TensorValue> & attention_mask,
    const TextEncoderLayerWeights & weights,
    const AceStepTextEncoderConfig & config) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj(
        {config.hidden_size, config.num_attention_heads * dim, false, GGML_PREC_F32});
    const modules::LinearModule k_proj(
        {config.hidden_size, config.num_key_value_heads * dim, false, GGML_PREC_F32});
    const modules::LinearModule v_proj(
        {config.hidden_size, config.num_key_value_heads * dim, false, GGML_PREC_F32});
    const modules::LinearModule o_proj(
        {config.num_attention_heads * dim, config.hidden_size, false, GGML_PREC_F32});
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});
    const modules::AddModule add;

    auto attn_in = hidden_norm.build(ctx, input, binding::norm_data(ctx, weights.input_norm));
    auto q = q_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.q_proj));
    auto k = k_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.k_proj));
    auto v = v_proj.build(ctx, attn_in, binding::linear_data(ctx, weights.v_proj));
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), binding::norm_data(ctx, weights.q_norm));
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), binding::norm_data(ctx, weights.k_norm));
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads =
        repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k), kv_repeats);
    auto v_heads =
        repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v), kv_repeats);
    auto context = modules::ScaledDotProductAttentionModule({
        dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        modules::AttentionCausality::Causal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = add.build(ctx, input, o_proj.build(ctx, context, binding::linear_data(ctx, weights.o_proj)));

    auto ff_in = hidden_norm.build(ctx, x, binding::norm_data(ctx, weights.post_norm));
    auto gate =
        modules::LinearModule(
            {config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
            .build(ctx, ff_in, binding::linear_data(ctx, weights.gate_proj));
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up =
        modules::LinearModule(
            {config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
            .build(ctx, ff_in, binding::linear_data(ctx, weights.up_proj));
    auto ff =
        modules::LinearModule(
            {config.intermediate_size, config.hidden_size, false, GGML_PREC_F32})
            .build(ctx, modules::MulModule{}.build(ctx, gate, up), binding::linear_data(ctx, weights.down_proj));
    return add.build(ctx, x, ff);
}

TextEncoderWeights load_text_encoder_weights(
    ggml_backend_t backend,
    core::BackendType backend_type,
    const AceStepAssets & assets,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.text_encoder;
    const auto & source = *assets.text_encoder_weights;
    const int64_t dim = config.head_dim;

    TextEncoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "ace_step.text_encoder.weights",
        64ull * 1024ull * 1024ull);

    weights.embed_tokens = weights.store->load_tensor(
        source,
        "embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});

    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "layers." + std::to_string(i);
        TextEncoderLayerWeights layer;
        layer.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        layer.post_norm =
            weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        layer.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        layer.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        layer.q_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {config.num_attention_heads * dim, config.hidden_size});
        layer.k_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        layer.v_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        layer.o_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.hidden_size, config.num_attention_heads * dim});
        layer.gate_proj =
            weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        layer.up_proj =
            weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        layer.down_proj =
            weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(layer));
    }

    weights.norm = weights.store->load_f32_tensor(source, "norm.weight", {config.hidden_size});
    weights.store->upload();
    return weights;
}

class GgmlContextDeleter {
public:
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

class AceStepQwenTextEncoderRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        assets::TensorStorageType weight_storage_type)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          weights_(load_text_encoder_weights(backend_, backend_type_, *assets_, weight_storage_type)) {
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step text encoder backend initialization failed");
        }
        assets_->text_encoder_weights->release_storage();
    }

    ~Graph() {
        release_runtime_graphs();
    }

    AceStepTextConditioning encode(const AceStepTokenizedText & tokens) const {
        const auto total_start = Clock::now();
        const auto ensure_start = Clock::now();
        ensure_built(static_cast<int64_t>(tokens.input_ids.size()));
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.encode.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_start, Clock::now()));
        const auto & config = assets_->config.text_encoder;
        if (tokens.input_ids.empty()) {
            throw std::runtime_error("ACE-Step text encoder requires at least one token");
        }
        const auto input_start = Clock::now();
        core::write_tensor_i32(input_ids_value_, tokens.input_ids);
        core::set_backend_threads(backend_, threads_);
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.encode.input_upload_ms",
            engine::debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step text encoder graph compute failed");
        }
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.encode.graph.compute_ms",
            engine::debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        std::vector<float> values = core::read_tensor_f32(output_);
        const int64_t valid_tokens =
            static_cast<int64_t>(std::count(tokens.attention_mask.begin(), tokens.attention_mask.end(), 1));
        AceStepTextConditioning out;
        out.tokens = valid_tokens;
        out.hidden_size = config.hidden_size;
        out.values.assign(
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(valid_tokens * config.hidden_size));
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.encode.output_read_ms",
            engine::debug::elapsed_ms(output_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.text_encoder.encode.tokens", valid_tokens);
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.encode.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    AceStepTextConditioning embed_tokens(const AceStepTokenizedText & tokens) const {
        const auto total_start = Clock::now();
        const auto & config = assets_->config.text_encoder;
        const auto ensure_start = Clock::now();
        ensure_embedding_graph(static_cast<int64_t>(tokens.input_ids.size()));
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.embed.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_start, Clock::now()));
        const auto input_start = Clock::now();
        std::vector<int32_t> padded(static_cast<size_t>(embedding_capacity_), 0);
        std::copy(tokens.input_ids.begin(), tokens.input_ids.end(), padded.begin());
        core::write_tensor_i32(embedding_input_ids_value_, padded);
        core::set_backend_threads(backend_, threads_);
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.embed.input_upload_ms",
            engine::debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, embedding_graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ACE-Step text encoder embedding graph compute failed");
        }
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.embed.graph.compute_ms",
            engine::debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        std::vector<float> values = core::read_tensor_f32(embedding_output_);
        const int64_t valid_tokens =
            static_cast<int64_t>(std::count(tokens.attention_mask.begin(), tokens.attention_mask.end(), 1));
        AceStepTextConditioning out;
        out.tokens = valid_tokens;
        out.hidden_size = config.hidden_size;
        out.values.assign(
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(valid_tokens * config.hidden_size));
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.embed.output_read_ms",
            engine::debug::elapsed_ms(output_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.text_encoder.embed.tokens", valid_tokens);
        engine::debug::timing_log_scalar(
            "ace_step.text_encoder.embed.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    void release_runtime_graphs() const {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
        }
        if (backend_ != nullptr && embedding_graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, embedding_graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (embedding_buffer_ != nullptr) {
            ggml_backend_buffer_free(embedding_buffer_);
        }
        buffer_ = nullptr;
        embedding_buffer_ = nullptr;
        ctx_.reset();
        embedding_ctx_.reset();
        input_ids_ = nullptr;
        positions_ = nullptr;
        output_ = nullptr;
        graph_ = nullptr;
        input_ids_value_ = {};
        encode_capacity_ = 0;
        embedding_input_ids_ = nullptr;
        embedding_output_ = nullptr;
        embedding_graph_ = nullptr;
        embedding_input_ids_value_ = {};
        embedding_capacity_ = 0;
    }

private:
    void ensure_built(int64_t tokens) const {
        if (tokens <= 0) {
            throw std::runtime_error("ACE-Step text encoder requires positive token count");
        }
        if (ctx_ != nullptr && encode_capacity_ == tokens) {
            return;
        }
        build(tokens);
    }

    void ensure_embedding_graph(int64_t tokens) const {
        if (tokens <= 0) {
            throw std::runtime_error("ACE-Step embedding graph requires positive token count");
        }
        if (embedding_ctx_ != nullptr && embedding_capacity_ >= tokens) {
            return;
        }
        if (backend_ != nullptr && embedding_graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, embedding_graph_);
        }
        if (embedding_buffer_ != nullptr) {
            ggml_backend_buffer_free(embedding_buffer_);
            embedding_buffer_ = nullptr;
        }
        embedding_ctx_.reset();
        embedding_input_ids_ = nullptr;
        embedding_output_ = nullptr;
        embedding_graph_ = nullptr;
        embedding_capacity_ = std::max<int64_t>(256, tokens);

        const auto & config = assets_->config.text_encoder;
        ggml_init_params params{16ull * 1024ull * 1024ull, nullptr, true};
        embedding_ctx_.reset(ggml_init(params));
        if (embedding_ctx_ == nullptr) {
            throw std::runtime_error("ACE-Step embedding ggml context initialization failed");
        }
        core::ModuleBuildContext build_ctx{embedding_ctx_.get()};
        embedding_input_ids_ = ggml_new_tensor_1d(embedding_ctx_.get(), GGML_TYPE_I32, embedding_capacity_);
        embedding_input_ids_value_ = core::wrap_tensor(
            embedding_input_ids_,
            core::TensorShape::from_dims({embedding_capacity_}),
            GGML_TYPE_I32);
        auto embedded = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(
            build_ctx,
            embedding_input_ids_value_,
            weights_.embed_tokens);
        embedding_output_ = core::reshape_tensor(
            build_ctx,
            embedded,
            core::TensorShape::from_dims({embedding_capacity_, config.hidden_size})).tensor;
        ggml_set_output(embedding_output_);
        embedding_graph_ = ggml_new_graph_custom(embedding_ctx_.get(), 16384, false);
        ggml_build_forward_expand(embedding_graph_, embedding_output_);
        embedding_buffer_ = ggml_backend_alloc_ctx_tensors(embedding_ctx_.get(), backend_);
        if (embedding_buffer_ == nullptr) {
            throw std::runtime_error("ACE-Step embedding backend buffer allocation failed");
        }
    }

    void build(int64_t tokens) const {
        const auto & config = assets_->config.text_encoder;
        if (tokens <= 0) {
            throw std::runtime_error("ACE-Step text encoder requires positive token count");
        }
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        ctx_.reset();
        input_ids_ = nullptr;
        positions_ = nullptr;
        output_ = nullptr;
        graph_ = nullptr;
        encode_capacity_ = tokens;

        ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("ACE-Step text encoder ggml context initialization failed");
        }

        input_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, encode_capacity_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, encode_capacity_);
        input_ids_value_ = core::wrap_tensor(input_ids_, core::TensorShape::from_dims({encode_capacity_}), GGML_TYPE_I32);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({encode_capacity_}), GGML_TYPE_I32);
        core::ModuleBuildContext build_ctx{ctx_.get()};
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(
            build_ctx,
            input_ids_value_,
            weights_.embed_tokens);
        x = core::reshape_tensor(build_ctx, x, core::TensorShape::from_dims({1, encode_capacity_, config.hidden_size}));
        for (int64_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
            const auto & layer = weights_.layers[static_cast<size_t>(layer_index)];
            x = decoder_layer(build_ctx, x, positions, std::nullopt, layer, config);
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(
            build_ctx,
            x,
            {weights_.norm, std::nullopt});

        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("ACE-Step text encoder backend buffer allocation failed");
        }

        std::vector<int32_t> position_values(static_cast<size_t>(encode_capacity_), 0);
        for (int64_t i = 0; i < encode_capacity_; ++i) {
            position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const AceStepAssets> assets_;
    TextEncoderWeights weights_;
    mutable std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    mutable ggml_tensor * input_ids_ = nullptr;
    mutable ggml_tensor * positions_ = nullptr;
    mutable core::TensorValue input_ids_value_;
    mutable ggml_tensor * output_ = nullptr;
    mutable ggml_cgraph * graph_ = nullptr;
    mutable ggml_backend_buffer_t buffer_ = nullptr;
    mutable int64_t encode_capacity_ = 0;
    mutable std::unique_ptr<ggml_context, GgmlContextDeleter> embedding_ctx_;
    mutable ggml_tensor * embedding_input_ids_ = nullptr;
    mutable core::TensorValue embedding_input_ids_value_;
    mutable ggml_tensor * embedding_output_ = nullptr;
    mutable ggml_cgraph * embedding_graph_ = nullptr;
    mutable ggml_backend_buffer_t embedding_buffer_ = nullptr;
    mutable int64_t embedding_capacity_ = 0;
};

AceStepQwenTextEncoderRuntime::AceStepQwenTextEncoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (assets_ == nullptr) {
        throw std::runtime_error("ACE-Step text encoder runtime requires assets");
    }
    if (execution_ == nullptr) {
        throw std::runtime_error("ACE-Step text encoder runtime requires execution context");
    }
}

AceStepQwenTextEncoderRuntime::~AceStepQwenTextEncoderRuntime() = default;

void AceStepQwenTextEncoderRuntime::prepare_runtime() const {
    if (!graph_) {
        graph_ = std::make_unique<Graph>(*execution_, assets_, weight_storage_type_);
    }
}

void AceStepQwenTextEncoderRuntime::release_runtime_graphs() const {
    if (graph_) {
        graph_->release_runtime_graphs();
    }
}

AceStepTextConditioning AceStepQwenTextEncoderRuntime::encode(const AceStepTokenizedText & tokens) const {
    const auto total_start = Clock::now();
    const auto ensure_runtime_start = Clock::now();
    prepare_runtime();
    engine::debug::timing_log_scalar(
        "ace_step.text_encoder.runtime.encode.graph.ensure_ms",
        engine::debug::elapsed_ms(ensure_runtime_start, Clock::now()));
    AceStepTextConditioning out = graph_->encode(tokens);
    engine::debug::timing_log_scalar(
        "ace_step.text_encoder.runtime.encode_ms",
        engine::debug::elapsed_ms(total_start, Clock::now()));
    return out;
}

AceStepTextConditioning AceStepQwenTextEncoderRuntime::embed_tokens(const AceStepTokenizedText & tokens) const {
    const auto total_start = Clock::now();
    const auto ensure_runtime_start = Clock::now();
    prepare_runtime();
    engine::debug::timing_log_scalar(
        "ace_step.text_encoder.runtime.embed.graph.ensure_ms",
        engine::debug::elapsed_ms(ensure_runtime_start, Clock::now()));
    AceStepTextConditioning out = graph_->embed_tokens(tokens);
    engine::debug::timing_log_scalar(
        "ace_step.text_encoder.runtime.embed_ms",
        engine::debug::elapsed_ms(total_start, Clock::now()));
    return out;
}

}  // namespace engine::models::ace_step
