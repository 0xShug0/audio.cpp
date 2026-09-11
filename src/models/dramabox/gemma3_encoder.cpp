#include "engine/models/dramabox/gemma3_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kGemmaPromptWeightContextBytes = 28ull * 1024ull * 1024ull * 1024ull;
constexpr size_t kGemmaPromptAggregateWeightContextBytes = 1024ull * 1024ull * 1024ull;
constexpr float kFeatureRmsNormEps = 1.0e-6F;
constexpr float kMaskNegInf = -std::numeric_limits<float>::max();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

const assets::TensorSource & source_for(
    const std::vector<std::shared_ptr<const assets::TensorSource>> & sources,
    std::string_view name) {
    for (const auto & source : sources) {
        if (source != nullptr && source->has_tensor(name)) {
            return *source;
        }
    }
    throw std::runtime_error("missing DramaBox Gemma tensor: " + std::string(name));
}

std::vector<modules::LinearWeights> load_aggregate_layers(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t output_size,
    int64_t hidden_size,
    int64_t hidden_states) {
    const int64_t flat_size = hidden_size * hidden_states;
    const auto raw = source.require_f32(prefix + ".weight", {output_size, flat_size});
    std::vector<modules::LinearWeights> layers;
    layers.reserve(static_cast<size_t>(hidden_states));
    for (int64_t layer_index = 0; layer_index < hidden_states; ++layer_index) {
        std::vector<float> packed(static_cast<size_t>(output_size * hidden_size), 0.0F);
        for (int64_t out = 0; out < output_size; ++out) {
            for (int64_t dim = 0; dim < hidden_size; ++dim) {
                packed[static_cast<size_t>(out * hidden_size + dim)] =
                    raw[static_cast<size_t>(out * flat_size + dim * hidden_states + layer_index)];
            }
        }
        modules::LinearWeights weights;
        weights.weight = store.make_from_f32(
            core::TensorShape::from_dims({output_size, hidden_size}),
            storage_type,
            std::move(packed));
        weights.bias = std::nullopt;
        layers.push_back(std::move(weights));
    }
    return layers;
}

modules::GemmaDecoderStackConfig gemma_decoder_config(const DramaBoxGemma3Config & config) {
    modules::GemmaDecoderStackConfig out;
    out.hidden_size = config.hidden_size;
    out.layers = config.num_hidden_layers;
    out.attention_heads = config.num_attention_heads;
    out.kv_heads = config.num_key_value_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.vocab_size = config.vocab_size;
    out.sliding_window_pattern = config.sliding_window_pattern;
    out.rope_theta = config.rope_theta;
    out.local_rope_theta = config.rope_local_base_freq;
    out.rope_freq_scale = 1.0F / config.rope_scaling_factor;
    out.rms_norm_eps = config.rms_norm_eps;
    out.query_pre_attn_scalar = config.query_pre_attn_scalar;
    out.scale_embeddings = true;
    out.use_fast_cuda_projection = true;
    return out;
}

modules::GemmaDecoderWeightBinding dramabox_gemma_binding(assets::TensorStorageType storage_type) {
    modules::GemmaDecoderWeightBinding binding;
    binding.model_prefix = "language_model.model";
    binding.projection_storage_type = storage_type;
    return binding;
}

core::TensorValue aggregate_hidden(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & accumulated,
    const core::TensorValue & hidden,
    const core::TensorValue & hidden_attention_mask,
    const DramaBoxGemma3PromptWeights & weights,
    const DramaBoxConfig & config,
    int64_t hidden_index) {
    const int64_t hidden_size = config.gemma.hidden_size;
    const int64_t output_size = config.transformer.cross_attention_dim;
    auto normalized = modules::RMSNormModule({hidden_size, kFeatureRmsNormEps, false, false}).build(ctx, hidden, {});
    auto mask = core::reshape_tensor(
        ctx,
        hidden_attention_mask,
        core::TensorShape::from_dims({hidden.shape.dims[0], hidden.shape.dims[1], 1}));
    mask = modules::RepeatModule({normalized.shape}).build(ctx, mask);
    normalized = modules::MulModule{}.build(ctx, normalized, mask);
    normalized = core::wrap_tensor(
        ggml_scale(
            ctx.ggml,
            core::ensure_backend_addressable_layout(ctx, normalized).tensor,
            std::sqrt(static_cast<float>(output_size) /
                      static_cast<float>(hidden_size))),
        normalized.shape,
        GGML_TYPE_F32);
    const auto & projection = weights.aggregate_layers.at(static_cast<size_t>(hidden_index));
    const ggml_prec precision = ggml_is_quantized(projection.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    auto contribution = modules::LinearModule({hidden_size, output_size, false, precision}).build(ctx, normalized, projection);
    if (!accumulated.valid()) {
        return contribution;
    }
    return modules::AddModule{}.build(ctx, accumulated, contribution);
}

std::vector<int32_t> positions(int64_t tokens) {
    std::vector<int32_t> out(static_cast<size_t>(tokens));
    for (int64_t i = 0; i < tokens; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return out;
}

std::vector<float> make_causal_key_padding_mask(
    const DramaBoxGemmaTokenBatch & tokens,
    int64_t max_batch,
    int64_t heads,
    bool keep_padded_query_diagonal) {
    std::vector<float> out(static_cast<size_t>(max_batch * heads * tokens.tokens * tokens.tokens), kMaskNegInf);
    for (int64_t b = 0; b < tokens.batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t q = 0; q < tokens.tokens; ++q) {
                if (keep_padded_query_diagonal &&
                    tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + q)] == 0) {
                    out[static_cast<size_t>(((b * heads + h) * tokens.tokens + q) * tokens.tokens + q)] = 0.0F;
                    continue;
                }
                for (int64_t k = 0; k <= q; ++k) {
                    const bool keep = tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + k)] != 0;
                    out[static_cast<size_t>(((b * heads + h) * tokens.tokens + q) * tokens.tokens + k)] =
                        keep ? 0.0F : kMaskNegInf;
                }
            }
        }
    }
    return out;
}

std::vector<float> make_hidden_attention_mask(const DramaBoxGemmaTokenBatch & tokens, int64_t max_batch) {
    std::vector<float> out(static_cast<size_t>(max_batch * tokens.tokens), 0.0F);
    for (int64_t b = 0; b < tokens.batch; ++b) {
        for (int64_t t = 0; t < tokens.tokens; ++t) {
            out[static_cast<size_t>(b * tokens.tokens + t)] =
                tokens.attention_mask[static_cast<size_t>(b * tokens.tokens + t)] != 0 ? 1.0F : 0.0F;
        }
    }
    return out;
}

}  // namespace

DramaBoxGemma3PromptWeights load_dramabox_gemma3_prompt_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType gemma_weight_storage_type,
    assets::TensorStorageType projection_weight_storage_type) {
    DramaBoxGemma3PromptWeights weights;
    const auto & config = assets.config.gemma;
    weights.encoder = modules::GemmaDecoderComponent::load_from_resolver(
        backend,
        backend_type,
        gemma_decoder_config(config),
        dramabox_gemma_binding(gemma_weight_storage_type),
        weight_context_bytes == 0 ? kGemmaPromptWeightContextBytes : weight_context_bytes,
        [&assets](std::string_view name) -> const assets::TensorSource & {
            return source_for(assets.gemma_weights, name);
        });
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.gemma3_prompt.aggregate_weights",
        kGemmaPromptAggregateWeightContextBytes);
    const std::string aggregate_prefix = "text_embedding_projection.audio_aggregate_embed";
    weights.aggregate_layers = load_aggregate_layers(
        *weights.store,
        *assets.audio_weights,
        aggregate_prefix,
        projection_weight_storage_type,
        assets.config.transformer.cross_attention_dim,
        config.hidden_size,
        config.num_hidden_layers + 1);
    weights.aggregate_bias = weights.store->load_tensor(
        *assets.audio_weights,
        aggregate_prefix + ".bias",
        assets::TensorStorageType::F32,
        {assets.config.transformer.cross_attention_dim});
    weights.store->upload();
    return weights;
}

core::TensorValue build_dramabox_gemma3_prompt_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & positions,
    const core::TensorValue & additive_attention_mask,
    const core::TensorValue & hidden_attention_mask,
    const DramaBoxGemma3PromptWeights & weights,
    const DramaBoxConfig & config) {
    if (weights.encoder.config().layers != config.gemma.num_hidden_layers) {
        throw std::runtime_error("DramaBox Gemma3 layer count mismatch");
    }
    core::TensorValue accumulated;
    const auto encoder_outputs = weights.encoder.build(
        ctx,
        input_ids,
        positions,
        additive_attention_mask,
        true);
    if (static_cast<int64_t>(encoder_outputs.captured_hidden_states.size()) != config.gemma.num_hidden_layers + 1) {
        throw std::runtime_error("DramaBox Gemma3 captured hidden state count mismatch");
    }
    for (int64_t hidden_index = 0; hidden_index <= config.gemma.num_hidden_layers; ++hidden_index) {
        accumulated = aggregate_hidden(
            ctx,
            accumulated,
            encoder_outputs.captured_hidden_states[static_cast<size_t>(hidden_index)],
            hidden_attention_mask,
            weights,
            config,
            hidden_index);
    }
    auto bias = core::reshape_tensor(
        ctx,
        weights.aggregate_bias,
        core::TensorShape::from_dims({1, 1, config.transformer.cross_attention_dim}));
    bias = modules::RepeatModule({accumulated.shape}).build(ctx, bias);
    return modules::AddModule{}.build(ctx, accumulated, bias);
}

class DramaBoxGemma3PromptRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxGemma3PromptWeights & weights,
        int64_t max_batch)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          max_batch_(max_batch),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt backend initialization failed");
        }
        if (assets_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt runtime requires assets");
        }
        if (max_batch_ <= 0) {
            throw std::runtime_error("DramaBox Gemma prompt max_batch must be positive");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch) const noexcept {
        return batch == max_batch_;
    }

    DramaBoxPromptEncoding encode(const DramaBoxGemmaTokenBatch & tokens) const {
        const auto total_start = Clock::now();
        const auto & config = assets_->config;
        if (tokens.batch <= 0 || tokens.batch > max_batch_) {
            throw std::runtime_error("DramaBox Gemma prompt batch exceeds prepared max_batch");
        }
        if (tokens.tokens != config.gemma.prompt_max_length) {
            throw std::runtime_error("DramaBox Gemma prompt token length mismatch");
        }
        std::vector<int32_t> padded_ids(static_cast<size_t>(max_batch_ * tokens.tokens), 0);
        for (int64_t b = 0; b < tokens.batch; ++b) {
            std::copy_n(
                tokens.input_ids.data() + static_cast<std::ptrdiff_t>(b * tokens.tokens),
                static_cast<size_t>(tokens.tokens),
                padded_ids.data() + static_cast<std::ptrdiff_t>(b * tokens.tokens));
        }
        core::write_tensor_i32(input_ids_, padded_ids);
        core::write_tensor_i32(positions_, positions(tokens.tokens));
        const bool use_flash_attention = true;
        const int64_t mask_heads = use_flash_attention ? 1 : config.gemma.num_attention_heads;
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.batch", tokens.batch);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.max_batch", max_batch_);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.tokens", tokens.tokens);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.flash_attention", use_flash_attention);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.mask_heads", mask_heads);
        const auto attention_mask = make_causal_key_padding_mask(tokens, max_batch_, mask_heads, use_flash_attention);
        if (use_flash_attention) {
            core::write_tensor_f16(attention_mask_, attention_mask);
        } else {
            core::write_tensor_f32(attention_mask_, attention_mask);
        }
        core::write_tensor_f32(hidden_attention_mask_, make_hidden_attention_mask(tokens, max_batch_));
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.gemma_prompt");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox Gemma prompt graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        const auto full = core::read_tensor_f32(output_);
        DramaBoxPromptEncoding out;
        out.batch = tokens.batch;
        out.tokens = tokens.tokens;
        out.hidden_size = config.transformer.cross_attention_dim;
        out.attention_mask = tokens.attention_mask;
        out.audio_features.resize(static_cast<size_t>(tokens.batch * tokens.tokens * out.hidden_size));
        const int64_t row_values = tokens.tokens * out.hidden_size;
        for (int64_t b = 0; b < tokens.batch; ++b) {
            std::copy_n(
                full.data() + static_cast<std::ptrdiff_t>(b * row_values),
                static_cast<size_t>(row_values),
                out.audio_features.data() + static_cast<std::ptrdiff_t>(b * row_values));
        }
        engine::debug::timing_log_scalar("dramabox.gemma_prompt.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.graph.max_batch", max_batch_);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.graph.tokens", config.gemma.prompt_max_length);
        engine::debug::trace_log_scalar("dramabox.gemma_prompt.graph.cuda", backend_type_ == core::BackendType::Cuda);
        ggml_init_params params{1536ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox Gemma prompt ggml context initialization failed");
        }
        const int64_t tokens = config.gemma.prompt_max_length;
        input_ids_ = core::wrap_tensor(
            ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, tokens, max_batch_),
            core::TensorShape::from_dims({max_batch_, tokens}),
            GGML_TYPE_I32);
        positions_ = core::wrap_tensor(
            ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, tokens),
            core::TensorShape::from_dims({tokens}),
            GGML_TYPE_I32);
        const bool use_flash_attention = true;
        attention_mask_ = use_flash_attention
            ? core::wrap_tensor(
                  ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, tokens, tokens, 1, max_batch_),
                  core::TensorShape::from_dims({max_batch_, 1, tokens, tokens}),
                  GGML_TYPE_F16)
            : core::wrap_tensor(
                  ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, tokens, tokens, config.gemma.num_attention_heads, max_batch_),
                  core::TensorShape::from_dims({max_batch_, config.gemma.num_attention_heads, tokens, tokens}),
                  GGML_TYPE_F32);
        hidden_attention_mask_ = core::wrap_tensor(
            ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, tokens, max_batch_),
            core::TensorShape::from_dims({max_batch_, tokens, 1}),
            GGML_TYPE_F32);
        ggml_set_input(input_ids_.tensor);
        ggml_set_input(positions_.tensor);
        ggml_set_input(attention_mask_.tensor);
        ggml_set_input(hidden_attention_mask_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.gemma_prompt", backend_type_};
        auto output = build_dramabox_gemma3_prompt_encoder(
            build_ctx,
            input_ids_,
            positions_,
            attention_mask_,
            hidden_attention_mask_,
            weights_,
            config);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox Gemma prompt backend buffer allocation failed");
        }
        core::write_tensor_i32(positions_, positions(tokens));
        engine::debug::timing_log_scalar(
            "dramabox.gemma_prompt.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t max_batch_ = 1;
    const DramaBoxGemma3PromptWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_ids_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    core::TensorValue hidden_attention_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxGemma3PromptRuntime::DramaBoxGemma3PromptRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType gemma_weight_storage_type,
    assets::TensorStorageType projection_weight_storage_type,
    int64_t max_batch)
    : execution_(&execution),
      assets_(std::move(assets)),
      gemma_weight_storage_type_(gemma_weight_storage_type),
      projection_weight_storage_type_(projection_weight_storage_type),
      max_batch_(max_batch) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox Gemma prompt runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox Gemma prompt runtime requires assets");
    }
    if (max_batch_ <= 0) {
        throw std::runtime_error("DramaBox Gemma prompt max_batch must be positive");
    }
}

DramaBoxGemma3PromptRuntime::~DramaBoxGemma3PromptRuntime() = default;

void DramaBoxGemma3PromptRuntime::prepare(int64_t batch) const {
    if (batch <= 0 || batch > max_batch_) {
        throw std::runtime_error("DramaBox Gemma prompt batch exceeds session max_batch");
    }
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxGemma3PromptWeights>(load_dramabox_gemma3_prompt_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            gemma_weight_storage_type_,
            projection_weight_storage_type_));
        for (const auto & source : assets_->gemma_weights) {
            source->release_storage();
        }
    }
    if (!graph_ || !graph_->matches(batch)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(
            *execution_,
            assets_,
            *weights_,
            batch);
    }
}

DramaBoxPromptEncoding DramaBoxGemma3PromptRuntime::encode(const DramaBoxGemmaTokenBatch & tokens) const {
    prepare(tokens.batch);
    return graph_->encode(tokens);
}

void DramaBoxGemma3PromptRuntime::release_runtime_state() const {
    graph_.reset();
    weights_.reset();
}

}  // namespace engine::models::dramabox
