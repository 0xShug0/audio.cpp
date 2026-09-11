#include "engine/models/dramabox/prompt_connector.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kConnectorWeightContextBytes = 1400ull * 1024ull * 1024ull;
constexpr float kRmsNormEps = 1.0e-6F;
constexpr double kPi = 3.14159265358979323846264338327950288;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

DramaBoxConnectorBlockWeights load_prompt_connector_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const DramaBoxTransformerConfig & config) {
    const int64_t hidden = config.connector_num_attention_heads * config.connector_attention_head_dim;
    DramaBoxConnectorBlockWeights block;
    block.attention.q = modules::binding::linear_from_source(store, source, prefix + ".attn1.to_q", storage_type, hidden, hidden, true);
    block.attention.k = modules::binding::linear_from_source(store, source, prefix + ".attn1.to_k", storage_type, hidden, hidden, true);
    block.attention.v = modules::binding::linear_from_source(store, source, prefix + ".attn1.to_v", storage_type, hidden, hidden, true);
    block.attention.out = modules::binding::linear_from_source(store, source, prefix + ".attn1.to_out.0", storage_type, hidden, hidden, true);
    block.attention.gate = modules::binding::linear_from_source(
        store,
        source,
        prefix + ".attn1.to_gate_logits",
        storage_type,
        config.connector_num_attention_heads,
        hidden,
        true);
    block.attention.q_norm =
        store.load_tensor(source, prefix + ".attn1.q_norm.weight", assets::TensorStorageType::Native, {hidden});
    block.attention.k_norm =
        store.load_tensor(source, prefix + ".attn1.k_norm.weight", assets::TensorStorageType::Native, {hidden});
    block.ff_in = modules::binding::linear_from_source(
        store, source, prefix + ".ff.net.0.proj", storage_type, hidden * 4, hidden, true);
    block.ff_out = modules::binding::linear_from_source(
        store, source, prefix + ".ff.net.2", storage_type, hidden, hidden * 4, true);
    return block;
}

std::vector<float> make_split_rope_values(
    int64_t sequence_length,
    int64_t heads,
    int64_t head_dim,
    int64_t hidden,
    int64_t max_pos,
    bool cosine) {
    const int64_t values = hidden / 2;
    const int64_t values_per_head = head_dim / 2;
    std::vector<float> out(static_cast<size_t>(heads * sequence_length * values_per_head));
    const double theta = 10000.0;
    const double start = 1.0;
    const double end = theta;
    for (int64_t t = 0; t < sequence_length; ++t) {
        const double fractional = static_cast<double>(t) / static_cast<double>(max_pos);
        for (int64_t i = 0; i < values; ++i) {
            double lin = 0.0;
            if (values > 1) {
                lin = static_cast<double>(i) / static_cast<double>(values - 1);
            }
            const double power = std::log(start) / std::log(theta) +
                                 lin * (std::log(end) / std::log(theta) - std::log(start) / std::log(theta));
            const double index = std::pow(theta, power) * kPi / 2.0;
            const double freq = index * (fractional * 2.0 - 1.0);
            const float value = static_cast<float>(cosine ? std::cos(freq) : std::sin(freq));
            const int64_t h = i / values_per_head;
            const int64_t d = i % values_per_head;
            out[static_cast<size_t>((h * sequence_length + t) * values_per_head + d)] = value;
        }
    }
    return out;
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DramaBoxConnectorAttentionWeights & weights,
    const DramaBoxPromptConnectorWeights & connector_weights,
    const DramaBoxTransformerConfig & config,
    DramaBoxPerfMode perf_mode) {
    const int64_t hidden = config.connector_num_attention_heads * config.connector_attention_head_dim;
    const int64_t heads = config.connector_num_attention_heads;
    const int64_t head_dim = config.connector_attention_head_dim;
    const ggml_prec q_precision = ggml_is_quantized(weights.q.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    const ggml_prec k_precision = ggml_is_quantized(weights.k.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    const ggml_prec v_precision = ggml_is_quantized(weights.v.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    auto q = modules::LinearModule({hidden, hidden, true, q_precision}).build(ctx, input, weights.q);
    auto k = modules::LinearModule({hidden, hidden, true, k_precision}).build(ctx, input, weights.k);
    auto v = modules::LinearModule({hidden, hidden, true, v_precision}).build(ctx, input, weights.v);
    q = modules::RMSNormModule({hidden, kRmsNormEps, true, false}).build(ctx, q, {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({hidden, kRmsNormEps, true, false}).build(ctx, k, {weights.k_norm, std::nullopt});
    auto q_heads = modules::SplitRoPEAttentionModule({heads, head_dim})
                       .build(ctx, q, connector_weights.rope_cos, connector_weights.rope_sin);
    auto k_heads = modules::SplitRoPEAttentionModule({heads, head_dim})
                       .build(ctx, k, connector_weights.rope_cos, connector_weights.rope_sin);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, v),
            core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], heads, head_dim})));
    core::TensorValue out;
    if (perf_mode == DramaBoxPerfMode::FlashAttention) {
        out = modules::ScaledDotProductAttentionModule({
            head_dim,
            modules::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32,
            modules::AttentionCausality::NonCausal,
        }).build(ctx, q_heads, k_heads, v_heads);
    } else {
        auto scores = modules::MatMulModule{}.build(
            ctx,
            q_heads,
            modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
        auto attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                core::ensure_backend_addressable_layout(ctx, scores).tensor,
                nullptr,
                1.0F / std::sqrt(static_cast<float>(head_dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
        out = modules::MatMulModule{}.build(ctx, attn, v_heads);
        out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    }
    out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, out), input.shape);
    const ggml_prec gate_precision = ggml_is_quantized(weights.gate.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    auto gates = modules::LinearModule({hidden, heads, true, gate_precision}).build(ctx, input, weights.gate);
    gates = modules::SigmoidModule{}.build(ctx, gates);
    gates = core::wrap_tensor(
        ggml_scale(ctx.ggml, gates.tensor, 2.0F),
        gates.shape,
        GGML_TYPE_F32);
    auto gate_shape = core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, 1});
    auto gate_repeated = modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim})})
                             .build(ctx, core::reshape_tensor(ctx, gates, gate_shape));
    out = modules::MulModule{}.build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, out),
            core::TensorShape::from_dims({out.shape.dims[0], out.shape.dims[1], heads, head_dim})),
        gate_repeated);
    out = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, out), input.shape);
    const ggml_prec out_precision = ggml_is_quantized(weights.out.weight.type) ? GGML_PREC_DEFAULT : GGML_PREC_F32;
    return modules::LinearModule({hidden, hidden, true, out_precision}).build(ctx, out, weights.out);
}

std::vector<float> make_connector_input(
    const DramaBoxPromptEncoding & prompt,
    int64_t max_batch,
    const DramaBoxPromptConnectorWeights & weights,
    const DramaBoxConfig & config) {
    const int64_t tokens = prompt.tokens;
    const int64_t hidden = prompt.hidden_size;
    const int64_t registers = config.transformer.connector_num_learnable_registers;
    if (tokens % registers != 0) {
        throw std::runtime_error("DramaBox connector token length must be divisible by learnable registers");
    }
    std::vector<float> out(static_cast<size_t>(max_batch * tokens * hidden), 0.0F);
    for (int64_t b = 0; b < prompt.batch; ++b) {
        int64_t valid = 0;
        for (int64_t t = 0; t < tokens; ++t) {
            if (prompt.attention_mask[static_cast<size_t>(b * tokens + t)] != 0) {
                const auto * src = prompt.audio_features.data() + static_cast<std::ptrdiff_t>((b * tokens + t) * hidden);
                auto * dst = out.data() + static_cast<std::ptrdiff_t>((b * tokens + valid) * hidden);
                std::copy_n(src, static_cast<size_t>(hidden), dst);
                ++valid;
            }
        }
        for (int64_t t = valid; t < tokens; ++t) {
            const int64_t register_index = t % registers;
            const auto * src = weights.learnable_registers_host.data() + static_cast<std::ptrdiff_t>(register_index * hidden);
            auto * dst = out.data() + static_cast<std::ptrdiff_t>((b * tokens + t) * hidden);
            std::copy_n(src, static_cast<size_t>(hidden), dst);
        }
    }
    return out;
}

}  // namespace

DramaBoxPromptConnectorWeights load_dramabox_prompt_connector_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type,
    int64_t sequence_length) {
    const auto & config = assets.config.transformer;
    const int64_t hidden = config.connector_num_attention_heads * config.connector_attention_head_dim;
    if (hidden != config.cross_attention_dim) {
        throw std::runtime_error("DramaBox prompt connector hidden size mismatch");
    }
    DramaBoxPromptConnectorWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.prompt_connector.weights",
        weight_context_bytes == 0 ? kConnectorWeightContextBytes : weight_context_bytes);
    const std::string prefix = "model.diffusion_model.audio_embeddings_connector";
    weights.learnable_registers_host = assets.audio_weights->require_f32(
        prefix + ".learnable_registers",
        {config.connector_num_learnable_registers, hidden});
    weights.blocks.reserve(static_cast<size_t>(config.connector_num_layers));
    for (int64_t i = 0; i < config.connector_num_layers; ++i) {
        weights.blocks.push_back(load_prompt_connector_block(
            *weights.store,
            *assets.audio_weights,
            prefix + ".transformer_1d_blocks." + std::to_string(i),
            weight_storage_type,
            config));
    }
    const int64_t max_pos = config.connector_positional_embedding_max_pos.empty()
        ? sequence_length
        : config.connector_positional_embedding_max_pos.front();
    weights.rope_cos = weights.store->make_f32(
        core::TensorShape::from_dims({1, config.connector_num_attention_heads, sequence_length, config.connector_attention_head_dim / 2}),
        make_split_rope_values(
            sequence_length,
            config.connector_num_attention_heads,
            config.connector_attention_head_dim,
            hidden,
            max_pos,
            true));
    weights.rope_sin = weights.store->make_f32(
        core::TensorShape::from_dims({1, config.connector_num_attention_heads, sequence_length, config.connector_attention_head_dim / 2}),
        make_split_rope_values(
            sequence_length,
            config.connector_num_attention_heads,
            config.connector_attention_head_dim,
            hidden,
            max_pos,
            false));
    weights.store->upload();
    return weights;
}

core::TensorValue build_dramabox_prompt_connector(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const DramaBoxPromptConnectorWeights & weights,
    const DramaBoxConfig & config,
    DramaBoxPerfMode perf_mode) {
    const int64_t hidden = config.transformer.connector_num_attention_heads * config.transformer.connector_attention_head_dim;
    auto x = input;
    for (const auto & block : weights.blocks) {
        auto norm = modules::RMSNormModule({hidden, kRmsNormEps, false, false}).build(ctx, x, {});
        auto attn = attention(ctx, norm, block.attention, weights, config.transformer, perf_mode);
        x = modules::AddModule{}.build(ctx, x, attn);
        norm = modules::RMSNormModule({hidden, kRmsNormEps, false, false}).build(ctx, x, {});
        auto ff = modules::FeedForwardGeluModule({hidden, hidden * 4, true}).build(ctx, norm, {
            block.ff_in.weight,
            block.ff_in.bias,
            block.ff_out.weight,
            block.ff_out.bias,
        });
        x = modules::AddModule{}.build(ctx, x, ff);
    }
    return modules::RMSNormModule({hidden, kRmsNormEps, false, false}).build(ctx, x, {});
}

std::vector<float> make_branch_conditioning_features(
    const DramaBoxConditioningEncoding & conditioning,
    int64_t branch_count,
    bool cfg_enabled,
    bool stg_enabled) {
    const int64_t row_values = conditioning.tokens * conditioning.hidden_size;
    const int64_t expected_conditioning = cfg_enabled ? 2 : 1;
    if (conditioning.batch < expected_conditioning) {
        throw std::runtime_error("DramaBox conditioning batch does not cover guidance branches");
    }
    std::vector<float> out(static_cast<size_t>(branch_count * row_values), 0.0F);
    for (int64_t branch = 0; branch < branch_count; ++branch) {
        int64_t src_branch = 0;
        if (cfg_enabled && branch == 1) {
            src_branch = 1;
        } else if (stg_enabled && branch == branch_count - 1) {
            src_branch = 0;
        }
        std::copy_n(
            conditioning.features.data() + static_cast<std::ptrdiff_t>(src_branch * row_values),
            static_cast<size_t>(row_values),
            out.data() + static_cast<std::ptrdiff_t>(branch * row_values));
    }
    return out;
}

DramaBoxConditioningEncoding select_conditioning_batch(const DramaBoxConditioningEncoding & conditioning, int64_t batch) {
    if (batch < 0 || batch >= conditioning.batch) {
        throw std::runtime_error("DramaBox conditioning batch slice is out of range");
    }
    const int64_t row_values = conditioning.tokens * conditioning.hidden_size;
    if (row_values <= 0 || static_cast<int64_t>(conditioning.features.size()) != conditioning.batch * row_values) {
        throw std::runtime_error("DramaBox conditioning batch shape mismatch");
    }
    DramaBoxConditioningEncoding out;
    out.batch = 1;
    out.tokens = conditioning.tokens;
    out.hidden_size = conditioning.hidden_size;
    out.features.resize(static_cast<size_t>(row_values));
    std::copy_n(
        conditioning.features.data() + static_cast<std::ptrdiff_t>(batch * row_values),
        static_cast<size_t>(row_values),
        out.features.data());
    return out;
}

DramaBoxConditioningEncoding join_positive_negative_conditioning(
    const DramaBoxConditioningEncoding & positive,
    const DramaBoxConditioningEncoding & negative) {
    const int64_t row_values = positive.tokens * positive.hidden_size;
    if (positive.batch != 1 ||
        negative.batch != 1 ||
        negative.tokens != positive.tokens ||
        negative.hidden_size != positive.hidden_size ||
        static_cast<int64_t>(positive.features.size()) != row_values ||
        static_cast<int64_t>(negative.features.size()) != row_values) {
        throw std::runtime_error("DramaBox positive/negative conditioning shape mismatch");
    }
    DramaBoxConditioningEncoding out;
    out.batch = 2;
    out.tokens = positive.tokens;
    out.hidden_size = positive.hidden_size;
    out.features.resize(static_cast<size_t>(2 * row_values));
    std::copy_n(positive.features.data(), static_cast<size_t>(row_values), out.features.data());
    std::copy_n(
        negative.features.data(),
        static_cast<size_t>(row_values),
        out.features.data() + static_cast<std::ptrdiff_t>(row_values));
    return out;
}

class DramaBoxPromptConnectorRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const DramaBoxAssets> assets,
        const DramaBoxPromptConnectorWeights & weights,
        int64_t max_batch,
        DramaBoxPerfMode perf_mode)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          max_batch_(max_batch),
          perf_mode_(perf_mode),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("DramaBox prompt connector backend initialization failed");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t batch) const noexcept {
        return batch == max_batch_;
    }

    DramaBoxConditioningEncoding encode(const DramaBoxPromptEncoding & prompt) const {
        const auto total_start = Clock::now();
        const auto & config = assets_->config;
        if (prompt.batch <= 0 || prompt.batch > max_batch_) {
            throw std::runtime_error("DramaBox prompt connector batch exceeds prepared max_batch");
        }
        if (prompt.tokens != config.gemma.prompt_max_length ||
            prompt.hidden_size != config.transformer.cross_attention_dim) {
            throw std::runtime_error("DramaBox prompt connector input shape mismatch");
        }
        const auto input_start = Clock::now();
        auto connector_input = make_connector_input(prompt, max_batch_, weights_, config);
        core::write_tensor_f32(input_, std::move(connector_input));
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.prompt_connector.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.prompt_connector");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox prompt connector graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.prompt_connector.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        const auto output_start = Clock::now();
        const auto full = core::read_tensor_f32(output_);
        DramaBoxConditioningEncoding out;
        out.batch = prompt.batch;
        out.tokens = prompt.tokens;
        out.hidden_size = config.transformer.cross_attention_dim;
        out.features.resize(static_cast<size_t>(out.batch * out.tokens * out.hidden_size));
        const int64_t row_values = out.tokens * out.hidden_size;
        for (int64_t b = 0; b < prompt.batch; ++b) {
            std::copy_n(
                full.data() + static_cast<std::ptrdiff_t>(b * row_values),
                static_cast<size_t>(row_values),
                out.features.data() + static_cast<std::ptrdiff_t>(b * row_values));
        }
        debug::timing_log_scalar("dramabox.prompt_connector.output_read_ms", debug::elapsed_ms(output_start, Clock::now()));
        debug::timing_log_scalar("dramabox.prompt_connector.total_ms", debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox prompt connector ggml context initialization failed");
        }
        input_ = core::wrap_tensor(
            ggml_new_tensor_3d(
                ctx_.get(),
                GGML_TYPE_F32,
                config.transformer.cross_attention_dim,
                config.gemma.prompt_max_length,
                max_batch_),
            core::TensorShape::from_dims({max_batch_, config.gemma.prompt_max_length, config.transformer.cross_attention_dim}),
            GGML_TYPE_F32);
        ggml_set_input(input_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "dramabox.prompt_connector", backend_type_};
        auto output = build_dramabox_prompt_connector(build_ctx, input_, weights_, config, perf_mode_);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox prompt connector backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.prompt_connector.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const DramaBoxAssets> assets_;
    int64_t max_batch_ = 1;
    DramaBoxPerfMode perf_mode_ = DramaBoxPerfMode::Exact;
    const DramaBoxPromptConnectorWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxPromptConnectorRuntime::DramaBoxPromptConnectorRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type,
    int64_t max_batch,
    DramaBoxPerfMode perf_mode)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type),
      max_batch_(max_batch),
      perf_mode_(perf_mode) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox prompt connector runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox prompt connector runtime requires assets");
    }
    if (max_batch_ <= 0) {
        throw std::runtime_error("DramaBox prompt connector max_batch must be positive");
    }
}

DramaBoxPromptConnectorRuntime::~DramaBoxPromptConnectorRuntime() = default;

void DramaBoxPromptConnectorRuntime::prepare(int64_t batch) const {
    if (batch <= 0 || batch > max_batch_) {
        throw std::runtime_error("DramaBox prompt connector batch exceeds session max_batch");
    }
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxPromptConnectorWeights>(load_dramabox_prompt_connector_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_,
            assets_->config.gemma.prompt_max_length));
    }
    if (!graph_ || !graph_->matches(batch)) {
        graph_.reset();
        graph_ = std::make_unique<Graph>(*execution_, assets_, *weights_, batch, perf_mode_);
    }
}

DramaBoxConditioningEncoding DramaBoxPromptConnectorRuntime::encode(const DramaBoxPromptEncoding & prompt) const {
    prepare(prompt.batch);
    return graph_->encode(prompt);
}

void DramaBoxPromptConnectorRuntime::release_runtime_state() const {
    graph_.reset();
    weights_.reset();
}

}  // namespace engine::models::dramabox
