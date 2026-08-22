#include "engine/models/midashenglm_gen/flow_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kTimeEmbeddingSize = 256;
constexpr int64_t kFlowSteps = 10;
constexpr float kFlowSigma = 0.25F;
constexpr float kFlowTemperature = 1.5F;
constexpr float kSwaySamplingCoef = -1.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> epss_schedule() {
    constexpr float kPi = 3.14159265358979323846F;
    const int values[] = {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32};
    std::vector<float> out;
    out.reserve(sizeof(values) / sizeof(values[0]));
    for (int value : values) {
        float t = static_cast<float>(value) / 32.0F;
        t = t + kSwaySamplingCoef * (std::cos(kPi * 0.5F * t) - 1.0F + t);
        out.push_back(t);
    }
    return out;
}

std::vector<float> sinus_time_embedding(float timestep, int64_t batch) {
    std::vector<float> out(static_cast<size_t>(batch * kTimeEmbeddingSize));
    const int64_t half = kTimeEmbeddingSize / 2;
    const float scale = 1000.0F;
    const float base = std::log(10000.0F) / static_cast<float>(half - 1);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < half; ++i) {
            const float value = scale * timestep * std::exp(static_cast<float>(i) * -base);
            out[static_cast<size_t>(b * kTimeEmbeddingSize + i)] = std::sin(value);
            out[static_cast<size_t>(b * kTimeEmbeddingSize + half + i)] = std::cos(value);
        }
    }
    return out;
}

void add_scaled(std::vector<float> & dst, const std::vector<float> & rhs, float scale) {
    if (dst.size() != rhs.size()) {
        throw std::runtime_error("MiDashengLM-Gen flow vector size mismatch");
    }
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += rhs[i] * scale;
    }
}

core::TensorValue linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features,
    bool bias) {
    return modules::LinearModule({in_features, out_features, bias, GGML_PREC_DEFAULT}).build(ctx, input, weights);
}

core::TensorValue dit_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const MiDashengLmGenConfig & config,
    const MiDashengLmGenFlowBlockWeights & weights) {
    const int64_t head_dim = config.flow_hidden_size / config.flow_heads;
    auto q = linear(ctx, input, {weights.attention.q_weight, weights.attention.q_bias}, config.flow_hidden_size, config.flow_hidden_size, true);
    auto k = linear(ctx, input, {weights.attention.k_weight, weights.attention.k_bias}, config.flow_hidden_size, config.flow_hidden_size, true);
    auto v = linear(ctx, input, {weights.attention.v_weight, weights.attention.v_bias}, config.flow_hidden_size, config.flow_hidden_size, true);
    q = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, q), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.flow_heads, head_dim}));
    k = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, k), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.flow_heads, head_dim}));
    v = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, v), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.flow_heads, head_dim}));
    q = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F}).build(ctx, k, positions);
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule(
        {head_dim, modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32, modules::AttentionCausality::NonCausal})
        .build(ctx, q, k, v);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.flow_hidden_size}));
    return linear(ctx, context, {weights.attention.out_weight, weights.attention.out_bias}, config.flow_hidden_size, config.flow_hidden_size, true);
}

core::TensorValue dit_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const MiDashengLmGenConfig & config,
    const MiDashengLmGenFlowBlockWeights & weights) {
    auto x = modules::RMSNormModule({config.flow_hidden_size, 1.0e-6F, true, false}).build(ctx, input, weights.norm1);
    x = dit_attention(ctx, x, positions, config, weights);
    x = modules::AddModule{}.build(ctx, input, x);
    auto ff = modules::RMSNormModule({config.flow_hidden_size, 1.0e-6F, true, false}).build(ctx, x, weights.norm2);
    ff = linear(ctx, ff, weights.mlp_in, config.flow_hidden_size, config.flow_intermediate_size, true);
    ff = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ff);
    ff = linear(ctx, ff, weights.mlp_out, config.flow_intermediate_size, config.flow_hidden_size, true);
    return modules::AddModule{}.build(ctx, x, ff);
}

MiDashengLmGenFlowBlockWeights load_block(
    core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const MiDashengLmGenConfig & config,
    engine::assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "model.flowloss.cfm.model.blocks." + std::to_string(layer);
    MiDashengLmGenFlowBlockWeights out;
    out.norm1 = binding::norm_weight_from_source(store, source, prefix + ".norm1", config.flow_hidden_size);
    out.attention.q_weight = store.load_tensor(source, prefix + ".attn.to_q.weight", storage_type, {config.flow_hidden_size, config.flow_hidden_size});
    out.attention.q_bias = store.load_f32_tensor(source, prefix + ".attn.to_q.bias", {config.flow_hidden_size});
    out.attention.k_weight = store.load_tensor(source, prefix + ".attn.to_k.weight", storage_type, {config.flow_hidden_size, config.flow_hidden_size});
    out.attention.k_bias = store.load_f32_tensor(source, prefix + ".attn.to_k.bias", {config.flow_hidden_size});
    out.attention.v_weight = store.load_tensor(source, prefix + ".attn.to_v.weight", storage_type, {config.flow_hidden_size, config.flow_hidden_size});
    out.attention.v_bias = store.load_f32_tensor(source, prefix + ".attn.to_v.bias", {config.flow_hidden_size});
    out.attention.out_weight = store.load_tensor(source, prefix + ".attn.to_out.0.weight", storage_type, {config.flow_hidden_size, config.flow_hidden_size});
    out.attention.out_bias = store.load_f32_tensor(source, prefix + ".attn.to_out.0.bias", {config.flow_hidden_size});
    out.norm2 = binding::norm_weight_from_source(store, source, prefix + ".norm2", config.flow_hidden_size);
    out.mlp_in = binding::linear_from_source(store, source, prefix + ".mlp.ff.0.0", storage_type, config.flow_intermediate_size, config.flow_hidden_size, true);
    out.mlp_out = binding::linear_from_source(store, source, prefix + ".mlp.ff.2", storage_type, config.flow_hidden_size, config.flow_intermediate_size, true);
    return out;
}

std::shared_ptr<const MiDashengLmGenFlowWeights> load_weights(
    const MiDashengLmGenAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type) {
    auto weights = std::make_shared<MiDashengLmGenFlowWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "midashenglm_gen.flow.weights",
        weight_context_bytes);
    const auto & source = *assets.weights;
    const auto & config = assets.config;
    weights->time_in = binding::linear_from_source(*weights->store, source, "model.flowloss.cfm.model.t_embedder.time_mlp.0", storage_type, config.flow_hidden_size, kTimeEmbeddingSize, true);
    weights->time_out = binding::linear_from_source(*weights->store, source, "model.flowloss.cfm.model.t_embedder.time_mlp.2", storage_type, config.flow_hidden_size, config.flow_hidden_size, true);
    weights->x_embedder = binding::linear_from_source(*weights->store, source, "model.flowloss.cfm.model.x_embedder", storage_type, config.flow_hidden_size, config.target_embedding_size, true);
    weights->c_embedder = binding::linear_from_source(*weights->store, source, "model.flowloss.cfm.model.c_embedder.cond_embedder", storage_type, config.flow_hidden_size, config.hidden_size, true);
    weights->fake_latent = weights->store->load_tensor(source, "model.flowloss.cfm.model.fake_latent", storage_type, {1, 1, config.hidden_size});
    weights->fake_latent_values = source.require_f32("model.flowloss.cfm.model.fake_latent", {1, 1, config.hidden_size});
    weights->blocks.reserve(static_cast<size_t>(config.flow_depth));
    for (int64_t layer = 0; layer < config.flow_depth; ++layer) {
        weights->blocks.push_back(load_block(*weights->store, source, config, storage_type, layer));
    }
    weights->final_norm = binding::norm_weight_from_source(*weights->store, source, "model.flowloss.cfm.model.final_layer.norm_final", config.flow_hidden_size);
    weights->final_linear = binding::linear_from_source(*weights->store, source, "model.flowloss.cfm.model.final_layer.linear", storage_type, config.target_embedding_size, config.flow_hidden_size, true);
    weights->store->upload();
    return weights;
}

}  // namespace

class MiDashengLmGenFlowRuntime::DenoiserGraph {
public:
    DenoiserGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const MiDashengLmGenFlowWeights> weights,
        MiDashengLmGenConfig config,
        int64_t batch,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          batch_(batch) {
        if (batch_ <= 0) {
            throw std::runtime_error("MiDashengLM-Gen flow graph requires positive batch");
        }
        const int64_t tokens = 1 + 2 * config_.patch_size;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiDashengLM-Gen flow graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "midashenglm_gen.flow", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{input_ctx_.get(), "midashenglm_gen.flow.inputs", execution_.backend_type()};

        x_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})).tensor;
        latent_history_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})).tensor;
        condition_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, config_.hidden_size})).tensor;
        time_embedding_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, kTimeEmbeddingSize})).tensor;
        positions_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({tokens})).tensor;
        ggml_set_input(x_);
        ggml_set_input(latent_history_);
        ggml_set_input(condition_);
        ggml_set_input(time_embedding_);

        auto x = core::wrap_tensor(x_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size}));
        auto history = core::wrap_tensor(latent_history_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size}));
        auto condition = core::wrap_tensor(condition_, core::TensorShape::from_dims({batch_, 1, config_.hidden_size}));
        auto time = core::wrap_tensor(time_embedding_, core::TensorShape::from_dims({batch_, kTimeEmbeddingSize}));
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({tokens}), GGML_TYPE_I32);

        auto time_hidden = linear(ctx, time, weights_->time_in, kTimeEmbeddingSize, config_.flow_hidden_size, true);
        time_hidden = modules::SiluModule{}.build(ctx, time_hidden);
        time_hidden = linear(ctx, time_hidden, weights_->time_out, config_.flow_hidden_size, config_.flow_hidden_size, true);
        time_hidden = core::reshape_tensor(ctx, time_hidden, core::TensorShape::from_dims({batch_, 1, config_.flow_hidden_size}));
        auto cond_hidden = linear(ctx, condition, weights_->c_embedder, config_.hidden_size, config_.flow_hidden_size, true);
        auto y = modules::AddModule{}.build(ctx, time_hidden, cond_hidden);
        auto history_hidden = linear(ctx, history, weights_->x_embedder, config_.target_embedding_size, config_.flow_hidden_size, true);
        auto x_hidden = linear(ctx, x, weights_->x_embedder, config_.target_embedding_size, config_.flow_hidden_size, true);
        auto sequence = modules::ConcatModule({1}).build(ctx, history_hidden, x_hidden);
        sequence = modules::ConcatModule({1}).build(ctx, y, sequence);
        for (const auto & block : weights_->blocks) {
            sequence = dit_block(ctx, sequence, positions, config_, block);
        }
        sequence = modules::RMSNormModule({config_.flow_hidden_size, 1.0e-6F, true, false}).build(ctx, sequence, weights_->final_norm);
        sequence = linear(ctx, sequence, weights_->final_linear, config_.flow_hidden_size, config_.target_embedding_size, true);
        auto last = modules::SliceModule({1, 1 + config_.patch_size, config_.patch_size}).build(ctx, sequence);
        output_ = core::ensure_backend_addressable_layout(ctx, last).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (input_buffer_ == nullptr || gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate MiDashengLM-Gen flow graph");
        }
        std::vector<int32_t> positions_values(static_cast<size_t>(tokens));
        for (int64_t i = 0; i < tokens; ++i) {
            positions_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(core::wrap_tensor(positions_, core::TensorShape::from_dims({tokens}), GGML_TYPE_I32), positions_values);
    }

    ~DenoiserGraph() {
        clear_graph();
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    std::vector<float> run(
        const std::vector<float> & x,
        const std::vector<float> & latent_history,
        const std::vector<float> & condition,
        float timestep) {
        const size_t patch_values = static_cast<size_t>(batch_ * config_.patch_size * config_.target_embedding_size);
        if (x.size() != patch_values || latent_history.size() != patch_values ||
            condition.size() != static_cast<size_t>(batch_ * config_.hidden_size)) {
            throw std::runtime_error("MiDashengLM-Gen flow input size mismatch");
        }
        core::write_tensor_f32(core::wrap_tensor(x_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})), x);
        core::write_tensor_f32(core::wrap_tensor(latent_history_, core::TensorShape::from_dims({batch_, config_.patch_size, config_.target_embedding_size})), latent_history);
        core::write_tensor_f32(core::wrap_tensor(condition_, core::TensorShape::from_dims({batch_, 1, config_.hidden_size})), condition);
        const auto time_values = sinus_time_embedding(timestep, batch_);
        core::write_tensor_f32(core::wrap_tensor(time_embedding_, core::TensorShape::from_dims({batch_, kTimeEmbeddingSize})), time_values);
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiDashengLM-Gen flow graph compute failed");
        }
        return core::read_tensor_float(output_);
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const MiDashengLmGenFlowWeights> weights_;
    MiDashengLmGenConfig config_;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * x_ = nullptr;
    ggml_tensor * latent_history_ = nullptr;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * time_embedding_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

MiDashengLmGenFlowRuntime::MiDashengLmGenFlowRuntime(
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen flow runtime requires assets");
    }
    rng_policy_ = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution.backend_type(),
        execution.config().device,
        "midashenglm_gen.flow.rng",
        "MiDashengLM-Gen");
    weights_ = load_weights(*assets_, execution.backend(), execution.backend_type(), weight_context_bytes, storage_type);
}

MiDashengLmGenFlowRuntime::~MiDashengLmGenFlowRuntime() = default;

std::vector<float> MiDashengLmGenFlowRuntime::sample(
    const MiDashengLmGenFlowInput & input,
    float cfg_scale,
    uint64_t seed,
    uint64_t & randn_offset_blocks) {
    if (input.batch <= 0) {
        throw std::runtime_error("MiDashengLM-Gen flow sample requires positive batch");
    }
    const auto & config = assets_->config;
    const int64_t effective_batch = cfg_scale < 1.0e-5F ? input.batch : (cfg_scale == 1.0F ? input.batch : 2 * input.batch);
    if (graph_ == nullptr || graph_->batch() != effective_batch) {
        graph_.reset();
        graph_ = std::make_unique<DenoiserGraph>(*execution_, weights_, config, effective_batch, graph_arena_bytes_);
    }
    const size_t patch_values = static_cast<size_t>(input.batch * config.patch_size * config.target_embedding_size);
    std::vector<float> y0;
    if (input.x.empty()) {
        y0 = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
            patch_values,
            seed,
            randn_offset_blocks,
            rng_policy_,
            engine::sampling::TorchRandnPrecision::Float32);
        randn_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(patch_values),
            rng_policy_);
    } else {
        if (input.x.size() != patch_values) {
            throw std::runtime_error("MiDashengLM-Gen flow initial noise size mismatch");
        }
        y0 = input.x;
    }
    std::vector<float> x(patch_values);
    for (int64_t b = 0; b < input.batch; ++b) {
        for (int64_t c = 0; c < config.target_embedding_size; ++c) {
            for (int64_t t = 0; t < config.patch_size; ++t) {
                x[static_cast<size_t>(b * config.patch_size * config.target_embedding_size + t * config.target_embedding_size + c)] =
                    y0[static_cast<size_t>(b * config.target_embedding_size * config.patch_size + c * config.patch_size + t)];
            }
        }
    }
    const auto schedule = epss_schedule();
    for (size_t step = 0; step + 1 < schedule.size(); ++step) {
        const float t0 = schedule[step];
        const float t1 = schedule[step + 1];
        const float dt = t1 - t0;
        std::vector<float> graph_x;
        std::vector<float> graph_history;
        std::vector<float> graph_condition;
        if (effective_batch == input.batch) {
            graph_x = x;
            graph_history = input.latent_history;
            graph_condition = input.condition;
        } else {
            graph_x.reserve(2 * x.size());
            graph_x.insert(graph_x.end(), x.begin(), x.end());
            graph_x.insert(graph_x.end(), x.begin(), x.end());
            graph_history.reserve(2 * input.latent_history.size());
            graph_history.insert(graph_history.end(), input.latent_history.begin(), input.latent_history.end());
            graph_history.insert(graph_history.end(), input.latent_history.begin(), input.latent_history.end());
            graph_condition.reserve(static_cast<size_t>(2 * input.batch * config.hidden_size));
            graph_condition.insert(graph_condition.end(), input.condition.begin(), input.condition.end());
            const size_t fake_count = static_cast<size_t>(input.batch * config.hidden_size);
            std::vector<float> fake(fake_count);
            for (int64_t b = 0; b < input.batch; ++b) {
                std::copy(
                    weights_->fake_latent_values.begin(),
                    weights_->fake_latent_values.end(),
                    fake.begin() + static_cast<std::ptrdiff_t>(b * config.hidden_size));
            }
            graph_condition.insert(graph_condition.end(), fake.begin(), fake.end());
        }
        auto pred = graph_->run(graph_x, graph_history, graph_condition, t0);
        if (effective_batch != input.batch) {
            for (size_t i = 0; i < patch_values; ++i) {
                pred[i] = pred[i] + (pred[i] - pred[i + patch_values]) * cfg_scale;
            }
            pred.resize(patch_values);
        }
        add_scaled(x, pred, dt);
        const float shift_scale = kFlowSigma * std::sqrt(kFlowTemperature) * std::sqrt(std::abs(dt));
        const auto noise_bct = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
            patch_values,
            seed,
            randn_offset_blocks,
            rng_policy_,
            engine::sampling::TorchRandnPrecision::Float32);
        randn_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(patch_values),
            rng_policy_);
        std::vector<float> noise(patch_values);
        for (int64_t b = 0; b < input.batch; ++b) {
            for (int64_t c = 0; c < config.target_embedding_size; ++c) {
                for (int64_t t = 0; t < config.patch_size; ++t) {
                    noise[static_cast<size_t>(b * config.patch_size * config.target_embedding_size + t * config.target_embedding_size + c)] =
                    noise_bct[static_cast<size_t>(b * config.target_embedding_size * config.patch_size + c * config.patch_size + t)];
                }
            }
        }
        if (step + 2 != schedule.size()) {
            add_scaled(x, noise, shift_scale);
        }
    }
    return x;
}

void MiDashengLmGenFlowRuntime::release_graphs() {
    graph_.reset();
}

}  // namespace engine::models::midashenglm_gen
