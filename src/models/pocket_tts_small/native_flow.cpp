#include "engine/models/pocket_tts_small/flow_lm.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/neon/execution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace engine::models::pocket_tts_small {
namespace {
using Plan = neon::ExecutionPlan;
using Value = Plan::Value;

// Output-channel tiles avoid a scalar reduction per row. Preparation is once per
// runtime; the inner loop is portable SIMD-friendly C++, without an ISA dependency.
class Matrix {
public:
    explicit Matrix(const modules::LinearWeights & weights)
        : inputs_(weights.weight.shape.dims[1]), outputs_(weights.weight.shape.dims[0]),
          packed_(((outputs_ + 7) / 8) * inputs_ * 8, 0.0F), result_(outputs_) {
        if (weights.weight.type != GGML_TYPE_F32)
            throw std::runtime_error("Native FlowLM currently requires F32 projection weights");
        const auto values = core::read_tensor_f32(weights.weight.tensor);
        if (values.size() != static_cast<size_t>(inputs_ * outputs_))
            throw std::runtime_error("Native FlowLM projection shape differs from its weights");
        for (int row = 0; row < outputs_; ++row)
            for (int col = 0; col < inputs_; ++col)
                packed_[static_cast<size_t>((row / 8) * inputs_ + col) * 8 + row % 8] = values[row * inputs_ + col];
        if (weights.bias) {
            bias_ = core::read_tensor_f32(weights.bias->tensor);
            if (bias_.size() != static_cast<size_t>(outputs_))
                throw std::runtime_error("Native FlowLM projection bias shape differs");
        }
    }
    const std::vector<float> & run(const std::vector<float> & input) {
        if (input.size() != static_cast<size_t>(inputs_)) throw std::runtime_error("Native FlowLM projection input differs");
        for (int row = 0; row < outputs_; row += 8) {
            std::array<float, 8> sums{};
            const auto * weight = packed_.data() + static_cast<size_t>(row / 8) * inputs_ * 8;
            for (int col = 0; col < inputs_; ++col) {
                const float value = input[col];
                for (int lane = 0; lane < 8; ++lane) sums[lane] += value * weight[col * 8 + lane];
            }
            for (int lane = 0; lane < std::min(8, outputs_ - row); ++lane)
                result_[row + lane] = sums[lane] + (bias_.empty() ? 0.0F : bias_[row + lane]);
        }
        return result_;
    }
private:
    int inputs_, outputs_;
    std::vector<float> packed_, bias_, result_;
};

neon::LayerNorm load_norm(const modules::NormWeights & weights, int channels, float epsilon, bool affine = false) {
    if (affine && (!weights.weight || !weights.bias))
        throw std::runtime_error("Native FlowLM requires affine normalization weights");
    return neon::LayerNorm(weights.weight ? core::read_tensor_f32(weights.weight->tensor)
        : std::vector<float>(channels, 1.0F),
        weights.bias ? core::read_tensor_f32(weights.bias->tensor) : std::vector<float>{}, epsilon);
}

struct Layer {
    Matrix q, k, v, out, fc1, fc2;
    Plan normalize, attention, middle, finish;
    std::optional<Value> key, residual;
    std::vector<float> activated;
    Layer(const FlowLMConfig & config, const modules::TransformerEncoderBlockWeights & weights)
        : q({weights.self_attention.q_weight, weights.self_attention.q_bias}),
          k({weights.self_attention.k_weight, weights.self_attention.k_bias}),
          v({weights.self_attention.v_weight, weights.self_attention.v_bias}),
          out({weights.self_attention.out_weight, weights.self_attention.out_bias}),
          fc1({weights.feed_forward.fc1_weight, weights.feed_forward.fc1_bias}),
          fc2({weights.feed_forward.fc2_weight, weights.feed_forward.fc2_bias}) {
        const auto x = normalize.input();
        normalize.set_output(normalize.add(load_norm(weights.norm1, config.hidden_size, config.norm_eps, true), {x}));
        const auto query = attention.input(), current_k = attention.input(), current_v = attention.input();
        const auto past_k = attention.input(), past_v = attention.input(), position = attention.input();
        const int dimension = static_cast<int>(config.attention_size / config.num_heads);
        const neon::RotaryEmbedding rope(dimension, dimension, 10000.0F);
        const auto qr = attention.add(rope, {query, position});
        key = attention.add(rope, {current_k, position});
        attention.set_output(attention.add(neon::Attention(config.num_heads, config.num_kv_heads, dimension),
            {qr, past_k, past_v, *key, current_v, position}));
        attention.retain(*key);
        const auto skip = middle.input(), projected = middle.input();
        residual = middle.add(neon::ScaleAdd(weights.layer_scale1 ?
            core::read_tensor_f32(weights.layer_scale1->scale.tensor) : std::vector<float>{1.0F}), {skip, projected});
        middle.retain(*residual);
        middle.set_output(middle.add(load_norm(weights.norm2, config.hidden_size, config.norm_eps, true), {*residual}));
        const auto retained = finish.input(), update = finish.input();
        finish.set_output(finish.add(neon::ScaleAdd(weights.layer_scale2 ?
            core::read_tensor_f32(weights.layer_scale2->scale.tensor) : std::vector<float>{1.0F}), {retained, update}));
    }
};

struct HeadBlock {
    Matrix modulation, fc1, fc2;
    Plan normalize, activate;
    HeadBlock(const modules::AdaLNResidualMLPWeights & weights, int channels, float epsilon)
        : modulation(weights.modulation.projection), fc1(weights.fc1), fc2(weights.fc2) {
        const auto x = normalize.input();
        normalize.set_output(normalize.add(load_norm(weights.norm, channels, epsilon), {x}));
        const auto h = activate.input();
        activate.set_output(activate.add(neon::Activation(neon::ActivationKind::SiLU), {h}));
    }
};
}

struct NativeFlowLM::Impl {
    FlowLMConfig config;
    int capacity, threads, kv_width;
    FlowLMState state;
    Matrix input, eos, head_input, head_condition, final_modulation, final_project;
    std::optional<Matrix> z_projection;
    Plan output_norm, activate_condition, final_norm;
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::unique_ptr<HeadBlock>> head;
    std::vector<float> hidden, modulated;

    Impl(FlowLMConfig cfg, const FlowLMWeights & weights, int64_t length, int workers)
        : config(cfg), capacity(static_cast<int>(length)), threads(workers),
          kv_width(config.num_kv_heads * (config.attention_size / config.num_heads)),
          input(weights.input_linear), eos(weights.out_eos), head_input(weights.flow_net.input_projection),
          head_condition(weights.flow_net.condition_projection),
          final_modulation(weights.flow_net.output_projection.modulation.projection),
          final_project(weights.flow_net.output_projection.projection) {
        if (length <= 0 || length > INT32_MAX || workers <= 0)
            throw std::runtime_error("Native FlowLM requires a positive cache capacity and thread count");
        for (const auto & layer : weights.transformer_layers)
            layers.push_back(std::make_unique<Layer>(config, layer));
        const auto x = output_norm.input();
        output_norm.set_output(output_norm.add(load_norm(weights.out_norm, config.hidden_size, config.norm_eps, true), {x}));
        if (weights.z_projection) z_projection.emplace(*weights.z_projection);
        const auto c = activate_condition.input();
        activate_condition.set_output(activate_condition.add(neon::Activation(neon::ActivationKind::SiLU), {c}));
        for (const auto & block : weights.flow_net.residual_layers)
            head.push_back(std::make_unique<HeadBlock>(block, config.flow_hidden_size, config.flow_eps));
        const auto last = final_norm.input();
        final_norm.set_output(final_norm.add(neon::LayerNorm(
            std::vector<float>(config.flow_hidden_size, 1.0F), {}, config.flow_eps), {last}));
    }

    void forward(const std::vector<float> & embeddings) {
        const int32_t position = static_cast<int32_t>(state.current_end);
        if (position >= capacity) throw std::runtime_error("Native FlowLM cache is full");
        hidden = embeddings;
        for (size_t i = 0; i < layers.size(); ++i) {
            auto & layer = *layers[i];
            auto & cache = state.layers[i];
            const auto & normalized = layer.normalize.run({neon::TensorView(hidden, {static_cast<int>(hidden.size()), 1})}, threads);
            const auto & q = layer.q.run(normalized.values), & k = layer.k.run(normalized.values), & v = layer.v.run(normalized.values);
            const auto & context = layer.attention.run({
                neon::TensorView(q, {static_cast<int>(q.size()), 1}),
                neon::TensorView(k, {static_cast<int>(k.size()), 1}),
                neon::TensorView(v, {static_cast<int>(v.size()), 1}),
                neon::TensorView(cache.key, neon::Shape({kv_width, capacity}, neon::Layout::FrameMajor)),
                neon::TensorView(cache.value, neon::Shape({kv_width, capacity}, neon::Layout::FrameMajor)),
                neon::TensorView(&position, 1, {1})}, threads);
            const auto & projected = layer.out.run(context.values);
            const auto & norm = layer.middle.run({neon::TensorView(hidden, {static_cast<int>(hidden.size()), 1}),
                neon::TensorView(projected, {static_cast<int>(projected.size()), 1})}, threads);
            layer.activated = layer.fc1.run(norm.values);
            for (auto & value : layer.activated)
                value = 0.5F * value * (1.0F + std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value)));
            const auto & update = layer.fc2.run(layer.activated);
            hidden = layer.finish.run({layer.middle.view(*layer.residual),
                neon::TensorView(update, {static_cast<int>(update.size()), 1})}, threads).values;
            const auto rotated_key = layer.attention.view(*layer.key);
            std::copy_n(rotated_key.data, kv_width, cache.key.data() + position * kv_width);
            std::copy_n(v.data(), kv_width, cache.value.data() + position * kv_width);
            cache.valid_steps = position + 1;
        }
        ++state.current_end;
    }
};

NativeFlowLM::NativeFlowLM(FlowLMConfig config, const FlowLMWeights & weights, int64_t capacity, int threads)
    : impl_(std::make_unique<Impl>(config, weights, capacity, threads)) {}
NativeFlowLM::~NativeFlowLM() = default;

void NativeFlowLM::apply_prompt(const std::vector<float> & embeddings, int64_t steps, const FlowLMState & state) {
    auto & p = *impl_;
    if (steps <= 0 || state.layers.size() != p.layers.size() || state.current_end < 0 || state.current_end + steps > p.capacity ||
        embeddings.size() != static_cast<size_t>(steps * p.config.hidden_size))
        throw std::runtime_error("Native FlowLM prompt dimensions or cache capacity differ");
    p.state = state;
    for (auto & layer : p.state.layers) {
        const auto prefix = static_cast<size_t>(state.current_end * p.kv_width);
        if (layer.valid_steps < state.current_end || layer.key.size() < prefix || layer.value.size() < prefix)
            throw std::runtime_error("Native FlowLM voice cache is incomplete");
        layer.key.resize(static_cast<size_t>(p.capacity * p.kv_width), 0.0F);
        layer.value.resize(static_cast<size_t>(p.capacity * p.kv_width), 0.0F);
    }
    for (int64_t i = 0; i < steps; ++i) {
        const auto start = embeddings.begin() + i * p.config.hidden_size;
        p.forward(std::vector<float>(start, start + p.config.hidden_size));
    }
}

FlowLMStepResult NativeFlowLM::step(const std::vector<float> & latent, const std::vector<float> & noise) {
    auto & p = *impl_;
    if (p.state.layers.size() != p.layers.size()) throw std::runtime_error("Native FlowLM must be primed before decoding");
    p.forward(p.input.run(latent));
    const auto & norm = p.output_norm.run({neon::TensorView(p.hidden, {static_cast<int>(p.hidden.size()), 1})}, p.threads);
    const auto & c = p.z_projection ? p.z_projection->run(norm.values) : norm.values;
    FlowLMStepResult result;
    result.eos_logit = p.eos.run(c)[0];
    const auto & condition = p.head_condition.run(c);
    const auto & activated = p.activate_condition.run({neon::TensorView(condition, {static_cast<int>(condition.size()), 1})}, p.threads);
    auto x = p.head_input.run(noise);
    for (auto & block : p.head) {
        const auto & normalized = block->normalize.run({neon::TensorView(x, {static_cast<int>(x.size()), 1})}, p.threads);
        const auto & modulation = block->modulation.run(activated.values);
        const size_t width = x.size(); p.modulated.resize(width);
        for (size_t i = 0; i < width; ++i)
            p.modulated[i] = normalized.values[i] + normalized.values[i] * modulation[width + i] + modulation[i];
        const auto & first = block->fc1.run(p.modulated);
        const auto & hidden = block->activate.run({neon::TensorView(first, {static_cast<int>(first.size()), 1})}, p.threads);
        const auto & delta = block->fc2.run(hidden.values);
        for (size_t i = 0; i < width; ++i) x[i] += modulation[2 * width + i] * delta[i];
    }
    const auto & normalized = p.final_norm.run({neon::TensorView(x, {static_cast<int>(x.size()), 1})}, p.threads);
    const auto & modulation = p.final_modulation.run(activated.values);
    p.modulated.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        p.modulated[i] = normalized.values[i] + normalized.values[i] * modulation[x.size() + i] + modulation[i];
    result.next_latent = p.final_project.run(p.modulated);
    if (p.config.sampler == FlowSamplerKind::Pointwise)
        for (size_t i = 0; i < noise.size(); ++i) result.next_latent[i] += noise[i];
    return result;
}

FlowLMState NativeFlowLM::export_state() const {
    auto state = impl_->state;
    for (auto & layer : state.layers) {
        layer.key.resize(static_cast<size_t>(state.current_end * impl_->kv_width));
        layer.value.resize(static_cast<size_t>(state.current_end * impl_->kv_width));
    }
    return state;
}
}
