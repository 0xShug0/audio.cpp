#include "engine/models/moonshine_asr/neon_decoder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace engine::models::moonshine_asr {
namespace cpu = engine::neon;
namespace {
cpu::ExecutionPlan::Value project(cpu::ExecutionPlan &plan, cpu::ExecutionPlan::Value input,
                                  MoonshineNeonLinear &weight) {
    return std::visit(
        [&](auto &values) {
            if constexpr (std::is_same_v<std::decay_t<decltype(values)>, std::vector<float>>) {
                return plan.add(cpu::Linear(weight.input_features, weight.output_features, std::move(values),
                                            std::move(weight.bias)),
                                {input});
            } else {
                return plan.add(cpu::QuantizedLinear(weight.input_features, weight.output_features, std::move(values),
                                                     std::move(weight.bias)),
                                {input});
            }
        },
        weight.weight);
}
} // namespace

MoonshineNeonDecoder::MoonshineNeonDecoder(MoonshineDecoderConfig config, MoonshineNeonDecoderWeights weights)
    : config_(std::move(config)) {
    if (config_.hidden_size <= 0 || config_.hidden_size > INT32_MAX || config_.vocab_size <= 0 ||
        config_.vocab_size > INT32_MAX || config_.layers <= 0 || weights.layers.size() != size_t(config_.layers) ||
        weights.token_embedding.input_features != config_.hidden_size ||
        weights.token_embedding.output_features != config_.vocab_size) {
        throw std::invalid_argument("Moonshine native decoder configuration or embedding mismatch");
    }
    self_cache_.resize(weights.layers.size());
    step_inputs_.reserve(2 + 4 * weights.layers.size());
    const auto memory = cross_plan_.input();
    const auto input = step_plan_.input(), position = step_plan_.input();
    auto value = std::visit(
        [&](auto &table) {
            return step_plan_.add(cpu::Embedding(static_cast<int>(config_.hidden_size),
                                                 static_cast<int>(config_.vocab_size), std::move(table)),
                                  {input});
        },
        weights.token_embedding.weight);
    const cpu::RotaryEmbedding rotary(static_cast<int>(config_.head_dim), static_cast<int>(config_.rotary_dim),
                                      config_.rope_theta);
    const cpu::Attention attention(static_cast<int>(config_.heads), static_cast<int>(config_.kv_heads),
                                   static_cast<int>(config_.head_dim));
    for (auto &layer : weights.layers) {
        const auto cross_key = cross_plan_.add(cpu::SplitHeads(static_cast<int>(config_.kv_heads)),
                                               {project(cross_plan_, memory, layer.cross_key)});
        const auto cross_value = cross_plan_.add(cpu::SplitHeads(static_cast<int>(config_.kv_heads)),
                                                 {project(cross_plan_, memory, layer.cross_value)});
        cross_plan_.retain(cross_key);
        cross_plan_.retain(cross_value);
        cross_results_.push_back({cross_key, cross_value});
        const auto past_key = step_plan_.input(), past_value = step_plan_.input();
        const auto key_memory = step_plan_.input(), value_memory = step_plan_.input();
        auto hidden = step_plan_.add(cpu::LayerNorm(layer.input_norm, {}, 1e-5F), {value});
        const auto query = step_plan_.add(rotary, {project(step_plan_, hidden, layer.self_query), position});
        const auto key = step_plan_.add(rotary, {project(step_plan_, hidden, layer.self_key), position});
        const auto val = project(step_plan_, hidden, layer.self_value);
        step_plan_.retain(key);
        step_plan_.retain(val);
        self_results_.push_back({key, val});
        hidden = step_plan_.add(attention, {query, past_key, past_value, key, val, position});
        hidden = project(step_plan_, hidden, layer.self_output);
        value = step_plan_.add(cpu::ScaleAdd({1}), {value, hidden});
        hidden = step_plan_.add(cpu::LayerNorm(layer.cross_norm, {}, 1e-5F), {value});
        hidden = step_plan_.add(attention, {project(step_plan_, hidden, layer.cross_query), key_memory, value_memory});
        hidden = project(step_plan_, hidden, layer.cross_output);
        value = step_plan_.add(cpu::ScaleAdd({1}), {value, hidden});
        hidden = step_plan_.add(cpu::LayerNorm(layer.mlp_norm, {}, 1e-5F), {value});
        hidden = project(step_plan_, hidden, layer.mlp_input);
        hidden = step_plan_.add(cpu::SwiGLU(), {hidden});
        hidden = project(step_plan_, hidden, layer.mlp_output);
        value = step_plan_.add(cpu::ScaleAdd({1}), {value, hidden});
    }
    cross_plan_.set_output(cross_results_.back()[1]);
    value = step_plan_.add(cpu::LayerNorm(weights.norm, {}, 1e-5F), {value});
    step_plan_.set_output(project(step_plan_, value, weights.head));
}

void MoonshineNeonDecoder::prepare_memory(const std::vector<float> &memory, int capacity, int threads) {
    if (memory.empty() || memory.size() % config_.hidden_size || memory.size() / config_.hidden_size > INT32_MAX ||
        capacity <= 0 || capacity > config_.max_position_embeddings) {
        throw std::invalid_argument("Moonshine native decoder memory or capacity invalid");
    }
    capacity_ = 0;
    const int frames = static_cast<int>(memory.size() / config_.hidden_size);
    cross_plan_.run(
        {cpu::TensorView(memory, cpu::Shape({static_cast<int>(config_.hidden_size), frames}, cpu::Layout::FrameMajor))},
        threads);
    const int kv_features = static_cast<int>(config_.kv_heads * config_.head_dim);
    const cpu::Shape cache_shape({kv_features, capacity}, cpu::Layout::FrameMajor);
    step_inputs_.clear();
    step_inputs_.push_back(cpu::TensorView(&token_, 1, {1}));
    step_inputs_.push_back(cpu::TensorView(&position_, 1, {1}));
    for (size_t i = 0; i < self_cache_.size(); ++i) {
        self_cache_[i].key.resize(cache_shape.elements());
        self_cache_[i].value.resize(cache_shape.elements());
        step_inputs_.push_back(cpu::TensorView(self_cache_[i].key, cache_shape));
        step_inputs_.push_back(cpu::TensorView(self_cache_[i].value, cache_shape));
        step_inputs_.push_back(cross_plan_.view(cross_results_[i][0]));
        step_inputs_.push_back(cross_plan_.view(cross_results_[i][1]));
    }
    capacity_ = capacity;
    next_position_ = 0;
}

const std::vector<float> &MoonshineNeonDecoder::run_step(int32_t token, int32_t position, int threads) {
    if (token < 0 || token >= config_.vocab_size || position != next_position_ || position < 0 ||
        position >= capacity_) {
        throw std::invalid_argument("Moonshine native decoder token or cache position invalid");
    }
    position_ = position;
    token_ = token;
    const auto &output = step_plan_.run(step_inputs_, threads);
    const size_t kv_features = static_cast<size_t>(config_.kv_heads * config_.head_dim);
    for (size_t layer = 0; layer < self_cache_.size(); ++layer) {
        const auto key = step_plan_.view(self_results_[layer][0]), value = step_plan_.view(self_results_[layer][1]);
        std::copy_n(key.data, kv_features, self_cache_[layer].key.data() + size_t(position) * kv_features);
        std::copy_n(value.data, kv_features, self_cache_[layer].value.data() + size_t(position) * kv_features);
    }
    ++next_position_;
    return output.values;
}

} // namespace engine::models::moonshine_asr
