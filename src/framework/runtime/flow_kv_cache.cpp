#include "engine/framework/runtime/flow_kv_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace engine::runtime {

namespace {

void validate_cache_type(ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        return;
    }
    throw std::runtime_error("RollingFlowKVCache supports only f32/f16/bf16 cache storage");
}

std::vector<std::byte> float_values_to_bytes(const std::vector<float> & values, ggml_type type) {
    validate_cache_type(type);
    if (type == GGML_TYPE_F32) {
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        ggml_fp32_to_fp16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
        std::vector<std::byte> bytes(converted.size() * sizeof(ggml_fp16_t));
        std::memcpy(bytes.data(), converted.data(), bytes.size());
        return bytes;
    }
    std::vector<ggml_bf16_t> converted(values.size());
    ggml_fp32_to_bf16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
    std::vector<std::byte> bytes(converted.size() * sizeof(ggml_bf16_t));
    std::memcpy(bytes.data(), converted.data(), bytes.size());
    return bytes;
}

}  // namespace

RollingFlowKVCache::RollingFlowKVCache(
    const std::vector<int64_t> & layer_capacities,
    int64_t step_values) {
    reset(layer_capacities, step_values);
}

RollingFlowKVCache::RollingFlowKVCache(
    const std::vector<int64_t> & layer_capacities,
    int64_t step_values,
    ggml_type type) {
    reset(layer_capacities, step_values, type);
}

void RollingFlowKVCache::reset(
    const std::vector<int64_t> & layer_capacities,
    int64_t step_values) {
    reset(layer_capacities, step_values, GGML_TYPE_F32);
}

void RollingFlowKVCache::reset(
    const std::vector<int64_t> & layer_capacities,
    int64_t step_values,
    ggml_type type) {
    validate_cache_type(type);
    if (step_values <= 0) {
        throw std::runtime_error("RollingFlowKVCache requires positive step values");
    }
    layers_.resize(layer_capacities.size());
    for (size_t i = 0; i < layer_capacities.size(); ++i) {
        const int64_t capacity = layer_capacities[i];
        if (capacity <= 0) {
            throw std::runtime_error("RollingFlowKVCache requires positive layer capacity");
        }
        auto & layer = layers_[i];
        layer.type = type;
        layer.capacity_steps = capacity;
        layer.step_values = step_values;
        layer.step_bytes = static_cast<size_t>(step_values) * ggml_type_size(type);
        layer.valid_steps = 0;
        layer.key_bytes.assign(static_cast<size_t>(capacity) * layer.step_bytes, std::byte{0});
        layer.value_bytes.assign(static_cast<size_t>(capacity) * layer.step_bytes, std::byte{0});
    }
}

void RollingFlowKVCache::clear_valid_steps() {
    for (auto & layer : layers_) {
        std::fill(layer.key_bytes.begin(), layer.key_bytes.end(), std::byte{0});
        std::fill(layer.value_bytes.begin(), layer.value_bytes.end(), std::byte{0});
        layer.valid_steps = 0;
    }
}

const FlowKVLayerCache & RollingFlowKVCache::layer(size_t index) const {
    if (index >= layers_.size()) {
        throw std::runtime_error("RollingFlowKVCache layer index is out of range");
    }
    return layers_[index];
}

FlowKVLayerCache & RollingFlowKVCache::layer(size_t index) {
    if (index >= layers_.size()) {
        throw std::runtime_error("RollingFlowKVCache layer index is out of range");
    }
    return layers_[index];
}

void RollingFlowKVCache::append_tail(
    size_t layer_index,
    const std::vector<float> & current_key,
    const std::vector<float> & current_value,
    int64_t current_steps,
    int64_t append_steps,
    const std::string & label) {
    auto & cache = layer(layer_index);
    append_tail_bytes(
        layer_index,
        float_values_to_bytes(current_key, cache.type),
        float_values_to_bytes(current_value, cache.type),
        current_steps,
        append_steps,
        label);
}

void RollingFlowKVCache::append_tail_bytes(
    size_t layer_index,
    const std::vector<std::byte> & current_key,
    const std::vector<std::byte> & current_value,
    int64_t current_steps,
    int64_t append_steps,
    const std::string & label) {
    auto & cache = layer(layer_index);
    if (current_steps <= 0 || append_steps <= 0 || append_steps > current_steps) {
        throw std::runtime_error(label + " received invalid KV append size");
    }
    const size_t current_bytes = static_cast<size_t>(current_steps) * cache.step_bytes;
    if (current_key.size() != current_bytes || current_value.size() != current_bytes) {
        throw std::runtime_error(label + " current KV shape mismatch");
    }

    std::vector<std::byte> logical_key;
    std::vector<std::byte> logical_value;
    logical_key.reserve(static_cast<size_t>(cache.valid_steps + append_steps) * cache.step_bytes);
    logical_value.reserve(logical_key.capacity());

    const int64_t start = cache.capacity_steps - cache.valid_steps;
    for (int64_t step = start; step < cache.capacity_steps; ++step) {
        const auto byte_offset = static_cast<std::ptrdiff_t>(static_cast<size_t>(step) * cache.step_bytes);
        const auto key_begin = cache.key_bytes.begin() + byte_offset;
        logical_key.insert(logical_key.end(), key_begin, key_begin + static_cast<std::ptrdiff_t>(cache.step_bytes));
        const auto value_begin = cache.value_bytes.begin() + byte_offset;
        logical_value.insert(logical_value.end(), value_begin, value_begin + static_cast<std::ptrdiff_t>(cache.step_bytes));
    }

    logical_key.insert(
        logical_key.end(),
        current_key.begin(),
        current_key.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(append_steps) * cache.step_bytes));
    logical_value.insert(
        logical_value.end(),
        current_value.begin(),
        current_value.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(append_steps) * cache.step_bytes));

    const int64_t new_valid = std::min<int64_t>(
        cache.capacity_steps,
        static_cast<int64_t>(logical_key.size() / cache.step_bytes));
    cache.valid_steps = new_valid;
    std::fill(cache.key_bytes.begin(), cache.key_bytes.end(), std::byte{0});
    std::fill(cache.value_bytes.begin(), cache.value_bytes.end(), std::byte{0});
    const size_t copy_bytes = static_cast<size_t>(new_valid) * cache.step_bytes;
    std::copy(
        logical_key.end() - static_cast<std::ptrdiff_t>(copy_bytes),
        logical_key.end(),
        cache.key_bytes.end() - static_cast<std::ptrdiff_t>(copy_bytes));
    std::copy(
        logical_value.end() - static_cast<std::ptrdiff_t>(copy_bytes),
        logical_value.end(),
        cache.value_bytes.end() - static_cast<std::ptrdiff_t>(copy_bytes));
}

}  // namespace engine::runtime
