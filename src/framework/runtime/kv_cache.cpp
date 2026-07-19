#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine::runtime {

namespace {

void write_cache_tensor(const core::TensorValue & tensor, const std::vector<float> & values) {
    if (tensor.type == GGML_TYPE_F32) {
        core::write_tensor_f32(tensor, values);
    } else if (tensor.type == GGML_TYPE_F16) {
        core::write_tensor_f16(tensor, values);
    } else {
        throw std::runtime_error("TransformerKVCache supports only f32 and f16 cache tensors");
    }
}

std::vector<float> read_cache_tensor(const core::TensorValue & tensor) {
    if (tensor.type == GGML_TYPE_F32) {
        return core::read_tensor_f32(tensor.tensor);
    }
    if (tensor.type == GGML_TYPE_F16) {
        return core::read_tensor_f16(tensor.tensor);
    }
    throw std::runtime_error("TransformerKVCache supports only f32 and f16 cache tensors");
}

}  // namespace

TransformerKVCache::TransformerKVCache(
    int64_t cache_steps,
    int64_t step_elems,
    std::vector<core::TensorValue> keys,
    std::vector<core::TensorValue> values)
    : cache_steps_(std::max<int64_t>(0, cache_steps)),
      step_elems_(std::max<int64_t>(0, step_elems)) {
    if (step_elems_ <= 0) {
        throw std::runtime_error("TransformerKVCache requires positive step_elems");
    }
    if (keys.size() != values.size()) {
        throw std::runtime_error("TransformerKVCache key/value layer counts must match");
    }
    const size_t cache_elems = static_cast<size_t>(cache_steps_ * step_elems_);
    layers_.reserve(keys.size());
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        layers_.push_back(LayerCache{
            std::move(keys[layer]),
            std::move(values[layer]),
            std::vector<float>(cache_elems, 0.0F),
            std::vector<float>(cache_elems, 0.0F),
        });
    }
}

void TransformerKVCache::import_state(const TransformerKVState & state) {
    current_end_ = state.current_end;
    if (layers_.empty()) {
        valid_steps_ = 0;
        return;
    }
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("TransformerKVCache state layer count does not match cache layer count");
    }
    const int64_t state_steps = state.layers.empty() ? 0 : state.layers.front().valid_steps;
    if (state_steps > cache_steps_) {
        throw std::runtime_error("TransformerKVCache state valid_steps exceeds cache capacity");
    }
    valid_steps_ = state_steps;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & cache = layers_[layer];
        const auto & source = state.layers[layer];
        if (source.valid_steps != state_steps) {
            throw std::runtime_error("TransformerKVCache requires consistent valid_steps across all layers");
        }
        const size_t keep_elems = static_cast<size_t>(source.valid_steps * step_elems_);
        if (source.key.size() != source.value.size()) {
            throw std::runtime_error("TransformerKVCache source key/value sizes must match");
        }
        if (source.key.size() != keep_elems) {
            throw std::runtime_error("TransformerKVCache source tensors do not match valid_steps * step_elems");
        }
        if (cache_steps_ > 0) {
            std::fill(cache.import_key_scratch.begin(), cache.import_key_scratch.end(), 0.0F);
            std::fill(cache.import_value_scratch.begin(), cache.import_value_scratch.end(), 0.0F);
            if (keep_elems > 0) {
                std::copy(source.key.begin(), source.key.end(), cache.import_key_scratch.begin());
                std::copy(source.value.begin(), source.value.end(), cache.import_value_scratch.begin());
            }
            write_cache_tensor(cache.key_tensor, cache.import_key_scratch);
            write_cache_tensor(cache.value_tensor, cache.import_value_scratch);
        }
    }
}

TransformerKVState TransformerKVCache::export_state() const {
    TransformerKVState state;
    state.current_end = current_end_;
    state.layers.resize(layers_.size());
    const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & out = state.layers[layer];
        out.valid_steps = valid_steps_;
        if (keep_elems == 0) {
            continue;
        }
        const auto key_values = read_cache_tensor(layers_[layer].key_tensor);
        const auto value_values = read_cache_tensor(layers_[layer].value_tensor);
        out.key.assign(key_values.begin(), key_values.begin() + static_cast<ptrdiff_t>(keep_elems));
        out.value.assign(value_values.begin(), value_values.begin() + static_cast<ptrdiff_t>(keep_elems));
    }
    return state;
}

void TransformerKVCache::advance_after_direct_append(int64_t steps) {
    if (steps <= 0) {
        return;
    }
    if (valid_steps_ + steps > cache_steps_) {
        throw std::runtime_error("TransformerKVCache direct append exceeds cache capacity");
    }
    valid_steps_ += steps;
    current_end_ += steps;
}

void TransformerKVCache::retain_prefix(int64_t prefix_steps) {
    if (prefix_steps < 0 || prefix_steps > valid_steps_) {
        throw std::runtime_error("TransformerKVCache prefix length exceeds current state");
    }
    valid_steps_ = prefix_steps;
    current_end_ = prefix_steps;
}

int64_t TransformerKVCache::valid_steps() const noexcept {
    return valid_steps_;
}

int64_t TransformerKVCache::current_end() const noexcept {
    return current_end_;
}

int64_t TransformerKVCache::cache_steps() const noexcept {
    return cache_steps_;
}

const core::TensorValue & TransformerKVCache::key_tensor(size_t layer) const {
    return layers_.at(layer).key_tensor;
}

const core::TensorValue & TransformerKVCache::value_tensor(size_t layer) const {
    return layers_.at(layer).value_tensor;
}

void TransformerKVCache::trace_log_state(const std::string & name, int64_t num_heads, int64_t head_dim) const {
    if (!debug::trace_log_enabled()) {
        return;
    }
    debug::trace_log_scalar(name + ".current_end", current_end_);
    if (layers_.empty() || valid_steps_ <= 0) {
        return;
    }
    const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
    const auto first_key = read_cache_tensor(layers_.front().key_tensor);
    std::vector<float> first_key_keep(first_key.begin(), first_key.begin() + static_cast<ptrdiff_t>(keep_elems));
    debug::trace_log_f32(name + ".layer0.key", {1, valid_steps_, num_heads, head_dim}, first_key_keep);
    if (layers_.size() > 1) {
        const auto last_key = read_cache_tensor(layers_.back().key_tensor);
        std::vector<float> last_key_keep(last_key.begin(), last_key.begin() + static_cast<ptrdiff_t>(keep_elems));
        debug::trace_log_f32(name + ".layer_last.key", {1, valid_steps_, num_heads, head_dim}, last_key_keep);
    }
}

core::TensorValue view_transformer_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    const char * label) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error(std::string(label) + " cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, head_dim}),
        cache.type);
}

}  // namespace engine::runtime
