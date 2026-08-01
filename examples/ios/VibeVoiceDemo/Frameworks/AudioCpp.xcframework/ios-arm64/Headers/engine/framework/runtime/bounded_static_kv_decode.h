#pragma once

#include "engine/framework/runtime/kv_cache.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::runtime {

struct BoundedStaticKVDecodeConfig {
    int64_t sliding_window = 0;
    int64_t min_cache_steps = 256;
    std::string label = "bounded static KV decode";
};

struct BoundedStaticKVDecodeStep {
    int64_t position = 0;
    int64_t valid_steps = 0;
    int64_t cache_steps = 0;
    int32_t cache_slot = 0;
};

int64_t bounded_static_kv_cache_steps_for_required(
    const BoundedStaticKVDecodeConfig & config,
    int64_t required_steps);

class BoundedStaticKVDecodeCursor {
public:
    void import_state(const TransformerKVState & state, int64_t cache_steps);
    BoundedStaticKVDecodeStep next_step() const;
    void advance_after_direct_append(int64_t steps);
    void reset() noexcept;

private:
    int64_t valid_steps_ = 0;
    int64_t absolute_end_ = 0;
    int64_t cache_steps_ = 0;
};

template <typename Graph>
class BoundedStaticKVDecodeRuntime {
public:
    using Factory = std::function<std::unique_ptr<Graph>(int64_t cache_steps)>;

    BoundedStaticKVDecodeRuntime() = default;

    explicit BoundedStaticKVDecodeRuntime(BoundedStaticKVDecodeConfig config)
        : config_(std::move(config)) {}

    bool prepare_for_prefill(int64_t required_steps, const Factory & factory) {
        const int64_t target_steps = target_cache_steps(required_steps);
        if (graph_ != nullptr && graph_->cache_steps() == target_steps) {
            return false;
        }
        graph_ = factory(target_steps);
        cursor_.reset();
        return true;
    }

    bool grow_for_next_step(int64_t required_steps, const Factory & factory) {
        const int64_t target_steps = target_cache_steps(required_steps);
        if (graph_ != nullptr && graph_->cache_steps() >= target_steps) {
            return false;
        }
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " requires an initialized graph before cache growth");
        }
        auto state = graph_->export_state();
        graph_ = factory(target_steps);
        import_state(state);
        return true;
    }

    void import_state(const TransformerKVState & state) {
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " requires a graph before importing KV state");
        }
        graph_->import_state(state);
        cursor_.import_state(state, graph_->cache_steps());
    }

    BoundedStaticKVDecodeStep next_step() const {
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " requires an initialized graph before decode");
        }
        return cursor_.next_step();
    }

    void advance_after_direct_append(int64_t steps) {
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " requires an initialized graph before cache advance");
        }
        const auto before = cursor_.next_step();
        if (before.valid_steps < before.cache_steps) {
            graph_->advance_cache_after_direct_append(steps);
        }
        cursor_.advance_after_direct_append(steps);
    }

    bool has_graph() const noexcept {
        return graph_ != nullptr;
    }

    Graph & graph() {
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " graph is not initialized");
        }
        return *graph_;
    }

    const Graph & graph() const {
        if (graph_ == nullptr) {
            throw std::runtime_error(config_.label + " graph is not initialized");
        }
        return *graph_;
    }

    int64_t target_cache_steps(int64_t required_steps) const {
        return bounded_static_kv_cache_steps_for_required(config_, required_steps);
    }

private:
    BoundedStaticKVDecodeConfig config_;
    std::unique_ptr<Graph> graph_;
    BoundedStaticKVDecodeCursor cursor_;
};

}  // namespace engine::runtime
