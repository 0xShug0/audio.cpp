#pragma once

#include "engine/framework/runtime/cache_slots.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace engine::models::tiny_voca {

// A graph is made for one stack and one length.
struct GraphKey {
    int stack = 0;
    int64_t length = 0;

    bool operator==(const GraphKey & other) const noexcept {
        return stack == other.stack && length == other.length;
    }
};

struct GraphCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    size_t kept_graphs = 0;
    size_t kept_bytes = 0;
};

// The graphs a session keeps from one sentence to the next, least recently
// used first out. Two limits hold at all times: the number of graphs, and the
// bytes their buffers take together. The second one is needed because a graph
// grows with its length, from about 0.3 MB to about 25 MB at the model's
// limit; a count alone would let a few long sentences hold more memory than
// the rest of the model. A graph larger than the byte limit is never kept.
template <typename Graph>
class GraphCache {
public:
    GraphCache(size_t max_graphs, size_t max_bytes)
        : slots_(max_graphs), max_graphs_(max_graphs), max_bytes_(max_bytes) {}

    // The kept graph for `key`, or nullptr. Counts a hit or a miss.
    Graph * find(const GraphKey & key) {
        if (auto * found = slots_.find(key)) {
            ++hits_;
            return found->graph.get();
        }
        ++misses_;
        return nullptr;
    }

    // Keeps `graph`, which takes `bytes`, and returns nullptr; older graphs go
    // until both limits hold. A graph that cannot be kept comes back to the
    // caller, who frees it after use.
    std::unique_ptr<Graph> put(const GraphKey & key, std::unique_ptr<Graph> graph, size_t bytes) {
        if (max_graphs_ == 0 || bytes > max_bytes_) {
            return graph;
        }
        while (slots_.size() >= max_graphs_ || kept_bytes() + bytes > max_bytes_) {
            drop_oldest();
        }
        slots_.put(key, Entry{std::move(graph), bytes});
        return nullptr;
    }

    GraphCacheStats stats() {
        return {hits_, misses_, slots_.size(), kept_bytes()};
    }

private:
    struct Entry {
        std::unique_ptr<Graph> graph;
        size_t bytes = 0;
    };

    size_t kept_bytes() {
        size_t total = 0;
        slots_.for_each_value([&](Entry & entry) { total += entry.bytes; });
        return total;
    }

    void drop_oldest() {
        slots_.set_capacity(slots_.size() - 1);
        slots_.set_capacity(max_graphs_);
    }

    runtime::CacheSlots<GraphKey, Entry> slots_;
    size_t max_graphs_ = 0;
    size_t max_bytes_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
};

}  // namespace engine::models::tiny_voca
