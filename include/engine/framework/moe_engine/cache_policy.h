#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
#include <unordered_set>
#include <vector>

namespace engine::core {

// Backend-independent eviction. Candidate order is stable; IDs are logical
// experts, while the returned candidate index is resolved by the weight store.
// The policy owns eviction history, never weight storage or execution resources.
class MoeCachePolicy {
public:
    virtual ~MoeCachePolicy() = default;
    virtual void access(const std::unordered_set<int32_t> & required, uint64_t tick) = 0;
    virtual size_t victim(const std::vector<int32_t> & candidates,
        const std::unordered_set<int32_t> & required) const = 0;
    virtual void touch(size_t candidate, uint64_t tick) = 0;
    virtual void reset() = 0;
};

using MoeCacheFactory = std::function<std::unique_ptr<MoeCachePolicy>(size_t experts, size_t slots)>;
std::unique_ptr<MoeCachePolicy> make_moe_lru_cache(size_t experts, size_t slots);
std::unique_ptr<MoeCachePolicy> make_moe_frequency_cache(size_t experts, size_t slots, uint64_t decay_steps);

MoeCacheFactory prepare_moe_frequency_cache(uint64_t decay_steps);

} // namespace engine::core
