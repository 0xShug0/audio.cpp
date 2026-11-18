#include "engine/framework/moe_engine/moe_policy.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <stdexcept>

namespace engine::moe {
namespace {
class Lru final : public EvictionPolicy {
public:
    size_t select(const EvictionInput & input) const override {
        const auto & candidates = input.candidates;
        if (candidates.empty()) throw std::logic_error("MoE cache has no eligible victim");
        return std::min_element(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
            return a.last_used < b.last_used;
        })->slot;
    }
};
class Frequency final : public EvictionPolicy {
public:
    Frequency(size_t experts, uint64_t decay) : counts_(experts), decay_(decay) {}
    size_t select(const EvictionInput & input) const override {
        const auto & candidates = input.candidates;
        if (candidates.empty()) throw std::logic_error("MoE cache has no eligible victim");
        const auto score = [&](const auto & candidate) { return candidate.expert < 0 ? uint64_t{0} : counts_.at(static_cast<size_t>(candidate.expert)); };
        return std::min_element(candidates.begin(), candidates.end(), [&](const auto & a, const auto & b) {
            const auto x = score(a), y = score(b);
            return x != y ? x < y : a.last_used < b.last_used;
        })->slot;
    }
    void accessed(const CacheAccess & access, const CacheContext &) override {
        counts_.at(static_cast<size_t>(access.expert)) += access.assignments;
        if (++steps_ == decay_) { for (auto & count : counts_) count /= 2; steps_ = 0; }
    }
private:
    std::vector<uint64_t> counts_;
    uint64_t decay_, steps_ = 0;
};
}
PreparedPolicy prepare_lru(const PolicyParameters & parameters) {
    if (!parameters.empty()) throw std::invalid_argument("default MoE eviction has no parameters");
    PreparedPolicy prepared;
    prepared.eviction = [](size_t) { return std::make_unique<Lru>(); };
    return prepared;
}
PreparedPolicy prepare_frequency(const PolicyParameters & parameters) {
    for (const auto & item : parameters)
        if (item.first != "decay_steps") throw std::invalid_argument("unknown MoE eviction parameter: " + item.first);
    const auto decay = runtime::parse_positive_i64_option(parameters, {"decay_steps"}, 128);
    PreparedPolicy prepared;
    prepared.requirements.cache_accesses = true;
    prepared.eviction = [decay](size_t experts) { return std::make_unique<Frequency>(experts, static_cast<uint64_t>(decay)); };
    return prepared;
}
} // namespace engine::moe
