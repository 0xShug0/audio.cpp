#include "engine/framework/moe_engine/cache_policy.h"

#include <algorithm>
#include <stdexcept>

namespace engine::core {
namespace {

class LruCachePolicy : public MoeCachePolicy {
public:
    explicit LruCachePolicy(size_t slots) : last_used_(slots, 0) {}
    void access(const std::unordered_set<int32_t> &, uint64_t) override {}
    size_t victim(const std::vector<int32_t> & ids, const std::unordered_set<int32_t> & required) const override {
        size_t slot = ids.size();
        for (size_t candidate = 0; candidate < ids.size(); ++candidate) {
            if (required.count(ids[candidate])) continue;
            if (slot == ids.size() || last_used_[candidate] < last_used_[slot]) slot = candidate;
        }
        return slot;
    }
    void touch(size_t candidate, uint64_t tick) override { last_used_[candidate] = tick; }
    void reset() override {
        std::fill(last_used_.begin(), last_used_.end(), 0);
    }
protected:
    std::vector<uint64_t> last_used_;
};

class FrequencyCachePolicy final : public LruCachePolicy {
public:
    FrequencyCachePolicy(size_t experts, size_t slots, uint64_t decay_steps)
        : LruCachePolicy(slots), frequency_(experts, 0), decay_steps_(decay_steps) {}
    void access(const std::unordered_set<int32_t> & required, uint64_t) override {
        if (++accesses_ % decay_steps_ == 0)
            for (auto & count : frequency_) count /= 2;
        for (auto id : required) ++frequency_[id];
    }
    size_t victim(const std::vector<int32_t> & ids, const std::unordered_set<int32_t> & required) const override {
        size_t slot = ids.size();
        for (size_t candidate = 0; candidate < ids.size(); ++candidate) {
            if (required.count(ids[candidate])) continue;
            if (ids[candidate] < 0) return candidate;
            if (slot == ids.size() || frequency_[ids[candidate]] < frequency_[ids[slot]] ||
                (frequency_[ids[candidate]] == frequency_[ids[slot]] && last_used_[candidate] < last_used_[slot]))
                slot = candidate;
        }
        return slot;
    }
    void reset() override {
        LruCachePolicy::reset();
        std::fill(frequency_.begin(), frequency_.end(), 0);
        accesses_ = 0;
    }
private:
    std::vector<uint64_t> frequency_;
    uint64_t decay_steps_;
    uint64_t accesses_ = 0;
};

} // namespace

std::unique_ptr<MoeCachePolicy> make_moe_lru_cache(size_t, size_t slots) {
    return std::make_unique<LruCachePolicy>(slots);
}
std::unique_ptr<MoeCachePolicy> make_moe_frequency_cache(size_t experts, size_t slots, uint64_t decay_steps) {
    if (!decay_steps) throw std::invalid_argument("frequency cache requires positive decay_steps");
    return std::make_unique<FrequencyCachePolicy>(experts, slots, decay_steps);
}

MoeCacheFactory prepare_moe_frequency_cache(uint64_t decay_steps) {
    if (!decay_steps) throw std::invalid_argument("frequency cache requires positive decay_steps");
    return [decay_steps](size_t experts, size_t slots) {
        return make_moe_frequency_cache(experts, slots, decay_steps);
    };
}

} // namespace engine::core
