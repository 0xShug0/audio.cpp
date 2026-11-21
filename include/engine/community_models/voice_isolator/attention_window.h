#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::voice_isolator {

// Where a stream's new frames go in a time-attention cache and which cached
// frames each of them sees. The cache is a ring: frame f sits in slot
// f mod capacity and keeps the rotary position it was written at, counted from
// a base, so the stream never moves its keys. Once positions pass
// rebase_after, every cached key is rotated back to a later base, keeping
// positions small however long the stream runs.
class AttentionWindow {
public:
    // window: frames a query sees, itself included (0: everything retained);
    // retained: past frames kept for attention; block: the most frames a step
    // adds; rebase_after: the largest rotary position written before a rebase.
    AttentionWindow(int64_t window, int64_t retained, int64_t block, int64_t rebase_after);

    struct Step {
        int64_t first = 0;
        int64_t frames = 0;
        // Before this step, rotate every cached key back by rebase positions (0: none).
        int64_t rebase = 0;
        // Per new frame: its slot and its rotary position.
        std::vector<int32_t> slots;
        std::vector<int32_t> positions;
    };

    Step advance(int64_t frames);
    // Additive mask [frames][capacity] for the step just taken: 0 where query q
    // sees the slot, -infinity elsewhere.
    void fill_mask(const Step & step, float * mask) const;
    void reset() noexcept;

    int64_t capacity() const noexcept { return capacity_; }

private:
    int64_t window_ = 0;
    int64_t retained_ = 0;
    int64_t capacity_ = 0;
    int64_t rebase_after_ = 0;
    int64_t written_ = 0;
    int64_t base_ = 0;
};

}  // namespace engine::models::voice_isolator
