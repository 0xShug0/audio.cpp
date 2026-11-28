#include "engine/community_models/voice_isolator/attention_window.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace engine::models::voice_isolator {

AttentionWindow::AttentionWindow(int64_t window, int64_t retained, int64_t block, int64_t rebase_after)
    : window_(window), retained_(retained), capacity_(retained + block), rebase_after_(rebase_after) {
    if (window < 0 || retained < 1 || block < 1 || rebase_after < capacity_) {
        throw std::runtime_error("Voice Isolator attention window needs retained frames, a block and room to rebase");
    }
}

AttentionWindow::Step AttentionWindow::advance(int64_t frames) {
    if (frames < 1 || frames > capacity_ - retained_) {
        throw std::runtime_error("Voice Isolator attention block does not fit the cache");
    }
    Step step;
    step.first = written_;
    step.frames = frames;
    if (written_ + frames - 1 - base_ > rebase_after_) {
        // Frames before written_ - retained are never seen again.
        const int64_t base = std::max(base_, written_ - retained_);
        step.rebase = base - base_;
        base_ = base;
    }
    for (int64_t i = 0; i < frames; ++i) {
        step.slots.push_back(static_cast<int32_t>((written_ + i) % capacity_));
        step.positions.push_back(static_cast<int32_t>(written_ + i - base_));
    }
    written_ += frames;
    return step;
}

void AttentionWindow::fill_mask(const Step & step, float * mask) const {
    const float hidden = -std::numeric_limits<float>::infinity();
    const int64_t last = step.first + step.frames - 1;
    const int64_t oldest = step.first - std::min(step.first, retained_);
    for (int64_t q = 0; q < step.frames; ++q) {
        const int64_t newest = step.first + q;
        const int64_t lo = window_ > 0 ? std::max(oldest, newest - window_ + 1) : oldest;
        float * row = mask + q * capacity_;
        for (int64_t slot = 0; slot < capacity_; ++slot) {
            // The latest frame written to the slot.
            const int64_t frame = last - ((last - slot) % capacity_ + capacity_) % capacity_;
            row[slot] = frame >= lo && frame <= newest ? 0.0F : hidden;
        }
    }
}

void AttentionWindow::reset() noexcept {
    written_ = 0;
    base_ = 0;
}

}  // namespace engine::models::voice_isolator
