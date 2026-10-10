#include "engine/framework/audio/probability_segmenter.h"

namespace engine::audio {

ProbabilitySegmenter::ProbabilitySegmenter(ProbabilitySegmenterConfig config) : config_(config) {}

std::optional<ProbabilitySegment> ProbabilitySegmenter::push(float probability, int64_t start, int64_t end) {
    if (probability >= config_.threshold) {
        if (!active_) {
            active_ = true;
            start_ = start;
            confidence_sum_ = 0.0;
            confidence_count_ = 0;
        }
        end_ = end;
        confidence_sum_ += probability;
        ++confidence_count_;
    } else if (active_ && start - end_ >= config_.min_silence) {
        return finish();
    }
    return std::nullopt;
}

std::optional<ProbabilitySegment> ProbabilitySegmenter::finish() {
    if (!active_) return std::nullopt;
    active_ = false;
    if (end_ - start_ < config_.min_speech) return std::nullopt;
    return ProbabilitySegment{start_, end_,
        static_cast<float>(confidence_sum_ / static_cast<double>(confidence_count_))};
}

}  // namespace engine::audio
