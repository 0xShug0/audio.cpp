#pragma once

#include <cstdint>
#include <vector>

namespace engine::decoders {

struct CtcTokenSpan {
    int32_t id = 0;
    int64_t start_frame = 0;
    int64_t end_frame = 0; // exclusive, excluding trailing blanks
};

struct CtcAlignment {
    std::vector<int> path; // target state per searched frame; -1 when no path exists
    float score = 0.0f;
};

enum class CtcAlignmentSearch { Full, AnchoredBand };

// Host sequence processing only. Neural heads and device argmax stay in their
// producer graphs. Labels are the blank-interleaved target state sequence;
// positions select emission rows and prior adds a score to nonblank states.
class CtcHostDecoder {
public:
    explicit CtcHostDecoder(int32_t blank_id = 0);
    static std::vector<int32_t> frame_ids(const float * logits, int64_t frames, int64_t classes);
    std::vector<CtcTokenSpan> decode(const std::vector<int32_t> & frame_ids) const;
    CtcAlignment align(const float * log_probs, int columns, const std::vector<int> & labels,
        const std::vector<int> & positions, const float * prior, const std::vector<int> & greedy,
        CtcAlignmentSearch search = CtcAlignmentSearch::AnchoredBand);
    void release_workspace();

private:
    int32_t blank_;
    std::vector<float> previous_, padded_;
    std::vector<uint8_t> pointers_;
};

} // namespace engine::decoders
