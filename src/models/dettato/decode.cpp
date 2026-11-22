#include "decode.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

namespace engine::models::dettato::detail {
namespace {
bool peak(const std::vector<float> &values, int64_t t, int lane, int64_t frames, float threshold) {
    const float value = values[static_cast<size_t>(t * 88 + lane)];
    const float left = t ? values[static_cast<size_t>((t - 1) * 88 + lane)] : -1;
    const float right = t + 1 < frames ? values[static_cast<size_t>((t + 1) * 88 + lane)] : -1;
    return value >= threshold && value > left && value >= right;
}

double refine(const std::vector<float> &values, int64_t t, int lane, int64_t frames) {
    const float value = values[static_cast<size_t>(t * 88 + lane)];
    const float left = t ? values[static_cast<size_t>((t - 1) * 88 + lane)] : value;
    const float right = t + 1 < frames ? values[static_cast<size_t>((t + 1) * 88 + lane)] : value;
    const float denominator = 2 * (value - std::min(left, right));
    return denominator <= 1e-6f ? 0 : std::clamp((right - left) / denominator, -0.5f, 0.5f);
}
} // namespace

std::vector<Note> decode_notes(const std::array<std::vector<float>, 6> &maps, int64_t frames, double hop,
                               const NoteThresholds &thresholds) {
    if (frames <= 0 || hop <= 0 || thresholds.low_frames <= 0)
        throw std::runtime_error("Invalid Dettato note-decoding shape or thresholds");
    for (const auto &values : maps)
        if (values.size() != static_cast<size_t>(frames * 88))
            throw std::runtime_error("Dettato probability-map shape mismatch");
    std::vector<Note> notes;
    for (int voice = 0; voice < 2; ++voice) {
        const auto &on = maps[voice * 3];
        const auto &fr = maps[voice * 3 + 1];
        const auto &off = maps[voice * 3 + 2];
        for (int lane = 0; lane < 88; ++lane) {
            std::vector<int64_t> starts, ends, lows;
            for (int64_t t = 0; t < frames; ++t) {
                if (peak(on, t, lane, frames, voice ? thresholds.second_onset : thresholds.onset))
                    starts.push_back(t);
                if (peak(off, t, lane, frames, thresholds.offset))
                    ends.push_back(t);
                bool low = true;
                for (int k = 0; k < thresholds.low_frames && t + k < frames; ++k)
                    low = low && fr[static_cast<size_t>((t + k) * 88 + lane)] < thresholds.frame;
                if (low)
                    lows.push_back(t);
            }
            if (thresholds.onset_nms > 0 && starts.size() > 1) {
                std::vector<size_t> order(starts.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return on[static_cast<size_t>(starts[a] * 88 + lane)] >
                           on[static_cast<size_t>(starts[b] * 88 + lane)];
                });
                std::vector<bool> keep(starts.size(), true);
                for (const auto i : order) {
                    if (!keep[i])
                        continue;
                    for (size_t j = 0; j < starts.size(); ++j)
                        if (i != j && std::abs(starts[i] - starts[j]) <= thresholds.onset_nms)
                            keep[j] = false;
                }
                size_t count = 0;
                for (size_t i = 0; i < starts.size(); ++i)
                    if (keep[i])
                        starts[count++] = starts[i];
                starts.resize(count);
            }
            for (size_t i = 0; i < starts.size(); ++i) {
                const auto t = starts[i], next = i + 1 < starts.size() ? starts[i + 1] : frames;
                const auto low = std::lower_bound(lows.begin(), lows.end(), t + 1);
                const auto end = std::lower_bound(ends.begin(), ends.end(), t + 2);
                const auto frame_end = low == lows.end() ? frames : *low;
                const auto offset_end = end == ends.end() ? frames : *end;
                const double onset = t + refine(on, t, lane, frames);
                double stop;
                if (offset_end < frames && offset_end <= std::min(frame_end, next))
                    stop = offset_end + refine(off, offset_end, lane, frames);
                else if (frame_end <= next)
                    stop = frame_end - 0.5;
                else
                    stop = next + (next < frames ? refine(on, next, lane, frames) : 0);
                const auto confidence_frame =
                    std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(onset)), 0, frames - 1);
                notes.push_back({onset * hop, std::max(stop * hop, onset * hop + 0.02), 21 + lane, 100,
                                 on[static_cast<size_t>(confidence_frame * 88 + lane)], voice + 1, false});
            }
        }
    }
    std::sort(notes.begin(), notes.end(), [](const Note &a, const Note &b) {
        return std::tie(a.onset, a.offset, a.pitch) < std::tie(b.onset, b.offset, b.pitch);
    });
    return notes;
}
} // namespace engine::models::dettato::detail
