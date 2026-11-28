// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Anchored CTC search adapted from NVIDIA/NeMo-Speech.cpp, revision 66d50d91.
#include "engine/framework/decoders/ctc_host_decoder.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace engine::decoders {
namespace {
void require_log_values(const float * values, size_t count) {
    if (std::any_of(values, values + count, [](float value) {
        return std::isnan(value) || value == std::numeric_limits<float>::infinity();
    })) throw std::invalid_argument("CTC: invalid log probabilities");
}
} // namespace

CtcHostDecoder::CtcHostDecoder(int32_t blank_id) : blank_(blank_id) {
    if (blank_id < 0) throw std::invalid_argument("CTC blank ID must be nonnegative");
}

std::vector<int32_t> CtcHostDecoder::frame_ids(const float * logits, int64_t frames, int64_t classes) {
    if (frames < 0 || classes <= 0 || classes > INT_MAX || (frames && !logits))
        throw std::invalid_argument("CTC: invalid logits shape");
    std::vector<int32_t> ids(static_cast<size_t>(frames));
    for (int64_t t = 0; t < frames; ++t) {
        const auto * row = logits + static_cast<size_t>(t) * classes;
        int32_t best = 0;
        float best_value = row[0];
        for (int64_t v = 1; v < classes; ++v) {
            if (row[v] > best_value) {
                best_value = row[v];
                best = static_cast<int32_t>(v);
            }
        }
        ids[static_cast<size_t>(t)] = best;
    }
    return ids;
}

std::vector<CtcTokenSpan> CtcHostDecoder::decode(const std::vector<int32_t> & ids) const {
    std::vector<CtcTokenSpan> spans;
    int32_t previous = blank_;
    for (size_t t = 0; t < ids.size(); ++t) {
        const auto id = ids[t];
        if (id < 0) throw std::invalid_argument("CTC: negative frame token ID");
        if (id != blank_) {
            if (id != previous) spans.push_back({id, static_cast<int64_t>(t), static_cast<int64_t>(t + 1)});
            else spans.back().end_frame = static_cast<int64_t>(t + 1);
        }
        previous = id;
    }
    return spans;
}

void CtcHostDecoder::release_workspace() {
    std::vector<float>().swap(previous_);
    std::vector<float>().swap(padded_);
    std::vector<uint8_t>().swap(pointers_);
}

namespace {

// Start positions of every token trigram: first position and count per trigram.
std::map<std::array<int, 3>, std::pair<int, int>>
trigram_positions(const std::vector<int>& tokens) {
    std::map<std::array<int, 3>, std::pair<int, int>> table;
    for (size_t i = 0; i + 2 < tokens.size(); ++i) {
        auto [it, inserted] =
            table.try_emplace({tokens[i], tokens[i + 1], tokens[i + 2]}, static_cast<int>(i), 0);
        ++it->second.second;
    }
    return table;
}

// NeMo's _ctc_anchor_band_starts.
std::vector<int64_t>
anchor_band_starts(
    const std::vector<int>& greedy, const std::vector<int>& target, int64_t states, int64_t half,
    int64_t width, int blank) {
    const int64_t frames = static_cast<int64_t>(greedy.size());
    std::vector<int> observed, times;
    for (int64_t i = 0; i < frames; ++i)
        if (greedy[i] != blank && (i == 0 || greedy[i] != greedy[i - 1])) {
            observed.push_back(greedy[i]);
            times.push_back(static_cast<int>(i));
        }
    const auto left = trigram_positions(observed), right = trigram_positions(target);
    std::vector<std::pair<int64_t, int64_t>> pairs;
    for (const auto& [key, position] : left) {
        if (position.second != 1)
            continue;
        const auto it = right.find(key);
        if (it != right.end() && it->second.second == 1)
            pairs.emplace_back(times[position.first + 1], 2 * (it->second.first + 1) + 1);
    }
    std::sort(pairs.begin(), pairs.end());
    // Longest chain increasing in state (patience sort with parents).
    std::vector<int64_t> tails;
    std::vector<int> tail_ids, parent;
    for (size_t i = 0; i < pairs.size(); ++i) {
        const size_t j =
            std::lower_bound(tails.begin(), tails.end(), pairs[i].second) - tails.begin();
        parent.push_back(j ? tail_ids[j - 1] : -1);
        if (j == tails.size()) {
            tails.push_back(pairs[i].second);
            tail_ids.push_back(static_cast<int>(i));
        } else {
            tails[j] = pairs[i].second;
            tail_ids[j] = static_cast<int>(i);
        }
    }
    std::vector<std::pair<int64_t, int64_t>> chosen;
    for (int k = tail_ids.empty() ? -1 : tail_ids.back(); k >= 0; k = parent[k])
        chosen.push_back(pairs[k]);
    std::reverse(chosen.begin(), chosen.end());
    std::vector<double> xs{0.0}, ys{0.0};
    for (const auto& [t, s] : chosen)
        if (0 < t && t < frames - 1) {
            xs.push_back(static_cast<double>(t));
            ys.push_back(static_cast<double>(s));
        }
    xs.push_back(static_cast<double>(frames - 1));
    ys.push_back(static_cast<double>(states - 1));
    // np.interp at 0..frames-1, then rint, clip and the reachability limits.
    std::vector<int64_t> starts(frames);
    const size_t n = xs.size();
    for (int64_t t = 0; t < frames; ++t) {
        const double x = static_cast<double>(t);
        const size_t j = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin() - 1;
        double center;
        if (j == n - 1 || xs[j] == x)
            center = ys[j];
        else
            center = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (x - xs[j]) + ys[j];
        const int64_t start = static_cast<int64_t>(std::nearbyint(center)) - half;
        starts[t] = std::min(std::max<int64_t>(start, 0), states - width);
    }
    int64_t running = std::numeric_limits<int64_t>::max();
    for (int64_t t = 0; t < frames; ++t) {
        running = std::min(running, starts[t] - 2 * t);
        starts[t] =
            std::max(running + 2 * t, std::max<int64_t>(0, states - width - 2 * (frames - 1 - t)));
    }
    return starts;
}

}  // namespace

CtcAlignment
CtcHostDecoder::align(
    const float* log_probs, int columns, const std::vector<int>& labels,
    const std::vector<int>& positions, const float* prior, const std::vector<int>& greedy, CtcAlignmentSearch search) {
    if (columns < 1 || blank_ >= columns || (!positions.empty() && !log_probs) || labels.empty() ||
        greedy.size() > size_t(INT_MAX) || labels.size() > size_t(INT_MAX))
        throw std::invalid_argument("banded_ctc_search: invalid input shape");
    for (int label : labels)
        if (label < 0 || label >= columns)
            throw std::invalid_argument("banded_ctc_search: class ID outside columns");
    for (int label : greedy)
        if (label < 0 || label >= columns)
            throw std::invalid_argument("banded_ctc_search: invalid greedy class ID");
    int previous_position = -1;
    for (int position : positions) {
        if (position <= previous_position || size_t(position) >= greedy.size())
            throw std::invalid_argument("banded_ctc_search: invalid searched position");
        previous_position = position;
        require_log_values(log_probs + size_t(position) * columns, size_t(columns));
        if (prior) require_log_values(prior + position, 1);
    }
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int64_t frames = static_cast<int64_t>(positions.size());
    const int64_t states = static_cast<int64_t>(labels.size());
    CtcAlignment out{std::vector<int>(frames, -1), neg_inf};
    const int64_t half =
        search == CtcAlignmentSearch::Full || frames < 1000
            ? states
            : std::min<int64_t>(states, static_cast<int64_t>(std::ceil(0.10 * states + 100)));
    const int64_t width = std::min<int64_t>(states, 2 * half + 1);
    std::vector<int> target;
    for (int64_t s = 1; s < states; s += 2) target.push_back(labels[s]);
    int64_t needed = static_cast<int64_t>(target.size());
    for (size_t k = 1; k < target.size(); ++k) needed += target[k] == target[k - 1];
    if (frames < needed || frames == 0)
        return out;
    std::vector<int> searched(frames);
    for (int64_t t = 0; t < frames; ++t) searched[t] = greedy[positions[t]];
    const auto starts = search == CtcAlignmentSearch::Full ? std::vector<int64_t>(frames, 0) :
        anchor_band_starts(searched, target, states, half, width, blank_);
    const auto row = [&](int64_t t) {
        return log_probs + static_cast<size_t>(positions[t]) * columns;
    };
    // Forward: float32 scores, 2-bit backpointers (stay 0, step 1, skip 2).
    previous_.assign(static_cast<size_t>(width), neg_inf);
    padded_.assign(static_cast<size_t>(width + 4), neg_inf);
    auto & previous = previous_;
    auto & padded = padded_;
    const size_t row_bytes = static_cast<size_t>(width + 3) / 4;
    pointers_.assign(static_cast<size_t>(frames) * row_bytes, 0);
    auto & pointers = pointers_;
    for (int64_t i = 0; i < std::min<int64_t>(2, width); ++i) {
        previous[i] = row(0)[labels[i]];
        if (prior)
            previous[i] += labels[i] != blank_ ? prior[positions[0]] : 0.0f;
    }
    std::vector<uint8_t> can_skip(states, 0);
    for (int64_t s = 2; s < states; ++s) can_skip[s] = labels[s] != blank_ && labels[s] != labels[s - 2];
    for (int64_t t = 1; t < frames; ++t) {
        const int64_t start = starts[t], delta = start - starts[t - 1];
        std::copy(previous.begin(), previous.end(), padded.begin() + 2);
        const float* emissions = row(t);
        const float p = prior ? prior[positions[t]] : 0.0f;
        uint8_t* back = pointers.data() + static_cast<size_t>(t) * row_bytes;
        for (int64_t i = 0; i < width; ++i) {
            float best = padded[2 + delta + i];
            uint8_t move = 0;
            if (padded[1 + delta + i] > best)
                best = padded[1 + delta + i], move = 1;
            if (padded[delta + i] > best && can_skip[start + i])
                best = padded[delta + i], move = 2;
            const int label = labels[start + i];
            float score = best + emissions[label];
            if (prior)
                score += label != blank_ ? p : 0.0f;
            previous[i] = score;
            back[i >> 2] |= static_cast<uint8_t>(move << (2 * (i & 3)));
        }
    }
    // Backtrace from the better terminal state (the last blank on a tie).
    int64_t end_state = -1;
    for (const int64_t candidate : {states - 1, states - 2}) {
        const int64_t local = candidate - starts[frames - 1];
        if (local < 0 || local >= width)
            continue;
        if (end_state < 0 || previous[local] > out.score) {
            end_state = candidate;
            out.score = previous[local];
        }
    }
    if (end_state < 0 || !std::isfinite(out.score)) {
        out.score = neg_inf;
        return out;
    }
    for (int64_t t = frames - 1, state = end_state; t >= 0; --t) {
        out.path[t] = static_cast<int>(state);
        if (t) {
            const int64_t i = state - starts[t];
            state -= (pointers[static_cast<size_t>(t) * row_bytes + (i >> 2)] >> (2 * (i & 3))) & 3;
        }
    }
    return out;
}


} // namespace engine::decoders
