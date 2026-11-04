// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "engine/community_models/nemotron_transcribe/repetition_guard.h"

#include <algorithm>

namespace engine::models::nemotron_transcribe {
namespace {

bool
periodic(const std::vector<int32_t>& t, size_t base, size_t n, size_t p) {
    for (size_t i = base; i + p < n; ++i)
        if (t[i] != t[i + p])
            return false;
    return true;
}

constexpr size_t kPeriodMax = 32, kSpan = 32, kMinRepeats = 4;
// Four repeats for short phrases: speech repeats a phrase three times ("I'm like X, I'm like X, I'm
// like X"), a loop does not stop.
constexpr size_t kShortPeriodMin = 4, kShortPeriodMax = 12, kShortSpan = 15, kShortMinRepeats = 4;
constexpr size_t kMinWindow = kShortSpan, kEvery = 4;
constexpr size_t kReplayNgram = 16, kReplayWindow = 256;
constexpr double kReplayFraction = 0.9;

int
loop_period(const std::vector<int32_t>& t, size_t n) {
    if (n < kMinWindow)
        return 0;
    const int last = t[n - 1];
    for (size_t p = 1; p <= kPeriodMax; ++p) {
        const bool short_rule = p >= kShortPeriodMin && p <= kShortPeriodMax;
        const size_t need = short_rule ? std::max(kShortSpan, p * kShortMinRepeats)
                                       : std::max(kSpan, p * kMinRepeats);
        if (n < need || t[n - 1 - p] != last)
            continue;
        const size_t base = n - need;
        if (!periodic(t, base, n, p))
            continue;
        if (short_rule) {
            bool divisor = false;
            for (size_t q = 1; q < p && !divisor; ++q)
                divisor = p % q == 0 && periodic(t, base, n, q);
            if (divisor)
                continue;  // a shorter period explains the window and its own rule declined it
        }
        return static_cast<int>(p);
    }
    return 0;
}

}  // namespace

bool
RepetitionGuard::replay(size_t n) {
    for (size_t i = scanned_; i < n; ++i) {
        if (i + 1 < kReplayNgram) {
            hashes_.push_back(0);
            continue;
        }
        uint64_t h = 1469598103934665603ull;  // FNV-1a over the n-gram's token ids
        for (size_t k = i + 1 - kReplayNgram; k <= i; ++k) {
            h ^= static_cast<uint32_t>(tokens_[k]);
            h *= 1099511628211ull;
        }
        hashes_.push_back(h);
        if (i >= kReplayNgram &&
            i - kReplayNgram + 1 >= kReplayNgram)  // the last n-gram not overlapping this one
            seen_.insert(hashes_[i - kReplayNgram]);
        const char flag = seen_.count(h) ? 1 : 0;
        recent_.push_back(flag);
        repeats_ += flag;
        if (recent_.size() > kReplayWindow) {
            repeats_ -= recent_.front();
            recent_.pop_front();
        }
    }
    scanned_ = std::max(scanned_, n);
    return recent_.size() >= kReplayWindow &&
           static_cast<double>(repeats_) / kReplayWindow >= kReplayFraction;
}

bool
RepetitionGuard::fires(size_t n) {
    if (n > scanned_ && replay(n))
        return true;
    if (n < kMinWindow || n - last_check_ < kEvery)
        return false;
    last_check_ = n;
    return loop_period(tokens_, n) > 0;
}

bool
RepetitionGuard::push(int32_t t) {
    if (!fired_)
        fired_ = fires(tokens_.size());
    tokens_.push_back(t);
    return fired_;
}

void
RepetitionGuard::reset() {
    *this = RepetitionGuard{};
}

}  // namespace engine::models::nemotron_transcribe
