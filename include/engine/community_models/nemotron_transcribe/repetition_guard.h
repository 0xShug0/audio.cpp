// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted from NVIDIA/NeMo-Speech.cpp salm_guards at 536b17a41bef62d2a67a40ed3df61df26a4389fe.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_set>
#include <vector>

namespace engine::models::nemotron_transcribe {

// The smallest period p (1..32) the last tokens repeat with over the window p needs, else 0.
// Periods 4-12 need max(15, 4p) tokens (a phrase said four times) unless a divisor period explains
// the window; the others need max(32, 4p).
// One sequence's guard. Fires when the output repeats with a short period (checked every 4 tokens
// from the 15th) or when at least 90% of the last 256 16-token n-grams already appeared earlier in
// the output.
class RepetitionGuard {
   public:
    // Commit token `t`; true when the sequence must end after it. The check covers the tokens
    // before `t`, so the newest token is kept, and the guard stays fired.
    bool push(int32_t t);
    void reset();

   private:
    bool fires(size_t n);
    bool replay(size_t n);

    std::vector<int32_t> tokens_;
    size_t last_check_ = 0;
    bool fired_ = false;
    // replay
    std::vector<uint64_t> hashes_;
    std::unordered_set<uint64_t> seen_;
    std::deque<char> recent_;
    int repeats_ = 0;
    size_t scanned_ = 0;
};

}  // namespace engine::models::nemotron_transcribe
