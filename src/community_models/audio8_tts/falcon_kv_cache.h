#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace engine::models::audio8_tts {

// Appends one token's K (or V) vectors to a host-side KV cache stored in
// ggml's col-major [head_dim, seq, n_kv] layout: element (d, t, h) lives at
// d + head_dim*(t + seq*h), i.e. the per-head stride is the CURRENT sequence
// length. Because that stride grows with every appended token, the cached
// tokens must be re-laid into the new stride before `fresh` (the new token's
// n_kv*head_dim values, head-major) is written at t = seq. Appending without
// the re-layout makes the new token overwrite the previous head blocks and
// silently corrupts attention context from the second token on.
inline void append_falcon_kv_token(
    std::vector<float> & cache,
    int64_t seq,
    int64_t n_kv,
    int64_t head_dim,
    const float * fresh) {
    const int64_t new_seq_len = seq + 1;
    std::vector<float> old;
    old.swap(cache);
    cache.assign(static_cast<size_t>(new_seq_len * n_kv * head_dim), 0.0F);
    for (int64_t h = 0; h < n_kv; ++h) {
        for (int64_t t = 0; t < seq; ++t) {
            std::copy_n(old.data() + head_dim * (t + seq * h),
                        static_cast<size_t>(head_dim),
                        cache.data() + head_dim * (t + new_seq_len * h));
        }
        std::copy_n(fresh + head_dim * h,
                    static_cast<size_t>(head_dim),
                    cache.data() + head_dim * (seq + new_seq_len * h));
    }
}

}  // namespace engine::models::audio8_tts
