// Regression test for the Falcon-H1 host KV cache append (audio8_tts 0.1B).
//
// The cache is stored in ggml's col-major [head_dim, seq, n_kv] layout where
// the per-head stride is the CURRENT sequence length. Appending a token
// without re-laying the existing entries into the new stride makes the new
// token overwrite the previous head blocks: with n_kv=2, appending token 1
// wrote head 0 at floats [64,128) — exactly where token 0's head 1 lived —
// so from the second token on, attention read corrupted keys/values for every
// head past the first (argmax diverged from the HF reference at prompt step 3
// and the recurrent state blew up on long sequences).
//
// This test feeds recognizable per-(token, head, dim) values through
// append_falcon_kv_token and checks the full cache contents after every
// append; the historical implementation fails from the second append on.

#include "falcon_kv_cache.h"

#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

float marker(int64_t token, int64_t head, int64_t dim) {
    return static_cast<float>(token * 100000 + head * 1000 + dim);
}

}  // namespace

int main() {
    using engine::test::require;
    using engine::test::require_eq;
    using engine::models::audio8_tts::append_falcon_kv_token;

    constexpr int64_t n_kv = 2;
    constexpr int64_t head_dim = 64;
    constexpr int64_t n_tokens = 8;

    std::vector<float> cache;
    for (int64_t t = 0; t < n_tokens; ++t) {
        std::vector<float> fresh(static_cast<size_t>(n_kv * head_dim));
        for (int64_t h = 0; h < n_kv; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                fresh[static_cast<size_t>(d + head_dim * h)] = marker(t, h, d);
            }
        }
        append_falcon_kv_token(cache, t, n_kv, head_dim, fresh.data());

        const int64_t seq = t + 1;
        require_eq(static_cast<int64_t>(cache.size()), seq * n_kv * head_dim, "cache size after append");
        for (int64_t h = 0; h < n_kv; ++h) {
            for (int64_t tt = 0; tt < seq; ++tt) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    const float actual = cache[static_cast<size_t>(d + head_dim * (tt + seq * h))];
                    require(actual == marker(tt, h, d),
                            "cache entry corrupted after appending token " + std::to_string(t));
                }
            }
        }
    }

    std::cout << "audio8_tts_falcon_kv_cache_test passed\n";
    return 0;
}
