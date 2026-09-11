#include "engine/community_models/audio8_asr/types.h"
#include "test_assert.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

namespace test = engine::test;
using engine::community_models::audio8_asr::audio8_asr_prompt_audio_token_count;
using engine::community_models::audio8_asr::audio8_asr_round_f32_to_bf16;

void test_prompt_token_count_matches_reference() {
    // Reference processor: downsampled = (mel_frames + 1) // 2;
    // tokens = max(downsampled // merge_factor, 1). Values observed from the
    // Audio8-ASR-0.1B reference runs on repo test audio.
    test::require_eq(audio8_asr_prompt_audio_token_count(1407, 4), int64_t{176}, "1407 frames");
    test::require_eq(audio8_asr_prompt_audio_token_count(595, 4), int64_t{74}, "595 frames");
    test::require_eq(audio8_asr_prompt_audio_token_count(3000, 4), int64_t{375}, "3000 frames");
    test::require_eq(audio8_asr_prompt_audio_token_count(31, 4), int64_t{4}, "31 frames");
    test::require_eq(audio8_asr_prompt_audio_token_count(32, 4), int64_t{4}, "32 frames");
    // Short audio clamps to a single audio token.
    test::require_eq(audio8_asr_prompt_audio_token_count(1, 4), int64_t{1}, "1 frame");
    test::require_eq(audio8_asr_prompt_audio_token_count(6, 4), int64_t{1}, "6 frames");
    test::require_eq(audio8_asr_prompt_audio_token_count(7, 4), int64_t{1}, "7 frames");
    bool threw = false;
    try {
        (void) audio8_asr_prompt_audio_token_count(0, 4);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "zero mel frames must be rejected");
}

void test_bf16_rounding() {
    // Exactly representable in bfloat16.
    test::require_eq(audio8_asr_round_f32_to_bf16(1.0F), 1.0F, "1.0");
    test::require_eq(audio8_asr_round_f32_to_bf16(0.5F), 0.5F, "0.5");
    test::require_eq(audio8_asr_round_f32_to_bf16(-2.75F), -2.75F, "-2.75");
    // Round to nearest even: 0.3f (0x3E99999A) rounds up to 0x3E9A0000.
    test::require_eq(audio8_asr_round_f32_to_bf16(0.3F), 0.30078125F, "0.3f");
    // Round-to-nearest-even tie: 1.0 + 2^-8 (bits 0x3F808000) rounds the
    // trailing mantissa back down to 1.0.
    const uint32_t tie_bits = 0x3F808000u;
    float tie = 0.0F;
    std::memcpy(&tie, &tie_bits, sizeof(tie));
    test::require_eq(audio8_asr_round_f32_to_bf16(tie), 1.0F, "rne tie");
}

}  // namespace

int main() {
    try {
        test_prompt_token_count_matches_reference();
        test_bf16_rounding();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: audio8_asr unit checks\n";
    return 0;
}
