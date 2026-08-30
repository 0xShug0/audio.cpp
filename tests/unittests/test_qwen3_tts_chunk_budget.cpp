// F6.2 -- Qwen3-TTS's default text chunk must fit inside its token cap.
//
// The old default was a fixed 8192 codepoints against a 2048-frame cap. At the
// 12.5 Hz codec frame rate 2048 frames is 163.8 s of audio, roughly a fifth of
// what 8192 characters of prose needs, so any long-form input was truncated
// mid-sentence on a bare `break`. These tests pin the replacement arithmetic;
// no assets and no backend are needed.

#include "engine/models/qwen3_tts/types.h"

#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::qwen3_tts::kQwen3TTSChunkSafetyMargin;
using engine::models::qwen3_tts::kQwen3TTSCodecFrameRateHz;
using engine::models::qwen3_tts::kQwen3TTSMinTextChunkSize;
using engine::models::qwen3_tts::qwen3_tts_default_text_chunk_size;
using engine::models::qwen3_tts::qwen3_tts_frames_for_codepoints;

// The frame rate is 24000 / 1920 (tokenizer_speech_encoder.cpp:45-46).
void test_frame_rate_matches_the_codec() {
    engine::test::require_close(
        static_cast<float>(kQwen3TTSCodecFrameRateHz),
        12.5F,
        1.0e-6F,
        "codec frame rate");
}

// The contract: whatever the cap, a full default chunk needs fewer frames than
// the cap can emit. This is the assertion the old 8192 constant failed.
void test_default_chunk_always_fits_the_token_cap() {
    const std::vector<int64_t> caps = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    for (const int64_t cap : caps) {
        const int64_t chunk = qwen3_tts_default_text_chunk_size(cap);
        const int64_t frames = qwen3_tts_frames_for_codepoints(chunk);
        if (chunk > kQwen3TTSMinTextChunkSize && frames > cap) {
            throw std::runtime_error(
                "default chunk " + std::to_string(chunk) + " codepoints needs " +
                std::to_string(frames) + " frames, past the cap of " + std::to_string(cap));
        }
    }
}

// The shipped cap. 2048 frames is 163.8 s of audio; the chunk must be a small
// four-figure number of codepoints, not five.
void test_shipped_cap_produces_a_synthesisable_chunk() {
    const int64_t chunk = qwen3_tts_default_text_chunk_size(2048);
    engine::test::require(chunk < 8192, "the shipped cap no longer defaults to the old 8192 chunk");
    engine::test::require(chunk > 512, "the chunk is not so small that every sentence is split");
    // 2048 / 12.5 = 163.84 s, x 13 codepoints/s x 0.75 margin = 1597.
    engine::test::require_eq(chunk, int64_t{1597}, "default chunk for a 2048-frame cap");

    const int64_t frames = qwen3_tts_frames_for_codepoints(chunk);
    engine::test::require_eq(frames, int64_t{1536}, "frames a full default chunk needs");
    engine::test::require(frames < 2048, "a full default chunk finishes inside the cap");
    // The safety margin is real headroom, not a rounding accident.
    const double headroom = 1.0 - (static_cast<double>(frames) / 2048.0);
    engine::test::require(
        headroom > 0.15,
        "at least 15% of the token budget is left as headroom");
}

// The old default is exactly the failure the finding describes: 8192 codepoints
// is several times what a 2048-frame cap can synthesise.
void test_old_default_is_provably_unsynthesisable() {
    const int64_t frames = qwen3_tts_frames_for_codepoints(8192);
    engine::test::require(
        frames > 2048,
        "8192 codepoints needs more frames than the 2048 cap can emit");
    engine::test::require(
        frames > 3 * 2048,
        "8192 codepoints needs more than three times the shipped cap");
    engine::test::require_eq(frames, int64_t{7877}, "frames 8192 codepoints would need");
}

// Chunk size tracks the cap, so raising --max-tokens raises the chunk too.
void test_chunk_size_scales_with_the_cap() {
    engine::test::require(
        qwen3_tts_default_text_chunk_size(4096) > qwen3_tts_default_text_chunk_size(2048),
        "a larger cap allows a larger chunk");
    engine::test::require(
        qwen3_tts_default_text_chunk_size(1024) < qwen3_tts_default_text_chunk_size(2048),
        "a smaller cap forces a smaller chunk");
    // Linear in the cap, up to one codepoint of truncation on each side.
    const int64_t doubled = qwen3_tts_default_text_chunk_size(4096);
    const int64_t twice = 2 * qwen3_tts_default_text_chunk_size(2048);
    engine::test::require(
        doubled >= twice - 1 && doubled <= twice + 2,
        "chunk size is linear in the cap up to truncation");
}

// Degenerate caps must not produce a zero or negative chunk, which would make
// the splitter loop forever or throw.
void test_degenerate_caps_clamp_to_the_floor() {
    engine::test::require_eq(
        qwen3_tts_default_text_chunk_size(0),
        kQwen3TTSMinTextChunkSize,
        "a zero cap falls back to the floor");
    engine::test::require_eq(
        qwen3_tts_default_text_chunk_size(-1),
        kQwen3TTSMinTextChunkSize,
        "a negative cap falls back to the floor");
    engine::test::require_eq(
        qwen3_tts_default_text_chunk_size(1),
        kQwen3TTSMinTextChunkSize,
        "a one-frame cap falls back to the floor");
    engine::test::require_eq(
        qwen3_tts_frames_for_codepoints(0),
        int64_t{0},
        "an empty chunk needs no frames");
    engine::test::require(kQwen3TTSChunkSafetyMargin < 1.0, "the safety margin leaves headroom");
}

}  // namespace

int main() {
    try {
        test_frame_rate_matches_the_codec();
        test_default_chunk_always_fits_the_token_cap();
        test_shipped_cap_produces_a_synthesisable_chunk();
        test_old_default_is_provably_unsynthesisable();
        test_chunk_size_scales_with_the_cap();
        test_degenerate_caps_clamp_to_the_floor();
        std::cout << "qwen3_tts_chunk_budget_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "qwen3_tts_chunk_budget_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
