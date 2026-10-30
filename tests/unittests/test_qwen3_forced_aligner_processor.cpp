#include "engine/models/qwen3_forced_aligner/processor.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/tokenizer_text.h"

#include "test_assert.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Word timestamps are consumed by the framework in the uploaded audio's sample
// domain, so the millisecond predictions must be scaled by that rate, never by
// the frontend's 16 kHz model rate.
std::shared_ptr<engine::models::qwen3_asr::Qwen3ASRAssets> aligner_assets(int64_t segment_ms) {
    const auto dir = std::filesystem::temp_directory_path() / "qwen3_forced_aligner_processor_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "tokenizer_config.json") << "{}";
    std::ofstream(dir / "vocab.json") << "{\"hello\": 0, \"world\": 1}";
    std::ofstream(dir / "merges.txt") << "";

    auto assets = std::make_shared<engine::models::qwen3_asr::Qwen3ASRAssets>();
    assets->resources.add_file("tokenizer_config", dir / "tokenizer_config.json");
    assets->resources.add_file("vocab", dir / "vocab.json");
    assets->resources.add_file("merges", dir / "merges.txt");
    assets->config.sample_rate = 16000;
    assets->config.timestamp_token_id = 2;
    assets->config.timestamp_segment_time_ms = segment_ms;
    return assets;
}

void test_spans_are_in_the_input_audio_sample_domain() {
    const auto assets = aligner_assets(20);
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer(assets);
    engine::models::qwen3_forced_aligner::Qwen3ForcedAlignProcessor processor(*assets, tokenizer);

    // id * 20 ms: 2000, 2200, 3000, 3200 ms.
    const auto spans = processor.parse_timestamps({"hello", "world"}, {100, 110, 150, 160}, 24000, 0, false);
    engine::test::require_eq(spans[0].span.start_sample, 48000, "2000 ms at 24 kHz");
    engine::test::require_eq(spans[0].span.end_sample, 52800, "2200 ms at 24 kHz");
    engine::test::require_eq(spans[1].span.start_sample, 72000, "3000 ms at 24 kHz");
    engine::test::require_eq(spans[1].span.end_sample, 76800, "3200 ms at 24 kHz");
}

void test_zero_length_span_repair_and_clamping_use_the_input_rate() {
    const auto assets = aligner_assets(20);
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer(assets);
    engine::models::qwen3_forced_aligner::Qwen3ForcedAlignProcessor processor(*assets, tokenizer);

    // Degenerate equal timestamps are widened by one segment: 20 ms of the
    // 24 kHz input, not of the model's 16 kHz frame.
    const auto repaired = processor.parse_timestamps({"hello"}, {100, 100}, 24000, 0, false);
    engine::test::require_eq(repaired[0].span.end_sample - repaired[0].span.start_sample, 480,
                             "zero-length span repaired in the input sample domain");

    // 3000 ms at 24 kHz is 72000 samples; the clamp bound is the 24 kHz audio length.
    const auto clamped = processor.parse_timestamps({"hello", "world"}, {100, 150, 160, 170}, 24000, 60000, true);
    engine::test::require(clamped[0].span.end_sample <= 60000, "clamped span must stay inside the audio");
    engine::test::require(clamped[1].span.end_sample <= 60000, "clamped span must stay inside the audio");
}

void test_punctuation_only_text_has_no_alignable_words() {
    for (const std::string text : {".", "...", "?", " "}) {
        engine::test::require(
            !engine::models::qwen3_forced_aligner::has_alignable_words(text, "English"),
            "punctuation-only forced-aligner text should not be alignable");
    }
}

void test_word_text_has_alignable_words() {
    engine::test::require(
        engine::models::qwen3_forced_aligner::has_alignable_words("Some call me nature.", "English"),
        "English words should be alignable");
    engine::test::require(
        engine::models::qwen3_forced_aligner::has_alignable_words("你好。", "Chinese"),
        "Chinese words should be alignable");
}

}  // namespace

int main() {
    try {
        test_punctuation_only_text_has_no_alignable_words();
        test_word_text_has_alignable_words();
        test_spans_are_in_the_input_audio_sample_domain();
        test_zero_length_span_repair_and_clamping_use_the_input_rate();
        std::cout << "qwen3_forced_aligner_processor_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "qwen3_forced_aligner_processor_test failed: " << ex.what() << "\n";
        return 1;
    }
}
