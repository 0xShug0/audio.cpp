#include "engine/models/qwen3_forced_aligner/processor.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/tokenizer_text.h"
#include "engine/models/qwen3_asr/postprocess.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"

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

void test_language_specific_word_segmentation() {
    const auto assets = aligner_assets(20);
    const auto dictionary = assets->resources.require_file("vocab").parent_path() / "korean_dict_jieba.dict";
    std::ofstream(dictionary) << "가나다 10 x\n가나다라 10 x\n한국어 10 x\n가 10 x\n";
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer(assets);
    engine::models::qwen3_forced_aligner::Qwen3ForcedAlignProcessor processor(*assets, tokenizer);
    engine::test::require(processor.tokenize_words("한국어를 말합니다", "ko") ==
        std::vector<std::string>{"한국어를", "말합니다"}, "old packages retain legacy Korean segmentation");
    assets->resources.add_file("korean_dictionary", dictionary);
    for (const std::string language : {"ko", "Korean"}) {
        const auto words = processor.tokenize_words("한국어를,　가나다라마 모르는말 가나 가", language);
        engine::test::require(words == std::vector<std::string>{"한국어", "를", "가나다라", "마", "모르는말", "가나", "가"},
                              "Korean longest-prefix selection, remainder, unknown and short tokens");
    }
    for (const std::string language : {"ja", "Japanese"}) {
        engine::test::require(processor.tokenize_words("というおじいさんから", language) ==
            std::vector<std::string>{"というおじいさんから"}, "old packages retain legacy Japanese segmentation");
    }
    engine::test::require(processor.tokenize_words("Some call me nature.", "English") ==
                              std::vector<std::string>{"Some", "call", "me", "nature"}, "English word boundaries unchanged");
    engine::test::require(processor.tokenize_words("你好。world!", "Chinese") ==
                              std::vector<std::string>{"你", "好", "world"}, "Chinese word boundaries unchanged");
}

void test_punctuated_overlap_text() {
    using engine::runtime::WordTimestamp;
    const auto trim_chunk_text = [](const std::string & text, const std::vector<WordTimestamp> & words,
                                   engine::runtime::TimeSpan source, engine::runtime::TimeSpan keep) {
        std::vector<WordTimestamp> retained;
        engine::audio::append_chunk_word_timestamps(retained, words, source, keep);
        return engine::models::qwen3_asr::trim_chunk_text(
            text, words, retained.data(), retained.size(), source.start_sample);
    };
    const std::vector<WordTimestamp> repeated{
        {{0, 10}, "おられたそうです"}, {{10, 20}, "おられたそうです"},
        {{20, 30}, "次です"}};
    const std::string text = "おられたそうです。おられたそうです。次です。";
    engine::test::require(trim_chunk_text(text, repeated, {100, 130}, {110, 120}) ==
        "おられたそうです。", "select by time, retaining a legitimate repeated phrase");
    engine::test::require(trim_chunk_text(text, repeated, {100, 130}, {100, 120}) ==
        "おられたそうです。おられたそうです。", "keep both genuine repetitions in the keep window");
    engine::test::require(trim_chunk_text(text, repeated, {100, 130}, {100, 130}) == text,
        "full keep window preserves original punctuation and text");
    engine::test::require(trim_chunk_text(text, {{{0, 5}, "おられたそうです"}}, {100, 130}, {110, 120}).empty(),
        "no retained words produces no overlap text");
    engine::test::require(trim_chunk_text("Old. Twenty-two, café! Next.",
        {{{0, 10}, "Old"}, {{10, 20}, "Twentytwo"}, {{20, 30}, "café"}, {{30, 40}, "Next"}},
        {0, 40}, {10, 30}) == "Twenty-two, café!", "preserve internal hyphens, accents and punctuation");
    engine::test::require(trim_chunk_text("前。ＡＢＣ、ｶﾞ！後。",
        {{{0, 10}, "前"}, {{10, 20}, "ABC"}, {{20, 30}, "ガ"}, {{30, 40}, "後"}},
        {0, 40}, {10, 30}) == "ＡＢＣ、ｶﾞ！", "map Nagisa normalization back to original text");
    engine::test::require(trim_chunk_text("Old. ﷺ! Next.",
        {{{0, 10}, "Old"}, {{10, 20}, "صلىاللهعليهوسلم"}, {{20, 30}, "Next"}},
        {0, 30}, {10, 20}) == "ﷺ!", "grow the reusable buffer for long compatibility decompositions");
    bool rejected = false;
    try { trim_chunk_text("Hello.", {{{0, 10}, "different"}}, {0, 20}, {0, 10}); }
    catch (const std::runtime_error &) { rejected = true; }
    engine::test::require(rejected, "unmappable alignment must fail explicitly, without silently returning duplicated text");
}

// Optional real-tokenizer comparison against external upstream-generated cases.
// Checkpoints and large fixture collections stay outside the code repository.
void test_upstream_segmentation_cases(const std::filesystem::path & directory, const std::filesystem::path & cases_path) {
    const auto assets = aligner_assets(20);
    assets->resources.add_file("nagisa_config", directory / "nagisa.json");
    assets->resources.add_tensor_source("nagisa_weights", directory / "nagisa.safetensors");
    assets->resources.add_file("korean_dictionary", directory / "korean_dict_jieba.dict");
    engine::models::qwen3_asr::Qwen3ASRTextTokenizer tokenizer(assets);
    engine::models::qwen3_forced_aligner::Qwen3ForcedAlignProcessor processor(*assets, tokenizer);
    const auto cases = engine::io::json::parse_file(cases_path);
    size_t index = 0;
    for (const auto & item : cases.as_array()) {
        std::vector<std::string> expected;
        for (const auto & word : item.require("words").as_array()) expected.push_back(word.as_string());
        const auto words = processor.tokenize_words(item.require("text").as_string(), item.require("language").as_string());
        engine::test::require(words == expected, "upstream segmentation case " + std::to_string(index));
        ++index;
    }
    std::cout << "Upstream segmentation cases passed: " << index << '\n';
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::filesystem::path directory, cases;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("Missing test argument value");
            if (arg == "--log") engine::debug::configure_logging({true, std::string(argv[++i])});
            else if (arg == "--tokenizer-dir") directory = argv[++i];
            else if (arg == "--cases") cases = argv[++i];
            else throw std::runtime_error("Unknown test argument: " + arg);
        }
        test_punctuation_only_text_has_no_alignable_words();
        test_word_text_has_alignable_words();
        test_language_specific_word_segmentation();
        test_punctuated_overlap_text();
        test_spans_are_in_the_input_audio_sample_domain();
        test_zero_length_span_repair_and_clamping_use_the_input_rate();
        if (!directory.empty() || !cases.empty()) {
            if (directory.empty() || cases.empty()) throw std::runtime_error("Both --tokenizer-dir and --cases are required");
            test_upstream_segmentation_cases(directory, cases);
        }
        std::cout << "qwen3_forced_aligner_processor_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "qwen3_forced_aligner_processor_test failed: " << ex.what() << "\n";
        return 1;
    }
}
