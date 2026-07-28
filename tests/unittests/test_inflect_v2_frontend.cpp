#include "engine/community_models/inflect_v2/frontend.h"
#include "engine/community_models/inflect_v2/runtime.h"
#include "engine/community_models/inflect_v2/session.h"

#include "test_assert.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main() try {
    using engine::models::inflect_v2::InflectV2Frontend;
    using engine::models::inflect_v2::apply_inflect_v2_edge_fade;

    engine::test::require_eq(
        InflectV2Frontend::normalize(
            "Dr. Smith paid $12.50 on 3/4/2026 at 9:05 PM."),
        std::string(
            "doctor Smith paid twelve dollars and fifty cents on March fourth "
            "two thousand and twenty six at nine oh five p m."),
        "normalization");
    engine::test::require_eq(
        InflectV2Frontend::normalize("Call 555-0123 at 8 PM."),
        std::string("Call five five five, zero one two three at eight p m."),
        "phone normalization");
    engine::test::require_eq(
        InflectV2Frontend::normalize("Version 2.10.3 costs $1.05."),
        std::string(
            "Version two point ten point three costs one dollar and five cents."),
        "version and money normalization");
    engine::test::require_eq(
        InflectV2Frontend::normalize(
            "NASA uses RTX 5090 and USB-C."),
        std::string(
            "en ay ess ay uses ar tee ex fifty ninety and you ess bee see."),
        "acronym and pronunciation override normalization");

    const std::string phoneme_golden =
        std::string("h") + "\xC9\x99" + "l" + "\xCB\x88" + "o" +
        "\xCA\x8A" + ".";
    const auto token_ids =
        InflectV2Frontend::tokens_from_phonemes(phoneme_golden);
    engine::test::require(
        token_ids ==
            std::vector<int32_t>(
                {0, 50, 0, 83, 0, 54, 0, 156, 0, 57, 0, 135, 0, 4, 0}),
        "phoneme ids with blanks");
    bool unknown_phoneme_rejected = false;
    try {
        (void)InflectV2Frontend::tokens_from_phonemes("#");
    } catch (const std::runtime_error &) {
        unknown_phoneme_rejected = true;
    }
    engine::test::require(
        unknown_phoneme_rejected,
        "unknown phoneme must be rejected");

    const auto missing_library =
        std::filesystem::temp_directory_path() /
        "audio_cpp_inflect_v2_missing_espeak_library";
    engine::test::require(
        !std::filesystem::exists(missing_library),
        "missing eSpeak test path must not exist");
    bool missing_library_rejected = false;
    try {
        InflectV2Frontend frontend(missing_library, {});
    } catch (const std::runtime_error &) {
        missing_library_rejected = true;
    }
    engine::test::require(
        missing_library_rejected,
        "missing eSpeak library must be rejected");

    auto assets =
        std::make_shared<engine::models::inflect_v2::InflectV2Assets>();
    auto contract =
        std::make_shared<engine::model_spec::ModelContract>();
    contract->session_option_keys = {
        "inflect_v2.espeak_library_path",
        "inflect_v2.espeak_data_path",
    };

    engine::runtime::SessionOptions invalid_task_options;
    invalid_task_options.options["inflect_v2.espeak_library_path"] =
        missing_library.string();
    bool invalid_task_reported_first = false;
    try {
        engine::models::inflect_v2::InflectV2Session session(
            {
                engine::runtime::VoiceTaskKind::Vad,
                engine::runtime::RunMode::Offline,
            },
            invalid_task_options,
            assets,
            contract);
    } catch (const std::runtime_error & error) {
        invalid_task_reported_first =
            std::string(error.what()).find("offline TTS") != std::string::npos;
    }
    engine::test::require(
        invalid_task_reported_first,
        "task validation must run before eSpeak initialization");

    engine::runtime::SessionOptions invalid_option_options;
    invalid_option_options.options["inflect_v2.espeak_library_path"] =
        missing_library.string();
    invalid_option_options.options["inflect_v2.unknown"] = "1";
    bool invalid_option_reported_first = false;
    try {
        engine::models::inflect_v2::InflectV2Session session(
            {
                engine::runtime::VoiceTaskKind::Tts,
                engine::runtime::RunMode::Offline,
            },
            invalid_option_options,
            assets,
            contract);
    } catch (const std::runtime_error & error) {
        invalid_option_reported_first =
            std::string(error.what()).find("unknown Inflect v2 session option") !=
            std::string::npos;
    }
    engine::test::require(
        invalid_option_reported_first,
        "session option validation must run before eSpeak initialization");

    const auto missing_data =
        std::filesystem::temp_directory_path() /
        "audio_cpp_inflect_v2_missing_espeak_data";
    engine::test::require(
        !std::filesystem::exists(missing_data),
        "missing eSpeak data test path must not exist");
    bool missing_data_rejected = false;
    try {
        InflectV2Frontend frontend({}, missing_data);
    } catch (const std::runtime_error &) {
        missing_data_rejected = true;
    }
    engine::test::require(
        missing_data_rejected,
        "missing eSpeak data must be rejected");

    const auto chunks = InflectV2Frontend::split_text(
        "First sentence. Second sentence! Last one?",
        280);
    engine::test::require_eq(chunks.size(), size_t{3}, "sentence chunk count");
    engine::test::require_eq(
        chunks[0],
        std::string("First sentence."),
        "first sentence chunk");
    engine::test::require_eq(
        chunks[2],
        std::string("Last one?"),
        "last sentence chunk");
    engine::test::require(
        InflectV2Frontend::split_text(" \n\t", 280).empty(),
        "whitespace-only text must not produce chunks");

    for (const auto & [text, expected] :
         std::vector<std::pair<std::string, double>>{
             {"Question?", 0.28},
             {"Exclamation!", 0.24},
             {"Sentence.", 0.22},
             {"Clause;", 0.16},
             {"Label:", 0.13},
             {"Phrase,", 0.09},
             {"Words", 0.08},
         }) {
        engine::test::require(
            std::abs(InflectV2Frontend::boundary_pause_seconds(text) - expected) <
                1.0e-12,
            "punctuation pause");
    }

    std::vector<float> samples(480, 1.0F);
    apply_inflect_v2_edge_fade(samples, 24000);
    engine::test::require_eq(samples.front(), 0.0F, "fade-in starts at zero");
    engine::test::require_eq(samples[119], 1.0F, "fade-in reaches unity");
    engine::test::require_eq(samples[360], 1.0F, "fade-out starts at unity");
    engine::test::require_eq(samples.back(), 0.0F, "fade-out ends at zero");

    std::cout << "inflect_v2_frontend_test passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "inflect_v2_frontend_test failed: " << error.what() << "\n";
    return 1;
}
