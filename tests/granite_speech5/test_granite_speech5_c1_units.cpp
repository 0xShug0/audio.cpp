#include "engine/framework/debug/profiler.h"
#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/framework/io/json.h"
#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace json = engine::io::json;

std::filesystem::path fixtures_dir() {
    const char * directory = std::getenv("GRANITE_SPEECH5_FIXTURES");
    if (directory == nullptr || directory[0] == '\0') {
        throw std::runtime_error("reference parity requires GRANITE_SPEECH5_FIXTURES");
    }
    return directory;
}

json::Value load_fixture(const std::string & name) {
    const auto path = fixtures_dir() / name;
    test::require(std::filesystem::is_regular_file(path), "fixture present: " + name);
    return json::parse_file(path);
}

std::vector<int32_t> ids_from_json(const json::Value & value) {
    std::vector<int32_t> ids;
    for (const auto & item : value.as_array()) {
        ids.push_back(static_cast<int32_t>(item.as_i64()));
    }
    return ids;
}

void require_ids_eq(const std::vector<int32_t> & actual,
                    const std::vector<int32_t> & expected,
                    const std::string & label) {
    if (actual == expected) {
        return;
    }
    std::ostringstream oss;
    oss << label << " mismatch: expected len=" << expected.size() << " actual len=" << actual.size();
    const size_t common = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < common; ++i) {
        if (actual[i] != expected[i]) {
            oss << " first diff at " << i << ": expected=" << expected[i]
                << " actual=" << actual[i];
            break;
        }
    }
    throw std::runtime_error(oss.str());
}

void require_floats_eq(const std::vector<float> & actual,
                       const std::vector<float> & expected,
                       const std::string & label) {
    test::require_eq(actual.size(), expected.size(), label + " len");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(label + " mismatch at " + std::to_string(i));
        }
    }
}

std::optional<std::filesystem::path> find_model_dir() {
    if (const char * env = std::getenv("GRANITE_SPEECH5_MODEL")) {
        const std::filesystem::path candidate(env);
        if (std::filesystem::is_directory(candidate)) {
            return candidate;
        }
    }

    return std::nullopt;
}

// --- always-run, no-weight rule tests ---

void test_language_codes() {
    using g2b::granite_speech5_build_user_content;
    using g2b::granite_speech5_language_name;
    using g2b::granite_speech5_normalize_language_code;
    test::require_eq(granite_speech5_normalize_language_code(""), std::string(""), "blank");
    test::require_eq(granite_speech5_normalize_language_code("   \t\n "),
                     std::string(""), "whitespace blank");
    test::require_eq(granite_speech5_normalize_language_code("en"), std::string("en"), "en");
    test::require_eq(granite_speech5_normalize_language_code(" EN "), std::string("en"),
                     "trim+lower");
    test::require_eq(granite_speech5_normalize_language_code("Ja"), std::string("ja"),
                     "mixed case");
    for (const char * code : {"en", "fr", "de", "es", "pt", "ja"}) {
        (void) granite_speech5_normalize_language_code(code);
    }
    for (const char * bad : {"xx", "eng", "e", "en-US", "english", "jp", "zh"}) {
        bool threw = false;
        try {
            (void) granite_speech5_normalize_language_code(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, std::string("reject unsupported: ") + bad);
    }
    for (const char * bad : {"日本語", "én"}) {
        bool threw = false;
        try {
            (void) granite_speech5_normalize_language_code(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "reject non-ASCII hint");
    }
    test::require_eq(
        std::string(granite_speech5_language_name("en")), std::string("English"), "en name");
    test::require_eq(
        std::string(granite_speech5_language_name("fr")), std::string("French"), "fr name");
    test::require_eq(
        std::string(granite_speech5_language_name("de")), std::string("German"), "de name");
    test::require_eq(
        std::string(granite_speech5_language_name("es")), std::string("Spanish"), "es name");
    test::require_eq(std::string(granite_speech5_language_name("pt")),
                     std::string("Portuguese"),
                     "pt name");
    test::require_eq(std::string(granite_speech5_language_name("ja")),
                     std::string("Japanese"),
                     "ja name");
    // No-hint content is byte-identical to the reference.py smoke prompt body.
    test::require_eq(
        granite_speech5_build_user_content(""),
        std::string("<|audio|>transcribe the speech with proper punctuation and "
                    "capitalization."),
        "smoke user content");
    test::require_eq(granite_speech5_build_user_content("en"),
                     std::string("The speech is in English. Transcribe it exactly; do not "
                                 "translate. <|audio|>transcribe the speech with proper "
                                 "punctuation and capitalization."),
                     "en hint content");
    // Every hint differs (language name span) and carries the marker once.
    std::vector<std::string> hinted;
    for (const char * code : {"en", "fr", "de", "es", "pt", "ja"}) {
        hinted.push_back(granite_speech5_build_user_content(code));
    }
    test::require(std::adjacent_find(hinted.begin(), hinted.end()) == hinted.end(),
                    "hints distinct");
    for (const auto & content : hinted) {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = content.find("<|audio|>", pos)) != std::string::npos) {
            ++count;
            pos += 9;
        }
        test::require_eq(count, size_t{1}, "one structural marker");
    }
}

void test_template_bytes() {
    using g2b::granite_speech5_render_single_turn_prompt;
    test::require_eq(granite_speech5_render_single_turn_prompt("X"),
                     std::string("USER: X\n ASSISTANT:"),
                     "single-turn render");
    // The space preceding the assistant cue is part of the template.
    const std::string prompt = granite_speech5_render_single_turn_prompt("hello");
    test::require_eq(prompt.substr(prompt.size() - 12), std::string("\n ASSISTANT:"),
                     "assistant cue bytes");
    test::require_eq(prompt.substr(0, 6), std::string("USER: "), "user cue bytes");
}

void test_reserved_markers() {
    using g2b::granite_speech5_require_no_reserved_markers;
    const std::vector<std::string> markers = {"<|audio|>", "<|end_of_text|>", "<|pad|>"};
    granite_speech5_require_no_reserved_markers("plain transcription text", markers);
    granite_speech5_require_no_reserved_markers("", markers);
    granite_speech5_require_no_reserved_markers("anything <|goes|>", {});
    for (const auto & marker : markers) {
        bool threw = false;
        try {
            granite_speech5_require_no_reserved_markers(
                "transcribe <Marker> " + marker + " here", markers);
        } catch (const std::runtime_error & error) {
            threw = true;
            test::require(
                std::string(error.what()).find(marker) != std::string::npos, "names marker");
        }
        test::require(threw, "reject " + marker);
    }
}

void check_fixture_splice(const std::string & name) {
    const auto fixture = load_fixture(name);
    const auto one = ids_from_json(fixture.require("one_marker_ids"));
    const auto expanded = ids_from_json(fixture.require("expanded_ids"));
    const int64_t audio_tokens = fixture.require("audio_tokens").as_i64();
    test::require_eq(audio_tokens, int64_t{33}, name + " A=33");
    const auto out =
        g2b::granite_speech5_expand_audio_markers(one, g2b::kGraniteSpeech5AudioTokenId, 33);
    require_ids_eq(out.input_ids, expanded, name + " splice");
    test::require_eq(static_cast<int64_t>(out.audio_positions.size()), int64_t{33},
                     name + " position count");
    for (size_t i = 0; i < out.audio_positions.size(); ++i) {
        test::require_eq(out.audio_positions[i], out.audio_positions[0] + static_cast<int32_t>(i),
                         name + " contiguous");
        test::require_eq(expanded[static_cast<size_t>(out.audio_positions[i])],
                         g2b::kGraniteSpeech5AudioTokenId, name + " marker at position");
    }
}

void test_fixture_splice_and_hint_invariant() {
    check_fixture_splice("c1_prompt_none.json");
    check_fixture_splice("c1_prompt_en.json");
    check_fixture_splice("c1_prompt_ja.json");
    const auto none = ids_from_json(load_fixture("c1_prompt_none.json").require("expanded_ids"));
    const auto en = ids_from_json(load_fixture("c1_prompt_en.json").require("expanded_ids"));
    const auto ja = ids_from_json(load_fixture("c1_prompt_ja.json").require("expanded_ids"));
    // IDs differ only in the hint span: shared "USER:" head, shared
    // audio-run + tail suffix, hint text in between.
    for (const auto * hinted : {&en, &ja}) {
        test::require_eq((*hinted)[0], none[0], "shared head 0");
        test::require_eq((*hinted)[1], none[1], "shared head 1");
        const auto audio_at = [&](const std::vector<int32_t> & ids) {
            return std::find(ids.begin(), ids.end(), g2b::kGraniteSpeech5AudioTokenId);
        };
        const auto none_run = audio_at(none);
        const auto hint_run = audio_at(*hinted);
        test::require(none_run != none.end() && hint_run != hinted->end(), "audio runs");
        test::require(
            std::equal(none_run, none.end(), hint_run, hinted->end()), "shared audio+tail");
        // Head before the audio marker differs only by the hint insertion:
        // both open with "USER:" and close with the space before the marker.
        const std::vector<int32_t> none_head(none.begin(), none_run);
        const std::vector<int32_t> hint_head(hinted->begin(), hint_run);
        require_ids_eq(none_head, std::vector<int32_t>({6584, 25, 220}), "no-hint head");
        test::require(hint_head.size() > none_head.size(), "hint head longer");
        test::require_eq(hint_head[0], int32_t{6584}, "hint head opens");
        test::require_eq(hint_head[1], int32_t{25}, "hint head opens 2");
        test::require_eq(hint_head.back(), int32_t{220}, "hint head closes");
        test::require(hint_head != none_head, "hint span differs");
    }
    // en vs ja differ in exactly the language-name token.
    test::require_eq(en.size(), ja.size(), "en/ja same length");
    size_t diffs = 0;
    for (size_t i = 0; i < en.size(); ++i) {
        diffs += (en[i] != ja[i]) ? 1 : 0;
    }
    test::require_eq(diffs, size_t{1}, "en/ja one-token diff");
}

void test_synthetic_tiny_fixture() {
    const std::vector<int32_t> one{7, 21, 100352, 9, 5};
    const std::vector<int32_t> expanded{7, 21, 100352, 100352, 100352, 9, 5};
    const std::vector<int32_t> positions{2, 3, 4};
    const int64_t hidden = 4;
    const std::vector<float> projected{0, .25F, .5F, .75F, 1, 1.25F, 1.5F, 1.75F, 2, 2.25F, 2.5F, 2.75F};
    g2b::GraniteSpeech5PromptIds prompt;
    prompt.input_ids = expanded;
    prompt.audio_positions = positions;
    prompt.audio_tokens = 3;
    const auto budget = g2b::granite_speech5_prompt_budget(prompt);
    test::require_eq(budget.prompt_len, int64_t{7}, "tiny len");
    test::require_eq(budget.expansion_overhead, int64_t{2}, "tiny overhead");
    g2b::granite_speech5_require_prompt_fits(budget);
    auto decoder_prompt = g2b::granite_speech5_build_decoder_prompt(
        prompt, projected, hidden, g2b::kGraniteSpeech5AudioTokenId);
    require_ids_eq(decoder_prompt.input_ids, expanded, "tiny ids");
    test::require_eq(decoder_prompt.injection.tokens, int64_t{3}, "tiny tokens");
    require_ids_eq(decoder_prompt.injection.positions, positions, "tiny positions");
    require_floats_eq(decoder_prompt.injection.values, projected, "tiny values passthrough");
    // Splice from the one-marker form reproduces the independent expected expansion.
    const auto spliced =
        g2b::granite_speech5_expand_audio_markers(one, g2b::kGraniteSpeech5AudioTokenId, 3);
    require_ids_eq(spliced.input_ids, expanded, "tiny splice");
    require_ids_eq(spliced.audio_positions, positions, "tiny splice positions");
}

void test_injection_rules() {
    g2b::GraniteSpeech5PromptIds base;
    base.input_ids = {10, 20, g2b::kGraniteSpeech5AudioTokenId, g2b::kGraniteSpeech5AudioTokenId,
                      30};
    base.audio_positions = {2, 3};
    base.audio_tokens = 2;
    const std::vector<float> values(2 * 4, 0.5F);
    (void) g2b::granite_speech5_build_decoder_prompt(base, values, 4);
    auto expect_throw = [&](g2b::GraniteSpeech5PromptIds prompt, std::vector<float> vals,
                            int64_t hidden, const std::string & label) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_build_decoder_prompt(
                prompt, std::move(vals), hidden, g2b::kGraniteSpeech5AudioTokenId);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, label);
    };
    {
        auto bad = base;
        bad.audio_positions = {3, 2};
        expect_throw(bad, values, 4, "unsorted positions");
    }
    {
        auto bad = base;
        bad.input_ids = {10, g2b::kGraniteSpeech5AudioTokenId, 20,
                         g2b::kGraniteSpeech5AudioTokenId, 30};
        bad.audio_positions = {1, 3};
        expect_throw(bad, values, 4, "non-contiguous positions");
    }
    {
        auto bad = base;
        bad.audio_positions = {2, 5};
        expect_throw(bad, values, 4, "position out of range");
    }
    {
        auto bad = base;
        bad.input_ids[3] = 30;
        expect_throw(bad, values, 4, "position without marker");
    }
    {
        auto bad = base;
        bad.audio_tokens = 3;
        expect_throw(bad, values, 4, "count disagrees");
    }
    {
        auto bad = base;
        expect_throw(bad, std::vector<float>(7, 0.0F), 4, "values length");
    }
    {
        auto bad = base;
        bad.audio_positions.clear();
        expect_throw(bad, values, 4, "empty positions");
    }
    {
        auto bad = base;
        expect_throw(bad, values, 0, "nonpositive hidden");
    }
    {
        auto bad = base;
        bad.input_ids.clear();
        expect_throw(bad, values, 4, "empty prompt");
    }
    // Positive-zero audio expansion and marker-free prompts are rejected.
    for (int64_t bad_a : {0, -2}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_expand_audio_markers({1, 2, 3},
                                                               g2b::kGraniteSpeech5AudioTokenId,
                                                               bad_a);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "expansion rejects A=" + std::to_string(bad_a));
    }
    {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_expand_audio_markers({1, 2, 3},
                                                               g2b::kGraniteSpeech5AudioTokenId,
                                                               2);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "expansion rejects marker-free prompt");
    }
    {
        // A two-marker run is rejected before splicing, so injection
        // positions can never straddle two runs.
        bool threw = false;
        try {
            (void) g2b::granite_speech5_expand_audio_markers(
                {1, g2b::kGraniteSpeech5AudioTokenId, g2b::kGraniteSpeech5AudioTokenId, 2},
                g2b::kGraniteSpeech5AudioTokenId, 3);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "expansion rejects two-marker input pre-splice");
    }
}

void test_budget_rules() {
    g2b::GraniteSpeech5PromptIds prompt;
    prompt.input_ids = {1, 2, g2b::kGraniteSpeech5AudioTokenId, 4};
    prompt.audio_positions = {2};
    prompt.audio_tokens = 1;
    const auto budget = g2b::granite_speech5_prompt_budget(prompt);
    test::require_eq(budget.prompt_len, int64_t{4}, "len");
    test::require_eq(budget.audio_tokens, int64_t{1}, "audio");
    test::require_eq(budget.text_tokens, int64_t{3}, "text");
    test::require_eq(budget.expansion_overhead, int64_t{0}, "A=1 overhead");
    g2b::granite_speech5_require_prompt_fits(budget);
    {
        auto tight = budget;
        tight.max_position_embeddings = 3;
        bool threw = false;
        try {
            g2b::granite_speech5_require_prompt_fits(tight);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "budget overflow throws");
    }
    {
        g2b::GraniteSpeech5PromptIds empty;
        bool threw = false;
        try {
            g2b::granite_speech5_require_prompt_fits(
                g2b::granite_speech5_prompt_budget(empty));
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "empty prompt throws");
    }
    {
        auto bad = prompt;
        bad.audio_tokens = 7;
        bool threw = false;
        try {
            (void) g2b::granite_speech5_prompt_budget(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "budget/count mismatch throws");
    }
}

void test_eos_rule() {
    using g2b::granite_speech5_strip_observed_eos;
    constexpr int32_t eos = g2b::kGraniteSpeech5EosTokenId;
    require_ids_eq(granite_speech5_strip_observed_eos({1, 2, eos}, eos),
                     std::vector<int32_t>({1, 2}), "strip trailing eos");
    require_ids_eq(granite_speech5_strip_observed_eos({1, 2}, eos), std::vector<int32_t>({1, 2}),
                     "no eos unchanged");
    require_ids_eq(granite_speech5_strip_observed_eos({eos}, eos), std::vector<int32_t>({}),
                     "eos only");
    require_ids_eq(granite_speech5_strip_observed_eos({eos, 5}, eos),
                     std::vector<int32_t>({eos, 5}), "leading eos kept");
    require_ids_eq(
        granite_speech5_strip_observed_eos({1, eos, eos}, eos), std::vector<int32_t>({1, eos}),
        "strip at most one");
    require_ids_eq(
        granite_speech5_strip_observed_eos({}, eos), std::vector<int32_t>({}), "empty unchanged");
}

// --- checkpoint-gated parity tests (need the HF snapshot) ---

engine::assets::ResourceBundle prompt_bundle(const std::filesystem::path & model_dir) {
    engine::assets::ResourceBundle resources(model_dir);
    resources.add_model_files({
        {"tokenizer_json", "tokenizer.json", true},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"chat_template", "chat_template.jinja", true},
        {"generation_config", "generation_config.json", true},
    });
    return resources;
}

void test_tokenizer_parity(const std::filesystem::path & model_dir) {
    auto resources = prompt_bundle(model_dir);
    auto tokenizer = g2b::load_granite_speech5_tokenizer(resources);
    {
        const auto audio_id = tokenizer->find_token_id("<|audio|>");
        test::require(audio_id.has_value(), "audio marker resolves");
        test::require_eq(*audio_id, g2b::kGraniteSpeech5AudioTokenId, "audio marker id");
    }
    test::require(tokenizer->is_control_token_id(g2b::kGraniteSpeech5AudioTokenId),
                  "audio is control");
    {
        const auto pad_id = tokenizer->configured_pad_token_id();
        test::require(pad_id.has_value(), "pad id configured");
        test::require_eq(*pad_id, g2b::kGraniteSpeech5PadTokenId, "pad id");
    }
    const g2b::GraniteSpeech5PromptBuilder builder(tokenizer);
    const auto reserved = g2b::granite_speech5_reserved_markers_from_tokenizer_json(
        model_dir / "tokenizer.json");
    // The reserved check is wired into the text path: the fixed C1 content
    // passes the real 85-marker list, while a list naming content words
    // throws (and the structural audio marker itself is excised, not flagged).
    {
        const auto ok = builder.build_prompt_ids("", 1, reserved);
        test::require_eq(static_cast<int64_t>(ok.audio_positions.size()), int64_t{1},
                         "reserved list passes clean content");
        const auto audio_only = builder.build_prompt_ids("", 1, {"<|audio|>"});
        test::require_eq(static_cast<int64_t>(audio_only.audio_positions.size()), int64_t{1},
                         "structural marker excised");
        bool threw = false;
        try {
            (void) builder.build_prompt_ids("", 1, {"transcribe"});
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "reserved marker via build_prompt_ids throws");
    }
    for (const char * name : {"c1_prompt_none.json", "c1_prompt_en.json",
                              "c1_prompt_ja.json"}) {
        const auto fixture = load_fixture(name);
        const std::string code = fixture.require("language").is_null()
            ? ""
            : g2b::granite_speech5_normalize_language_code(
                  fixture.require("language").as_string());
        const std::string prompt_text = fixture.require("prompt").as_string();
        const auto one_expected = ids_from_json(fixture.require("one_marker_ids"));
        const auto expanded_expected = ids_from_json(fixture.require("expanded_ids"));
        // Framework encode of the rendered one-marker prompt == Python exactly.
        const auto single = g2b::granite_speech5_encode_single_marker_prompt(
            *tokenizer, g2b::granite_speech5_build_user_content(code));
        test::require_eq(single.prompt_text, prompt_text, std::string(name) + " prompt bytes");
        require_ids_eq(single.input_ids, one_expected, std::string(name) + " one-marker ids");
        // Builder expansion (A=33) == committed Python expansion.
        const auto built = builder.build_prompt_ids(code, 33, reserved);
        require_ids_eq(built.input_ids, expanded_expected, std::string(name) + " expanded");
        test::require_eq(static_cast<int64_t>(built.audio_positions.size()), int64_t{33},
                         std::string(name) + " A=33");
        const auto budget = g2b::granite_speech5_prompt_budget(built);
        g2b::granite_speech5_require_prompt_fits(budget);
        // Whole-sequence decode reproduces the prompt (split code points + marker).
        test::require_eq(tokenizer->decode(single.input_ids, false), prompt_text,
                         std::string(name) + " decode round-trip");
    }
    // UTF-8 / codepoint-splitting samples: exact IDs + round-trip.
    const std::vector<std::pair<std::string, std::vector<int32_t>>> samples = {
        {"日本語テスト 12345 hello",
         {9080, 22656, 45918, 252, 57933, 71634, 220, 4513, 1774, 24748}},
        {"絵文字😀テスト", {58254, 113, 88435, 76460, 222, 57933, 71634}},
        {"USER: こんにちは <|audio|> test 9876543210\n ASSISTANT:",
         {6584, 25, 220, 90115, 220, 100352, 1296, 220, 22207, 21969, 14423, 15, 198,
          36660, 3931, 2891, 25}},
    };
    for (const auto & [text, expected] : samples) {
        const auto ids = tokenizer->encode(text);
        require_ids_eq(ids, expected, "utf-8 encode: " + text);
        test::require_eq(tokenizer->decode(ids, false), text, "utf-8 round-trip: " + text);
    }
    // Expanded runs on the other two fixtures match reference trace input_ids.
    const char * traces = std::getenv("GRANITE_SPEECH5_TRACES");
    if (traces == nullptr) return;
    const std::filesystem::path trace_root(traces);
    for (const auto & [trace, audio_tokens] :
         {std::pair<std::string, int64_t>{"librispeech-0000-hf", 36},
          std::pair<std::string, int64_t>{"multilingual-hf", 252}}) {
        const auto trace_path = trace_root / trace / "trace.json";
        if (!std::filesystem::is_regular_file(trace_path)) {
            std::cout << "NOTE: C1 skips " << trace << " (no trace.json)\n";
            continue;
        }
        const auto parsed = json::parse_file(trace_path);
        const auto reference_ids = ids_from_json(parsed.require("input_ids"));
        const auto built = builder.build_prompt_ids("", audio_tokens, reserved);
        require_ids_eq(built.input_ids, reference_ids, trace + " expanded ids");
        // Injection positions/count mirror the HF audio mask.
        test::require_eq(static_cast<int64_t>(built.audio_positions.size()), audio_tokens,
                         trace + " mask count");
        for (int64_t i = 0; i < audio_tokens; ++i) {
            test::require_eq(built.audio_positions[static_cast<size_t>(i)],
                             built.audio_positions[0] + static_cast<int32_t>(i),
                             trace + " contiguous");
        }
        // Generated IDs decode to the reference text after observed-EOS strip.
        const auto gen_ids = ids_from_json(parsed.require("generated_ids"));
        test::require(!gen_ids.empty() && gen_ids.back() == g2b::kGraniteSpeech5EosTokenId,
                      trace + " observed eos preserved");
        test::require_eq(builder.decode_generated(gen_ids), parsed.require("text").as_string(),
                         trace + " visible text");
    }
    // ja generated sequence (17 IDs incl. EOS) decodes to the reference text.
    const auto ja_trace = trace_root / "ja-hf" / "trace.json";
    if (std::filesystem::is_regular_file(ja_trace)) {
        const auto parsed = json::parse_file(ja_trace);
        const auto gen_ids = ids_from_json(parsed.require("generated_ids"));
        require_ids_eq(gen_ids,
                         std::vector<int32_t>({9080, 22656, 45918, 252, 16556, 27552, 116, 32149,
                                               33121, 28713, 22656, 30512, 98499, 64121, 33541,
                                               1811, 100257}),
                         "ja generated ids");
        test::require_eq(builder.decode_generated(gen_ids), parsed.require("text").as_string(),
                         "ja visible text");
    } else {
        std::cout << "NOTE: C1 skips ja generated decode (no trace.json)\n";
    }
    // Decoder prompt carries the exact C2 types with A*hidden values.
    std::vector<float> projected(static_cast<size_t>(33 * 2048), 0.5F);
    auto decoder_prompt = builder.build_decoder_prompt("ja", 33, projected);
    test::require_eq(decoder_prompt.injection.tokens, int64_t{33}, "injection tokens");
    test::require_eq(static_cast<int64_t>(decoder_prompt.injection.values.size()),
                     int64_t{33 * 2048}, "injection values");
    test::require_eq(static_cast<int64_t>(decoder_prompt.injection.positions.size()),
                     int64_t{33}, "injection positions");
}

void test_prompt_assets(const std::filesystem::path & model_dir) {
    auto resources = prompt_bundle(model_dir);
    g2b::check_granite_speech5_prompt_assets(resources);
    {
        const auto tokenizer_json = json::parse_file(model_dir / "tokenizer.json");
        const auto & added = tokenizer_json.require("added_tokens").as_array();
        test::require_eq(static_cast<int64_t>(added.size()), int64_t{97},
                         "total added tokens");
        int64_t specials = 0;
        for (const auto & entry : added) {
            const auto * special = entry.find("special");
            if (special != nullptr && special->is_bool() && special->as_bool()) {
                ++specials;
            }
        }
        test::require_eq(specials, int64_t{85}, "special added tokens");
    }
    const auto markers = g2b::granite_speech5_reserved_markers_from_tokenizer_json(
        model_dir / "tokenizer.json");
    test::require_eq(static_cast<int64_t>(markers.size()), int64_t{85}, "special marker count");
    test::require(std::find(markers.begin(), markers.end(), "<|audio|>") != markers.end(),
                  "audio reserved");
    g2b::granite_speech5_require_no_reserved_markers("clean instruction text", markers);
    bool threw = false;
    try {
        g2b::granite_speech5_require_no_reserved_markers("evil <|end_of_text|> text", markers);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "control marker rejected");
    // Tampered template / tokenizer settings fail loudly. Each edit copies
    // the four prompt resources and flips one pinned byte/flag.
    const auto tmp = std::filesystem::temp_directory_path() / "g2b_c1_assets";
    auto copy_bundle_with_edit = [&](const std::string & file, const std::string & from,
                                     const std::string & to,
                                     bool expect_unique_needle) {
        namespace fs = std::filesystem;
        fs::remove_all(tmp);
        fs::create_directories(tmp);
        for (const char * name :
             {"tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja"}) {
            if (name != file) {
                fs::copy_file(model_dir / name, tmp / name);
                continue;
            }
            std::ifstream in(model_dir / name, std::ios::binary);
            test::require(!!in, "read " + std::string(name));
            const std::string original((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
            const size_t first = original.find(from);
            test::require(first != std::string::npos, "edit needle: " + file);
            if (expect_unique_needle) {
                test::require(original.find(from, first + 1) == std::string::npos,
                              "unique needle: " + file);
            }
            std::string mutated = original;
            mutated.replace(first, from.size(), to);
            std::ofstream out(tmp / name, std::ios::binary);
            out << mutated;
        }
        engine::assets::ResourceBundle edited(tmp);
        edited.add_model_files({
            {"tokenizer_json", "tokenizer.json", true},
            {"tokenizer_config", "tokenizer_config.json", true},
            {"chat_template", "chat_template.jinja", true},
            {"generation_config", "generation_config.json", true},
        });
        return edited;
    };
    auto expect_check_throw = [&](const std::string & file, const std::string & from,
                                  const std::string & to, bool unique,
                                  const std::string & label) {
        auto edited = copy_bundle_with_edit(file, from, to, unique);
        bool threw = false;
        try {
            g2b::check_granite_speech5_prompt_assets(edited);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, label);
    };
    // Drop the pinned space before the assistant cue.
    {
        auto bad = copy_bundle_with_edit(
            "chat_template.jinja", "\n ASSISTANT:", "\nASSISTANT:", true);
        bool threw = false;
        try {
            g2b::check_granite_speech5_prompt_assets(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "tampered template rejected");
        // The loader fails closed too: it runs the asset check itself.
        threw = false;
        try {
            (void) g2b::load_granite_speech5_tokenizer(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "loader fails closed on tampered template");
    }
    // Drifted tokenizer flags throw.
    expect_check_throw(
        "tokenizer.json", "\"use_regex\": false", "\"use_regex\": true", true,
        "bytelevel use_regex drift rejected");
    expect_check_throw(
        "tokenizer.json", "\"byte_fallback\": false", "\"byte_fallback\": true", true,
        "byte_fallback drift rejected");
    expect_check_throw(
        "tokenizer.json", "\"lstrip\": false", "\"lstrip\": true", false,
        "added-token lstrip drift rejected");
    std::filesystem::remove_all(tmp);
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_language_codes();
        test_template_bytes();
        test_reserved_markers();
        if (std::getenv("GRANITE_SPEECH5_FIXTURES")) {
            test_fixture_splice_and_hint_invariant();
        } else {
            std::cout << "NOTE: reference prompt parity requires GRANITE_SPEECH5_FIXTURES\n";
        }
        test_synthetic_tiny_fixture();
        test_injection_rules();
        test_budget_rules();
        test_eos_rule();
        if (const auto model_dir = find_model_dir()) {
            if (std::getenv("GRANITE_SPEECH5_FIXTURES")) test_tokenizer_parity(*model_dir);
            test_prompt_assets(*model_dir);
            std::cout << "PASS: granite_speech5 C1 unit checks (checkpoint checks; reference parity requires explicit references)\n";
        } else {
            std::cout << "NOTE: C1 checkpoint parity skipped (set GRANITE_SPEECH5_MODEL)\n";
            std::cout << "PASS: granite_speech5 C1 unit checks\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
