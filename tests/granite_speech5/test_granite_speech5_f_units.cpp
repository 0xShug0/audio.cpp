#include "engine/framework/debug/profiler.h"
// granite_speech5 F unit checks: keyword biasing + speech translation
// prompt builders, session target_language/keywords resolution, budget gates,
// and checkpoint-gated prompt-ID parity against the pinned Python processor.
//
// Always-run (no checkpoint): target code/name validation, keyword
// parse/join edges, composer byte-exactness for every committed f_prompt
// fixture, empty-context builder identity with the plain ASR path, reserved
// rejection on terms, budget overflow messages naming the 16384 limit.
// Checkpoint-gated (GRANITE_SPEECH5_MODEL):
// one-marker token-ID parity per fixture, expanded parity on the real-A
// fixtures, 2000-term budget fit (IBM's cited keyword-list max) and a
// 20000-term over-context rejection through the exact gates transcribe_span
// uses.

#include "engine/community_models/granite_speech5/decoder.h"
#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/community_models/granite_speech5/session.h"
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
#include <unordered_map>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace json = engine::io::json;
namespace runtime = engine::runtime;



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

std::vector<std::string> strings_from_json(const json::Value & value) {
    std::vector<std::string> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_string());
    }
    return out;
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

void require_strings_eq(const std::vector<std::string> & actual,
                         const std::vector<std::string> & expected,
                         const std::string & label) {
    test::require_eq(actual.size(), expected.size(), label + " size");
    for (size_t i = 0; i < expected.size(); ++i) {
        test::require_eq(actual[i], expected[i], label + "[" + std::to_string(i) + "]");
    }
}

runtime::TaskRequest request_with_text_options(
    std::unordered_map<std::string, std::string> options,
    const std::string & text = "",
    bool with_text_input = false) {
    runtime::TaskRequest request;
    request.options = std::move(options);
    if (with_text_input) {
        request.text_input = runtime::Transcript{text, ""};
    }
    return request;
}

// --- always-run, no-weight rule tests ---

void test_target_languages() {
    using g2b::granite_speech5_normalize_target_language_code;
    test::require_eq(granite_speech5_normalize_target_language_code(""), std::string(""),
                     "blank target");
    test::require_eq(granite_speech5_normalize_target_language_code("   \t\n "),
                     std::string(""), "whitespace blank");
    test::require_eq(granite_speech5_normalize_target_language_code(" FR "), std::string("fr"),
                     "trim+lower");
    test::require_eq(granite_speech5_normalize_target_language_code("Zh"),
                     std::string("zh"), "mixed case");
    for (const char * code : {"en", "fr", "de", "es", "pt", "ja", "it", "zh"}) {
        (void) granite_speech5_normalize_target_language_code(code);
    }
    for (const char * bad :
         {"xx", "english", "jp", "cmn", "zh-CN", "pt-BR", "mandarin", "italian"}) {
        bool threw = false;
        try {
            (void) granite_speech5_normalize_target_language_code(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, std::string("reject unsupported target: ") + bad);
    }
    for (const char * bad : {"日本語", "français"}) {
        bool threw = false;
        try {
            (void) granite_speech5_normalize_target_language_code(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "reject non-ASCII target");
    }
    using g2b::granite_speech5_target_language_name;
    test::require_eq(std::string(granite_speech5_target_language_name("fr")),
                     std::string("French"), "fr name");
    test::require_eq(std::string(granite_speech5_target_language_name("pt")),
                     std::string("Portuguese"), "pt name");
    test::require_eq(std::string(granite_speech5_target_language_name("it")),
                     std::string("Italian"), "it name");
    test::require_eq(std::string(granite_speech5_target_language_name("zh")),
                     std::string("Mandarin"), "zh name");
    for (const char * code : {"en", "de", "es", "ja"}) {
        (void) granite_speech5_target_language_name(code);
    }
    bool threw = false;
    try {
        (void) granite_speech5_target_language_name("xx");
    } catch (const std::runtime_error & error) {
        threw = true;
        test::require(std::string(error.what()).find("xx") != std::string::npos,
                      "names the code");
    }
    test::require(threw, "reject unknown target name");
}

void test_keyword_parse() {
    using g2b::granite_speech5_parse_keyword_terms;
    require_strings_eq(granite_speech5_parse_keyword_terms("Concord, tents"),
                       {"Concord", "tents"}, "basic");
    require_strings_eq(granite_speech5_parse_keyword_terms("  a ,b\t,\n c  "),
                       {"a", "b", "c"}, "trim each");
    require_strings_eq(granite_speech5_parse_keyword_terms(""), {}, "empty");
    require_strings_eq(granite_speech5_parse_keyword_terms("   \t\n "), {}, "whitespace");
    require_strings_eq(granite_speech5_parse_keyword_terms("a,,b,"), {"a", "b"},
                       "skip empties");
    require_strings_eq(granite_speech5_parse_keyword_terms(",,,"), {}, "all commas");
    // UTF-8, multiword, and repeats are preserved verbatim (no folding/dedup).
    require_strings_eq(
        granite_speech5_parse_keyword_terms("café, 日本語, returned to its place"),
        {"café", "日本語", "returned to its place"}, "preserve");
    require_strings_eq(granite_speech5_parse_keyword_terms("a, a, b"), {"a", "a", "b"},
                       "repeats kept");
    test::require_eq(g2b::granite_speech5_join_keyword_terms({"a", "b c"}), std::string("a, b c"),
                     "join");
    test::require_eq(g2b::granite_speech5_join_keyword_terms({}), std::string(""),
                     "join empty");
}

void test_composers_against_fixtures() {
    // Every external f_prompt reference: C++ composer output must be
    // byte-identical to the pinned-Python user_content (string parity before
    // any token-ID comparison).
    const std::vector<std::string> names = {
        "f_prompt_kw_rare_hint", "f_prompt_kw_rare_nohint", "f_prompt_kw_absent",
        "f_prompt_kw_accented", "f_prompt_kw_multiword", "f_prompt_tr_en_fr",
        "f_prompt_tr_en_de", "f_prompt_tr_en_es", "f_prompt_tr_en_pt",
        "f_prompt_tr_en_ja", "f_prompt_tr_en_it", "f_prompt_tr_en_zh",
        "f_prompt_tr_fr_en", "f_prompt_tr_de_en", "f_prompt_tr_es_en",
        "f_prompt_tr_pt_en", "f_prompt_tr_ja_en", "f_prompt_tr_en_fr_src",
        "f_prompt_tr_en_fr_kw", "f_prompt_tr_ja_en_src",
    };
    for (const auto & name : names) {
        const auto fixture = load_fixture(name + ".json");
        const std::string source = fixture.require("source").as_string();
        const std::string target = fixture.require("target").as_string();
        const auto terms = strings_from_json(fixture.require("terms"));
        std::string composed;
        if (target.empty()) {
            composed = g2b::granite_speech5_build_keyword_user_content(source, terms);
        } else {
            composed = g2b::granite_speech5_build_translation_user_content(
                source, target, terms);
        }
        test::require_eq(composed, fixture.require("user_content").as_string(),
                         name + " user content");
        test::require_eq(
            g2b::granite_speech5_render_single_turn_prompt(composed),
            fixture.require("prompt").as_string(), name + " prompt render");
    }
 }

void test_composer_contracts() {
    // Spot-check the exact bytes: the card's instruction text with the
    // model-owned marker, C1 hint prefix where applicable.
    test::require_eq(
        g2b::granite_speech5_build_keyword_user_content("en", {"a", "b c"}),
        "The speech is in English. Transcribe it exactly; do not translate. "
        "<|audio|>transcribe the speech to text. Keywords: a, b c",
        "keyword hint bytes");
    test::require_eq(
        g2b::granite_speech5_build_translation_user_content("", "fr", {}),
        "<|audio|>translate the speech to French with proper punctuation and "
        "capitalization.",
        "translation bytes");
    test::require_eq(
        g2b::granite_speech5_build_translation_user_content("ja", "en", {"x"}),
        "The speech is in Japanese. <|audio|>translate the speech to English with "
        "proper punctuation and capitalization. Keywords: x",
        "translation src+kw bytes");
    // Empty term lists and bad codes throw loudly (session routes these away).
    for (int mode = 0; mode < 2; ++mode) {
        bool threw = false;
        try {
            if (mode == 0) {
                (void) g2b::granite_speech5_build_keyword_user_content("en", {});
            } else {
                (void) g2b::granite_speech5_build_translation_user_content(
                    "en", "xx", {});
            }
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, mode == 0 ? "empty keywords throw" : "bad target throws");
    }
    bool threw = false;
    try {
        (void) g2b::granite_speech5_build_keyword_user_content("xx", {"a"});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "bad source hint throws");
}

void test_resolve_target_and_keywords() {
    using g2b::granite_speech5_resolve_request_options;
    {
        // Absent target + no text = plain ASR (D behavior unchanged).
        const auto resolved = granite_speech5_resolve_request_options(
            request_with_text_options({{"language", "en"}}));
        test::require_eq(resolved.target_language, "", "absent target");
        test::require(resolved.keywords.empty(), "no text input = no keywords");
        test::require_eq(resolved.language, "en", "source kept");
    }
    {
        // Target from options only; keywords from text_input.text.
        auto resolved = granite_speech5_resolve_request_options(
            request_with_text_options({{"target_language", " FR "}}, "Concord, tents", true));
        test::require_eq(resolved.target_language, "fr", "target normalized");
        require_strings_eq(resolved.keywords, {"Concord", "tents"}, "keywords parsed");
    }
    {
        // Whitespace text = no biasing; blank option = ASR.
        auto resolved = granite_speech5_resolve_request_options(
            request_with_text_options({{"target_language", "  "}}, "  ", true));
        test::require_eq(resolved.target_language, "", "blank target = ASR");
        test::require(resolved.keywords.empty(), "whitespace = no keywords");
    }
    for (const char * bad : {"xx", "english", "pt-BR"}) {
        bool threw = false;
        try {
            (void) granite_speech5_resolve_request_options(
                request_with_text_options({{"target_language", bad}}));
        } catch (const std::runtime_error & error) {
            threw = true;
            // The message names the normalized (trimmed, lowercased) code.
            test::require(std::string(error.what()).find("unsupported target language") !=
                              std::string::npos,
                          "names the failure");
        }
        test::require(threw, std::string("reject target: ") + bad);
    }
    // Source-hint precedence (D) is untouched by the new keys.
    const auto resolved = granite_speech5_resolve_request_options(
        request_with_text_options({{"language", "EN"}, {"target_language", "de"}}, "a", true));
    test::require_eq(resolved.language, "en", "source still normalized");
    test::require_eq(resolved.target_language, "de", "target alongside source");
}

void test_reserved_and_budget_messages() {
    // A caller term smuggling a control marker is rejected naming the marker.
    const std::vector<std::string> markers = {"<|audio|>", "<|end_of_text|>", "<|pad|>"};
    g2b::granite_speech5_require_no_reserved_markers("café, 日本語", markers);
    for (const auto & evil :
         {std::string("ok <|audio|> term"), std::string("evil <|end_of_text|> text")}) {
        bool threw = false;
        try {
            g2b::granite_speech5_require_no_reserved_markers(evil, markers);
        } catch (const std::runtime_error & error) {
            threw = true;
            test::require(std::string(error.what()).find("<|") != std::string::npos,
                          "names marker");
        }
        test::require(threw, "reject: " + evil);
    }
    // Over-context errors name the 16384 processor bound (the gates
    // transcribe_span applies after the F builders).
    for (int mode = 0; mode < 2; ++mode) {
        bool threw = false;
        try {
            if (mode == 0) {
                g2b::granite_speech5_require_generation_budget(16000, 512);
            } else {
                g2b::GraniteSpeech5PromptBudget over;
                over.prompt_len = 16385;
                over.max_position_embeddings = 16384;
                g2b::granite_speech5_require_prompt_fits(over);
            }
        } catch (const std::runtime_error & error) {
            threw = true;
            test::require(std::string(error.what()).find("16384") != std::string::npos,
                          "names the limit");
        }
        test::require(threw, mode == 0 ? "generation budget throws" : "prompt fits throws");
    }
    // Boundary: prompt_len + max_tokens == 16384 fits exactly.
    g2b::granite_speech5_require_generation_budget(15872, 512);
}

// --- checkpoint-gated parity tests ---

std::optional<std::filesystem::path> find_model_dir() {
    if (const char * env = std::getenv("GRANITE_SPEECH5_MODEL")) {
        const std::filesystem::path candidate(env);
        if (std::filesystem::is_regular_file(candidate / "model.safetensors")) {
            return candidate;
        }
    }

    return std::nullopt;
}

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

void test_fixture_token_parity(const std::filesystem::path & model_dir) {
    auto resources = prompt_bundle(model_dir);
    auto tokenizer = g2b::load_granite_speech5_tokenizer(resources);
    const g2b::GraniteSpeech5PromptBuilder builder(tokenizer);
    const auto reserved = g2b::granite_speech5_reserved_markers_from_tokenizer_json(
        model_dir / "tokenizer.json");
    test::require_eq(static_cast<int64_t>(reserved.size()), int64_t{85}, "reserved count");
    const std::vector<std::string> names = {
        "f_prompt_kw_rare_hint", "f_prompt_kw_rare_nohint", "f_prompt_kw_absent",
        "f_prompt_kw_accented", "f_prompt_kw_multiword", "f_prompt_tr_en_fr",
        "f_prompt_tr_en_de", "f_prompt_tr_en_es", "f_prompt_tr_en_pt",
        "f_prompt_tr_en_ja", "f_prompt_tr_en_it", "f_prompt_tr_en_zh",
        "f_prompt_tr_fr_en", "f_prompt_tr_de_en", "f_prompt_tr_es_en",
        "f_prompt_tr_pt_en", "f_prompt_tr_ja_en", "f_prompt_tr_en_fr_src",
        "f_prompt_tr_en_fr_kw", "f_prompt_tr_ja_en_src",
    };
    for (const auto & name : names) {
        const auto fixture = load_fixture(name + ".json");
        const std::string source = fixture.require("source").as_string();
        const std::string target = fixture.require("target").as_string();
        const auto terms = strings_from_json(fixture.require("terms"));
        const auto one_expected = ids_from_json(fixture.require("one_marker_ids"));
        // Framework encode of the composed content == pinned Python IDs.
        const auto built = builder.build_prompt_ids_with_context(source, target, terms, 33,
                                                                  reserved);
        const auto single = g2b::granite_speech5_encode_single_marker_prompt(
            *tokenizer,
            target.empty()
                ? g2b::granite_speech5_build_keyword_user_content(source, terms)
                : g2b::granite_speech5_build_translation_user_content(source, target,
                                                                        terms));
        require_ids_eq(single.input_ids, one_expected, name + " one-marker ids");
        test::require_eq(single.prompt_text, fixture.require("prompt").as_string(),
                         name + " prompt bytes");
        // with_context dispatches to the same composer (33-token self-check).
        const auto spliced = g2b::granite_speech5_expand_audio_markers(
            one_expected, g2b::kGraniteSpeech5AudioTokenId, 33);
        require_ids_eq(built.input_ids, spliced.input_ids, name + " with_context splice");
        if (fixture.find("expanded_ids") != nullptr) {
            const auto audio_tokens = fixture.require("audio_tokens").as_i64();
            const auto expanded_expected = ids_from_json(fixture.require("expanded_ids"));
            const auto full = builder.build_prompt_ids_with_context(source, target, terms,
                                                                     audio_tokens, reserved);
            require_ids_eq(full.input_ids, expanded_expected, name + " expanded ids");
        }
    }
    // Empty context reproduces the plain ASR builder exactly (plain path
    // unchanged at the ID level, not just the branch level).
    const auto plain = builder.build_prompt_ids("en", 36, reserved);
    const auto via_context = builder.build_prompt_ids_with_context("en", "", {}, 36, reserved);
    require_ids_eq(via_context.input_ids, plain.input_ids, "empty context == plain");
    // The unified decoder entry serves every mode: translation/keyword
    // decoder prompts carry the with_context IDs (never a silent plain
    // prompt), with injection positions/values passed through; empty
    // context reproduces the plain decoder prompt exactly.
    std::vector<float> values(static_cast<size_t>(36 * 2048), 0.25F);
    for (const auto & [source, target, terms] :
         {std::tuple<std::string, std::string, std::vector<std::string>>{"en", "fr", {"Concord"}},
          {"en", "", {"Concord", "tents"}}, {"", "", {}}}) {
        const auto ids =
            builder.build_prompt_ids_with_context(source, target, terms, 36, reserved);
        auto decoder_prompt = builder.build_decoder_prompt_with_context(
            source, target, terms, 36, values, reserved);
        require_ids_eq(decoder_prompt.input_ids, ids.input_ids,
                       "decoder with_context ids (" + target + ")");
        test::require_eq(decoder_prompt.injection.tokens, int64_t{36},
                         "decoder with_context tokens (" + target + ")");
        require_ids_eq(decoder_prompt.injection.positions, ids.audio_positions,
                       "decoder with_context positions (" + target + ")");
        test::require_eq(static_cast<int64_t>(decoder_prompt.injection.values.size()),
                         int64_t{36 * 2048}, "decoder with_context values (" + target + ")");
    }
    const auto plain_decoder = builder.build_decoder_prompt("en", 36, values, reserved);
    const auto empty_decoder =
        builder.build_decoder_prompt_with_context("en", "", {}, 36, values, reserved);
    require_ids_eq(empty_decoder.input_ids, plain_decoder.input_ids,
                   "empty decoder context == plain");
    require_ids_eq(empty_decoder.injection.positions, plain_decoder.injection.positions,
                   "empty decoder positions == plain");
    // Reserved markers inside keyword terms are rejected at encode time.
    bool threw = false;
    try {
        (void) builder.build_prompt_ids_with_context(
            "en", "", {"ok", "evil <|end_of_text|> term"}, 36, reserved);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "reserved term rejected at encode");
}

void test_keyword_budget_scale(const std::filesystem::path & model_dir) {
    auto resources = prompt_bundle(model_dir);
    auto tokenizer = g2b::load_granite_speech5_tokenizer(resources);
    const g2b::GraniteSpeech5PromptBuilder builder(tokenizer);
    const auto reserved = g2b::granite_speech5_reserved_markers_from_tokenizer_json(
        model_dir / "tokenizer.json");
    // IBM cites up to 2000 keywords: 2000 short terms must still fit the
    // runtime budget on a short clip (A=36, default max_tokens=512).
    std::vector<std::string> two_k;
    for (int i = 0; i < 2000; ++i) {
        two_k.push_back("kw" + std::to_string(i));
    }
    const auto big = builder.build_prompt_ids_with_context("en", "", two_k, 36, reserved);
    const auto budget = g2b::granite_speech5_prompt_budget(big);
    g2b::granite_speech5_require_prompt_fits(budget);
    g2b::granite_speech5_require_generation_budget(budget.prompt_len, 512);
    std::cout << "NOTE: F 2000-term prompt_len=" << budget.prompt_len << " (text_tokens="
              << budget.text_tokens << ")\n";
    // 20000 terms overflows the same budget through the exact session gates.
    std::vector<std::string> twenty_k;
    for (int i = 0; i < 20000; ++i) {
        twenty_k.push_back("keyword" + std::to_string(i));
    }
    const auto over = builder.build_prompt_ids_with_context("en", "", twenty_k, 36, reserved);
    const auto over_budget = g2b::granite_speech5_prompt_budget(over);
    bool threw = false;
    try {
        g2b::granite_speech5_require_prompt_fits(over_budget);
        g2b::granite_speech5_require_generation_budget(over_budget.prompt_len, 512);
    } catch (const std::runtime_error & error) {
        threw = true;
        test::require(std::string(error.what()).find("16384") != std::string::npos,
                      "over-context names the limit");
    }
    test::require(threw, "20000-term list rejected");
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_target_languages();
        test_keyword_parse();
        if (std::getenv("GRANITE_SPEECH5_FIXTURES")) {
            test_composers_against_fixtures();
        } else {
            std::cout << "NOTE: reference composer parity requires GRANITE_SPEECH5_FIXTURES\n";
        }
        test_composer_contracts();
        test_resolve_target_and_keywords();
        test_reserved_and_budget_messages();
        if (const auto model_dir = find_model_dir()) {
            if (std::getenv("GRANITE_SPEECH5_FIXTURES")) test_fixture_token_parity(*model_dir);
            test_keyword_budget_scale(*model_dir);
            std::cout << "PASS: granite_speech5 F unit checks (checkpoint checks; reference parity requires explicit references)\n";
        } else {
            std::cout << "NOTE: F checkpoint parity skipped (set GRANITE_SPEECH5_MODEL)\n";
            std::cout << "PASS: granite_speech5 F unit checks\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
