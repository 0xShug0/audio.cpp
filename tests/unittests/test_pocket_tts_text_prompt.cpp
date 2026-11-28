#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/routing.h"
#include "engine/models/pocket_tts/text_conditioner.h"

#include "test_assert.h"

#include <exception>
#include <iostream>
#include <string>

namespace {

namespace pocket_tts = engine::models::pocket_tts;

struct Case {
    const char * text;
    const char * prepared;
};

void require_prepared(const pocket_tts::PocketTTSModelConfig & config, const Case & c, const std::string & label) {
    const std::string actual =
        pocket_tts::prepare_prompt_text(config, c.text, pocket_tts::PromptPunctuation::PocketTTS);
    engine::test::require(
        actual == c.prepared,
        label + " prompt for '" + c.text + "': expected '" + c.prepared + "' actual '" + actual + "'");
}

// Expected values are pocket-tts prepare_text_prompt with the settings of
// myned-ai/pocket-tts-greek greek.yaml (append_terminal_punctuation: false).
void test_greek_prompt_matches_reference() {
    pocket_tts::PocketTTSModelConfig config;
    config.append_terminal_punctuation = false;
    const Case cases[] = {
        {"καλημέρα σας", "Καλημέρα σας"},
        {"έλα εδώ", "Έλα εδώ"},
        // ";" is the Greek question mark and must survive.
        {"Τι κάνεις;", "Τι κάνεις;"},
        {"σήμερα; ναι", "Σήμερα; ναι"},
        // str.upper() has no precomposed capital for ΐ.
        {"ΐσως", "Ϊ́σως"},
        {"ύστερα, είπε,", "Ύστερα, είπε,"},
        {"ωραία\nμέρα", "Ωραία μέρα"},
        {"Ήδη.", "Ήδη."},
        // audio.cpp skips opening quotes before upper-casing, as for the other languages.
        {"«γεια», είπε", "«Γεια», είπε"},
    };
    for (const auto & c : cases) {
        require_prepared(config, c, "greek");
    }
}

// The default settings keep the reference terminal punctuation step.
void test_default_prompt_unchanged() {
    const pocket_tts::PocketTTSModelConfig config;
    const Case cases[] = {
        {"hello there", "Hello there."},
        {"état", "État."},
        {"wait,", "Wait."},
        {"Τι κάνεις;", "Τι κάνεις."},
        {"καλημέρα σας", "Καλημέρα σας."},
    };
    for (const auto & c : cases) {
        require_prepared(config, c, "default");
    }
}

}  // namespace

int main() {
    try {
        test_greek_prompt_matches_reference();
        test_default_prompt_unchanged();
        std::cout << "pocket_tts_text_prompt_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "pocket_tts_text_prompt_test failed: " << ex.what() << "\n";
        return 1;
    }
}
