// PocketTTS text chunking against cases from the reference's tests
// (pocket-tts tests/test_split_sentences.py). tokenizers/tokenizer-1.model is the
// tokenizer of the English PocketTTS model.
#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/text_conditioner.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::pocket_tts::PocketTTSAssets;
using engine::models::pocket_tts::TextConditioner;

std::string join(const std::vector<std::string> & chunks) {
    std::string out;
    for (const auto & chunk : chunks) {
        out += "[" + chunk + "]";
    }
    return out;
}

void expect_chunks(
    const PocketTTSAssets & assets,
    const std::string & text,
    int64_t max_tokens,
    const std::vector<std::string> & expected) {
    const auto chunks = TextConditioner().split_into_sentence_chunks(assets, text, max_tokens);
    if (chunks != expected) {
        throw std::runtime_error(
            "\"" + text + "\" at " + std::to_string(max_tokens) + " tokens: expected " + join(expected) + ", got " +
            join(chunks));
    }
}

// Punctuation followed by a letter or digit ends nothing, so the chunks keep the text as
// written: joining the pieces with a space used to give "www. kyutai. org".
void test_no_cut_inside_words(const PocketTTSAssets & assets) {
    expect_chunks(
        assets,
        "Visit www.kyutai.org for the docs. Then write to jo@icloud.com if it breaks.",
        50,
        {"Visit www.kyutai.org for the docs. Then write to jo@icloud.com if it breaks."});
    expect_chunks(assets, "We met at 4 p.m. on Tuesday. It rained.", 50, {"We met at 4 p.m. on Tuesday. It rained."});
    expect_chunks(
        assets,
        "You can view the details of your consultation and of the invoice at www.actu.com/tech, "
        "or send an email to liam_jones@icloud.com, and we will reply before 4 p.m. on Monday.",
        50,
        {"You can view the details of your consultation and of the invoice at www.actu.com/tech, "
         "or send an email to liam_jones@icloud.com,",
         "and we will reply before 4 p.m. on Monday."});
    expect_chunks(
        assets,
        "The train leaves at 10:30, arrives at 12:45, and returns at 18:05.",
        8,
        {"The train leaves at 10:30,", "arrives at 12:45,", "and returns at 18:05."});
    expect_chunks(
        assets,
        "The technician reported that the centrifuge needs 1,994.42 kW and an extra 1,874 kW to "
        "function on 06-30-2028 at 14 Lab Court, E8 5TT Manchester, MU9S 3VC, le taux est de 3,5 %",
        50,
        {"The technician reported that the centrifuge needs 1,994.42 kW and an extra 1,874 kW to "
         "function on 06-30-2028 at 14 Lab Court,",
         "E8 5TT Manchester, MU9S 3VC, le taux est de 3,5 %."});
    expect_chunks(
        assets, "We live in the U.S. It is big. Very big.", 8, {"We live in the U.S.", "It is big. Very big."});
}

// A period that ends a title stays with the name after it.
void test_no_cut_after_titles(const PocketTTSAssets & assets) {
    expect_chunks(
        assets,
        "Hi, I'm calling to follow up on your latest lab results, the ones we discussed with Dr. "
        "Miller and Prof. Chen, the specialists in cardiology, and the general practitioner.",
        50,
        {"Hi, I'm calling to follow up on your latest lab results, the ones we discussed with Dr. "
         "Miller and Prof. Chen, the specialists in cardiology,",
         "and the general practitioner."});
    expect_chunks(
        assets,
        "Dr. Miller and Mrs. Jones arrived. Prof. Smith was late, as usual.",
        8,
        {"Dr. Miller and Mrs. Jones arrived.", "Prof. Smith was late,", "as usual."});
}

// A chunk never ends on a decimal number, which the model would read as the end of the
// sentence.
void test_no_chunk_ends_on_a_decimal_number(const PocketTTSAssets & assets) {
    const std::string reimbursement =
        "The decision will be processed by the S C H E L L I N G department and the result "
        "will be a reimbursement of $822.95, which should be reflected in your account within "
        "a few business days.";
    expect_chunks(assets, reimbursement, 50, {reimbursement});
    expect_chunks(
        assets,
        "The value is 3.14. The next value is larger. And the third one is the largest.",
        10,
        {"The value is 3.14. The next value is larger.", "And the third one is the largest."});
    expect_chunks(
        assets,
        "The fee is $822.95, payable by Friday, and the deposit is $100.50, due today.",
        8,
        {"The fee is $822.95, payable by Friday,", "and the deposit is $100.50, due today."});
}

// A period followed by a space still ends a sentence, also between two numbers, and the rest
// of the splitting is unchanged.
void test_sentence_ends(const PocketTTSAssets & assets) {
    expect_chunks(assets, "I counted to 3. 14 people were there.", 10, {"I counted to 3.", "14 people were there."});
    expect_chunks(
        assets,
        "First sentence here. Second sentence here. Third one.",
        10,
        {"First sentence here.", "Second sentence here.", "Third one."});
    expect_chunks(
        assets, "We met in 2008. It was raining. The bus was late.", 8,
        {"We met in 2008.", "It was raining.", "The bus was late."});
    expect_chunks(
        assets,
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
        "age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was "
        "the season of Light, it was the season of Darkness",
        20,
        {"It was the best of times, it was the worst of times,",
         "it was the age of wisdom, it was the age of foolishness,",
         "it was the epoch of belief,",
         "it was the epoch of incredulity,",
         "it was the season of Light, it was the season of Darkness."});
}

}  // namespace

int main() {
    try {
        PocketTTSAssets assets;
        assets.tokenizer_pieces = engine::tokenizers::load_sentencepiece_model(
            std::filesystem::path(ENGINE_TEST_ASSET_ROOT) / "tokenizers" / "tokenizer-1.model");
        test_no_cut_inside_words(assets);
        test_no_cut_after_titles(assets);
        test_no_chunk_ends_on_a_decimal_number(assets);
        test_sentence_ends(assets);
    } catch (const std::exception & error) {
        std::cerr << "pocket_tts_text_chunking_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "pocket_tts_text_chunking_test passed\n";
    return 0;
}
