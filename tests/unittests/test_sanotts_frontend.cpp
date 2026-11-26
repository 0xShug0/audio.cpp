#include "engine/community_models/sanotts/frontend.h"
#include "engine/community_models/sanotts/runtime.h"

#include "test_assert.h"

#include <iostream>
#include <string>
#include <vector>

int main() try {
    using engine::models::sanotts::SanoTtsFrontend;
    using engine::models::sanotts::sanotts_text_seed;

    // sha256 known-answer vectors: the seed is the first 8 digest bytes,
    // big-endian, exactly int.from_bytes(sha256(text).digest()[:8], "big").
    engine::test::require(
        sanotts_text_seed("abc") == 0xBA7816BF8F01CFEAULL,
        "sha256 seed of 'abc'");
    engine::test::require(
        sanotts_text_seed("") == 0xE3B0C44298FC1C14ULL,
        "sha256 seed of the empty string");
    // 64 bytes forces the two-block tail path (rem >= 56).
    engine::test::require(
        sanotts_text_seed(std::string(64, 'a')) == 0xFFE054FE7AE0CB6DULL,
        "sha256 seed across the two-block tail");

    const auto chunks = SanoTtsFrontend::split_text(
        "First sentence ends here. Second sentence is also short. "
        "And a third one rounds it out.",
        40);
    engine::test::require(chunks.size() >= 2, "long text must split");
    for (const auto & chunk : chunks) {
        engine::test::require(!chunk.empty(), "no empty chunks");
    }

    const auto single = SanoTtsFrontend::split_text("Tiny.", 280);
    engine::test::require(
        single.size() == 1 && single[0] == "Tiny.",
        "short text stays one chunk");

    engine::test::require(
        SanoTtsFrontend::boundary_pause_seconds("Ends with a period.") == 0.20,
        "sentence end pause");
    engine::test::require(
        SanoTtsFrontend::boundary_pause_seconds("ends with a comma,") == 0.08,
        "clause pause");
    engine::test::require(
        SanoTtsFrontend::boundary_pause_seconds("trailing space. ") == 0.20,
        "pause looks through trailing whitespace");

    const auto keep = [](const std::string & text) {
        return SanoTtsFrontend::keep_contractions(text);
    };
    engine::test::require(
        keep("Don't stop, it's fine.") ==
            "Don\xE2\x80\x99t stop, it\xE2\x80\x99s fine.",
        "apostrophes inside words are kept with the word");
    engine::test::require(
        keep("rock'n'roll") == "rock\xE2\x80\x99n\xE2\x80\x99roll",
        "two apostrophes in one word");
    engine::test::require(
        keep("'quoted' dogs' 'til 5'6 a ' b '") == "'quoted' dogs' 'til 5'6 a ' b '",
        "other apostrophes stay punctuation");
    engine::test::require(
        keep("").empty() && keep("'") == "'", "empty and lone apostrophe");

    std::cout << "sanotts frontend tests passed\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "sanotts frontend test failed: " << error.what() << "\n";
    return 1;
}
