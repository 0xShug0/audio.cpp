/*
 * The `voice_samples` speaker list.
 *
 * The entry count is the speaker count, and an entry with no path is a speaker
 * that is named but not cloned -- "[S<n>]: None" in the prompt. That is how a
 * dialogue clones one voice and lets the model invent the other, and the
 * spelling for it is a trailing separator: "a.wav,".
 *
 * Pinned because the first version of this discarded a trailing blank as a
 * typo, which silently turned exactly that spelling into a single-speaker
 * prompt. The documentation said one thing and the parser did the other, and
 * nothing failed -- the model simply cloned one voice and never heard about the
 * second speaker.
 */

#include "engine/community_models/moss_ttsd/session.h"
#include "test_assert.h"

#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using engine::models::moss_ttsd::parse_speaker_paths;

// "a.wav" for a cloned speaker, "-" for one that is named but not cloned.
std::string describe(const std::string & value) {
    std::string out;
    for (const auto & entry : parse_speaker_paths(value)) {
        if (!out.empty()) {
            out += "|";
        }
        out += entry.value_or("-");
    }
    return out;
}

void check(const std::string & value, const std::string & expected) {
    const auto actual = describe(value);
    engine::test::require(
        actual == expected,
        "voice_samples \"" + value + "\" parsed as [" + actual + "], expected [" + expected + "]");
}

}  // namespace

int main() {
    try {
        // The case the option exists for: clone [S1], invent [S2].
        check("a.wav,", "a.wav|-");
        // Both cloned.
        check("a.wav,b.wav", "a.wav|b.wav");
        // One speaker, cloned. No second speaker is implied.
        check("a.wav", "a.wav");
        // Invent the first, clone the second.
        check(",b.wav", "-|b.wav");
        // Three speakers, the middle one invented.
        check("a.wav,,c.wav", "a.wav|-|c.wav");
        // Two speakers, neither cloned -- pointless but well defined.
        check(",", "-|-");
        // No option value is no speakers at all, which is a different prompt
        // from a list of uncloned ones: it renders "Reference(s): None".
        check("", "");
        // Surrounding whitespace is the shell's, not part of a path.
        check(" a.wav , b.wav ", "a.wav|b.wav");
        check("a.wav,   ", "a.wav|-");

        std::cout << "moss_ttsd_speakers_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "moss_ttsd_speakers_test failed: " << error.what() << "\n";
        return 1;
    }
}
