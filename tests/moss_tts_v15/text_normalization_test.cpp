/*
 * Text-normalisation parity for the MOSS-TTS family.
 *
 * The vectors are the ones that ship inside the reference
 * tts_robust_normalizer_single_script.py, exported verbatim, so the contract is
 * upstream's rather than one invented here.
 *
 *   moss_tts_v15_text_normalization_test [--vectors <json>]
 */

#include "engine/framework/io/json.h"
#include "engine/framework/text/moss_tts_normalization.h"

#include <exception>
#include <iostream>
#include <string>

namespace {

namespace json = engine::io::json;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto path = arg_value(
            argc, argv, "--vectors",
            "tests/moss_tts_v15/reference/text_normalization_vectors.json");
        const auto vectors = json::parse_file(path);

        int64_t passed = 0;
        int64_t failed = 0;
        for (const auto & entry : vectors.as_array()) {
            const auto name = json::require_string(entry, "name");
            const auto input = json::require_string(entry, "input");
            const auto expected = json::require_string(entry, "expected");
            const auto actual = engine::text::normalize_moss_tts_text(input);
            if (actual == expected) {
                ++passed;
                continue;
            }
            ++failed;
            if (failed <= 8) {
                std::cout << "FAIL " << name << "\n   in: " << input
                          << "\n  out: " << actual << "\n  exp: " << expected << "\n";
            }
        }

        // Idempotence is a property the reference relies on: the processor may
        // normalise text that has already been normalised.
        int64_t non_idempotent = 0;
        for (const auto & entry : vectors.as_array()) {
            const auto expected = json::require_string(entry, "expected");
            if (engine::text::normalize_moss_tts_text(expected) != expected) {
                ++non_idempotent;
                if (non_idempotent <= 4) {
                    std::cout << "NOT IDEMPOTENT: " << expected << "\n             -> "
                              << engine::text::normalize_moss_tts_text(expected) << "\n";
                }
            }
        }

        std::cout << "\n=== TEXT NORMALISATION: " << passed << " passed, " << failed
                  << " failed, " << non_idempotent << " non-idempotent ===\n";
        const bool ok = failed == 0 && non_idempotent == 0;
        std::cout << (ok ? "PASS" : "FAIL") << "\n";
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "text_normalization_test failed: " << error.what() << "\n";
        return 1;
    }
}
