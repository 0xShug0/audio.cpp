#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sanotts {

struct SanoTtsEncoded {
    std::vector<int32_t> token_ids;
    std::string dropped;   // symbols outside the vocabulary, for tracing
};

/**
 * Text -> the 62-symbol phoneme ids the sanoTTS front end was trained on.
 *
 * eSpeak-ng produces the IPA; misaki's E2M table then rewrites it into the
 * character-level inventory this model uses. Both steps are reproduced from
 * the project's own JavaScript and Python front ends so the three agree
 * symbol for symbol.
 *
 * eSpeak-ng is opened at runtime and never linked, matching how inflect_v2
 * treats it: it is GPL-3.0 and must not be embedded in this project.
 */
class SanoTtsFrontend {
public:
    SanoTtsFrontend(
        std::filesystem::path espeak_library_path,
        std::filesystem::path espeak_data_path,
        int64_t max_tokens);
    ~SanoTtsFrontend();

    [[nodiscard]] SanoTtsEncoded encode(const std::string & text) const;

    /** Long-form splitting on sentence punctuation, then a codepoint budget. */
    [[nodiscard]] static std::vector<std::string> split_text(
        const std::string & text,
        int64_t max_codepoints);

    /** Pause inserted between chunks, longer after a sentence end. */
    [[nodiscard]] static double boundary_pause_seconds(const std::string & chunk);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t max_tokens_;
};

}  // namespace engine::models::sanotts
