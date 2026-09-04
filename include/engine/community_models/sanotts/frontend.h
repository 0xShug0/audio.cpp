#pragma once

#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::sanotts {

/** Thrown by encode() when a chunk phonemizes past the duration model's
 *  token limit; the session responds by bisecting the chunk. */
struct SanoTtsTooLongError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

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

/**
 * Text -> Piper phoneme ids for the piperlite voices.
 *
 * Reproduces piper's phonemize_espeak / phonemes_to_ids convention through
 * the same eSpeak-ng library: phonemizer-style punctuation preservation,
 * NFD decomposition to single codepoints, then the voice's phoneme_id_map
 * with [BOS, PAD, (id, PAD)..., EOS] framing. Deterministic per voice.
 */
class SanoTtsPiperFrontend {
public:
    SanoTtsPiperFrontend(
        std::filesystem::path espeak_library_path,
        std::filesystem::path espeak_data_path,
        std::string espeak_voice,
        std::unordered_map<std::string, int32_t> phoneme_id_map,
        int64_t max_tokens);
    ~SanoTtsPiperFrontend();

    [[nodiscard]] SanoTtsEncoded encode(const std::string & text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unordered_map<std::string, int32_t> id_map_;
    int64_t max_tokens_;
};

}  // namespace engine::models::sanotts
