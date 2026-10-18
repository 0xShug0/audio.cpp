#pragma once

// Raw Vietnamese/English text → the phonemes the model reads, through sea-g2p.
//
// The model is trained on SEA-G2P phonemes, so `--text` has to be phonemes and
// every caller has had to run the Python package first. sea-g2p is a Rust crate
// with a C ABI (`--features capi`), and this loads it the way the eSpeak front
// end loads espeak-ng: at runtime, by name or by an explicit path, so nothing is
// added to the build and a host without the library keeps working exactly as
// before — it just has to pass phonemes itself.
//
// Re-implementing the rules in C++ was the alternative and is the one to avoid:
// number, date, unit and abbreviation handling would then exist twice, and the
// two copies drift the first time either is corrected.

#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::vieneu_v3_turbo {

class TextFrontend {
public:
    /// Loads `library` (empty: the usual names next to the binary and on the
    /// search path) and opens the `sea_g2p.bin` dictionary. Throws when either
    /// step fails, so a caller that asked for text input hears why.
    TextFrontend(const std::filesystem::path & library, const std::filesystem::path & dictionary);
    ~TextFrontend();

    TextFrontend(const TextFrontend &) = delete;
    TextFrontend & operator=(const TextFrontend &) = delete;

    /// Normalise (numbers, dates, units, abbreviations) then phonemise.
    /// `punctuation_norm` applies sea-g2p's trailing-punctuation rule first.
    std::string phonemize(const std::string & text, bool punctuation_norm = true) const;

    /// Normalisation alone — the length a chunker should measure, because what
    /// matters is the length after "3,5 triệu" has become words.
    std::string normalize(const std::string & text, bool punctuation_norm = false) const;

    /// Where the library and the dictionary came from, for logs and errors.
    const std::filesystem::path & library_path() const noexcept;
    const std::filesystem::path & dictionary_path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::vieneu_v3_turbo
