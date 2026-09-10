#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace engine::audio {

// Raw eSpeak-ng phonemization only. Normalization, voice fallback policy,
// punctuation restoration and token mapping belong to each model frontend.
class EspeakPhonemizer {
public:
    // data_directory is espeak-ng-data itself, not its parent. Empty paths
    // use the system library/data defaults. Voices are tried in order.
    EspeakPhonemizer(std::filesystem::path library,
                     std::filesystem::path data_directory,
                     std::vector<std::string> voices);
    std::string phonemize(const std::string & text, int phonemes_mode,
                          const std::string & clause_separator = " ") const;
private:
    std::filesystem::path library_;
    std::filesystem::path data_;
    std::vector<std::string> voices_;
};

} // namespace engine::audio
