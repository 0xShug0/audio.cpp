#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::piper_tts {

struct PiperTtsEncoded {
    std::vector<int32_t> token_ids;
};

class PiperTtsFrontend {
public:
    PiperTtsFrontend(
        std::filesystem::path espeak_library_path,
        std::filesystem::path espeak_data_path,
        std::string espeak_voice,
        std::unordered_map<std::string, int32_t> phoneme_id_map,
        int64_t max_tokens);
    ~PiperTtsFrontend();

    [[nodiscard]] PiperTtsEncoded encode(const std::string & text) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unordered_map<std::string, int32_t> id_map_;
    int64_t max_tokens_;
};

}  // namespace engine::models::piper_tts
