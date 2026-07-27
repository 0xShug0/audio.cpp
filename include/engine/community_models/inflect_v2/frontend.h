#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::inflect_v2 {

struct InflectV2FrontendOutput {
    std::string normalized_text;
    std::string phoneme_text;
    std::vector<int32_t> token_ids;
};

class InflectV2Frontend {
public:
    InflectV2Frontend(
        std::filesystem::path espeak_library_path = {},
        std::filesystem::path espeak_data_path = {});
    ~InflectV2Frontend();

    InflectV2Frontend(InflectV2Frontend &&) noexcept;
    InflectV2Frontend & operator=(InflectV2Frontend &&) noexcept;
    InflectV2Frontend(const InflectV2Frontend &) = delete;
    InflectV2Frontend & operator=(const InflectV2Frontend &) = delete;

    InflectV2FrontendOutput encode(const std::string & text) const;
    static std::string normalize(const std::string & text);
    static std::vector<int32_t> tokens_from_phonemes(
        const std::string & phoneme_text);
    static std::vector<std::string> split_text(
        const std::string & text,
        int64_t limit = 280);
    static double boundary_pause_seconds(const std::string & chunk);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::inflect_v2
