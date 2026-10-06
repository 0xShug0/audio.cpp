#pragma once

#include <string_view>

namespace minitts::server {

struct NeutralSpeechOptionRule {
    std::string_view name;
    double value;
};

// Top-level OpenAI speech compatibility exceptions: these neutral values behave
// like an omitted field, even when the model does not support the option.
// Check after value validation and alias normalization, before model forwarding.
inline constexpr NeutralSpeechOptionRule kNeutralSpeechOptions[] = {
    {"speed", 1.0},
};

inline bool is_neutral_speech_option(std::string_view name, double value) {
    for (const auto & rule : kNeutralSpeechOptions) {
        if (rule.name == name && rule.value == value) return true;
    }
    return false;
}

} // namespace minitts::server
