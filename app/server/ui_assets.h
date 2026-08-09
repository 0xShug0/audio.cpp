#pragma once

#include <array>
#include <string_view>

namespace minitts::server {

struct EmbeddedDemoVoice {
    std::string_view name;
    std::string_view wav_bytes;
};

std::string_view embedded_ui_html() noexcept;
const std::array<EmbeddedDemoVoice, 4> & embedded_demo_voices() noexcept;
std::string_view embedded_demo_voice_prompt_text() noexcept;

}  // namespace minitts::server
