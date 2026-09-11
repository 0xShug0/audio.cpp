#include "ui_assets.h"

#include "audiocpp_demo_voice_assets.h"
#include "audiocpp_ui_asset.h"

namespace minitts::server {

namespace {

template <std::size_t Size>
std::string_view byte_view(const unsigned char (&bytes)[Size]) noexcept {
    return {reinterpret_cast<const char *>(bytes), Size};
}

}  // namespace

std::string_view embedded_ui_html() noexcept {
    return {
        reinterpret_cast<const char *>(kAudioCppUiHtml),
        kAudioCppUiHtmlSize,
    };
}

const std::array<EmbeddedDemoVoice, 4> & embedded_demo_voices() noexcept {
    static const std::array<EmbeddedDemoVoice, 4> voices{{
        {"demo_1_man", byte_view(kAudioCppDemo1ManWav)},
        {"demo_2_man", byte_view(kAudioCppDemo2ManWav)},
        {"demo_3_woman", byte_view(kAudioCppDemo3WomanWav)},
        {"demo_4_woman", byte_view(kAudioCppDemo4WomanWav)},
    }};
    return voices;
}

std::string_view embedded_demo_voice_prompt_text() noexcept {
    return byte_view(kAudioCppDemoVoicePromptText);
}

}  // namespace minitts::server
