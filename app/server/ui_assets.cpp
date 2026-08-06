#include "ui_assets.h"

#include "audiocpp_ui_asset.h"

namespace minitts::server {

std::string_view embedded_ui_html() noexcept {
    return {
        reinterpret_cast<const char *>(kAudioCppUiHtml),
        kAudioCppUiHtmlSize,
    };
}

}  // namespace minitts::server
