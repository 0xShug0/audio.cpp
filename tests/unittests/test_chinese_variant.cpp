#include "engine/framework/text/chinese_variant.h"
#include "engine/framework/debug/profiler.h"
#include "test_assert.h"

#include <iostream>

int main() {
    engine::debug::configure_logging({true, std::nullopt});
    const auto started = std::chrono::steady_clock::now();
    try {
        using engine::test::require_eq;
        using namespace engine::text;
        const std::string chinese = "傳統中文，東京與電視";
        const std::string simplified = "传统中文，东京与电视";
        for (const std::string language : {"zh", "zh-CN", "ZH_hant_TW", "cmn", "cmn-Hans", "zho", "chi", "Chinese", "Mandarin", "中文"}) {
            require_eq(maybe_convert_traditional_to_simplified(chinese, language), simplified,
                "Mandarin conversion: " + language);
        }
        for (const std::string language : {"yue", "yue-Hant", "zh-yue", "Cantonese", "zh-HK", "zh_MO"}) {
            require_eq(maybe_convert_traditional_to_simplified(chinese, language), chinese,
                "Cantonese preservation: " + language);
        }
        for (const std::string language : {"ja", "ja-JP", "Japanese", "ko", "en"}) {
            for (const std::string text : {"日本語の新聞を読みます。", "電気と水道の料金を払います。", "今日は東京で会議があります。", "東京"}) {
                require_eq(maybe_convert_traditional_to_simplified(text, language), text,
                    "Non-Chinese preservation: " + language);
            }
        }
        for (const std::optional<std::string> & language : {std::optional<std::string>(), std::optional<std::string>(""),
                std::optional<std::string>("auto"), std::optional<std::string>("None")}) {
            for (const std::string text : {"日本語の新聞を読みます。", "東京カメラ", "東京ｶﾒﾗ", "東京ㇰ"}) {
                require_eq(maybe_convert_traditional_to_simplified_opt(text, language), text,
                    "Auto kana preservation");
            }
            require_eq(maybe_convert_traditional_to_simplified_opt(chinese, language), simplified,
                "Auto Mandarin conversion");
            require_eq(maybe_convert_traditional_to_simplified_opt("東京", language), std::string("东京"),
                "Kanji-only auto retains existing conversion; explicit ja is required");
        }
        require_eq(maybe_convert_traditional_to_simplified("東京カメラ", "zh"), std::string("东京カメラ"),
            "Explicit Mandarin takes precedence over kana detection");
        require_eq(convert_traditional_to_simplified("日本語"), std::string("日本语"),
            "Unconditional conversion remains unchanged");
        require_eq(maybe_convert_traditional_to_simplified("Hello world", "auto"), std::string("Hello world"),
            "ASCII passthrough");
        std::cout << "chinese_variant_test passed\n";
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
