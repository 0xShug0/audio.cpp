#include "text_chunking.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect_size(const char * name, const std::string & text, int64_t expected) {
    const auto actual = engine::models::moss_tts_nano::detail::default_text_chunk_size(text);
    if (actual != expected) {
        std::fprintf(stderr, "FAIL: %s expected %lld, got %lld\n", name,
                     static_cast<long long>(expected), static_cast<long long>(actual));
        ++failures;
    }
}

}  // namespace

int main() {
    std::string supplementary_cjk_b;
    std::string supplementary_cjk_g;
    std::string bmp_cjk;
    for (int i = 0; i < 60; ++i) {
        supplementary_cjk_b += "\xF0\xA0\x80\x80";  // U+20000, CJK Extension B
        supplementary_cjk_g += "\xF0\xB0\x80\x80";  // U+30000, CJK Extension G
        bmp_cjk += "漢";
    }
    expect_size("supplementary CJK Extension B", supplementary_cjk_b, 50);
    expect_size("supplementary CJK Extension G", supplementary_cjk_g, 50);
    expect_size("BMP CJK remains chunked", bmp_cjk, 50);
    expect_size("Latin text keeps the general default", std::string(60, 'a'), 256);

    bool rejected_invalid_utf8 = false;
    try {
        engine::models::moss_tts_nano::detail::default_text_chunk_size("\xF0\xA0");
    } catch (const std::runtime_error &) {
        rejected_invalid_utf8 = true;
    }
    if (!rejected_invalid_utf8) {
        std::fprintf(stderr, "FAIL: malformed UTF-8 was not rejected\n");
        ++failures;
    }

    std::printf("%s\n", failures == 0 ? "PASS: MOSS-TTS-Nano text chunking" : "FAIL: MOSS-TTS-Nano text chunking");
    return failures == 0 ? 0 : 1;
}
