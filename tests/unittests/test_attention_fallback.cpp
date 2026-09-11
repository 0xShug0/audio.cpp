#include "engine/framework/core/attention_fallback.h"

#include "test_assert.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using engine::core::AttentionPreference;
using engine::test::require;
using engine::test::require_eq;

void test_parse_attention_preference() {
    require_eq(
        static_cast<int>(engine::core::parse_attention_preference("auto", "attention")),
        static_cast<int>(AttentionPreference::Auto),
        "parse auto");
    require_eq(
        static_cast<int>(engine::core::parse_attention_preference("flash", "attention")),
        static_cast<int>(AttentionPreference::Flash),
        "parse flash");
    require_eq(
        static_cast<int>(engine::core::parse_attention_preference("eager", "attention")),
        static_cast<int>(AttentionPreference::Eager),
        "parse eager");
    bool threw = false;
    try {
        engine::core::parse_attention_preference("sometimes", "breeze_tts.attention");
    } catch (const std::runtime_error & error) {
        threw = true;
        require(
            std::string(error.what()).find("breeze_tts.attention") != std::string::npos,
            "parse error names the option");
    }
    require(threw, "parse invalid must throw");
}

void test_resolve_flash_attention() {
    require(
        engine::core::resolve_flash_attention(nullptr, 128, AttentionPreference::Flash),
        "explicit flash resolves true");
    require(
        !engine::core::resolve_flash_attention(nullptr, 128, AttentionPreference::Eager),
        "explicit eager resolves false");
    // Null backend preserves historical behavior regardless of head_dim.
    require(
        engine::core::resolve_flash_attention(nullptr, 128, AttentionPreference::Auto),
        "auto with null backend preserves flash");
    require(
        engine::core::resolve_flash_attention(nullptr, -1, AttentionPreference::Auto),
        "auto with bad head_dim preserves flash");
}

}  // namespace

int main() {
    try {
        test_parse_attention_preference();
        test_resolve_flash_attention();
    } catch (const std::exception & error) {
        std::cerr << "attention_fallback_test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "attention_fallback_test passed\n";
    return 0;
}
