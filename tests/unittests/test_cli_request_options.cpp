#include "../../app/cli/request.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"

#include "test_assert.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::string option_from_json_number(const std::string & number) {
    const auto root = engine::io::json::parse("{\"value\":" + number + "}");
    const auto * value = root.find("value");
    engine::test::require(value != nullptr, "json value field parsed");
    return minitts::cli::json_option_string(*value);
}

uint64_t parse_seed(const std::string & value) {
    const std::unordered_map<std::string, std::string> options{{"seed", value}};
    const auto parsed = engine::runtime::parse_u64_option(options, {"seed"});
    if (!parsed.has_value()) {
        throw std::runtime_error("seed option was not parsed");
    }
    return *parsed;
}

void require_seed_rejected(const std::string & value) {
    try {
        (void) parse_seed(value);
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("unsafe seed option was parsed unexpectedly: " + value);
}

void test_large_safe_integer_numbers_stay_decimal() {
    const std::string one_quadrillion = option_from_json_number("1000000000000000");
    engine::test::require_eq(one_quadrillion, "1000000000000000", "1e15 option string");
    engine::test::require_eq(parse_seed(one_quadrillion), uint64_t{1000000000000000}, "1e15 seed");

    const std::string nearby = option_from_json_number("1000000000000001");
    engine::test::require_eq(nearby, "1000000000000001", "1e15+1 option string");
    engine::test::require_eq(parse_seed(nearby), uint64_t{1000000000000001}, "1e15+1 seed");

    const std::string negative = option_from_json_number("-1000000000000000");
    engine::test::require_eq(negative, "-1000000000000000", "negative 1e15 option string");
}

void test_float_numbers_keep_json_formatting() {
    engine::test::require_eq(option_from_json_number("0.7"), "0.7", "float option string");
    engine::test::require_eq(option_from_json_number("1.25"), "1.25", "fractional option string");
}

void test_unsafe_integer_numbers_are_not_silently_rounded() {
    const std::string boundary = option_from_json_number("9007199254740992");
    engine::test::require(
        boundary.find('e') != std::string::npos || boundary.find('E') != std::string::npos,
        "2^53 boundary keeps scientific formatting");
    require_seed_rejected(boundary);

    const std::string rounded = option_from_json_number("9007199254740993");
    engine::test::require(
        rounded.find('e') != std::string::npos || rounded.find('E') != std::string::npos,
        "2^53+1 does not become a rounded decimal option");
    require_seed_rejected(rounded);
}

void test_audio_only_language_is_request_option() {
    const char * argv[] = {"audiocpp_cli", "--task", "asr", "--family", "qwen3_asr", "--language", "en"};
    const auto request = minitts::cli::build_request_from_cli(7, const_cast<char **>(argv));

    engine::test::require(!request.text_input.has_value(), "audio-only language does not synthesize text input");
    engine::test::require_eq(request.options.at("language"), std::string("en"), "audio-only language request option");
}

}  // namespace

int main() {
    test_large_safe_integer_numbers_stay_decimal();
    test_float_numbers_keep_json_formatting();
    test_unsafe_integer_numbers_are_not_silently_rounded();
    test_audio_only_language_is_request_option();
    std::cout << "cli_request_options_test passed\n";
    return 0;
}
