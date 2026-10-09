#include "../../app/common/base64.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using minitts::app::base64_decode;
using minitts::app::base64_decode_bytes;
using minitts::app::base64_encode;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// True when both decoders turn the input away.
bool decode_throws(const std::string & input) {
    bool as_uint8 = false;
    bool as_byte = false;
    try {
        (void)base64_decode(input);
    } catch (const std::runtime_error &) {
        as_uint8 = true;
    }
    try {
        (void)base64_decode_bytes(input);
    } catch (const std::runtime_error &) {
        as_byte = true;
    }
    return as_uint8 && as_byte;
}

bool same_bytes(const std::vector<std::byte> & bytes, const std::vector<uint8_t> & expected) {
    return std::equal(bytes.begin(), bytes.end(), expected.begin(), expected.end(),
                      [](std::byte a, uint8_t b) { return std::to_integer<uint8_t>(a) == b; });
}

void test_roundtrip() {
    for (size_t size = 0; size <= 64; ++size) {
        std::vector<uint8_t> bytes(size);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<uint8_t>(i * 37 + size);
        }
        const auto encoded = base64_encode(bytes);
        require(base64_decode(encoded) == bytes, "roundtrip size " + std::to_string(size));
        require(same_bytes(base64_decode_bytes(encoded), bytes), "std::byte roundtrip size " + std::to_string(size));
    }
}

void test_known_vectors() {
    const auto hello = base64_decode("aGVsbG8=");
    require(std::string(hello.begin(), hello.end()) == "hello", "known vector hello");
    require(base64_decode("").empty(), "empty input decodes to empty");
    require(base64_decode("Zg==").size() == 1, "single byte with double padding");
    require(base64_decode("Zm8=").size() == 2, "two bytes with single padding");
}

void test_whitespace_and_data_uri() {
    const auto spaced = base64_decode("aGVs\nbG8=");
    require(std::string(spaced.begin(), spaced.end()) == "hello", "whitespace tolerated");
    const auto uri = base64_decode("data:audio/wav;base64,aGVsbG8=");
    require(std::string(uri.begin(), uri.end()) == "hello", "data URI prefix stripped");
    const auto charset_uri = base64_decode("data:audio/wav;charset=utf-8;base64,aGVsbG8=");
    require(std::string(charset_uri.begin(), charset_uri.end()) == "hello", "data URI with extra params");
    require(same_bytes(base64_decode_bytes("data:audio/wav;base64,aGVs\nbG8="), uri), "std::byte data URI");
}

void test_malformed_inputs() {
    require(decode_throws("aGVsbG8*"), "invalid character rejected");
    require(decode_throws("aGVsbG8=== "), "excess padding rejected");
    require(decode_throws("aG==bG8="), "data after padding rejected");
    require(decode_throws("a"), "single leftover character rejected");
    require(decode_throws("data:audio/wav,aGVsbG8="), "data URI without base64 marker rejected");
    require(decode_throws("data:audio/wav;base64"), "data URI without payload rejected");
}

}  // namespace

int main() {
    test_roundtrip();
    test_known_vectors();
    test_whitespace_and_data_uri();
    test_malformed_inputs();
    std::cout << "base64_test passed\n";
    return 0;
}
