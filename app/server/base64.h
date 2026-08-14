#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace minitts::server {

std::string base64_encode(const uint8_t * data, size_t size);
std::string base64_encode(const std::vector<uint8_t> & bytes);
std::string base64_encode(const std::vector<std::byte> & bytes);

// Decodes a base64 payload. Accepts an optional "data:<mime>;base64," prefix so
// clients can send data URIs verbatim. Throws std::runtime_error on malformed input.
std::vector<uint8_t> base64_decode(std::string_view input);

}  // namespace minitts::server
