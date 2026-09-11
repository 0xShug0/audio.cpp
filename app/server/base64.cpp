#include "base64.h"

#include <stdexcept>

namespace minitts::server {

namespace {

constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int decode_char(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

}  // namespace

std::string base64_encode(const uint8_t * data, size_t size) {
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t i = 0; i < size; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < size ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < size ? data[i + 2] : 0;
        const uint32_t chunk = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(chunk >> 18) & 0x3f]);
        out.push_back(kAlphabet[(chunk >> 12) & 0x3f]);
        out.push_back(i + 1 < size ? kAlphabet[(chunk >> 6) & 0x3f] : '=');
        out.push_back(i + 2 < size ? kAlphabet[chunk & 0x3f] : '=');
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t> & bytes) {
    return base64_encode(bytes.data(), bytes.size());
}

std::string base64_encode(const std::vector<std::byte> & bytes) {
    return base64_encode(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
}

std::vector<uint8_t> base64_decode(std::string_view input) {
    if (input.rfind("data:", 0) == 0) {
        const auto comma = input.find(',');
        if (comma == std::string_view::npos || input.find(";base64") == std::string_view::npos) {
            throw std::runtime_error("malformed base64 data URI");
        }
        input = input.substr(comma + 1);
    }

    std::vector<uint8_t> out;
    out.reserve((input.size() / 4) * 3);
    uint32_t buffer = 0;
    int bits = 0;
    int padding = 0;
    for (const char c : input) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            continue;
        }
        if (c == '=') {
            ++padding;
            if (padding > 2) {
                throw std::runtime_error("malformed base64 payload: excess padding");
            }
            continue;
        }
        if (padding > 0) {
            throw std::runtime_error("malformed base64 payload: data after padding");
        }
        const int value = decode_char(c);
        if (value < 0) {
            throw std::runtime_error("malformed base64 payload: invalid character");
        }
        buffer = (buffer << 6) | static_cast<uint32_t>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xff));
        }
    }
    if (bits >= 6) {
        throw std::runtime_error("malformed base64 payload: truncated quartet");
    }
    return out;
}

}  // namespace minitts::server
