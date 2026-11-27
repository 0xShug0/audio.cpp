#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace engine::models::fish_drama3 {
namespace sha256_detail {

constexpr uint32_t kShaK[64] = {
    0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U,
    0x923F82A4U, 0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U,
    0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U,
    0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU,
    0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U,
    0x06CA6351U, 0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U,
    0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, 0xA2BFE8A1U, 0xA81A664BU,
    0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U,
    0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU,
    0x5B9CCA4FU, 0x682E6FF3U, 0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U,
    0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U,
};

inline uint32_t rotr32(uint32_t v, int n) {
    return (v >> n) | (v << (32 - n));
}

inline void sha256_block(uint32_t * h, const unsigned char * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
               (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
               static_cast<uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t hh = h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t1 = hh + s1 + ch + kShaK[i] + w[i];
        const uint32_t t2 = s0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

}  // namespace sha256_detail

class Sha256 {
public:
    void update(std::string_view bytes) {
        total_ += bytes.size();
        while (!bytes.empty()) {
            if (used_ == 0 && bytes.size() >= block_.size()) {
                sha256_detail::sha256_block(state_.data(), reinterpret_cast<const unsigned char *>(bytes.data()));
                bytes.remove_prefix(block_.size());
                continue;
            }
            const size_t count = std::min(block_.size() - used_, bytes.size());
            std::memcpy(block_.data() + used_, bytes.data(), count);
            used_ += count;
            bytes.remove_prefix(count);
            if (used_ == block_.size()) {
                sha256_detail::sha256_block(state_.data(), block_.data());
                used_ = 0;
            }
        }
    }

    std::array<uint32_t, 8> digest() const {
        auto state = state_;
        std::array<unsigned char, 128> tail{};
        std::copy_n(block_.data(), used_, tail.data());
        tail[used_] = 0x80;
        const size_t size = used_ < 56 ? 64 : 128;
        const auto bits = total_ * 8;
        for (unsigned i = 0; i < 8; ++i) tail[size - 1 - i] = static_cast<unsigned char>(bits >> (i * 8));
        sha256_detail::sha256_block(state.data(), tail.data());
        if (size == 128) sha256_detail::sha256_block(state.data(), tail.data() + 64);
        return state;
    }

    std::string hex() const {
        constexpr char digits[] = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (const auto word : digest()) {
            for (int shift = 28; shift >= 0; shift -= 4) result.push_back(digits[(word >> shift) & 15]);
        }
        return result;
    }

private:
    std::array<uint32_t, 8> state_{0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
                                  0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};
    std::array<unsigned char, 64> block_{};
    uint64_t total_ = 0;
    size_t used_ = 0;
};

}  // namespace engine::models::fish_drama3
