#pragma once

#include <cstddef>
#include <cstring>
#include <optional>
#include <vector>

namespace engine::models::fish_drama3 {

struct PrefillState {
    std::vector<std::byte> backbone;
    std::vector<std::byte> tower;
    std::vector<float> hidden;
};

inline std::vector<std::byte> pack_prefill(const std::vector<std::byte> & backbone,
    const std::vector<std::byte> & tower, const std::vector<float> & hidden) {
    std::vector<std::byte> payload;
    payload.reserve(backbone.size() + tower.size() + hidden.size() * sizeof(float));
    payload.insert(payload.end(), backbone.begin(), backbone.end());
    payload.insert(payload.end(), tower.begin(), tower.end());
    const auto * bytes = reinterpret_cast<const std::byte *>(hidden.data());
    if (!hidden.empty()) payload.insert(payload.end(), bytes, bytes + hidden.size() * sizeof(float));
    return payload;
}

inline std::optional<PrefillState> unpack_prefill(const std::vector<std::byte> & payload,
    size_t backbone_bytes, size_t tower_bytes, size_t hidden_count) {
    const auto hidden_bytes = hidden_count * sizeof(float);
    if (payload.size() != backbone_bytes + tower_bytes + hidden_bytes) return std::nullopt;
    const auto tower = payload.begin() + backbone_bytes;
    const auto hidden = tower + tower_bytes;
    PrefillState state{{payload.begin(), tower}, {tower, hidden}, std::vector<float>(hidden_count)};
    if (hidden_bytes) std::memcpy(state.hidden.data(), payload.data() + backbone_bytes + tower_bytes, hidden_bytes);
    return state;
}

}  // namespace engine::models::fish_drama3
