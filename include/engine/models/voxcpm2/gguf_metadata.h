#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct gguf_context;

namespace engine::assets {
class TensorSource;
}

namespace engine::models::voxcpm2 {

// Reads GGUF KV metadata (tokenizer.ggml.*, voxcpm_*) directly from the file
// backing a TensorSource. Only meaningful for GGUF sources: for any other
// source type valid() is false and all accessors return nullopt (optional_*)
// or throw (require_*). This keeps VoxCPM schema knowledge out of the
// framework TensorSource interface.
class GgufMetadataReader {
public:
    explicit GgufMetadataReader(const engine::assets::TensorSource & source);
    ~GgufMetadataReader();

    GgufMetadataReader(const GgufMetadataReader &) = delete;
    GgufMetadataReader & operator=(const GgufMetadataReader &) = delete;

    bool valid() const noexcept { return gguf_ != nullptr; }

    [[nodiscard]] std::optional<std::string> optional_string(std::string_view key) const;
    [[nodiscard]] std::optional<uint32_t> optional_u32(std::string_view key) const;
    [[nodiscard]] std::optional<float> optional_f32(std::string_view key) const;
    [[nodiscard]] std::optional<std::vector<std::string>> optional_string_array(std::string_view key) const;
    [[nodiscard]] std::optional<std::vector<int32_t>> optional_i32_array(std::string_view key) const;
    [[nodiscard]] std::optional<std::vector<float>> optional_f32_array(std::string_view key) const;

    [[nodiscard]] std::string require_string(std::string_view key) const;
    [[nodiscard]] uint32_t require_u32(std::string_view key) const;
    [[nodiscard]] std::vector<std::string> require_string_array(std::string_view key) const;
    [[nodiscard]] std::vector<int32_t> require_i32_array(std::string_view key) const;

private:
    struct gguf_context * gguf_ = nullptr;
};

}  // namespace engine::models::voxcpm2