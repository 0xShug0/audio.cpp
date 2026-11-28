#pragma once

#include <ggml-cpp.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace engine::assets {

// Checkpoint metadata only: no tensor storage, model configuration, or backend state.
class GgufMetadata {
public:
    explicit GgufMetadata(const std::filesystem::path & path);

    [[nodiscard]] bool has(const char * key) const;
    [[nodiscard]] int64_t tensor_count() const;
    [[nodiscard]] std::string require_string(const char * key) const;
    [[nodiscard]] std::optional<std::string> find_string(const char * key) const;
    [[nodiscard]] bool require_bool(const char * key) const;
    [[nodiscard]] std::optional<bool> find_bool(const char * key) const;
    [[nodiscard]] int64_t require_int(const char * key) const;
    [[nodiscard]] int64_t require_int(const char * key, int64_t min, int64_t max) const;
    [[nodiscard]] float require_f32(const char * key) const;
    [[nodiscard]] double require_number(const char * key) const;
    [[nodiscard]] std::optional<double> find_number(const char * key) const;
    [[nodiscard]] std::vector<int64_t> require_int_array(const char * key, bool allow_bool = false) const;
    [[nodiscard]] std::vector<std::string> require_string_array(const char * key) const;

private:
    [[nodiscard]] int64_t require_key(const char * key) const;
    [[noreturn]] void fail(const char * key, const char * message) const;

    std::filesystem::path path_;
    gguf_context_ptr context_;
};

}  // namespace engine::assets
