#pragma once

#include "engine/framework/core/module.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::assets {

enum class TensorStorageType {
    Native,
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
};

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
};

struct TensorData {
    core::TensorShape shape;
    ggml_type type = GGML_TYPE_F32;
    std::vector<std::byte> bytes;
};

struct TensorDataF32 {
    core::TensorShape shape;
    std::vector<float> values;
};

struct RawTensorData {
    TensorMetadata metadata;
    std::vector<std::byte> bytes;
};

class TensorSource {
public:
    virtual ~TensorSource() = default;

    [[nodiscard]] virtual const std::filesystem::path & source_path() const noexcept = 0;
    [[nodiscard]] virtual bool has_tensor(std::string_view name) const noexcept = 0;
    [[nodiscard]] virtual TensorMetadata require_metadata(std::string_view name) const = 0;
    [[nodiscard]] virtual std::vector<TensorMetadata> tensors() const = 0;
    virtual void release_storage() const {}
    [[nodiscard]] virtual RawTensorData require_tensor_data(std::string_view name) const = 0;
    [[nodiscard]] virtual std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const = 0;
    [[nodiscard]] virtual std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape = std::nullopt) const = 0;
    [[nodiscard]] std::vector<float> require_f32(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const {
        return require_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>(expected_shape)));
    }
    [[nodiscard]] std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const {
        return optional_f32(name, std::optional<std::vector<int64_t>>(std::vector<int64_t>(expected_shape)));
    }
    [[nodiscard]] TensorDataF32 require_f32_tensor(std::string_view name) const;
    [[nodiscard]] TensorDataF32 require_f32_tensor(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] TensorData require_tensor(
        std::string_view name,
        TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] TensorData require_tensor(
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const;
    [[nodiscard]] TensorData require_tensor_as_shape(
        std::string_view name,
        TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_source_shape,
        std::initializer_list<int64_t> tensor_shape) const;
    virtual void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const;
    virtual void set_backend_f32_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        const std::vector<int64_t> & expected_shape) const;
    [[nodiscard]] std::optional<TensorDataF32> optional_f32_tensor(std::string_view name) const;
    [[nodiscard]] std::optional<TensorDataF32> optional_f32_tensor(
        std::string_view name,
        std::initializer_list<int64_t> expected_shape) const;
    [[nodiscard]] std::optional<RawTensorData> optional_tensor_data(std::string_view name) const;
    [[nodiscard]] std::optional<std::string> find_tensor_name(
        std::initializer_list<std::string_view> candidates) const;
    [[nodiscard]] std::string require_tensor_name(
        std::initializer_list<std::string_view> candidates) const;
    [[nodiscard]] virtual int64_t require_i64_scalar(std::string_view name) const = 0;
};

[[nodiscard]] TensorStorageType parse_tensor_storage_type(std::string_view value);
[[nodiscard]] ggml_type ggml_type_for_tensor_storage(TensorStorageType storage_type);
[[nodiscard]] TensorStorageType tensor_storage_type_for_dtype(std::string_view dtype);
[[nodiscard]] std::vector<float> tensor_data_to_f32(
    std::string_view name,
    const TensorData & tensor);
[[nodiscard]] TensorStorageType resolve_tensor_storage_type(
    const TensorSource & source,
    std::string_view name,
    TensorStorageType requested_type);
void set_backend_tensor_from_f32_parallel(
    ggml_tensor * tensor,
    std::string_view name,
    const std::vector<float> & values,
    const core::TensorShape & shape,
    ggml_type type);
std::shared_ptr<const TensorSource> open_tensor_source(const std::filesystem::path & path);
std::vector<std::filesystem::path> indexed_tensor_source_shard_paths(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root);
std::shared_ptr<const TensorSource> open_indexed_tensor_source(
    const std::filesystem::path & index_path,
    const std::filesystem::path & model_root);

}  // namespace engine::assets
