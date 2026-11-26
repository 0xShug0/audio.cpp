#pragma once

#include <ggml-backend.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace engine::models::fish_drama3 {

inline size_t snapshot_size(const std::vector<ggml_tensor *> & tensors) {
    size_t bytes = 0;
    for (const auto * tensor : tensors) bytes += ggml_nbytes(tensor);
    return bytes;
}

inline std::vector<std::byte> capture_snapshot(const std::vector<ggml_tensor *> & tensors) {
    std::vector<std::byte> result(snapshot_size(tensors));
    size_t offset = 0;
    for (const auto * tensor : tensors) {
        const auto bytes = ggml_nbytes(tensor);
        ggml_backend_tensor_get(tensor, result.data() + offset, 0, bytes);
        offset += bytes;
    }
    return result;
}

inline void restore_snapshot(const std::vector<ggml_tensor *> & tensors, const std::vector<std::byte> & snapshot) {
    if (snapshot.size() != snapshot_size(tensors))
        throw std::invalid_argument("Drama 3 prefill snapshot size mismatch");
    size_t offset = 0;
    for (auto * tensor : tensors) {
        const auto bytes = ggml_nbytes(tensor);
        ggml_backend_tensor_set(tensor, snapshot.data() + offset, 0, bytes);
        offset += bytes;
    }
}

} // namespace engine::models::fish_drama3
