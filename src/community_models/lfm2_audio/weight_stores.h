#pragma once

// The weights of one LFM2-Audio component. With lfm2_audio.cpu_repack on the
// CPU, each matmul weight goes into a BackendWeightStore of its own for the
// first of ggml's CPU extra buffer types (core::cpu_extra_buffer_types) whose
// kernels take its type and shape, as llama.cpp places its weights. Every
// other weight, and every weight on other backends, goes into the plain store.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {

class WeightStores {
public:
    WeightStores(core::ExecutionContext & execution, std::string name, size_t context_bytes, bool cpu_repack);

    WeightStores(const WeightStores &) = delete;
    WeightStores & operator=(const WeightStores &) = delete;

    [[nodiscard]] core::BackendWeightStore & plain() noexcept { return plain_; }

    // A native weight of [rows, cols] that is only ever src0 of
    // LinearModule's 2-D ggml_mul_mat, whose input is a contiguous F32 tensor:
    // never viewed, reshaped, gathered, copied or read back, as an extra
    // buffer lays it out for its own kernels only. Its graphs validate with
    // core::validate_backend_graph_with_cpu_extra_buffers.
    core::TensorValue load_matmul(const assets::TensorSource & source, const std::string & name, std::initializer_list<int64_t> shape);
    core::TensorValue load_matmul(const assets::TensorSource & source, const std::string & name, const std::vector<int64_t> & shape);

    // The extra buffer type a matmul weight of `type` and [rows, cols] goes
    // into, or nullptr for the plain store.
    [[nodiscard]] ggml_backend_buffer_type_t matmul_buffer_type(ggml_type type, int64_t rows, int64_t cols) const;

    void upload();

    // The extra buffer types in use ("CPU_REPACK", ...), in the order of
    // their first weights, and how many weights each holds.
    [[nodiscard]] std::vector<std::pair<std::string, size_t>> extra_buffers() const;

private:
    struct Extra {
        ggml_backend_buffer_type_t buffer_type = nullptr;
        std::unique_ptr<core::BackendWeightStore> store;
        size_t weights = 0;
    };

    core::ExecutionContext & execution_;
    std::string name_;
    size_t context_bytes_;
    core::BackendWeightStore plain_;
    std::vector<ggml_backend_buffer_type_t> extra_buffer_types_;
    std::vector<Extra> extra_;
};

}  // namespace engine::community_models::lfm2_audio
