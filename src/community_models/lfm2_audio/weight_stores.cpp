#include "weight_stores.h"

#include "engine/framework/core/backend.h"

#include <stdexcept>

namespace engine::community_models::lfm2_audio {

WeightStores::WeightStores(core::ExecutionContext & execution, std::string name, size_t context_bytes, bool cpu_repack)
    : execution_(execution),
      name_(std::move(name)),
      context_bytes_(context_bytes),
      plain_(execution.backend(), execution.backend_type(), name_, context_bytes) {
    if (cpu_repack) {
        extra_buffer_types_ = core::cpu_extra_buffer_types(execution.backend());
    }
}

core::TensorValue WeightStores::load_matmul(
    const assets::TensorSource & source, const std::string & name, std::initializer_list<int64_t> shape) {
    return load_matmul(source, name, std::vector<int64_t>(shape));
}

core::TensorValue WeightStores::load_matmul(const assets::TensorSource & source, const std::string & name, const std::vector<int64_t> & shape) {
    const auto native = assets::TensorStorageType::Native;
    if (extra_buffer_types_.empty() || shape.size() != 2) {
        return plain_.load_tensor(source, name, native, shape);
    }

    const ggml_type type = assets::ggml_type_for_tensor_storage(assets::tensor_storage_type_for_dtype(source.require_metadata(name).dtype));
    ggml_backend_buffer_type_t buffer_type = matmul_buffer_type(type, shape[0], shape[1]);
    if (buffer_type == nullptr) {
        return plain_.load_tensor(source, name, native, shape);
    }

    Extra * extra = nullptr;
    for (auto & candidate : extra_) {
        if (candidate.buffer_type == buffer_type) {
            extra = &candidate;
            break;
        }
    }
    if (extra == nullptr) {
        extra_.push_back({buffer_type,
                          std::make_unique<core::BackendWeightStore>(
                              execution_.backend(), execution_.backend_type(), name_ + "." + ggml_backend_buft_name(buffer_type), context_bytes_,
                              buffer_type),
                          0});
        extra = &extra_.back();
    }

    auto value = extra->store->load_tensor(source, name, native, shape);
    // The probe above took the source's type; the store keeps it on the CPU.
    if (value.tensor->type != type) {
        throw std::logic_error(name_ + " " + name + " changed type on its way into " + ggml_backend_buft_name(buffer_type));
    }
    // Named for ggml's log of the weights it repacks, and for errors.
    ggml_set_name(value.tensor, name.c_str());
    ++extra->weights;
    return value;
}

ggml_backend_buffer_type_t WeightStores::matmul_buffer_type(ggml_type type, int64_t rows, int64_t cols) const {
    for (auto * buffer_type : extra_buffer_types_) {
        if (core::buffer_type_supports_matmul_weight(execution_.backend(), buffer_type, type, cols, rows)) {
            return buffer_type;
        }
    }

    return nullptr;
}

void WeightStores::upload() {
    plain_.upload();
    // An extra buffer's set_tensor converts the layout, and takes each weight
    // in one whole write, as the store's are.
    for (auto & extra : extra_) {
        extra.store->upload();
    }
}

std::vector<std::pair<std::string, size_t>> WeightStores::extra_buffers() const {
    std::vector<std::pair<std::string, size_t>> out;
    for (const auto & extra : extra_) {
        out.emplace_back(ggml_backend_buft_name(extra.buffer_type), extra.weights);
    }

    return out;
}

}  // namespace engine::community_models::lfm2_audio
