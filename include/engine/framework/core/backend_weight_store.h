#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/trace.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::core {

// How a loaded weight is used, which decides the buffers it may go into.
enum class WeightUse {
    Any,  // any op, views and readback
    // Only ever src0 of a 2-D ggml_mul_mat whose src1 is a contiguous F32
    // tensor (as LinearModule's input is): never viewed, reshaped, gathered,
    // copied or read back. Such a weight may go into a CPU extra buffer
    // (BackendWeightStoreOptions::cpu_extra_buffers), whose layout only that
    // buffer type's own matmul kernels read.
    MatMulOnly,
};

struct BackendWeightStoreOptions {
    // The buffer type for every weight; nullptr is the backend's default.
    ggml_backend_buffer_type_t buffer_type = nullptr;
    // On the CPU backend, put each MatMulOnly weight into the first of ggml's
    // CPU extra buffer types (cpu_extra_buffer_types) that runs its matmul, as
    // llama.cpp does. Ignored with a buffer_type.
    bool cpu_extra_buffers = false;
};

class BackendWeightStore {
public:
    struct BufferInfo {
        std::string buffer_type;
        size_t bytes = 0;
        size_t tensors = 0;
    };

    BackendWeightStore(
        ggml_backend_t backend,
        BackendType backend_type,
        std::string name,
        size_t context_bytes,
        ggml_backend_buffer_type_t buffer_type = nullptr)
        : BackendWeightStore(backend, backend_type, std::move(name), context_bytes, BackendWeightStoreOptions{buffer_type, false}) {}

    BackendWeightStore(
        ggml_backend_t backend,
        BackendType backend_type,
        std::string name,
        size_t context_bytes,
        const BackendWeightStoreOptions & options)
        : backend_(backend),
          backend_type_(backend_type),
          name_(std::move(name)),
          buffer_type_(options.buffer_type),
          context_bytes_(context_bytes) {
        if (backend_ == nullptr) {
            throw std::runtime_error(name_ + " backend is not initialized");
        }
        if (context_bytes == 0) {
            throw std::runtime_error(name_ + " context bytes must be non-zero");
        }
        ggml_init_params params{context_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize " + name_ + " weight context");
        }
        if (options.cpu_extra_buffers && buffer_type_ == nullptr) {
            extra_buffer_types_ = cpu_extra_buffer_types(backend_);
        }
    }

    BackendWeightStore(const BackendWeightStore &) = delete;
    BackendWeightStore & operator=(const BackendWeightStore &) = delete;

    ~BackendWeightStore() {
        for (auto & extra : extra_buffers_) {
            if (extra.buffer != nullptr) {
                ggml_backend_buffer_free(extra.buffer);
            }
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    TensorValue load_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        std::initializer_list<int64_t> expected_shape,
        WeightUse use = WeightUse::Any) {
        return load_tensor(source, tensor_name, storage_type, std::vector<int64_t>(expected_shape), use);
    }

    TensorValue load_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape,
        WeightUse use = WeightUse::Any) {
        const auto shape = shape_from_dims(expected_shape);
        const auto resolved_type = type_for_storable_loaded_tensor(source, tensor_name, storage_type, shape);
        const ggml_type type = assets::ggml_type_for_tensor_storage(resolved_type);
        auto * extra_buffer_type = use == WeightUse::MatMulOnly ? matmul_buffer_type(type, shape) : nullptr;
        auto value = make_backend_tensor(shape, type, extra_buffer_type);
        if (extra_buffer_type != nullptr) {
            // Named for ggml's log of the weights it repacks, and for errors.
            ggml_set_name(value.tensor, std::string(tensor_name).c_str());
        }
        PendingUpload upload;
        upload.tensor = value.tensor;
        upload.source = &source;
        upload.name = std::string(tensor_name);
        upload.storage_type = resolved_type;
        upload.expected_shape = std::vector<int64_t>(expected_shape);
        upload.kind = PendingUploadKind::Tensor;
        pending_.push_back(std::move(upload));
        return value;
    }

    TensorValue load_tensor_as_shape(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & expected_source_shape,
        const TensorShape & tensor_shape) {
        const auto resolved_type = type_for_storable_loaded_tensor(source, tensor_name, storage_type, tensor_shape);
        auto value = make_backend_tensor(tensor_shape, assets::ggml_type_for_tensor_storage(resolved_type));
        const auto data = require_tensor_as_shape(source, tensor_name, resolved_type, expected_source_shape, tensor_shape);
        append_upload(value.tensor, data.bytes);
        return value;
    }

    TensorValue load_f32_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        std::initializer_list<int64_t> expected_shape) {
        return load_f32_tensor(source, tensor_name, std::vector<int64_t>(expected_shape));
    }

    TensorValue load_f32_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        const std::vector<int64_t> & expected_shape) {
        const auto shape = shape_from_dims(expected_shape);
        auto value = make_backend_tensor(shape, GGML_TYPE_F32);
        PendingUpload upload;
        upload.tensor = value.tensor;
        upload.source = &source;
        upload.name = std::string(tensor_name);
        upload.storage_type = assets::TensorStorageType::F32;
        upload.expected_shape = std::vector<int64_t>(expected_shape);
        upload.kind = PendingUploadKind::F32;
        pending_.push_back(std::move(upload));
        return value;
    }

    TensorValue make_tensor(
        const TensorShape & shape,
        ggml_type type,
        const void * data,
        size_t bytes) {
        auto value = make_backend_tensor(shape, type);
        std::vector<std::byte> owned(bytes);
        std::memcpy(owned.data(), data, bytes);
        append_upload(value.tensor, std::move(owned));
        return value;
    }

    TensorValue make_f32(const TensorShape & shape, std::vector<float> values) {
        auto value = make_backend_tensor(shape, GGML_TYPE_F32);
        if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
            throw std::runtime_error(name_ + " F32 tensor value count does not match shape");
        }
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        append_upload(value.tensor, std::move(bytes));
        return value;
    }

    TensorValue make_from_f32(
        const TensorShape & shape,
        assets::TensorStorageType storage_type,
        std::vector<float> values) {
        const ggml_type type = type_for_storable_derived_tensor(shape, storage_type);
        auto value = make_backend_tensor(shape, type);
        if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
            throw std::runtime_error(name_ + " derived tensor value count does not match shape");
        }
        append_upload(value.tensor, values_to_bytes(shape, type, values));
        return value;
    }

    // The buffer type a MatMulOnly weight of this type and shape goes into:
    // the first CPU extra buffer type that runs its matmul, or nullptr for
    // the store's own buffer.
    ggml_backend_buffer_type_t matmul_buffer_type(ggml_type type, const TensorShape & shape) const {
        if (shape.rank != 2) {
            return nullptr;
        }
        for (auto * extra_buffer_type : extra_buffer_types_) {
            if (buffer_type_supports_matmul_weight(backend_, extra_buffer_type, type, shape.dims[1], shape.dims[0])) {
                return extra_buffer_type;
            }
        }
        return nullptr;
    }

    void upload() {
        if (uploaded()) {
            throw std::runtime_error(name_ + " weights were already uploaded");
        }
        // Skipped only when every weight went into an extra buffer, so an
        // empty store still fails here.
        if (extra_buffers_.empty() || ggml_get_first_tensor(ctx_.get()) != nullptr) {
            buffer_ = buffer_type_ != nullptr
                ? ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), buffer_type_)
                : ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate " + name_ + " backend weight buffer");
            }
            ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // The header context is host RAM by construction; the buffer is where
            // the weights actually landed (device, or a host fallback), by name.
            debug::timing_log_context_reservation(name_, ctx_.get());
            debug::timing_log_scalar(
                name_ + ".buffer_mb",
                static_cast<double>(ggml_backend_buffer_get_size(buffer_)) / (1024.0 * 1024.0));
            debug::timing_log_scalar(name_ + ".buffer_name", std::string_view(ggml_backend_buffer_name(buffer_)));
        }
        for (auto & extra : extra_buffers_) {
            const std::string type_name = ggml_backend_buft_name(extra.buffer_type);
            extra.buffer = ggml_backend_alloc_ctx_tensors_from_buft(extra.ctx.get(), extra.buffer_type);
            if (extra.buffer == nullptr) {
                throw std::runtime_error("failed to allocate " + name_ + " " + type_name + " weight buffer");
            }
            ggml_backend_buffer_set_usage(extra.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            debug::timing_log_scalar(
                name_ + "." + type_name + ".buffer_mb",
                static_cast<double>(ggml_backend_buffer_get_size(extra.buffer)) / (1024.0 * 1024.0));
        }
        // An extra buffer's set_tensor converts the layout, and takes each
        // weight in one whole write, as these are.
        for (auto & upload : pending_) {
            if (upload.kind == PendingUploadKind::Tensor) {
                upload.source->set_backend_tensor(
                    upload.tensor,
                    upload.name,
                    upload.storage_type,
                    upload.expected_shape);
            } else if (upload.kind == PendingUploadKind::F32) {
                upload.source->set_backend_f32_tensor(upload.tensor, upload.name, upload.expected_shape);
            } else {
                upload_bytes(upload.tensor, upload.bytes);
                upload.bytes.clear();
                upload.bytes.shrink_to_fit();
            }
        }
        pending_.clear();
        pending_.shrink_to_fit();
    }

    // The buffers upload() allocated: the store's own, then one per extra
    // buffer type, in the order their first weights were loaded.
    std::vector<BufferInfo> buffers() const {
        std::vector<BufferInfo> out;
        const auto add = [&](ggml_backend_buffer_t buffer, ggml_context * ctx) {
            if (buffer == nullptr) {
                return;
            }
            BufferInfo info{ggml_backend_buffer_name(buffer), ggml_backend_buffer_get_size(buffer), 0};
            for (auto * tensor = ggml_get_first_tensor(ctx); tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
                ++info.tensors;
            }
            out.push_back(std::move(info));
        };
        add(buffer_, ctx_.get());
        for (const auto & extra : extra_buffers_) {
            add(extra.buffer, extra.ctx.get());
        }
        return out;
    }

private:
    struct GgmlContextDeleter {
        void operator()(ggml_context * ctx) const noexcept {
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
    };

    enum class PendingUploadKind {
        Tensor,
        F32,
        Bytes,
    };

    struct ExtraBuffer {
        ggml_backend_buffer_type_t buffer_type = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_backend_buffer_t buffer = nullptr;
    };

    struct PendingUpload {
        ggml_tensor * tensor = nullptr;
        const assets::TensorSource * source = nullptr;
        std::string name;
        assets::TensorStorageType storage_type = assets::TensorStorageType::F32;
        std::vector<int64_t> expected_shape;
        PendingUploadKind kind = PendingUploadKind::Bytes;
        std::vector<std::byte> bytes;
    };

    static TensorShape shape_from_dims(const std::vector<int64_t> & dims) {
        if (dims.size() == 0 || dims.size() > kMaxTensorRank) {
            throw std::runtime_error("weight tensor rank must be between 1 and 4");
        }
        auto it = dims.begin();
        switch (dims.size()) {
            case 1:
                return TensorShape::from_dims({*it});
            case 2: {
                const int64_t d0 = *it++;
                return TensorShape::from_dims({d0, *it});
            }
            case 3: {
                const int64_t d0 = *it++;
                const int64_t d1 = *it++;
                return TensorShape::from_dims({d0, d1, *it});
            }
            case 4: {
                const int64_t d0 = *it++;
                const int64_t d1 = *it++;
                const int64_t d2 = *it++;
                return TensorShape::from_dims({d0, d1, d2, *it});
            }
        }
        throw std::runtime_error("weight tensor rank is unsupported");
    }

    ggml_type type_for_derived_storage(assets::TensorStorageType storage_type) const {
        if (storage_type == assets::TensorStorageType::Native) {
            return GGML_TYPE_F32;
        }
        return assets::ggml_type_for_tensor_storage(storage_type);
    }

    assets::TensorStorageType type_for_storable_loaded_tensor(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const TensorShape & shape) const {
        const auto resolved_type = backend_safe_loaded_storage_type(source, tensor_name, storage_type);
        const ggml_type type = assets::ggml_type_for_tensor_storage(resolved_type);
        if (!ggml_is_quantized(type)) {
            return resolved_type;
        }
        // A GGUF may store a source matrix in quantized form even when a model
        // reshapes it into a convolution kernel. Quantized GGML rows cannot end
        // in a non-block-sized dimension (for example a 1-wide Conv1D kernel),
        // so decode such native tensors to F32 for the reshaped backend weight.
        if (shape.rank < 2 || shape.last_dim() % ggml_blck_size(type) != 0) {
            return assets::TensorStorageType::F32;
        }
        return resolved_type;
    }

    assets::TensorStorageType backend_safe_loaded_storage_type(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type) const {
        if (storage_type != assets::TensorStorageType::Native) {
            return storage_type;
        }
        const auto native_type = assets::tensor_storage_type_for_dtype(source.require_metadata(tensor_name).dtype);
        if ((backend_type_ == BackendType::Vulkan || backend_type_ == BackendType::Metal) &&
            native_type == assets::TensorStorageType::BF16) {
            return assets::TensorStorageType::F16;
        }
        return native_type;
    }

    ggml_type type_for_storable_derived_tensor(
        const TensorShape & shape,
        assets::TensorStorageType storage_type) const {
        const ggml_type type = type_for_derived_storage(storage_type);
        if (!ggml_is_quantized(type)) {
            return type;
        }
        // GGML quantized tensors are row-block encoded; small conv kernels and 1-D tensors
        // cannot be represented in Q8_0 layout, so derived F32 weights keep exact storage.
        if (shape.rank < 2 || shape.last_dim() % ggml_blck_size(type) != 0) {
            return GGML_TYPE_F32;
        }
        return type;
    }

    assets::TensorData require_tensor_as_shape(
        const assets::TensorSource & source,
        std::string_view tensor_name,
        assets::TensorStorageType storage_type,
        const std::vector<int64_t> & source_shape,
        const TensorShape & tensor_shape) const {
        if (source_shape.empty() || source_shape.size() > kMaxTensorRank) {
            throw std::runtime_error(name_ + " source tensor rank must be between 1 and 4");
        }
        if (tensor_shape.rank == 1) {
            if (source_shape.size() == 1) {
                return source.require_tensor_as_shape(
                    tensor_name, storage_type, {source_shape[0]}, {tensor_shape.dims[0]});
            }
        } else if (tensor_shape.rank == 2) {
            if (source_shape.size() == 2) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1]},
                    {tensor_shape.dims[0], tensor_shape.dims[1]});
            }
            if (source_shape.size() == 3) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2]},
                    {tensor_shape.dims[0], tensor_shape.dims[1]});
            }
        } else if (tensor_shape.rank == 3) {
            if (source_shape.size() == 2) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2]});
            }
            if (source_shape.size() == 3) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2]});
            }
        } else if (tensor_shape.rank == 4) {
            if (source_shape.size() == 4) {
                return source.require_tensor_as_shape(
                    tensor_name,
                    storage_type,
                    {source_shape[0], source_shape[1], source_shape[2], source_shape[3]},
                    {tensor_shape.dims[0], tensor_shape.dims[1], tensor_shape.dims[2], tensor_shape.dims[3]});
            }
        }
        throw std::runtime_error(name_ + " unsupported tensor reshape rank");
    }

    std::vector<std::byte> values_to_bytes(
        const TensorShape & shape,
        ggml_type type,
        const std::vector<float> & values) const {
        if (type == GGML_TYPE_F32) {
            std::vector<std::byte> bytes(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        }
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> converted(values.size());
            ggml_fp32_to_fp16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
            std::vector<std::byte> bytes(converted.size() * sizeof(ggml_fp16_t));
            std::memcpy(bytes.data(), converted.data(), bytes.size());
            return bytes;
        }
        if (type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> converted(values.size());
            ggml_fp32_to_bf16_row(values.data(), converted.data(), static_cast<int64_t>(values.size()));
            std::vector<std::byte> bytes(converted.size() * sizeof(ggml_bf16_t));
            std::memcpy(bytes.data(), converted.data(), bytes.size());
            return bytes;
        }
        if (!ggml_is_quantized(type)) {
            throw std::runtime_error(name_ + " unsupported derived tensor storage type");
        }
        if (ggml_quantize_requires_imatrix(type)) {
            throw std::runtime_error(name_ + " derived tensor storage type requires an importance matrix");
        }
        if (shape.rank < 2) {
            throw std::runtime_error(name_ + " cannot quantize rank-1 derived tensor");
        }
        const int64_t elements_per_row = shape.last_dim();
        if (elements_per_row % ggml_blck_size(type) != 0) {
            throw std::runtime_error(name_ + " derived tensor row size is not divisible by quant block size");
        }
        const int64_t rows = shape.prefix_elements();
        std::vector<std::byte> bytes(static_cast<size_t>(rows) * ggml_row_size(type, elements_per_row));
        const size_t written = ggml_quantize_chunk(
            type,
            values.data(),
            bytes.data(),
            0,
            rows,
            elements_per_row,
            nullptr);
        if (written != bytes.size()) {
            throw std::runtime_error(name_ + " derived quantized tensor byte size mismatch");
        }
        return bytes;
    }

    bool uploaded() const {
        if (buffer_ != nullptr) {
            return true;
        }
        for (const auto & extra : extra_buffers_) {
            if (extra.buffer != nullptr) {
                return true;
            }
        }
        return false;
    }

    // The store's own context, or the one for an extra buffer type, made on
    // its first weight.
    ggml_context * context_for(ggml_backend_buffer_type_t extra_buffer_type) {
        if (extra_buffer_type == nullptr) {
            return ctx_.get();
        }
        for (auto & extra : extra_buffers_) {
            if (extra.buffer_type == extra_buffer_type) {
                return extra.ctx.get();
            }
        }
        ExtraBuffer extra;
        extra.buffer_type = extra_buffer_type;
        extra.ctx.reset(ggml_init({context_bytes_, nullptr, true}));
        if (extra.ctx == nullptr) {
            throw std::runtime_error("failed to initialize " + name_ + " " + ggml_backend_buft_name(extra_buffer_type) + " weight context");
        }
        extra_buffers_.push_back(std::move(extra));
        return extra_buffers_.back().ctx.get();
    }

    TensorValue make_backend_tensor(const TensorShape & shape, ggml_type type, ggml_backend_buffer_type_t extra_buffer_type = nullptr) {
        if (uploaded()) {
            throw std::runtime_error(name_ + " cannot add weights after upload");
        }
        const auto dims = to_ggml_dims(shape);
        auto * tensor = ggml_new_tensor(context_for(extra_buffer_type), type, static_cast<int>(shape.rank), dims.data());
        if (tensor == nullptr) {
            throw std::runtime_error("failed to create " + name_ + " weight tensor");
        }
        return wrap_tensor(tensor, shape, type);
    }

    void append_upload(ggml_tensor * tensor, std::vector<std::byte> bytes) {
        if (tensor == nullptr) {
            throw std::runtime_error(name_ + " cannot upload a null tensor");
        }
        if (bytes.size() != ggml_nbytes(tensor)) {
            throw std::runtime_error(name_ + " tensor byte size mismatch");
        }
        PendingUpload upload;
        upload.tensor = tensor;
        upload.kind = PendingUploadKind::Bytes;
        upload.bytes = std::move(bytes);
        pending_.push_back(std::move(upload));
    }

    void upload_bytes(ggml_tensor * tensor, const std::vector<std::byte> & bytes) const {
        if (tensor == nullptr) {
            throw std::runtime_error(name_ + " cannot upload a null tensor");
        }
        if (bytes.size() != ggml_nbytes(tensor)) {
            throw std::runtime_error(name_ + " tensor byte size mismatch");
        }
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    }

    ggml_backend_t backend_ = nullptr;
    BackendType backend_type_ = BackendType::Cpu;
    std::string name_;
    ggml_backend_buffer_type_t buffer_type_ = nullptr;
    size_t context_bytes_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<ggml_backend_buffer_type_t> extra_buffer_types_;
    std::vector<ExtraBuffer> extra_buffers_;
    std::vector<PendingUpload> pending_;
};

}  // namespace engine::core
