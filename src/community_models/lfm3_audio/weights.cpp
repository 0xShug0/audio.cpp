#include "engine/community_models/lfm3_audio/weights.h"

#include <ggml-cpp.h>

#include <stdexcept>

namespace engine::community_models::lfm3_audio {

namespace {

ggml_context_ptr probe_context(size_t tensors) {
    ggml_context_ptr ctx(ggml_init({tensors * ggml_tensor_overhead(), nullptr, true}));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the LFM3-Audio op probe context");
    }
    return ctx;
}

// Probes matmul and get_rows, as embed_tokens is both the embedding table and the tied head.
bool backend_runs_bf16(ggml_backend_t backend) {
    const auto ctx = probe_context(8);
    auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 256, 32);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 4);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    return ggml_backend_supports_op(backend, ggml_mul_mat(ctx.get(), weight, x)) &&
           ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), weight, ids));
}

// ggml's CPU BF16 dot product has SIMD paths for x86, RISC-V and POWER only (ggml-cpu/vec.cpp), so on an arm64
// CPU BF16 matmuls are scalar: about 20x slower than F16 on an M3 Ultra.
bool cpu_bf16_is_scalar(ggml_backend_t backend) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
#else
    (void)backend;
    return false;
#endif
}

}  // namespace

assets::TensorStorageType lfm3_storage_type(ggml_backend_t backend, ggml_type stored, const Lfm3WeightOptions & options) {
    switch (options.type) {
        case Lfm3WeightType::F32:
            return assets::TensorStorageType::F32;
        case Lfm3WeightType::F16:
            return assets::TensorStorageType::F16;
        case Lfm3WeightType::Native:
            break;
    }

    if (stored == GGML_TYPE_BF16 && (cpu_bf16_is_scalar(backend) || !backend_runs_bf16(backend))) {
        return assets::TensorStorageType::F16;
    }
    return assets::TensorStorageType::Native;
}

bool lfm3_reference_precision(const Lfm3WeightOptions & options) {
    return options.type == Lfm3WeightType::F32;
}

bool lfm3_backend_gathers(ggml_backend_t backend, ggml_type type) {
    const auto ctx = probe_context(4);
    auto * table = ggml_new_tensor_2d(ctx.get(), type, ggml_blck_size(type), 1);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    return ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), table, ids));
}

}  // namespace engine::community_models::lfm3_audio
