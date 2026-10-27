#pragma once

// LFM3-Audio weight storage, chosen by the lfm3_audio.weight_type and
// lfm3_audio.cpu_repack session options.

#include "engine/framework/assets/tensor_source.h"

#include <ggml-backend.h>
#include <ggml.h>

namespace engine::community_models::lfm3_audio {

enum class Lfm3WeightType { Native, F16, F32 };

struct Lfm3WeightOptions {
    Lfm3WeightType type = Lfm3WeightType::Native;
    bool cpu_repack = true;
};

// Storage type for matmul/lookup weights on `backend`: Native keeps the stored type unless it is BF16 and the
// backend lacks BF16 mul_mat/get_rows, then F16.
// F32 also selects F32 KV caches and explicit (non-flash) attention: the "reference" mode used by parity tests.
assets::TensorStorageType lfm3_storage_type(ggml_backend_t backend, ggml_type stored, const Lfm3WeightOptions & options);
bool lfm3_reference_precision(const Lfm3WeightOptions & options);
// Whether `backend` can get_rows from a table of `type` (CUDA has no K-quant get_rows).
bool lfm3_backend_gathers(ggml_backend_t backend, ggml_type type);

}  // namespace engine::community_models::lfm3_audio
