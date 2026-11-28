#pragma once

// LFM3-Audio weight storage, chosen by the lfm3_audio.weight_type and
// lfm3_audio.cpu_repack session options.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {

enum class Lfm3WeightType { Native, F16, F32 };

struct Lfm3WeightOptions {
    Lfm3WeightType type = Lfm3WeightType::Native;
    bool cpu_repack = true;
};

// Model placement/precision policy. Framework stores own allocation and upload;
// extra CPU buffers hold only projection weights supported by their kernels.
class Lfm3Weights {
public:
    Lfm3Weights(core::ExecutionContext & execution, std::string name, size_t context_bytes, Lfm3WeightOptions options);
    core::BackendWeightStore & plain() noexcept { return plain_; }
    core::TensorValue load_tensor(const assets::TensorSource & source, const std::string & name,
                                 assets::TensorStorageType storage, const std::vector<int64_t> & shape);
    core::TensorValue load_f32_tensor(const assets::TensorSource & source, const std::string & name,
                                     const std::vector<int64_t> & shape);
    ggml_backend_buffer_type_t matmul_buffer_type(ggml_type type, int64_t rows, int64_t cols) const;
    void upload();
private:
    struct Extra {
        ggml_backend_buffer_type_t type;
        std::unique_ptr<core::BackendWeightStore> store;
    };
    Lfm3WeightOptions options_;
    core::ExecutionContext & execution_;
    std::string name_;
    size_t context_bytes_;
    core::BackendWeightStore plain_;
    std::vector<ggml_backend_buffer_type_t> extra_buffer_types_;
    std::vector<Extra> extra_;
};

// Storage type for matmul/lookup weights on `backend`: Native keeps the stored type unless it is BF16 and the
// backend lacks BF16 mul_mat/get_rows, then F16.
// F32 also selects F32 KV caches and explicit (non-flash) attention: the "reference" mode used by parity tests.
assets::TensorStorageType lfm3_storage_type(ggml_backend_t backend, ggml_type stored, const Lfm3WeightOptions & options);
bool lfm3_reference_precision(const Lfm3WeightOptions & options);
// Whether `backend` can get_rows from a table of `type` (CUDA has no K-quant get_rows).
bool lfm3_backend_gathers(ggml_backend_t backend, ggml_type type);

inline constexpr ggml_prec kLfm3MatmulPrecision = GGML_PREC_F32;
inline constexpr int64_t kLfm3KvAlignment = 256;
modules::LfmDecoderConfig lfm3_decoder_config(modules::LfmDecoderConfig config, const Lfm3WeightOptions & options);

}  // namespace engine::community_models::lfm3_audio
