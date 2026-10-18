#pragma once

#include "engine/models/niagara_asr/assets.h"
#include "engine/models/niagara_asr/frontend.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::niagara_asr {

struct NiagaraInferenceResult {
    std::vector<float> logits;
    int64_t frames = 0;
    int64_t vocab_size = 0;
    double elapsed_ms = 0.0;
};

class NiagaraRuntime {
public:
    NiagaraRuntime(
        std::shared_ptr<const NiagaraAsrAssets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_storage_type,
        size_t graph_arena_bytes,
        size_t weight_context_bytes);
    ~NiagaraRuntime();

    NiagaraInferenceResult infer(const NiagaraFeatures & features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::niagara_asr
