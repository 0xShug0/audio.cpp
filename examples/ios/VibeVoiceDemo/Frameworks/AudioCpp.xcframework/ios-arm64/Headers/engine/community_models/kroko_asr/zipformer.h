#pragma once

#include "engine/community_models/kroko_asr/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::kroko_asr {

struct KrokoEncoderChunk {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

class KrokoZipformerRuntime {
public:
    KrokoZipformerRuntime(
        std::shared_ptr<const KrokoASRAssets> assets,
        core::ExecutionContext & execution_context);
    ~KrokoZipformerRuntime();

    KrokoZipformerRuntime(const KrokoZipformerRuntime &) = delete;
    KrokoZipformerRuntime & operator=(const KrokoZipformerRuntime &) = delete;

    KrokoEncoderChunk encode_chunk(
        const std::vector<float> & subsampled_features) const;
    void reset() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kroko_asr
