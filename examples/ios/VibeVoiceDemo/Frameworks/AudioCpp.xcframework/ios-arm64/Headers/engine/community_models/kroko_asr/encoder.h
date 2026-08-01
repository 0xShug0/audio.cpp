#pragma once

#include "engine/community_models/kroko_asr/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::kroko_asr {

struct KrokoSubsampledChunk {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

// Native implementation of the exported streaming Conv2dSubsampling
// boundary.  Keeping this model-specific avoids changing the shared
// ZipEnhancer implementation while the complete Zipformer2 runtime is
// validated.
class KrokoEncoderRuntime {
public:
    KrokoEncoderRuntime(
        std::shared_ptr<const KrokoASRAssets> assets,
        core::ExecutionContext & execution_context);
    ~KrokoEncoderRuntime();

    KrokoEncoderRuntime(const KrokoEncoderRuntime &) = delete;
    KrokoEncoderRuntime & operator=(const KrokoEncoderRuntime &) = delete;

    KrokoSubsampledChunk encode_subsampled_chunk(
        const std::vector<float> & features) const;
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kroko_asr
