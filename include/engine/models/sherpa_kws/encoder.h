#pragma once

#include "engine/models/sherpa_kws/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::sherpa_kws {

struct SherpaSubsampledChunk {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

// Native implementation of the exported streaming Conv2dSubsampling
// boundary.  Keeping this model-specific avoids changing the shared
// ZipEnhancer implementation while the complete Zipformer2 runtime is
// validated.
class SherpaEncoderRuntime {
public:
    SherpaEncoderRuntime(
        std::shared_ptr<const SherpaKwsAssets> assets,
        core::ExecutionContext & execution_context);
    ~SherpaEncoderRuntime();

    SherpaEncoderRuntime(const SherpaEncoderRuntime &) = delete;
    SherpaEncoderRuntime & operator=(const SherpaEncoderRuntime &) = delete;

    SherpaSubsampledChunk encode_subsampled_chunk(
        const std::vector<float> & features) const;
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sherpa_kws
