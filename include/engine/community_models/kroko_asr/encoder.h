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

// Session-owned Conv2dSubsampling/ConvNeXt graphs. Streaming packages retain
// their convolution history; offline packages use symmetric full-sequence
// convolution and reuse the prepared graph while the input shape is unchanged.
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
