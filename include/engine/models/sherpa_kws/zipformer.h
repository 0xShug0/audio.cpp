#pragma once

#include "engine/models/sherpa_kws/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::sherpa_kws {

struct SherpaEncoderChunk {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

class SherpaZipformerRuntime {
public:
    SherpaZipformerRuntime(
        std::shared_ptr<const SherpaKwsAssets> assets,
        core::ExecutionContext & execution_context);
    ~SherpaZipformerRuntime();

    SherpaZipformerRuntime(const SherpaZipformerRuntime &) = delete;
    SherpaZipformerRuntime & operator=(const SherpaZipformerRuntime &) = delete;

    SherpaEncoderChunk encode_chunk(
        const std::vector<float> & subsampled_features) const;
    void reset() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sherpa_kws
