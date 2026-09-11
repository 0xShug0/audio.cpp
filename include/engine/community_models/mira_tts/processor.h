#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::mira_tts {

class MiraAcousticProcessor final {
public:
    MiraAcousticProcessor(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType linear_storage_type,
        assets::TensorStorageType conv_storage_type);
    ~MiraAcousticProcessor();

    std::vector<float> process(
        const std::vector<int32_t> & speech_codes,
        const std::vector<int32_t> & context_codes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::mira_tts
