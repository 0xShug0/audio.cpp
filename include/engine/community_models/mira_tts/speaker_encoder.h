#pragma once

#include "engine/community_models/mira_tts/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::mira_tts {

class MiraSpeakerEncoder final {
public:
    MiraSpeakerEncoder(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType linear_storage_type,
        assets::TensorStorageType conv_storage_type);
    ~MiraSpeakerEncoder();

    // Returns the 32 discrete context codes consumed by Mira's prompt and
    // acoustic processor.
    std::vector<int32_t> encode(const runtime::AudioBuffer & reference_audio);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::mira_tts
