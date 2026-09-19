#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/gpt.h"
#include "engine/models/xtts_v2/speaker_encoder.h"

#include <memory>
#include <vector>

namespace engine::models::xtts_v2 {

class XttsV2DecoderRuntime {
public:
    XttsV2DecoderRuntime(const XttsV2Assets & assets, core::ExecutionContext & execution,
        size_t weight_context_bytes, size_t graph_context_bytes,
        assets::TensorStorageType conv_storage_type);
    ~XttsV2DecoderRuntime();

    std::vector<float> decode(const std::vector<float> & latents, int64_t frames,
                              const XttsV2SpeakerEmbedding & speaker);

private:
    struct Weights;
    class Graph;
    core::ExecutionContext & execution_;
    size_t graph_context_bytes_;
    std::shared_ptr<const Weights> weights_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::xtts_v2
