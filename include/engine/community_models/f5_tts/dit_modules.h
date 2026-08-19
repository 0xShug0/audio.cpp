#pragma once

#include "engine/community_models/f5_tts/runtime.h"
#include "engine/community_models/f5_tts/weights.h"

#include "engine/framework/core/module.h"

namespace engine::models::f5_tts {

struct F5DiTGraphBuild {
    core::TensorValue x;          // leaf [1, N, 100]
    core::TensorValue cond;       // leaf [1, N, 100]
    core::TensorValue text_ids;   // leaf [NT] (i32)
    core::TensorValue time_input; // leaf [1, 256]
    core::TensorValue output;     // [1, N, 100]
};

// Builds the DiT velocity graph from framework modules (see dit_modules.cpp).
F5DiTGraphBuild build_dit_modules_graph(
    ggml_context * ggml,
    const F5DiTWeights & w,
    const F5Architecture & arch,
    int frames,
    int text_len,
    core::BackendType backend_type);

}  // namespace engine::models::f5_tts

namespace engine::models::f5_tts {
// CUDA build-time constant staging (internal; used by runtime.cpp)
struct ConstStage;
std::vector<ConstStage> * const_stage_begin();
void const_stage_upload(std::vector<ConstStage> * stage);
void const_stage_end(std::vector<ConstStage> * stage);
}  // namespace engine::models::f5_tts
