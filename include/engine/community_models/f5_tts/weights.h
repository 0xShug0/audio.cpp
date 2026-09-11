#pragma once

#include "engine/community_models/f5_tts/runtime.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::f5_tts {

// ---------------------------------------------------------------------------
// Weights, expressed with framework module weight types (dev-branch pattern:
// the weight store loads torch-shaped tensors once; modules consume them).
// ---------------------------------------------------------------------------

struct F5TextConvNextWeights {
    modules::DepthwiseConv1dWeights dwconv;   // [C, 1, 7] + [C]
    modules::NormWeights norm;                // [C] + [C]
    modules::LinearWeights pw1;               // [1024, 512]
    modules::LinearWeights pw2;               // [512, 1024]
    std::vector<float> grn_gamma;             // [1024] (host: GRN has no module)
    std::vector<float> grn_beta;              // [1024]
};

struct F5BlockWeights {
    modules::LinearWeights attn_norm;         // adaLN: [6*D, D]
    modules::LinearWeights to_q;              // [D, D]
    modules::LinearWeights to_k;
    modules::LinearWeights to_v;
    modules::LinearWeights to_out;
    modules::LinearWeights ff0;               // [2*D, D]
    modules::LinearWeights ff2;               // [D, 2*D]
};

struct F5DiTWeights {
    int64_t vocab_size = 2731;  // from the checkpoint's text embedding
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;         // [vocab, 512]
    modules::LinearWeights input_proj;        // [1024, 712]
    modules::Conv1dWeights cpe0;              // [1024, 64, 31] (grouped g=16)
    modules::Conv1dWeights cpe2;
    modules::LinearWeights time0;             // [256, 256]
    modules::LinearWeights time2;             // [1024, 256]
    std::vector<F5TextConvNextWeights> text_blocks;  // x4
    std::vector<F5BlockWeights> blocks;              // x22
    modules::LinearWeights norm_out;          // [2*D, D]
    modules::LinearWeights proj_out;          // [100, D]
};

// Loads the EMA weights (prefix stripped) into module weight types.
F5DiTWeights load_dit_weights(
    const assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type);

}  // namespace engine::models::f5_tts
