#pragma once

#include "engine/framework/io/json.h"
#include "engine/models/roformer/assets.h"

namespace engine::models::roformer {

struct RoformerImplementation {
    std::string family;
    int mask_output_projection_layers;
    bool fused_qkv;
    bool transformer_output_norm;
    bool required_final_norm;
    bool cuda_strided_norm_and_residual;
    bool cuda_packed_qkv_gates;
    bool cuda_packed_qk_rope;
    bool cuda_batched_mask;
    void (*fill_band_layout)(RoformerArchitectureConfig &, const engine::io::json::Value &);
};

}  // namespace engine::models::roformer
