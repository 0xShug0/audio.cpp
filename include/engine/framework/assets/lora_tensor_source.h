#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <unordered_map>

namespace engine::assets {

enum class LoraMergeMode {
    AccumulateF32,
    RoundedBF16Delta,
};

struct LoraTensorDelta {
    std::vector<float> a;
    std::vector<float> b;
    int64_t r = 0;
    int64_t in = 0;
    int64_t out = 0;
    float scale = 1.0F;
    LoraMergeMode merge_mode = LoraMergeMode::AccumulateF32;
};

struct TensorOverride {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

// Shapes are A=[rank,in], B=[out,rank], base=[out,in]. Names and the resolved
// scale come from the model; this helper does not interpret adapter metadata.
LoraTensorDelta load_lora_tensor_delta(
    const TensorSource & base, const TensorSource & adapter,
    const std::string & base_name, const std::string & a_name,
    const std::string & b_name, float scale);

// Opt-in, load-time overlay. Full overrides take precedence over LoRA deltas.
// Merge arithmetic preserves FP32 rank-wise accumulation: for each k,
// W[o,i] += (scale * B[o,k]) * A[k,i], skipping zero scaled B values.
// This is NOT a rounded matrix-product delta followed by one base addition.
// RoundedBF16Delta explicitly selects BF16(W + BF16(scale * (B @ A))),
// with the matrix product accumulated in F32 before scaling.
// Base metadata/storage policy is retained; merged raw exports are F32 and
// backend uploads use the requested dtype through the existing conversion API.
// Optional upload caching retains one dtype per adapted tensor until source
// destruction, including across release_storage(). A dtype change replaces it.
std::shared_ptr<const TensorSource> make_lora_tensor_source(
    std::shared_ptr<const TensorSource> base,
    std::unordered_map<std::string, LoraTensorDelta> deltas,
    std::unordered_map<std::string, TensorOverride> overrides = {},
    std::string log_prefix = "lora",
    bool cache_backend_weights = false);

}  // namespace engine::assets
