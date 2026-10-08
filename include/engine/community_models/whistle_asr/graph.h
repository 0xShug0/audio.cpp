#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_tensor;

namespace engine::community_models::whistle_asr {

constexpr int kWhistleLayers = 8;
constexpr int64_t kWhistleDimension = 512;
constexpr int64_t kWhistleLanes = 4;
constexpr int64_t kWhistleHeads = 8;
constexpr int64_t kWhistleKvHeads = 2;
constexpr int64_t kWhistleQkDim = 48;
constexpr int64_t kWhistleVDim = 64;
// 30 s of 10 ms mel frames through three stride-2 stem stages.
constexpr int64_t kWhistleMaximumFrames = 375;

// Conditioned Hadamard MLP: three Kronecker-factored 512x512 products with
// learned diagonal scales and two fixed permutations between them.
struct WhistleHadamardWeights {
    core::TensorValue d1;
    core::TensorValue d2;
    core::TensorValue d3;
    core::TensorValue d4;
    core::TensorValue b2;
    modules::LinearWeights cond_v;
    modules::LinearWeights cond_u;
    // Kronecker factors stored as [out, in] for ggml_mul_mat.
    core::TensorValue w1a;
    core::TensorValue w1b;
    core::TensorValue w2a;
    core::TensorValue w2b;
    core::TensorValue w3a;
    core::TensorValue w3b;
};

// Manifold hyper-connection mixing across the four residual lanes.
struct WhistleMhcWeights {
    modules::LinearWeights phi_pre;
    modules::LinearWeights phi_post;
    modules::LinearWeights phi_res;
    core::TensorValue pre_bias;
    core::TensorValue post_bias;
    core::TensorValue res_bias;
    float a_pre = 0.0f;
    float a_post = 0.0f;
    float a_res = 0.0f;
};

// Fixed tensors the shared graph helpers need. Per-row tensors are repeated for
// the largest number of rows (frames or decoder steps) one graph processes.
struct WhistleGraphConstants {
    core::TensorValue permutation_1;
    core::TensorValue permutation_2;
    core::TensorValue lane_mean;
    // Structural constants: ggml_mul_mat against these transposes small per-row
    // matrices exactly, which the generic strided copy behind ggml_cont does slowly.
    core::TensorValue transpose_4x4;
    core::TensorValue identity_16;
    core::TensorValue identity_32;
};

// Reads checkpoint tensors in their JAX layouts and uploads them in the shapes the
// framework modules expect. Layer-stacked tensors carry a leading [8] dimension.
class WhistleWeightLoader {
public:
    WhistleWeightLoader(core::BackendWeightStore & store, const assets::TensorSource & source);

    [[nodiscard]] std::vector<float> values(const std::string & name, std::vector<int64_t> shape, int layer = -1) const;
    [[nodiscard]] float scalar(const std::string & name, int layer = -1) const;
    core::TensorValue vector(const std::string & name, int64_t size, int layer = -1);
    // Gemma-style RMS norm weights. The checkpoint stores scale and the norm
    // multiplies by 1 + scale, so the stored weight already holds 1 + scale.
    modules::NormWeights norm(const std::string & name, int64_t size, int layer = -1);
    // JAX kernels are [in, out]; LinearModule and ggml_mul_mat take [out, in].
    core::TensorValue transposed(const std::string & name, int64_t in, int64_t out, int layer = -1);
    modules::LinearWeights linear(const std::string & name, int64_t in, int64_t out, int layer = -1);
    WhistleHadamardWeights hadamard(const std::string & prefix, int layer);
    // The lane matching the layer index is the active lane; the others are pushed
    // towards the sigmoid floor by a fixed logit offset folded into the biases.
    WhistleMhcWeights mhc(const std::string & prefix, int layer);
    WhistleGraphConstants constants(const WhistleAssets & assets, int64_t rows);

    [[nodiscard]] core::BackendWeightStore & store() const noexcept { return store_; }

private:
    core::TensorValue permutation(const std::array<uint16_t, 512> & permutation, int64_t rows);
    core::TensorValue identity(int64_t size, int64_t rows);

    core::BackendWeightStore & store_;
    const assets::TensorSource & source_;
};

// Whistle-specific graph pieces over ggml [features, rows] activations, shared by
// the encoder graph and the decoder step graph.
class WhistleGraphOps {
public:
    WhistleGraphOps(core::ModuleBuildContext & ctx, const WhistleGraphConstants & constants);

    ggml_tensor * linear(ggml_tensor * input, const modules::LinearWeights & weights, int64_t in, int64_t out);
    ggml_tensor * silu(ggml_tensor * input);
    ggml_tensor * sigmoid(ggml_tensor * input);
    ggml_tensor * softmax(ggml_tensor * input);
    ggml_tensor * rms_norm(ggml_tensor * input, int64_t size);
    // RMS norm times 1 + scale; weights come from WhistleWeightLoader::norm.
    ggml_tensor * gemma_norm(ggml_tensor * input, const modules::NormWeights & weights, int64_t size);
    // Per-head Gemma RMS norm over [heads * 48, rows], returned flat again.
    ggml_tensor * head_norm(ggml_tensor * input, int64_t heads, const modules::NormWeights & weights);
    // NeoX rotary embedding over [heads * 48, rows]; positions is I32[rows].
    core::TensorValue rope(ggml_tensor * input, int64_t heads, ggml_tensor * positions);
    ggml_tensor * hadamard(ggml_tensor * input, const WhistleHadamardWeights & weights);
    // Wraps one block in the four-lane mixing. state is [4 * 512, rows]; block maps
    // the mixed [512, rows] input to its [512, rows] output.
    ggml_tensor * mhc(ggml_tensor * state, const WhistleMhcWeights & weights,
                      const std::function<ggml_tensor *(ggml_tensor *)> & block);
    // Average of the four lanes, [512, rows].
    ggml_tensor * lane_mean(ggml_tensor * state);

    [[nodiscard]] core::ModuleBuildContext & context() const noexcept { return ctx_; }

private:
    ggml_tensor * log_normalize_rows(ggml_tensor * x);
    ggml_tensor * sinkhorn(ggml_tensor * logits);
    ggml_tensor * identity_view(const core::TensorValue & identity, int64_t rows);
    ggml_tensor * kronecker(ggml_tensor * input, const core::TensorValue & a, const core::TensorValue & b);
    ggml_tensor * permute_features(ggml_tensor * input, const core::TensorValue & indices);

    core::ModuleBuildContext & ctx_;
    const WhistleGraphConstants & constants_;
};

}  // namespace engine::community_models::whistle_asr
