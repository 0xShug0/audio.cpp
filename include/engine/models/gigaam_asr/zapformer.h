#pragma once

#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <vector>

namespace engine::models::gigaam_asr {

// GigaAM-v4 ZapformerEncoder, full-context (offline) path. Mirrors
// modeling_gigaam.ZapformerEncoder.forward with chunk_size = -1.
struct ZapformerStackConfig {
    int64_t dim = 0;
    int64_t layers = 0;
    int64_t heads = 0;
    int64_t query_head_dim = 32;
    int64_t value_head_dim = 32;
    int64_t feedforward_multiple = 4;
    int64_t conv_kernel = 0;  // 0: the layers have neither a convolution module nor a second feed-forward
    int64_t downsampling = 1;
};

struct ZapformerConfig {
    int64_t features = 64;
    int64_t frontend_channels[3] = {16, 32, 64};
    int64_t embed_dim = 768;
    std::vector<ZapformerStackConfig> stacks;
    float rope_theta = 10000.0f;
    int64_t max_downsampling() const;
    int64_t frontend_bins() const;  // mel bins after two stride-2 convolutions
};

struct ZapformerLayerWeights {
    modules::LinearWeights ff1_in, ff1_out, ff2_in, ff2_out;
    modules::NormWeights attention_norm, output_norm;
    modules::LinearWeights qk, vg, attention_out;
    modules::LinearWeights conv_in, conv_out;
    modules::DepthwiseConv1dWeights depthwise;
    core::TensorValue mean_weights;
    bool has_conv = false;
};

struct ZapformerStackWeights {
    modules::LinearWeights proj, proj_t;  // proj_t is proj.weight transposed by the converter
    float proj_scale = 1.0f;              // OrthogonalLinear: in_features^-0.5 / 0.3
    float input_scale = 1.0f;
    std::vector<float> residual_scales;   // layers + 1
    std::vector<ZapformerLayerWeights> layers;
};

struct ZapformerWeights {
    modules::Conv2dWeights conv1, conv2, conv3;
    modules::Conv2dWeights convnext_depthwise, convnext_pointwise1, convnext_pointwise2;
    modules::LinearWeights out;
    std::vector<ZapformerStackWeights> stacks;
    float out_scale = 1.0f;
    float out_eps = 0.1f;
};

struct ZapformerInputs {
    core::TensorValue features;   // [1, mel_bins, mel_frames]
    core::TensorValue positions;  // I32 [padded encoder frames]
    core::TensorValue valid;      // F32 [1, padded encoder frames, 1]: 1 for real frames, 0 for the even padding
};

// Number of encoder frames before / after the even padding.
int64_t zapformer_encoder_frames(int64_t mel_frames);
int64_t zapformer_padded_frames(const ZapformerConfig & config, int64_t mel_frames);

// Returns [1, padded_frames, embed_dim]; only the first zapformer_encoder_frames() frames are meaningful.
core::TensorValue build_zapformer_encoder(
    core::ModuleBuildContext & ctx,
    const ZapformerInputs & inputs,
    int64_t mel_frames,
    const ZapformerConfig & config,
    const ZapformerWeights & weights);

}  // namespace engine::models::gigaam_asr
