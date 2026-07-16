#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::citrinet_asr {

struct JasperBlockConfig {
    int64_t filters = 0;
    int64_t repeat = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    float dropout = 0.0f;
    bool residual = false;
    std::string residual_mode;
    bool separable = false;
    bool se = false;
    int64_t se_reduction_ratio = 8;
};

struct CitrinetConfig {
    int64_t sample_rate = 0;
    int64_t n_mels = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int64_t win_length = 0;
    int64_t pad_to = 0;
    int64_t vocab_size = 0;
    int64_t num_classes = 0;
    int64_t blank_id = 0;
    int64_t output_stride = 1;
    std::string normalize;
    std::string window;
    std::vector<JasperBlockConfig> jasper;
};

struct Conv1dWeights {
    std::string weight_name;
    std::vector<int64_t> weight_source_shape;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    int64_t groups = 1;
    bool use_bias = false;
    std::optional<std::string> bias_name;
};

struct BatchNorm1dWeights {
    int64_t channels = 0;
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

struct ConvBn {
    Conv1dWeights conv;
    BatchNorm1dWeights bn;
};

struct SeparableConvBn {
    Conv1dWeights depthwise;
    Conv1dWeights pointwise;
    BatchNorm1dWeights bn;
};

struct SqueezeExciteWeights {
    Conv1dWeights fc1;
    Conv1dWeights fc2;
};

struct JasperBlockWeights {
    bool separable = false;
    std::vector<SeparableConvBn> separable_repeats;
    std::vector<ConvBn> conv_repeats;
    bool has_residual = false;
    Conv1dWeights residual_conv;
    BatchNorm1dWeights residual_bn;
    bool has_se = false;
    SqueezeExciteWeights se;
};

struct CitrinetWeights {
    CitrinetConfig config;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<float> window;
    std::vector<float> fb;
    std::vector<JasperBlockWeights> blocks;
    Conv1dWeights decoder;
    std::vector<tokenizers::SentencePiecePiece> tokenizer_pieces;
};

std::shared_ptr<const CitrinetWeights> load_citrinet_weights_cached(const std::filesystem::path & model_path);

}  // namespace engine::models::citrinet_asr
