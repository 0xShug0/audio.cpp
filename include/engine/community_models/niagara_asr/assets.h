#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::niagara_asr {

struct NiagaraFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 400;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t n_mels = 80;
};

struct NiagaraEncoderConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t state_size = 0;
    int64_t state_channels = 0;
    int64_t dense_input_features = 0;
    int64_t subsampling_kernel_height = 4;
    int64_t subsampling_kernel_width = 3;
};

struct NiagaraAsrConfig {
    std::string model_type = "niagara_asr";
    int64_t vocab_size = 256;
    int64_t blank_id = 256;
    NiagaraFrontendConfig frontend;
    NiagaraEncoderConfig encoder;
};

struct NiagaraAsrAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    NiagaraAsrConfig config;
    std::vector<tokenizers::SentencePiecePiece> tokenizer_pieces;
};

std::shared_ptr<const NiagaraAsrAssets> load_niagara_asr_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::community_models::niagara_asr
