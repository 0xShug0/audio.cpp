#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/ebranchformer_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::owsm_ctc {

struct OWSMCTCV4Config {
    std::string variant;
    int64_t prompt_hidden_size = 0;
    int64_t prompt_num_heads = 0;
    int64_t prompt_layers = 0;
    int64_t prompt_intermediate_size = 0;
    std::vector<int64_t> interctc_layers;
    std::vector<int64_t> cross_attention_layers;
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t encoder_layers = 0;
    int64_t intermediate_size = 0;
    int64_t vocabulary_size = 50002;
    int64_t max_audio_samples = 480000;
    int64_t frontend_frames = 3001;
    int64_t encoder_frames = 374;
    int32_t blank_id = 0;
};

struct OWSMCTCV4Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<tokenizers::SentencePiecePiece> sentencepiece;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend;
    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    OWSMCTCV4Config config;

    int32_t token_id(const std::string & token) const;
    std::string decode_visible(const std::vector<int32_t> & ids) const;
};

struct OWSMCTCV4Weights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::EspnetConv2dSubsampling8Weights subsampling;
    std::vector<modules::EBranchformerBlockWeights> encoder;
    modules::NormWeights encoder_norm;
    core::TensorValue embedding;
    modules::LinearWeights output;
    core::TensorValue half_scale;
    core::TensorValue embedding_scale;
    std::vector<modules::TransformerEncoderBlockWeights> prompt_encoder;
    modules::NormWeights prompt_norm;
    modules::LinearWeights prefix_projection;
    modules::LinearWeights prompt_projection;
    modules::LinearWeights ctc_conditioning;
    core::TensorValue prompt_scale;
};

std::shared_ptr<const OWSMCTCV4Assets> load_owsm_ctc_assets(const std::filesystem::path & path);
std::unique_ptr<OWSMCTCV4Weights> load_owsm_ctc_weights(
    const OWSMCTCV4Assets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type);

class OWSMCTCV4EBranchformerRuntime {
public:
    OWSMCTCV4EBranchformerRuntime(const OWSMCTCV4Assets & assets, const OWSMCTCV4Weights & weights,
                     core::ExecutionContext & execution);
    ~OWSMCTCV4EBranchformerRuntime();

    std::vector<int32_t> frame_tokens(const std::vector<float> & samples,
                                     int32_t language, int32_t task);

private:
    struct Graphs;
    const OWSMCTCV4Assets & assets_;
    const OWSMCTCV4Weights & weights_;
    core::ExecutionContext & execution_;
    std::unique_ptr<Graphs> graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_owsm_ctc_loader();

}  // namespace engine::models::owsm_ctc
