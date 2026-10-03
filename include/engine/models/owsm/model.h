#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/ebranchformer_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/transformers/transformer_blocks.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::owsm {

struct OWSMV4Config {
    std::string variant;
    int64_t hidden_size = 0;
    int64_t num_heads = 0;
    int64_t encoder_layers = 0;
    int64_t decoder_layers = 0;
    int64_t intermediate_size = 0;
    int64_t vocabulary_size = 50002;
    int64_t max_audio_samples = 480000;
    int64_t frontend_frames = 3001;
    int64_t encoder_frames = 374;
    int64_t max_decode_tokens = 374;
    int32_t blank_id = 0;
    int32_t notimestamps_id = 181;
    int32_t first_timestamp_id = 182;
    int32_t last_timestamp_id = 1682;
    int32_t sos_id = 49999;
    int32_t eos_id = 50000;
    int32_t sop_id = 50001;
};

struct OWSMV4Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<tokenizers::SentencePiecePiece> sentencepiece;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend;
    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    OWSMV4Config config;

    int32_t token_id(const std::string & token) const;
    std::vector<int32_t> tokenize_text(const std::string & text) const;
    std::string decode_visible(const std::vector<int32_t> & ids) const;
};

struct OWSMV4SubsamplingWeights {
    modules::Conv2dWeights conv0;
    modules::Conv2dWeights conv1;
    modules::Conv2dWeights conv2;
    modules::LinearWeights projection;
};

struct OWSMV4Weights {
    std::unique_ptr<core::BackendWeightStore> store;
    OWSMV4SubsamplingWeights subsampling;
    std::vector<modules::EBranchformerBlockWeights> encoder;
    modules::NormWeights encoder_norm;
    core::TensorValue embedding;
    std::vector<modules::TransformerDecoderBlockWeights> decoder;
    modules::NormWeights decoder_norm;
    modules::LinearWeights output;
    core::TensorValue half_scale;
    core::TensorValue embedding_scale;
};

struct OWSMV4DecodeResult {
    std::vector<int32_t> tokens;
    std::string detected_language;
};

std::shared_ptr<const OWSMV4Assets> load_owsm_assets(const std::filesystem::path & path);
std::unique_ptr<OWSMV4Weights> load_owsm_weights(
    const OWSMV4Assets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type);

class OWSMV4Runtime {
public:
    OWSMV4Runtime(const OWSMV4Assets & assets, const OWSMV4Weights & weights,
                  core::ExecutionContext & execution);
    ~OWSMV4Runtime();

    OWSMV4DecodeResult decode(
        const std::vector<float> & samples,
        const std::vector<int32_t> & prompt,
        bool predict_timestamps,
        int64_t max_tokens,
        int64_t beam_size = 1,
        const std::function<void(const std::vector<int32_t> &)> & on_tokens = {});

private:
    struct Graphs;
    const OWSMV4Assets & assets_;
    const OWSMV4Weights & weights_;
    core::ExecutionContext & execution_;
    std::unique_ptr<Graphs> graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_owsm_loader();

}  // namespace engine::models::owsm
