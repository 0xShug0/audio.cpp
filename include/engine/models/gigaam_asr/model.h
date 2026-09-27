#pragma once

#include "engine/models/gigaam_asr/conformer.h"
#include "engine/models/gigaam_asr/frontend.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/sentencepiece.h"

namespace engine::models::gigaam_asr {

struct GigaAMAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    ConformerConfig encoder;
    int64_t layers = 0;
    bool rnnt = false;
    int64_t classes = 0;
    int64_t predictor_hidden = 0;
    int64_t joint_hidden = 0;
    int64_t max_symbols_per_step = 10;
    std::vector<std::string> characters;
    std::vector<tokenizers::SentencePiecePiece> pieces;
    std::unique_ptr<GigaAMFrontend> frontend;
    std::string decode(const std::vector<int32_t> & ids) const;
};

struct GigaAMWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    ConformerWeights encoder;
    modules::LinearWeights ctc, joint_encoder, joint_predictor, joint_output;
    modules::LSTMCellWeights predictor;
    core::TensorValue embedding;
};

std::shared_ptr<const GigaAMAssets> load_gigaam_assets(const std::filesystem::path & path);
std::unique_ptr<GigaAMWeights> load_gigaam_weights(
    const GigaAMAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type);

struct GigaAMDecodedTokens {
    std::vector<int32_t> ids;
    std::vector<int64_t> frames;
    int64_t encoder_frames = 0;
    double encoder_ms = 0.0;
    double inference_ms = 0.0;
};

class GigaAMRuntime {
public:
    GigaAMRuntime(const GigaAMAssets & assets, const GigaAMWeights & weights, core::ExecutionContext & execution);
    ~GigaAMRuntime();
    GigaAMDecodedTokens transcribe(const audio::AudioTensor & features, bool timestamps = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_gigaam_asr_loader();

}  // namespace engine::models::gigaam_asr
