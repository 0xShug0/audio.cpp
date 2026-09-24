#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::canary_asr {

struct CanaryAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<tokenizers::SentencePiecePiece> vocabulary;
    std::vector<float> window;
    audio::AudioTensor filterbank;
    std::shared_ptr<const audio::NemoMelFrontend> frontend;
    int32_t special_token(const std::string & text) const;
};

struct CanaryWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::DepthwiseConvSubsamplingWeights subsampling;
    modules::LinearWeights encoder_out, head;
    std::vector<modules::RelativeConformerBlockWeights> encoder;
    std::vector<modules::TransformerDecoderBlockWeights> decoder;
    core::TensorValue embedding, positions;
    modules::NormWeights embedding_norm, decoder_norm;
};

std::shared_ptr<const CanaryAssets> load_canary_assets(const std::filesystem::path & path);
std::unique_ptr<CanaryWeights> load_canary_weights(
    const CanaryAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type);

struct CanaryFrontendFeatures {
    std::vector<float> values;
    int64_t raw_frames = 0;
    int64_t valid_frames = 0;
};

CanaryFrontendFeatures extract_canary_frontend(
    const std::vector<float> & samples,
    const CanaryAssets & assets,
    size_t threads);

class CanaryRuntime {
public:
    CanaryRuntime(const CanaryAssets & assets, const CanaryWeights & weights, core::ExecutionContext & execution);
    ~CanaryRuntime();
    std::vector<int32_t> transcribe(const std::vector<float> & samples,
        const std::vector<int32_t> & prompt, int64_t max_tokens);

private:
    struct Graphs;
    const CanaryAssets & assets_;
    const CanaryWeights & weights_;
    core::ExecutionContext & execution_;
    std::unique_ptr<Graphs> graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_canary_asr_loader();

}  // namespace engine::models::canary_asr
