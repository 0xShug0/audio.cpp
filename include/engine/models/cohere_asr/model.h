#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/nemo_mel_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::cohere_asr {

struct CohereAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<tokenizers::SentencePiecePiece> vocabulary;
    std::vector<float> window;
    audio::SparseMelFilterbank filterbank;
    std::shared_ptr<const audio::NemoMelFrontend> frontend;
    int32_t special_token(const std::string & text) const;
};

struct CohereWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    modules::DepthwiseConvSubsamplingWeights subsampling;
    modules::LinearWeights encoder_out, head;
    std::vector<modules::RelativeConformerBlockWeights> encoder;
    std::vector<modules::TransformerDecoderBlockWeights> decoder;
    core::TensorValue embedding, positions;
    modules::NormWeights embedding_norm, decoder_norm;
};

std::shared_ptr<const CohereAssets> load_cohere_assets(const std::filesystem::path & path);
std::unique_ptr<CohereWeights> load_cohere_weights(
    const CohereAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type);

audio::AudioTensor extract_cohere_frontend(
    const std::vector<float> & samples,
    const CohereAssets & assets,
    size_t threads);

class CohereRuntime {
public:
    CohereRuntime(const CohereAssets & assets, const CohereWeights & weights, core::ExecutionContext & execution);
    ~CohereRuntime();
    std::vector<std::vector<int32_t>> transcribe(const std::vector<std::vector<float>> & samples,
        const std::vector<int32_t> & prompt, int64_t max_tokens);

private:
    struct Graphs;
    const CohereAssets & assets_;
    const CohereWeights & weights_;
    core::ExecutionContext & execution_;
    runtime::CacheSlots<std::pair<int64_t, int64_t>, std::unique_ptr<Graphs>> graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_cohere_asr_loader();

}  // namespace engine::models::cohere_asr
