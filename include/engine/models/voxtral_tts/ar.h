#pragma once

#include "engine/models/voxtral_tts/assets.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

#include <array>
#include <memory>

namespace engine::models::voxtral_tts {

class VoxtralMinistralARRuntime {
public:
    VoxtralMinistralARRuntime(const VoxtralTtsAssets & assets, core::ExecutionContext & execution,
                             assets::TensorStorageType storage, ggml_type cache_type);
    ~VoxtralMinistralARRuntime();
    modules::CausalDecoderStepResult prefill(const std::vector<int32_t> & prompt,
                                            const std::vector<float> & voice, int64_t max_frames);
    modules::CausalDecoderStepResult decode(const std::vector<int32_t> & audio_codes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

modules::DecoderLayerWeights load_transformer_layer(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, const TransformerConfig & config, assets::TensorStorageType storage);

}  // namespace engine::models::voxtral_tts
