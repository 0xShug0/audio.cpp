#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"

#include <memory>
#include <vector>

namespace engine::models::minimax_h3 {

struct AudioVaeWeightStore {
    engine::core::ExecutionContext & execution;
    std::shared_ptr<const engine::assets::TensorSource> source;
    engine::core::BackendWeightStore store;
    engine::core::TensorValue latent_mean;
    engine::core::TensorValue latent_std;
    engine::modules::Conv1dWeights dec_in;
    engine::modules::BigVganVocoderWeights decoder;

    AudioVaeWeightStore(
        engine::core::ExecutionContext & execution_context,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        const MiniMaxH3Config & cfg,
        size_t weight_context_bytes);
};

std::vector<float> run_audio_vae_decode_graph(
    AudioVaeWeightStore & weights,
    const MiniMaxH3Config & cfg,
    const std::vector<float> & audio_rows);

}  // namespace engine::models::minimax_h3
