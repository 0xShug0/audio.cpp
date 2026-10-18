#pragma once

#include "engine/models/midashenglm_gen/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenAudioTokenizerInput {
    std::vector<float> latents;
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct MiDashengLmGenAudioTokenizerOutput {
    std::vector<engine::runtime::AudioBuffer> audio;
};

struct MiDashengLmGenVocosBlockWeights {
    engine::modules::DepthwiseConv1dWeights depthwise;
    engine::modules::NormWeights norm;
    engine::modules::LinearWeights pointwise_in;
    engine::modules::LinearWeights pointwise_out;
    engine::core::TensorValue gamma;
};

struct MiDashengLmGenAudioTokenizerWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::ConvTranspose1dWeights upsampler;
    engine::modules::Conv1dWeights embed;
    engine::modules::NormWeights norm;
    std::vector<MiDashengLmGenVocosBlockWeights> blocks;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights head;
    std::vector<float> istft_window;
};

class MiDashengLmGenAudioTokenizerRuntime {
public:
    MiDashengLmGenAudioTokenizerRuntime(
        std::shared_ptr<const MiDashengLmGenAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~MiDashengLmGenAudioTokenizerRuntime();

    MiDashengLmGenAudioTokenizerRuntime(const MiDashengLmGenAudioTokenizerRuntime &) = delete;
    MiDashengLmGenAudioTokenizerRuntime & operator=(const MiDashengLmGenAudioTokenizerRuntime &) = delete;

    void prepare_decode(int64_t batch, int64_t frames);
    MiDashengLmGenAudioTokenizerOutput decode(const MiDashengLmGenAudioTokenizerInput & input);
    void release_graphs();

private:
    class DecodeGraph;

    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const MiDashengLmGenAudioTokenizerWeights> weights_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

}  // namespace engine::models::midashenglm_gen
