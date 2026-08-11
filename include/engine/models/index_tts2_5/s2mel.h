#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/index_tts2_5/assets.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2_5 {

struct IndexTTS25LengthRegulatorWeights {
    engine::modules::LinearWeights content_projection;
    std::vector<engine::modules::Conv1dWeights> convs;
    std::vector<engine::modules::NormWeights> norms;
    engine::modules::Conv1dWeights output;
};

struct IndexTTS25S2MelGptLayerWeights {
    engine::modules::LinearWeights linear0;
    engine::modules::LinearWeights linear1;
    engine::modules::LinearWeights linear2;
};

struct IndexTTS25AdaLayerNormWeights {
    engine::core::TensorValue norm_weight;
    engine::modules::LinearWeights project;
};

struct IndexTTS25DitLayerWeights {
    IndexTTS25AdaLayerNormWeights attention_norm;
    engine::modules::LinearWeights qkv;
    engine::modules::LinearWeights attention_out;
    IndexTTS25AdaLayerNormWeights ffn_norm;
    engine::modules::LinearWeights ffn_w1;
    engine::modules::LinearWeights ffn_w2;
    engine::modules::LinearWeights ffn_w3;
    engine::modules::LinearWeights skip_in;
};

struct IndexTTS25WaveNetLayerWeights {
    engine::modules::Conv1dWeights in_layer;
    engine::modules::Conv1dWeights res_skip_layer;
};

struct IndexTTS25S2MelCfmWeights {
    engine::modules::LinearWeights x_embedder;
    engine::modules::LinearWeights cond_projection;
    engine::modules::LinearWeights cond_x_merge;
    engine::modules::LinearWeights skip_linear;
    engine::modules::LinearWeights time_mlp0;
    engine::modules::LinearWeights time_mlp2;
    engine::core::TensorValue time_freqs;
    engine::modules::LinearWeights time2_mlp0;
    engine::modules::LinearWeights time2_mlp2;
    engine::core::TensorValue time2_freqs;
    std::vector<IndexTTS25DitLayerWeights> dit_layers;
    IndexTTS25AdaLayerNormWeights dit_norm;
    engine::modules::LinearWeights conv1;
    engine::modules::LinearWeights res_projection;
    engine::modules::Conv1dWeights wavenet_cond;
    std::vector<IndexTTS25WaveNetLayerWeights> wavenet_layers;
    engine::modules::LinearWeights final_modulation;
    engine::modules::LinearWeights final_linear;
    engine::modules::Conv1dWeights conv2;
};

struct IndexTTS25S2MelWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    IndexTTS25S2MelGptLayerWeights gpt_layer;
    IndexTTS25LengthRegulatorWeights length_regulator;
    IndexTTS25S2MelCfmWeights cfm;
};

struct IndexTTS25S2MelSequence {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS25S2MelMel {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 80;
};

std::shared_ptr<const IndexTTS25S2MelWeights> load_index_tts2_5_s2mel_weights(
    const IndexTTS25Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class IndexTTS25S2MelRuntime {
public:
    IndexTTS25S2MelRuntime(
        std::shared_ptr<const IndexTTS25Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~IndexTTS25S2MelRuntime();

    IndexTTS25S2MelRuntime(const IndexTTS25S2MelRuntime &) = delete;
    IndexTTS25S2MelRuntime & operator=(const IndexTTS25S2MelRuntime &) = delete;

    void prepare_gpt_layer(int64_t frames);
    void prepare_length_regulator(int64_t input_frames, int64_t output_frames);
    void prepare_cfm(int64_t total_frames, bool use_cfg);
    void release_pre_cfm_graphs();
    void release_cfm_graph();

    IndexTTS25S2MelSequence project_gpt_latent(const std::vector<float> & latent, int64_t frames);
    IndexTTS25S2MelSequence regulate_length(
        const std::vector<float> & content,
        int64_t input_frames,
        int64_t output_frames);
    IndexTTS25S2MelMel infer_mel(
        const std::vector<float> & condition,
        int64_t total_frames,
        const std::vector<float> & reference_mel,
        int64_t reference_frames,
        const std::vector<float> & style,
        int64_t diffusion_steps,
        float cfg_rate,
        uint32_t seed,
        uint64_t rng_offset_blocks);

private:
    class GptLayerGraph;
    class LengthRegulatorGraph;
    class CfmGraph;

    std::shared_ptr<const IndexTTS25Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS25S2MelWeights> weights_;
    std::unique_ptr<GptLayerGraph> gpt_layer_graph_;
    std::unique_ptr<LengthRegulatorGraph> length_regulator_graph_;
    std::unique_ptr<CfmGraph> cfm_graph_;
};

}  // namespace engine::models::index_tts2_5
