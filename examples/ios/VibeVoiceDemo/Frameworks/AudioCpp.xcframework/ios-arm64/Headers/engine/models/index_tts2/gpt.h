#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/models/index_tts2/assets.h"

#include "ggml-backend.h"

#include <memory>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2GptConditionSubsamplingWeights {
    engine::modules::Conv2dWeights conv;
    engine::modules::LinearWeights out;
    engine::core::TensorValue pos_enc;
};

struct IndexTTS2GptConditionLayerWeights {
    engine::modules::NormWeights norm_ff;
    engine::modules::NormWeights norm_mha;
    engine::modules::NormWeights norm_conv;
    engine::modules::NormWeights norm_final;
    engine::modules::LinearWeights feed_forward_in;
    engine::modules::LinearWeights feed_forward_out;
    engine::modules::RelativeAttentionWeights self_attn;
    engine::modules::Conv1dWeights conv_pointwise_in;
    engine::modules::DepthwiseConv1dWeights conv_depthwise;
    engine::modules::NormWeights conv_norm;
    engine::modules::Conv1dWeights conv_pointwise_out;
};

struct IndexTTS2GptConditionEncoderWeights {
    IndexTTS2GptConditionSubsamplingWeights subsampling;
    std::vector<IndexTTS2GptConditionLayerWeights> layers;
    engine::modules::NormWeights after_norm;
};

struct IndexTTS2PerceiverAttentionWeights {
    engine::modules::LinearWeights q;
    engine::modules::LinearWeights kv;
    engine::modules::LinearWeights out;
};

struct IndexTTS2PerceiverFeedForwardWeights {
    engine::modules::LinearWeights in;
    engine::modules::LinearWeights out;
};

struct IndexTTS2PerceiverLayerWeights {
    IndexTTS2PerceiverAttentionWeights attention;
    IndexTTS2PerceiverFeedForwardWeights feed_forward;
};

struct IndexTTS2PerceiverWeights {
    engine::core::TensorValue latents;
    engine::modules::LinearWeights project_context;
    std::vector<IndexTTS2PerceiverLayerWeights> layers;
    engine::core::TensorValue norm_gamma;
};

struct IndexTTS2Gpt2LayerWeights {
    engine::modules::NormWeights attn_norm;
    engine::modules::LinearWeights qkv;
    engine::modules::LinearWeights attn_out;
    engine::modules::NormWeights mlp_norm;
    engine::modules::LinearWeights mlp_in;
    engine::modules::LinearWeights mlp_out;
};

struct IndexTTS2GptWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    IndexTTS2GptConditionEncoderWeights speaker_conditioner;
    IndexTTS2GptConditionEncoderWeights emotion_conditioner;
    IndexTTS2PerceiverWeights speaker_perceiver;
    IndexTTS2PerceiverWeights emotion_perceiver;
    engine::core::TensorValue text_embedding;
    engine::core::TensorValue mel_embedding;
    engine::core::TensorValue text_pos_embedding;
    engine::core::TensorValue mel_pos_embedding;
    engine::modules::LinearWeights emotion_vec_projection;
    engine::modules::LinearWeights emotion_layer;
    std::vector<float> speed_embedding_values;
    std::vector<IndexTTS2Gpt2LayerWeights> gpt_layers;
    engine::modules::NormWeights gpt_final_norm;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights mel_head;
    engine::modules::LinearWeights text_head;
};

struct IndexTTS2GptLatent {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

struct IndexTTS2GptGeneration {
    std::vector<int32_t> codes;
    IndexTTS2GptLatent speech_conditioning_latent;
    uint64_t rng_offset_blocks = 0;
};

struct IndexTTS2GptGenerationRequest {
    std::vector<int32_t> text_tokens;
    std::vector<float> speaker_semantic;
    int64_t speaker_frames = 0;
    std::vector<float> emotion_semantic;
    int64_t emotion_frames = 0;
    std::vector<float> emotion_vector;
    float top_p = 0.8F;
    int top_k = 30;
    float temperature = 0.8F;
    float repetition_penalty = 10.0F;
    bool do_sample = true;
    float length_penalty = 0.0F;
    int num_beams = 3;
    int max_mel_tokens = 1500;
    uint32_t seed = 0;
};

std::shared_ptr<const IndexTTS2GptWeights> load_index_tts2_gpt_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

class IndexTTS2GptRuntime {
public:
    IndexTTS2GptRuntime(
        std::shared_ptr<const IndexTTS2Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType matmul_storage_type,
        engine::assets::TensorStorageType conv_storage_type);
    ~IndexTTS2GptRuntime();

    IndexTTS2GptRuntime(const IndexTTS2GptRuntime &) = delete;
    IndexTTS2GptRuntime & operator=(const IndexTTS2GptRuntime &) = delete;

    void prepare_speaker_conditioning(int64_t frames);
    void prepare_emotion_conditioning(int64_t frames);
    void prepare_generation(int64_t text_tokens, int64_t max_mel_tokens, int64_t num_beams);
    IndexTTS2GptLatent speaker_conditioning(const std::vector<float> & semantic_btc, int64_t frames);
    IndexTTS2GptLatent emotion_conditioning(const std::vector<float> & semantic_btc, int64_t frames);
    std::vector<float> project_emotion_vector(const IndexTTS2GptLatent & emotion_conditioning);
    std::vector<float> merge_emotion_vector(
        const std::vector<float> & speaker_semantic,
        int64_t speaker_frames,
        const std::vector<float> & emotion_semantic,
        int64_t emotion_frames,
        float alpha);
    IndexTTS2GptGeneration generate_speech(const IndexTTS2GptGenerationRequest & request);
    IndexTTS2GptLatent forward_latent(
        const IndexTTS2GptLatent & speech_conditioning_latent,
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & codes,
        const std::vector<float> & emotion_semantic,
        int64_t emotion_frames,
        const std::vector<float> & emotion_vector);
    void release_conditioning_graphs();
    void release_generation_graphs();

private:
    class ConditioningGraph;
    class EmotionVectorGraph;
    class PrefillGraph;
    class DecodeGraph;
    class ForwardGraph;

    std::shared_ptr<const IndexTTS2Assets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const IndexTTS2GptWeights> weights_;
    std::unique_ptr<ConditioningGraph> speaker_conditioning_graph_;
    std::unique_ptr<ConditioningGraph> emotion_conditioning_graph_;
    std::unique_ptr<EmotionVectorGraph> emotion_vector_graph_;
    std::unique_ptr<PrefillGraph> prefill_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
    std::unique_ptr<ForwardGraph> forward_graph_;
};

}  // namespace engine::models::index_tts2
