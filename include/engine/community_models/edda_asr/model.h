#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::community_models::edda_asr {

// DFlash2 block-diffusion drafter shipped as drafter.safetensors by
// syvai/edda-v0.2-large-hurtig. Hyperparameters come from drafter_config.json.
struct DrafterConfig {
    int64_t layers = 5;
    int64_t target_layers = 32;
    int64_t block = 16;
    int64_t conv_kernel = 2;
    int64_t conv_group = 16;
    int64_t selector_rank = 256;
    int64_t selector_top_k = 16;
    std::vector<int64_t> target_layer_ids;
};

struct EddaAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    std::shared_ptr<const assets::TensorSource> drafter_source;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
    modules::WhisperEmbeddingConfig encoder;
    int64_t decoder_layers = 0, decoder_heads = 0, decoder_ffn = 0;
    int64_t vocabulary_size = 0, max_positions = 0;
    int32_t eos = 0, start_of_transcript = 0, transcribe = 0, no_timestamps = 0;
    std::vector<int32_t> suppress, begin_suppress;
    std::vector<bool> special_tokens;
    std::optional<DrafterConfig> drafter;

    std::vector<int32_t> prompt(const std::string & language) const;
    std::string decode_text(const std::vector<int32_t> & tokens) const;
};

std::shared_ptr<const EddaAssets> load_assets(const std::filesystem::path & path);

struct DrafterConvWeights {
    core::TensorValue base;  // [2, kernel, hidden]: prepare and finish taps
    core::TensorValue projection;
};

struct DrafterLayerWeights {
    modules::NormWeights attention_norm, cross_norm, mlp_norm;
    core::TensorValue q_weight, k_weight, v_weight, out_weight;
    core::TensorValue q_bias, v_bias, out_bias;
    modules::AttentionWeights cross_attention;
    modules::FeedForwardWeights mlp;
    DrafterConvWeights attention_conv, mlp_conv;
};

struct DrafterWeights {
    std::vector<DrafterLayerWeights> layers;
    core::TensorValue context_fc, mask_embedding, selector_projection;
    modules::NormWeights context_norm, final_norm;
    // The candidate selector walks the block on the host, so its codebooks stay in host memory.
    std::vector<uint16_t> predecessor_codebook, successor_codebook;
};

struct TargetLayerWeights {
    modules::NormWeights attention_norm, cross_norm, mlp_norm;
    core::TensorValue qkv_weight, qkv_bias, out_weight, out_bias;
    modules::AttentionWeights cross_attention;
    modules::FeedForwardWeights mlp;
};

struct EncoderLayerWeights {
    modules::NormWeights attention_norm, mlp_norm;
    core::TensorValue qkv_weight, qkv_bias, out_weight, out_bias;
    modules::FeedForwardWeights mlp;
};

struct EncoderWeights {
    modules::Conv1dWeights conv1, conv2;
    core::TensorValue positions;
    std::vector<EncoderLayerWeights> layers;
    modules::NormWeights final_norm;
};

struct EddaWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    EncoderWeights encoder;
    std::vector<TargetLayerWeights> decoder;
    core::TensorValue embedding, positions;
    modules::NormWeights decoder_norm;
    std::optional<DrafterWeights> drafter;
};

std::unique_ptr<EddaWeights> load_weights(const EddaAssets & assets, core::ExecutionContext & execution,
    assets::TensorStorageType type, bool with_drafter);

struct DecodeOptions {
    int64_t max_tokens = 440;
    bool speculative = true;
    bool use_selector = true;
    int64_t block = 16;
};

struct DecodeResult {
    std::vector<int32_t> tokens;  // generated tokens, without the prompt or end-of-text
    bool reached_eos = false;
    int64_t target_passes = 0;
    int64_t drafted = 0, accepted = 0;
    double encoder_ms = 0.0, decoder_ms = 0.0, drafter_ms = 0.0;
};

class EddaRuntime {
public:
    EddaRuntime(const EddaAssets & assets, const EddaWeights & weights, core::ExecutionContext & execution);
    ~EddaRuntime();
    DecodeResult transcribe(const std::vector<float> & samples, const std::vector<int32_t> & prompt,
        const DecodeOptions & options);

private:
    struct State;
    struct TargetGraph;
    struct DrafterGraph;
    TargetGraph & target_graph(int64_t tokens);
    DrafterGraph & drafter_graph(int64_t block);
    std::vector<int32_t> select_path(const DrafterGraph & graph, int32_t anchor) const;

    const EddaAssets & assets_;
    const EddaWeights & weights_;
    core::ExecutionContext & execution_;
    audio::WhisperLogMelExtractor frontend_;
    std::unique_ptr<State> state_;
    std::vector<std::unique_ptr<TargetGraph>> target_graphs_;
    std::vector<std::unique_ptr<DrafterGraph>> drafter_graphs_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_edda_asr_loader();

}  // namespace engine::community_models::edda_asr
