#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/framework/text/chunking.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::dots_tts {

struct DotsTransformerConfig {
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t hidden_size = 0;
    int64_t ffn_hidden_size = 0;
    bool modulation = false;
    bool qkv_bias = false;
    bool qk_norm = true;
    float attn_dropout = 0.0F;
    float dropout = 0.0F;
    std::string norm_layer;
    bool alibi_bias = false;
    bool rotary_bias = true;
    float rotary_theta = 10000.0F;
    int64_t input_dim = 0;
    bool causal = false;
};

struct DotsVocoderConfig {
    int sample_rate = 0;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    int64_t upsample_initial_channel = 0;
    std::string resblock;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> resblock_dilation_sizes;
    std::vector<int64_t> downsample_rates;
    std::vector<int64_t> downsample_channels;
    std::string activation;
    bool snake_logscale = true;
    int64_t latent_dim = 0;
    bool causal = true;
    int64_t mi_num_layers = 0;
    bool causal_encoder = true;
    bool use_bias_at_final = false;
    bool use_tanh_at_final = false;
};

struct DotsMeanFlowConfig {
    bool enabled = false;
    bool use_duration_embedding = false;
};

struct DotsLlmConfig {
    int64_t vocab_size = 0;
    int64_t max_position_embeddings = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    float rms_norm_eps = 1e-6F;
    float rope_theta = 1000000.0F;
    int64_t bos_token_id = 151643;
    int64_t eos_token_id = 151643;
};

struct DotsConfig {
    int64_t latent_dim = 0;
    int64_t patch_size = 0;
    float cfg_droprate = 0.2F;
    DotsTransformerConfig patch_encoder;
    DotsTransformerConfig dit;
    DotsVocoderConfig vocoder;
    float fm_sigma = 0.0F;
    float xvec_drop_rate = 0.2F;
    int64_t campplus_embedding_size = 512;
    float xvec_max_audio_seconds = 10.0F;
    std::optional<DotsMeanFlowConfig> meanflow;
    DotsLlmConfig llm;
};

enum class DotsTemplateName {
    Tts,
    InstructionTts,
    TextToAudio,
    TtsInterleave,
    Edit,
};

enum class DotsOdeMethod {
    Euler,
    Midpoint,
    Rk4,
};

enum class DotsEditXVectorMode {
    Auto,
    On,
    Off,
};

struct DotsGenerationOptions {
    DotsTemplateName template_name = DotsTemplateName::Tts;
    std::string language = "none";
    int64_t num_inference_steps = 10;
    float guidance_scale = 1.2F;
    float speaker_scale = 1.5F;
    DotsOdeMethod ode_method = DotsOdeMethod::Euler;
    int64_t max_tokens = 500;
    int64_t text_chunk_size = 320;
    engine::text::TextChunkMode text_chunk_mode = engine::text::TextChunkMode::TagAware;
    int64_t vocoder_merge_steps = 4;
    uint64_t seed = 42;
};

struct DotsPromptReference {
    std::optional<runtime::AudioBuffer> audio;
    std::string reference_text;
    std::optional<float> duration_seconds;
};

struct DotsEditOptions {
    std::optional<runtime::AudioBuffer> source_audio;
    std::string instruction;
    std::string source_text;
    std::string target_text;
    DotsEditXVectorMode use_xvector = DotsEditXVectorMode::Auto;
};

struct DotsRequest {
    std::string text;
    DotsPromptReference reference;
    DotsGenerationOptions generation;
    DotsEditOptions edit;
};

struct DotsGenerationSchedule {
    std::vector<int32_t> token_ids;
    bool interleave = false;
};

}  // namespace engine::models::dots_tts
