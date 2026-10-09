#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::modules {

struct Qwen3AudioEncoderConfig {
    int64_t num_mel_bins = 128;
    int64_t encoder_layers = 0;
    int64_t encoder_attention_heads = 0;
    int64_t encoder_ffn_dim = 0;
    int64_t d_model = 0;
    int64_t max_source_positions = 0;
    int64_t n_window = 100;
    int64_t n_window_infer = 400;
    int64_t conv_chunksize = 500;
    int64_t downsample_hidden_size = 0;
    int64_t output_dim = 0;
    std::string activation_function = "gelu";
};

struct Qwen3AudioFeatures {
    std::vector<float> values;
    std::vector<int32_t> attention_mask;
    int64_t mel_bins = 0;
    int64_t frames = 0;
    int64_t encoder_tokens = 0;
};

struct Qwen3AudioEmbeddings {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

int64_t qwen3_audio_encoder_token_count(int64_t input_frames);

class Qwen3AudioEncoderGraph;
struct Qwen3AudioEncoderWeights;

class Qwen3AudioEncoderRuntime {
public:
    Qwen3AudioEncoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        Qwen3AudioEncoderConfig config,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type,
        bool hf_transformers_layout = false);
    ~Qwen3AudioEncoderRuntime();

    Qwen3AudioEmbeddings encode(const Qwen3AudioFeatures & features);

private:
    Qwen3AudioEncoderConfig config_;
    std::shared_ptr<const Qwen3AudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Qwen3AudioEncoderGraph> graph_;
};

}  // namespace engine::modules
