#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::midi {

struct SheetSage2DecoderConfig {
    int64_t vocab_size = 31678;
    int64_t hidden_size = 512;
    int64_t encoder_hidden_size = 1024;
    int64_t intermediate_size = 2048;
    int64_t decoder_layers = 6;
    int64_t num_attention_heads = 8;
    int64_t max_position_embeddings = 5120;
    int64_t pad_token_id = 1;
    float layer_norm_eps = 1.0e-5F;
};

struct SheetSage2DecoderRuntimeOptions {
    size_t graph_arena_bytes = 1536ull * 1024ull * 1024ull;
    size_t weight_context_bytes = 1024ull * 1024ull * 1024ull;
    assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native;
};

class SheetSage2DecoderRuntime {
public:
    SheetSage2DecoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution,
        SheetSage2DecoderConfig config = {},
        SheetSage2DecoderRuntimeOptions options = {});
    ~SheetSage2DecoderRuntime();

    SheetSage2DecoderRuntime(const SheetSage2DecoderRuntime &) = delete;
    SheetSage2DecoderRuntime & operator=(const SheetSage2DecoderRuntime &) = delete;
    SheetSage2DecoderRuntime(SheetSage2DecoderRuntime &&) noexcept;
    SheetSage2DecoderRuntime & operator=(SheetSage2DecoderRuntime &&) noexcept;

    std::vector<float> decode_logits(
        const std::vector<float> & mixed_encoder_state,
        int64_t memory_steps,
        const std::vector<int32_t> & decoder_input_ids);

    void prepare(int64_t batch, int64_t memory_steps, int64_t decoder_steps);
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::midi
