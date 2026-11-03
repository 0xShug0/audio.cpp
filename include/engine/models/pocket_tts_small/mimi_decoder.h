#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::pocket_tts_small {

struct Assets;
struct BackendWeights;

struct MimiDecoderConfig {
    int64_t latent_size = 32;
    int64_t hidden_size = 128;
    int64_t num_heads = 4;
    int64_t intermediate_size = 384;
    int64_t transformer_layers = 12;
    int64_t encoder_upsample_stride = 16;
    int64_t context = 50;
    int64_t base_filters = 32;
    int64_t input_kernel_size = 3;
};

class MimiDecoder {
public:
    explicit MimiDecoder(MimiDecoderConfig config = {});
    ~MimiDecoder();

    const MimiDecoderConfig & config() const noexcept;

    std::vector<float> decode(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & latents,
        int64_t steps,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes,
        int64_t chunk_frames,
        int64_t stage2_chunk_frames,
        bool use_full_sequence_path) const;

    void reset_streaming_state() const;

    std::vector<float> decode_streaming(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & latent,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes) const;

    void clear_runtime_cache() const noexcept;

private:
    struct RuntimeCache;
    struct StreamingState;
    MimiDecoderConfig config_;
    mutable std::unique_ptr<RuntimeCache> runtime_cache_;
    mutable std::unique_ptr<StreamingState> streaming_state_;
};

struct Assets;
struct BackendWeights;

class AudioDecoder {
public:
    explicit AudioDecoder(MimiDecoderConfig config = {}, int fade_in_samples = 0);

    std::vector<float> decode(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & normalized_latents,
        int64_t steps,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes,
        int64_t chunk_frames,
        int64_t stage2_chunk_frames,
        bool use_full_sequence_path) const;

    void reset_streaming_state() const;

    std::vector<float> decode_streaming_step(
        ggml_backend_t backend,
        int threads,
        const Assets & manifest,
        const BackendWeights & weights,
        const std::vector<float> & normalized_latent,
        size_t conv_graph_context_bytes,
        size_t transformer_graph_context_bytes,
        size_t tail_graph_context_bytes) const;

    void clear_runtime_cache() const noexcept;

private:
    MimiDecoder decoder_;
    std::vector<float> fade_in_;
    mutable bool first_stream_chunk_ = true;
};

}  // namespace engine::models::pocket_tts_small
