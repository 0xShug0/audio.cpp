#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace engine::core { class ConstantTensorCache; }

namespace engine::codecs {

// Row-major [frames, code_groups] codes for the Qwen3 12.5 Hz audio tokenizer.
struct Qwen3SpeechCodes {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t code_groups = 0;
};

struct Qwen3AudioTokenizerDecoderConfig {
    int64_t codebook_size = 0;
    int64_t codebook_dim = 0;
    int64_t latent_dim = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t decoder_dim = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t num_layers = 0;
    int64_t num_quantizers = 0;
    int64_t num_semantic_quantizers = 1;
    int64_t sliding_window = 0;
    int64_t head_dim = 0;
    float rope_theta = 10000.0F;
    float rms_norm_eps = 1.0e-5F;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsampling_ratios;
};

Qwen3AudioTokenizerDecoderConfig parse_qwen3_audio_tokenizer_decoder_config(const io::json::Value & root);

struct Qwen3SpeechTokenizerEncoderWeights;
class Qwen3SpeechTokenizerEncoderGraph;

// Session-owned encoder graph and weights; execution_context must outlive it.
class Qwen3SpeechTokenizerEncoderRuntime {
public:
    Qwen3SpeechTokenizerEncoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        bool flash_attention);
    ~Qwen3SpeechTokenizerEncoderRuntime();

    Qwen3SpeechCodes encode(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const Qwen3SpeechTokenizerEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    bool flash_attention_ = false;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerEncoderGraph> graph_;
};

struct Qwen3SpeechTokenizerDecoderWeights;
class Qwen3SpeechTokenizerDecoderGraph;
class Qwen3SpeechDecoderStreamState;

// Owns offline/windowed/stateful graphs and stream state for one session.
// execution_context must outlive the runtime; reset_stream() starts an utterance.
class Qwen3SpeechTokenizerDecoderRuntime {
public:
    static constexpr int kSampleRate = 24000;
    static constexpr int64_t kSamplesPerFrame = 1920;
    static constexpr int64_t kLeftContextFrames = 25;
    Qwen3SpeechTokenizerDecoderRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        const Qwen3AudioTokenizerDecoderConfig & config,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        bool flash_attention,
        bool stateful_stream = false);
    ~Qwen3SpeechTokenizerDecoderRuntime();

    runtime::AudioBuffer decode(const Qwen3SpeechCodes & codec_codes) const;
    runtime::AudioBuffer decode_stream_chunk(
        const Qwen3SpeechCodes & codec_codes,
        int64_t context_frames,
        int64_t frames_per_chunk) const;

    // Clear stateful convolution and attention history between utterances.
    void reset_stream() const;

private:
    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights_;
    size_t graph_arena_bytes_ = 0;
    bool flash_attention_ = false;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    std::unique_ptr<Qwen3SpeechDecoderStreamState> stream_state_;
    mutable int64_t stream_graph_capacity_ = 0;
    mutable std::map<int64_t, std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph>> stateful_graphs_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> graph_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> stream_graph_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> stream_initial_graph_;
    // Always present to keep this public class layout identical when the private
    // Strix Halo compile definition differs between translation units.
    mutable std::array<std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph>, 2> optimized_graphs_;
};

}  // namespace engine::codecs
