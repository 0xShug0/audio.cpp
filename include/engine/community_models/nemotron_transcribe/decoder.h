#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::nemotron_transcribe {

// M = Mamba2, E = routed and shared experts, A = attention without RoPE.
// The defaults describe the 52 trunk blocks, excluding the optional MTP head.
struct DecoderConfig {
    int64_t hidden_size = 2688;
    int64_t vocab_size = 131072;
    std::string block_pattern = "MEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMAEMEMEMEMAEMEMEMEME";
    int64_t attention_heads = 32;
    int64_t attention_kv_heads = 2;
    int64_t attention_head_dim = 128;
    int64_t mamba_heads = 64;
    int64_t mamba_head_dim = 64;
    int64_t mamba_state_size = 128;
    int64_t mamba_groups = 8;
    int64_t conv_kernel = 4;
    int64_t experts = 128;
    int64_t experts_per_token = 6;
    int64_t expert_width = 1856;
    int64_t shared_expert_width = 3712;
    float routed_scaling_factor = 2.5F;
    float rms_norm_eps = 1.0e-5F;
    int64_t max_context = 65536;
    int64_t prefill_batch_size = 64;
};

struct GenerationOptions {
    int64_t max_new_tokens = 512;
    bool do_sample = false;
    float temperature = 1.0F;
    int64_t top_k = 0;
    float top_p = 1.0F;
    float repetition_penalty = 1.0F;
    uint64_t seed = 0;
    std::vector<int32_t> stop_token_ids = {11};
    // Cooperative cancellation is checked between prefill chunks and tokens.
    std::function<bool()> cancelled;
    std::function<void(int32_t)> on_token;
    // Useful for component parity checks. This copies one vocabulary vector.
    bool return_last_logits = false;
};

enum class TerminationReason {
    StopToken,
    TokenLimit,
    Cancelled,
};

struct GenerationResult {
    // Stop tokens are excluded. stop_token_id records the token that ended it.
    std::vector<int32_t> token_ids;
    TerminationReason termination = TerminationReason::TokenLimit;
    int32_t stop_token_id = -1;
    int64_t prompt_tokens = 0;
    int64_t processed_prompt_tokens = 0;
    int64_t cache_capacity = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    // Logits used for the last sampling decision, when requested.
    std::vector<float> last_logits;
};

class NemotronHDecoderRuntime {
public:
    NemotronHDecoderRuntime(std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution, DecoderConfig config = {},
        assets::TensorStorageType storage = assets::TensorStorageType::Native);
    ~NemotronHDecoderRuntime();

    NemotronHDecoderRuntime(const NemotronHDecoderRuntime &) = delete;
    NemotronHDecoderRuntime & operator=(const NemotronHDecoderRuntime &) = delete;

    // audio_embeddings is row-major [audio_frames, config.hidden_size]. The
    // prompt is before_audio, these audio rows, then after_audio. Each request
    // starts with empty attention, convolution and recurrent state. A runtime
    // belongs to one session and must not execute concurrent requests.
    // Empty prompts, invalid IDs/options and prompt+output over context throw.
    GenerationResult generate(const std::vector<int32_t> & before_audio,
        const std::vector<float> & audio_embeddings, int64_t audio_frames,
        const std::vector<int32_t> & after_audio, const GenerationOptions & options = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::nemotron_transcribe
