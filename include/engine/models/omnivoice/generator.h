#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/types.h"

#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace engine::models::omnivoice {

struct OmniVoiceQwen3DiffusionRuntimeStats {
    bool graph_rebuilt = false;
    int64_t batch_capacity = 0;
    int64_t total_token_capacity = 0;
    int64_t target_frame_capacity = 0;
    double rebuild_ms = 0.0;
    double rebuild_clear_ms = 0.0;
    double rebuild_build_ms = 0.0;
    double rebuild_alloc_ms = 0.0;
    double rebuild_init_ms = 0.0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
};

enum class OmniVoiceGeneratorPerfMode {
    Standard,
    FlashAttention,
};

class OmniVoiceQwen3DiffusionRuntime {
public:
    OmniVoiceQwen3DiffusionRuntime(
        std::shared_ptr<const OmniVoiceAssets> assets,
        core::ExecutionContext & execution_context,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type,
        bool mem_saver,
        OmniVoiceGeneratorPerfMode perf_mode,
        bool packed_cfg = true,
        bool device_scoring = true,
        bool packed_projections = true);
    ~OmniVoiceQwen3DiffusionRuntime();

    // One request, drawing from the runtime RNG (see seed_rng()).
    OmniVoiceGeneratedAudioTokens generate(
        const OmniVoicePrompt & prompt,
        const OmniVoiceGenerationOptions & options);
    // Several requests in one batch: every decoding step runs one forward
    // pass for all of them. Request i draws only from rngs[i], so a request
    // seeded with S gives the same tokens as generate() after seed_rng(S),
    // up to floating-point differences of the larger batched kernels.
    // Requests may differ in length, guidance scale and step count.
    std::vector<OmniVoiceGeneratedAudioTokens> generate_batch(
        const std::vector<OmniVoicePrompt> & prompts,
        const std::vector<OmniVoiceGenerationOptions> & options,
        std::vector<std::mt19937> & rngs);
    // RNG for one batched request: seeded with `seed`, or from the runtime
    // RNG when no seed is given.
    std::mt19937 make_request_rng(const std::optional<uint32_t> & seed);
    void release_runtime_graphs();
    void seed_rng(uint32_t seed);
    const OmniVoiceQwen3DiffusionRuntimeStats & last_stats() const noexcept;

private:
    std::vector<OmniVoiceGeneratedAudioTokens> generate_items(
        const std::vector<OmniVoicePrompt> & prompts,
        const std::vector<OmniVoiceGenerationOptions> & options,
        const std::vector<std::mt19937 *> & rngs);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::omnivoice
