#pragma once

#include "engine/community_models/voice_isolator/backbone.h"
#include "engine/community_models/voice_isolator/config.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>

// The ggml adapter behind Backbone: the checkpoint's weights on a backend and
// the graphs that run them.
namespace engine::models::voice_isolator {

struct NetworkOptions {
    // Storage of the large projection matrices; norms, biases and the scan stay F32.
    // Native: BF16 on CPUs with AVX512-BF16, F32 elsewhere.
    assets::TensorStorageType weight_storage = assets::TensorStorageType::Native;
    // Streaming attention keys and values: F32, F16, Q8_0, Q5_0 or Q4_0; Native picks per backend.
    assets::TensorStorageType attention_cache_storage = assets::TensorStorageType::Native;
    // Streams rotate their cached keys back once rotary positions pass this
    // many frames (41 s), keeping the rotations exact in F32.
    int64_t rebase_frames = 4096;
};

class Network {
public:
    // The execution context is the session's and must outlive the network.
    Network(const Config & config, const assets::TensorSource & source, core::ExecutionContext & execution,
        const NetworkOptions & options);
    ~Network();

    Network(const Network &) = delete;
    Network & operator=(const Network &) = delete;

    const Config & config() const noexcept;
    const BandLayout & layout() const noexcept;

    // Two-sided checkpoints: each run is one whole chunk.
    std::unique_ptr<Backbone> offline_backbone() const;
    // Causal checkpoints: runs of up to block_frames frames carry the stream's
    // state, at one of the checkpoint's trained look-ahead positions.
    std::unique_ptr<Backbone> streaming_backbone(int64_t lookahead_frames, int64_t block_frames) const;

    struct State;

private:
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::voice_isolator
