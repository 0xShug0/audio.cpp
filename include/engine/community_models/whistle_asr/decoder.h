#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/encoder.h"
#include "engine/framework/core/execution_context.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::whistle_asr {

constexpr size_t kWhistleMaximumTextTokens = 320;
constexpr int64_t kWhistleVocabulary = 8199;

// Autoregressive decoder as one persistent GGML step graph. Each step feeds one
// token. Self-attention keys and values, and the history of raw projections for the
// three-tap query/key/value mixing, stay on the backend between steps. The n-gram
// engram tables stay on the host because their lookup hashes the token history.
// The host gathers the rows and uploads them each step.
//
// The runtime gives the decoder its own single-thread CPU execution context.
// A step is thousands of small graph nodes, so its cost is per-node overhead.
// Measured on the 14 s sample (42 steps): the decoder took 134-140 ms on one CPU
// thread, 550-560 ms on Metal, and 1014-1111 ms with 2 CPU threads, because
// each node synchronizes the threads. The encoder keeps the session backend and
// thread count.
class WhistleDecoderRuntime {
public:
    WhistleDecoderRuntime(std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context);
    ~WhistleDecoderRuntime();
    WhistleDecoderRuntime(const WhistleDecoderRuntime &) = delete;
    WhistleDecoderRuntime & operator=(const WhistleDecoderRuntime &) = delete;

    // Uploads the encoder's cross-attention keys and values and clears the state of
    // the previous request.
    void start(const WhistleEncoderOutput & encoder);

    // Runs the decoder at position tokens.size() - 1 with tokens.back() as input and
    // returns the logits over the full vocabulary. Positions must be consecutive
    // from 0 after start().
    [[nodiscard]] const std::vector<float> & step(const std::vector<int32_t> & tokens);

    // Writes the accumulated step timings of the current request to the timing log.
    void log_timings() const;

private:
    class Graph;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::community_models::whistle_asr
