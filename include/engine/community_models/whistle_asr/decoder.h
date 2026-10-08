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
// token; self-attention keys/values and the raw projection history for the
// three-tap query/key/value mixing stay on the backend between steps. The n-gram
// engram tables stay on the host because their lookup is integer hashing over the
// token history.
//
// The runtime gives the decoder its own single-thread CPU execution context. A step
// is about 2600 small graph nodes for one token, so it is bound by per-node cost:
// on Metal each step took about 6.5 ms against 3.3 ms on one CPU thread, and
// extra CPU threads only add per-node synchronization (2 threads roughly tripled
// the step time). The encoder keeps the session backend and thread count.
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
