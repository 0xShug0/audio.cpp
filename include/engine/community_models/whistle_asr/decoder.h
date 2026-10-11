#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/encoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::whistle_asr {

constexpr size_t kWhistleMaximumTextTokens = 320;
constexpr int64_t kWhistleVocabulary = 8199;

// Host autoregressive decoder. Self-attention caches and three-tap projection
// history are request-local; immutable decoder weights are retained across requests.
class WhistleDecoderRuntime {
public:
    explicit WhistleDecoderRuntime(std::shared_ptr<const WhistleAssets> assets);
    ~WhistleDecoderRuntime();
    WhistleDecoderRuntime(const WhistleDecoderRuntime &) = delete;
    WhistleDecoderRuntime & operator=(const WhistleDecoderRuntime &) = delete;

    // Takes ownership of the encoder's cross-attention outputs and resets request state.
    void start(WhistleEncoderOutput encoder);

    // Runs the decoder at position tokens.size() - 1 with tokens.back() as input and
    // returns the logits over the full vocabulary. Positions must be consecutive
    // from 0 after start().
    [[nodiscard]] const std::vector<float> & step(const std::vector<int32_t> & tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::community_models::whistle_asr
