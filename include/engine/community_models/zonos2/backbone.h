#pragma once

#include "engine/community_models/zonos2/assets.h"
#include "engine/community_models/zonos2/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/moe_engine/moe_engine.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos2 {

struct Zonos2BackboneOptions {
    moe::EngineConfig moe;
};

// The ZONOS2 28-layer GQA backbone: dense SwiGLU and EDA-routed MoE layers over a pre-norm residual stream, one
// sequence slot with an F16 KV cache.
class Zonos2Backbone {
public:
    Zonos2Backbone(std::shared_ptr<const Zonos2Assets> assets, core::ExecutionContext & execution,
                   Zonos2BackboneOptions options = {});
    ~Zonos2Backbone();
    Zonos2Backbone(const Zonos2Backbone &) = delete;
    Zonos2Backbone & operator=(const Zonos2Backbone &) = delete;

    // Starts a generation of at most `capacity_rows` positions: grows the KV cache when needed, then zeroes it and
    // invalidates the expert cache.
    void begin(int capacity_rows);
    // Runs the whole prompt in one graph; the speaker projection replaces prompt.speaker_row's embedding.
    // `speaker` is [spk_dim] and required exactly when prompt.speaker_row >= 0. `logits` receives the last row
    // [n_codebooks * audio_vocab], post-softcap.
    void prefill(const Zonos2Prompt & prompt, const float * speaker, std::vector<float> & logits);
    // One decode row [n_codebooks + 1]; `logits` receives [n_codebooks * audio_vocab].
    void step(const int32_t * row, std::vector<float> & logits);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos2
