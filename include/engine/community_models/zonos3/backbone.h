#pragma once

#include "engine/community_models/zonos3/assets.h"
#include "engine/community_models/zonos3/prompt.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/moe_resources.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::zonos3 {

struct Zonos3BackboneOptions {
    core::MoeExecutionPolicy moe{core::MoeExpertResidency::Backend, false, 16, 1, {}, 0};
    // Parity control: compare staged and resident experts with identical plain kernels.
    bool repack_experts = true;
};

// The ZONOS3 GDN / global-attention latent-MoE backbone with Full AttnRes. One or two sequence slots (the second is
// the speaker-CFG twin) share fixed-capacity per-slot KV caches and GDN states.
class Zonos3Backbone {
public:
    Zonos3Backbone(std::shared_ptr<const Zonos3Assets> assets, core::ExecutionContext & execution,
                   Zonos3BackboneOptions options = {});
    ~Zonos3Backbone();
    Zonos3Backbone(const Zonos3Backbone &) = delete;
    Zonos3Backbone & operator=(const Zonos3Backbone &) = delete;

    // Starts a generation: (re)allocates state for `slots` (1|2) sequences of `capacity_rows` positions when it
    // grows, then zeroes every cache and state.
    void begin(int slots, int capacity_rows);
    // Prefills `slot` with the prompt in 256-row chunks; `logits` receives the last row [n_codebooks * audio_vocab].
    void prefill(int slot, const Zonos3Prompt & prompt, std::vector<float> & logits);
    // One decode row [n_codebooks + 1] per active slot (slots 0..slots-1); `logits` receives [slots][10 * 4098].
    void step(const std::array<const int32_t *, 2> & rows, int slots, std::vector<float> & logits);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::zonos3
