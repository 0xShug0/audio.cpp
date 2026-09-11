#pragma once

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::moss_voicegen {

// The delay family reads 1 + n_vq heads off the same backbone hidden state each step:
// lm_heads.0 predicts the next text token, lm_heads.1..n_vq predict one RVQ code each.
struct MossVoiceGenStepLogits {
    std::vector<float> text;                      // [text_vocab_size]
    std::vector<std::vector<float>> audio;        // n_vq x [audio_vocab_size + 1]
};

class MossVoiceGenHeadsRuntime {
public:
    MossVoiceGenHeadsRuntime(
        std::shared_ptr<const MossVoiceGenAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~MossVoiceGenHeadsRuntime();

    MossVoiceGenHeadsRuntime(const MossVoiceGenHeadsRuntime &) = delete;
    MossVoiceGenHeadsRuntime & operator=(const MossVoiceGenHeadsRuntime &) = delete;

    // Evaluates every head for one position. The graph is built on first use and reused.
    void evaluate(const std::vector<float> & hidden_state, MossVoiceGenStepLogits & out) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_voicegen
