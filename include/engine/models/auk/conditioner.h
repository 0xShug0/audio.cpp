#pragma once

// AuK's conditioner: Qwen2.5-Omni's Thinker used as an ENCODER.
//
// cfm_edit.py:encode_text runs the Thinker forward once with output_hidden_states=True
// and never generates, so there is no decode loop, no KV cache to carry and no
// sampling -- far less than a Qwen runtime, and the reason this is tractable at all.
//
// ⚠ THE CONDITIONING IS NOT THE LAST HIDDEN STATE. It is an ELMo-style fusion: every
// one of the 36 layer outputs is LayerNorm'd with no affine parameters, weighted by a
// softmax over learned per-layer weights, summed, and scaled by a learned scalar
// (cfm_edit.py:134). Taking the last hidden state instead produces a plausible signal
// that is wrong everywhere, with no error to notice.
//
// Text-only for now: the Thinker also takes audio through a separate 32-layer tower,
// which is its own stage.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::auk {

struct AukConditionerConfig {
    int64_t hidden_size = 2048;
    int64_t layers = 36;
    int64_t attention_heads = 16;
    int64_t key_value_heads = 2;      // GQA
    int64_t head_dim = 128;
    int64_t intermediate_size = 11008;
    int64_t vocab_size = 151936;
    float rms_norm_eps = 1e-6F;
    float rope_theta = 1000000.0F;
    // F.layer_norm's default, and NOT rms_norm_eps -- the fusion's normalization is a
    // plain LayerNorm from torch, not the model's RMSNorm.
    float fusion_norm_eps = 1e-5F;

    // Native keeps the checkpoint's own dtype (bf16 for Qwen2.5-Omni); F32 upcasts at
    // load. The reference runs fp32, so this is the knob that says whether a residual
    // difference is the port or the precision.
    assets::TensorStorageType weight_storage = assets::TensorStorageType::Native;

    void validate() const;
};

// The learned fusion parameters live in AuK's checkpoint, not Qwen's: `layer_weights`
// (one per layer, pre-softmax) and `layer_scale`.
struct AukFusionParameters {
    std::vector<float> layer_weights;
    float layer_scale = 1.0F;
};

struct AukConditioning {
    std::vector<float> values;   // [tokens, hidden_size], row-major
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

class AukConditioner {
public:
    AukConditioner(
        AukConditionerConfig config,
        AukFusionParameters fusion,
        const assets::TensorSource & source,
        core::ExecutionContext & execution,
        std::string prefix = "thinker.model");
    ~AukConditioner();

    AukConditioner(const AukConditioner &) = delete;
    AukConditioner & operator=(const AukConditioner &) = delete;

    // `audio` holds the audio tower's output, one row per <|AUDIO|> token, and
    // `audio_positions` says where those tokens sit in `input_ids`. The Thinker does
    // not see audio any other way: the placeholder token's embedding is REPLACED by the
    // tower's, so a message with audio and one without differ only in these rows.
    AukConditioning encode(
        const std::vector<int32_t> & input_ids,
        const std::vector<float> & audio = {},
        const std::vector<int64_t> & audio_positions = {});

    int64_t loaded_tensor_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::auk
