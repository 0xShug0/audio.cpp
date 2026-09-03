#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/models/chatterbox/components.h"
#include "engine/models/chatterbox/s3gen_flow.h"
#include "engine/models/chatterbox/s3gen_inference.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::community_models::chatterbox_turbo {

// Loads and runs Chatterbox Turbo's S3Gen half (speech tokens -> waveform) by delegating to the
// existing chatterbox family's flow encoder/decoder and HiFT vocoder code (architecturally
// identical apart from the meanflow decoder branch already added to
// engine::models::chatterbox::S3FlowDecoderWeights). The native, repacked GGUF (see
// tools/community_models/chatterbox_turbo/repack_chatterbox_turbo_gguf.py) stores S3Gen tensors
// under the same names those loaders already expect, so no name translation is needed here.
//
// MVP scope: only the built-in default voice baked into the T3 GGUF's `conds.gen.*` tensors is
// supported (no custom voice cloning yet) -- that path needs the turbo checkpoint's `s3.se`
// (ResNet-style speaker encoder, not CAMPPlus) and `s3.tok` (S3 speech tokenizer) sections, which
// are unverified and out of scope for this pass. See ChatterboxTurboAssets.
class ChatterboxTurboS3Gen {
public:
    static std::shared_ptr<ChatterboxTurboS3Gen> load(
        std::shared_ptr<const engine::assets::TensorSource> s3gen_source,
        const engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

    // speech_tokens: T3 output (S3 codebook ids, < 6561; caller strips control tokens).
    engine::models::chatterbox::S3GenInferenceOutputs synthesize(
        const engine::models::chatterbox::EmbedReferenceOutputs & ref_dict,
        const std::vector<int32_t> & speech_tokens,
        uint64_t flow_seed,
        uint64_t vocoder_seed) const;

private:
    std::shared_ptr<const engine::models::chatterbox::S3FlowEncoderWeights> encoder_weights_;
    std::shared_ptr<const engine::models::chatterbox::S3FlowDecoderWeights> decoder_weights_;
    std::shared_ptr<engine::modules::HiftVocoderComponent> vocoder_;
    mutable engine::models::chatterbox::S3GenSessionCache cache_;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
};

}  // namespace engine::community_models::chatterbox_turbo
