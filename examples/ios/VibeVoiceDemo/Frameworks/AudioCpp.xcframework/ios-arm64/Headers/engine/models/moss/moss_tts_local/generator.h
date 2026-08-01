#pragma once

#include "engine/models/moss/moss_tts_local/assets.h"
#include "engine/models/moss/moss_tts_local/backbone.h"
#include "engine/models/moss/moss_tts_local/depth_transformer.h"
#include "engine/models/moss/shared/token_rows.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::moss_tts_local {

// Sampling controls for the autoregressive audio-token loop. The audio_* fields drive
// the RVQ codebook heads and the text_* fields drive the binary assistant/end gate,
// mirroring the arguments of MossTTSLocalModel.generate.
struct MossGenerationOptions {
    int64_t max_new_frames = 4096;
    bool do_sample = true;
    float audio_temperature = 1.7F;
    float audio_top_p = 0.8F;
    int audio_top_k = 25;
    float audio_repetition_penalty = 1.0F;
    float text_temperature = 1.0F;
    float text_top_p = 1.0F;
    int text_top_k = 50;
    uint64_t seed = 0;
};

// Drives the MOSS-TTS-Local generation loop: for every frame it runs the Qwen3 backbone
// over the decoder sequence, then walks the depth transformer to emit one RVQ code per
// codebook. Generation stops when the binary gate selects the audio-end token.
class MossGenerator {
public:
    MossGenerator(
        std::shared_ptr<const MossTTSLocalAssets> assets,
        core::ExecutionContext & execution_context,
        size_t projection_graph_arena_bytes,
        size_t projection_weight_context_bytes,
        const MossBackboneRuntime & backbone,
        const MossDepthTransformer & depth);
    ~MossGenerator();

    MossGenerator(const MossGenerator &) = delete;
    MossGenerator & operator=(const MossGenerator &) = delete;

    // text_tokens holds the text channel (input_ids[..., 0]) and audio_codes holds the
    // n_vq audio channels flattened row-major as [seq, n_vq] (input_ids[..., 1:], with
    // audio_pad_token_id where a position carries no code). Returns the generated frames
    // as [num_frames][n_vq] audio codes.
    std::vector<std::vector<int32_t>> generate(
        const std::vector<int32_t> & text_tokens,
        const std::vector<int32_t> & audio_codes,
        const MossGenerationOptions & options) const;

private:
    std::shared_ptr<const MossTTSLocalAssets> assets_;
    const MossBackboneRuntime & backbone_;
    const MossDepthTransformer & depth_;
    int64_t hidden_size_ = 0;
    int64_t num_codebooks_ = 0;
    std::unique_ptr<moss::AudioCodebookEmbeddings> audio_codebooks_;
    std::vector<float> local_text_head_;                // [2 * hidden]
    engine::sampling::TorchCudaSamplingPolicy sampling_policy_;
    struct ProjectionRuntime;
    std::unique_ptr<ProjectionRuntime> projection_;
};

}  // namespace engine::models::moss_tts_local
