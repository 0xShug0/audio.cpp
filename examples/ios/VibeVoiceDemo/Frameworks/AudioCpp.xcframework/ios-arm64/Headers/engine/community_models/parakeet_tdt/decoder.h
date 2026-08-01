#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/encoder.h"
#include "engine/community_models/parakeet_tdt/weights.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct ParakeetDecodeOptions {
    int64_t max_tokens = 0;
    bool keep_language_tags = false;
};

struct ParakeetDecodedText {
    std::string text;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> token_frame_indices;
    std::vector<int32_t> durations;
    std::vector<runtime::WordTimestamp> word_timestamps;
};

class ParakeetDecoderRuntime {
public:
    ParakeetDecoderRuntime(
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const ParakeetWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~ParakeetDecoderRuntime();

    void prepare();

    ParakeetDecodedText decode(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options);

    void reset_state();
    ParakeetDecodedText decode_incremental(
        const ParakeetEncodedAudio & encoded,
        const ParakeetDecodeOptions & options,
        int64_t frame_offset = 0);
    ParakeetDecodedText format_tokens(
        std::vector<int32_t> token_ids,
        std::vector<int32_t> token_frame_indices,
        std::vector<int32_t> durations,
        const ParakeetDecodeOptions & options,
        int64_t audio_end_frame) const;

    int32_t run_step(int32_t input_token, const float * encoder_frame, bool decoder_cache_valid, int32_t * out_dur_id = nullptr);
    int32_t run_joint_step(const float * encoder_frame, int32_t * out_dur_id = nullptr);

private:
    struct StepGraph;
    struct JointGraph;

    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const ParakeetWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<StepGraph> step_graph_;
    std::unique_ptr<JointGraph> joint_graph_;
    std::vector<float> logits_scratch_;
    std::vector<float> hidden_scratch_;
    std::vector<float> cell_scratch_;
    std::vector<float> decoder_cache_scratch_;
    std::vector<float> hidden_read_scratch_;
    std::vector<float> cell_read_scratch_;
    int32_t pending_input_token_ = 0;
    bool predictor_cache_valid_ = false;
    bool state_initialized_ = false;

    void ensure_step_graph();
    void ensure_joint_graph();
    std::string decode_text(const std::vector<int32_t> & token_ids, bool keep_language_tags) const;
    std::vector<runtime::WordTimestamp> build_word_timestamps(
        const std::vector<int32_t> & token_ids,
        const std::vector<int32_t> & token_frame_indices,
        const std::vector<int32_t> & durations,
        int64_t audio_end_frame) const;
};

}  // namespace engine::community_models::parakeet_tdt
