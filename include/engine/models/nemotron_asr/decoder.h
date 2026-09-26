#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/nemotron_asr/assets.h"
#include "engine/models/nemotron_asr/encoder.h"
#include "engine/models/nemotron_asr/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::nemotron_asr {

struct NemotronDecodeOptions {
    int64_t max_tokens = 0;
    bool keep_language_tags = false;
};

struct NemotronDecodedText {
    std::string text;
    std::vector<int32_t> token_ids;
    std::vector<int32_t> durations;
    std::vector<runtime::WordTimestamp> token_timestamps;
};

// RNNT prediction-network state carried between decoder steps.
struct NemotronPredictorState {
    std::vector<float> hidden;         // [layers, hidden]
    std::vector<float> cell;           // [layers, hidden]
    std::vector<float> decoder_cache;  // projected prediction output [hidden]
};

struct NemotronDecoderStreamState {
    NemotronDecodeOptions options;
    NemotronDecodedText decoded;
    int64_t encoded_frames = 0;
    int64_t symbols_at_frame = 0;
    int32_t input_token = 0;
    bool decoder_cache_initialized = false;
    NemotronPredictorState predictor;
};

class NemotronDecoderRuntime {
public:
    NemotronDecoderRuntime(
        std::shared_ptr<const NemotronASRAssets> assets,
        std::shared_ptr<const NemotronWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~NemotronDecoderRuntime();

    void prepare();
    NemotronDecodedText decode(const NemotronEncodedAudio & encoded, const NemotronDecodeOptions & options);
    NemotronDecoderStreamState make_stream_state(const NemotronDecodeOptions & options);
    void decode_stream_chunk(
        const NemotronEncodedAudio & encoded,
        NemotronDecoderStreamState & state);
    NemotronDecodedText stream_result(const NemotronDecoderStreamState & state) const;
    // NeMo Hypothesis views for the speaker-tagged segment builder: the decoded
    // text, and the cumulative encoder frame of every emitted (non-blank) token.
    std::string stream_text(const NemotronDecoderStreamState & state, bool keep_language_tags) const;
    std::vector<int64_t> stream_token_frames(const NemotronDecoderStreamState & state) const;
private:
    struct Graph;
    struct JointGraph;

    void ensure_graph();
    void ensure_joint_graph();
    NemotronPredictorState initial_predictor_state() const;
    int32_t run_step(
        int32_t input_token, const float * encoder_frame, bool decoder_cache_initialized, NemotronPredictorState & predictor);
    int32_t run_joint_step(const float * encoder_frame, const NemotronPredictorState & predictor);
    std::string decode_text(const std::vector<int32_t> & token_ids, bool keep_language_tags) const;

    std::shared_ptr<const NemotronASRAssets> assets_;
    std::shared_ptr<const NemotronWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<JointGraph> joint_graph_;
    std::vector<float> encoder_frame_scratch_;
    std::vector<float> logits_scratch_;
    std::vector<float> hidden_read_scratch_;
    std::vector<float> cell_read_scratch_;
    int32_t unk_token_id_ = -1;
};

}  // namespace engine::models::nemotron_asr
