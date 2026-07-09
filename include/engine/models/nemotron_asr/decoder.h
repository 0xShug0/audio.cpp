#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/nemotron_asr/assets.h"
#include "engine/models/nemotron_asr/encoder.h"
#include "engine/models/nemotron_asr/weights.h"

#include <cstddef>
#include <cstdint>
#include <functional>
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

using NemotronTextDeltaCallback = std::function<void(const std::string &)>;

class NemotronDecoderRuntime {
public:
    NemotronDecoderRuntime(
        std::shared_ptr<const NemotronAssets> assets,
        std::shared_ptr<const NemotronWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~NemotronDecoderRuntime();

    void prepare();
    NemotronDecodedText decode(const NemotronEncodedAudio & encoded, const NemotronDecodeOptions & options);
    NemotronDecodedText decode_streaming(
        const NemotronDecodeOptions & options,
        const std::function<bool(NemotronEncodedAudio &)> & next_chunk,
        const NemotronTextDeltaCallback & on_text_delta = nullptr);

private:
    struct Graph;
    struct JointGraph;

    void ensure_graph();
    void ensure_joint_graph();
    int32_t run_step(int32_t input_token, const float * encoder_frame, bool decoder_cache_initialized);
    int32_t run_joint_step(const float * encoder_frame);
    std::string decode_text(const std::vector<int32_t> & token_ids, bool keep_language_tags) const;

    std::shared_ptr<const NemotronAssets> assets_;
    std::shared_ptr<const NemotronWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Graph> graph_;
    std::unique_ptr<JointGraph> joint_graph_;
    std::vector<float> encoder_frame_scratch_;
    std::vector<float> hidden_scratch_;
    std::vector<float> cell_scratch_;
    std::vector<float> decoder_cache_scratch_;
    std::vector<float> logits_scratch_;
    std::vector<float> hidden_read_scratch_;
    std::vector<float> cell_read_scratch_;
};

}  // namespace engine::models::nemotron_asr
