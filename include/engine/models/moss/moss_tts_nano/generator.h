#pragma once

#include "engine/models/moss/moss_tts_nano/global_transformer.h"
#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <functional>

namespace engine::models::moss_tts_nano {

class MossTTSNanoGenerator {
public:
    MossTTSNanoGenerator(
        MossTTSNanoGlobalTransformerRuntime & global_transformer,
        MossTTSNanoLocalFrameDecoderRuntime & local_frame_decoder);

    MossTTSNanoAudioCodes generate(
        const MossTTSNanoPrompt & prompt,
        const MossTTSNanoGenerationOptions & options,
        const std::function<void(const std::vector<int32_t> &)> & on_frame = {});

private:
    MossTTSNanoGlobalTransformerRuntime & global_transformer_;
    MossTTSNanoLocalFrameDecoderRuntime & local_frame_decoder_;
    // Previous prompt, kept so the (identical) template prefix K/V can be reused instead of
    // being forwarded again on the next utterance.
    std::vector<int32_t> previous_prompt_ids_;
    int64_t previous_prompt_rows_ = 0;
    int64_t previous_row_width_ = 0;
};

}  // namespace engine::models::moss_tts_nano
