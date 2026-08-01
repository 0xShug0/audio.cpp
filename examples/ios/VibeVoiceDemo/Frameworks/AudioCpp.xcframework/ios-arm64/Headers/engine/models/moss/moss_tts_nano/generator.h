#pragma once

#include "engine/models/moss/moss_tts_nano/global_transformer.h"
#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"
#include "engine/models/moss/moss_tts_nano/types.h"

namespace engine::models::moss_tts_nano {

class MossTTSNanoGenerator {
public:
    MossTTSNanoGenerator(
        MossTTSNanoGlobalTransformerRuntime & global_transformer,
        MossTTSNanoLocalFrameDecoderRuntime & local_frame_decoder);

    MossTTSNanoAudioCodes generate(const MossTTSNanoPrompt & prompt, const MossTTSNanoGenerationOptions & options);

private:
    MossTTSNanoGlobalTransformerRuntime & global_transformer_;
    MossTTSNanoLocalFrameDecoderRuntime & local_frame_decoder_;
};

}  // namespace engine::models::moss_tts_nano
