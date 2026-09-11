#pragma once

#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/tokenizer_text.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <memory>

namespace engine::models::moss_tts_nano {

class MossTTSNanoPromptBuilder {
public:
    MossTTSNanoPromptBuilder(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        const MossTTSNanoTextTokenizer & tokenizer);

    MossTTSNanoPrompt build(const MossTTSNanoRequest & request, const MossTTSNanoAudioCodes * prompt_codes) const;

private:
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    const MossTTSNanoTextTokenizer & tokenizer_;
};

}  // namespace engine::models::moss_tts_nano
