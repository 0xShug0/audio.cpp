#pragma once

#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeTokenizer {
public:
    explicit VoxtralRealtimeTokenizer(std::shared_ptr<const VoxtralRealtimeAssets> assets);
    ~VoxtralRealtimeTokenizer();

    VoxtralRealtimePrompt build_transcription_prompt(int64_t audio_samples, bool streaming) const;
    std::string decode(const std::vector<int32_t> & token_ids) const;
    bool is_stream_text_token(int32_t token_id) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_realtime
