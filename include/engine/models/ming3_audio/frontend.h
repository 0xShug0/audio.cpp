#pragma once

#include "engine/models/ming3_audio/assets.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/tokenizers/llama_bpe.h"

namespace engine::models::ming3_audio {

struct AudioFeatures {
    std::vector<float> log_mel;
    int64_t frames = 0;
};

class Frontend {
public:
    explicit Frontend(std::shared_ptr<const MingAudioAssets> assets);

    AudioFeatures extract(const runtime::AudioBuffer & audio, size_t threads) const;
    std::vector<int32_t> prompt(const std::string & instruction, int64_t mel_frames = 0,
        const std::string & system_prompt = {}, bool enable_thinking = false) const;
    std::string decode(const std::vector<int32_t> & ids) const;

private:
    std::shared_ptr<const MingAudioAssets> assets_;
    tokenizers::LlamaBpeTokenizer tokenizer_;
    audio::WhisperLogMelExtractor mel_;
};

}  // namespace engine::models::ming3_audio
