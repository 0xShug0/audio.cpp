#pragma once

#include "engine/community_models/fish_drama3/prompt.h"

#include <functional>

namespace engine::models::fish_drama3 {

struct Drama3Sampling {
    float temperature = 1.0F;
    float top_p = 0.9F;
    int top_k = 30;
    int64_t max_frames = 1024;
    uint32_t seed = 0;
};

struct Drama3Generation {
    codecs::FishDacCodes codes;
    runtime::AudioBuffer audio;
    bool reached_limit = false;
};

class Drama3Generator {
public:
    Drama3Generator(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution,
                    const runtime::SessionOptions & options);
    ~Drama3Generator();
    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    Drama3Generation generate(const std::string & text, const std::vector<Drama3Take> & references,
        const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
        const std::function<void(const runtime::AudioBuffer &)> & on_audio = {});

private:
    std::shared_ptr<const Drama3Assets> assets_;
    fish_audio::FishAudioTextTokenizer tokenizer_;
    int64_t cache_steps_;
    std::unique_ptr<Drama3BackboneDecoderRuntime> backbone_;
    std::unique_ptr<Drama3Tower> tower_;
    std::unique_ptr<core::ExecutionContext> codec_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> codec_;
};

}  // namespace engine::models::fish_drama3
