#pragma once

#include "engine/community_models/fish_drama3/prompt.h"
#include "engine/framework/runtime/streaming_audio.h"

#include <functional>
#include <optional>

namespace engine::models::fish_drama3 {

struct Drama3Sampling {
    float temperature = 1.0F;
    float top_p = 0.9F;
    int top_k = 4097;
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
                    const runtime::SessionOptions & options, bool streaming);
    ~Drama3Generator();
    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    void warm_streaming_codec(const codecs::FishDacCodes & reference);
    Drama3Generation generate(const std::string & text, const std::vector<Drama3Take> & references,
        const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
        const std::function<void(const runtime::AudioBuffer &)> & on_audio = {}, bool continuation = false);

private:
    std::shared_ptr<const Drama3Assets> assets_;
    Drama3TextTokenizer tokenizer_;
    int64_t cache_steps_;
    bool profile_first_audio_ = false;
    bool disk_experts_ = false;
    bool device_sampling_ = false;
    size_t codec_batch_frames_ = 4;
    bool codec_async_decode_ = true;
    runtime::StreamingAudioController<int32_t> streaming_audio_;
    std::unique_ptr<Drama3BackboneDecoderRuntime> backbone_;
    std::unique_ptr<Drama3BackboneDecoderRuntime::DecodeSession> decoder_;
    std::vector<Drama3Frame> cached_prefix_;
    std::vector<float> cached_hidden_;
    std::unique_ptr<Drama3Tower> tower_;
    std::unique_ptr<core::ExecutionContext> codec_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> codec_;
    std::optional<runtime::AudioBuffer> cached_reference_audio_;
    codecs::FishDacCodes cached_reference_codes_;
};

}  // namespace engine::models::fish_drama3
