#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <string>

namespace engine::community_models::whistle_asr {

struct WhistleTranscript {
    std::string text;
    std::string language;
};

class WhistleWeights;

class WhistleRuntime {
public:
    explicit WhistleRuntime(std::shared_ptr<const WhistleAssets> assets, int threads = 1);
    ~WhistleRuntime();
    WhistleRuntime(const WhistleRuntime &) = delete;
    WhistleRuntime & operator=(const WhistleRuntime &) = delete;

    [[nodiscard]] WhistleTranscript transcribe(
        const runtime::AudioBuffer & audio, const std::string & language = "") const;

private:
    std::shared_ptr<const WhistleAssets> assets_;
    std::unique_ptr<WhistleWeights> weights_;
    int threads_;
};

}  // namespace engine::community_models::whistle_asr
