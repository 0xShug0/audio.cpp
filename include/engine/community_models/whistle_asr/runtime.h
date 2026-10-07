#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/encoder.h"
#include "engine/community_models/whistle_asr/frontend.h"
#include "engine/framework/core/execution_context.h"
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
    WhistleRuntime(std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context);
    ~WhistleRuntime();
    WhistleRuntime(const WhistleRuntime &) = delete;
    WhistleRuntime & operator=(const WhistleRuntime &) = delete;

    [[nodiscard]] WhistleTranscript transcribe(
        const runtime::AudioBuffer & audio, const std::string & language = "");

private:
    std::shared_ptr<const WhistleAssets> assets_;
    WhistleEncoderRuntime encoder_;
    WhistleFrontend frontend_;
    std::unique_ptr<WhistleWeights> weights_;
    int threads_;
};

}  // namespace engine::community_models::whistle_asr
