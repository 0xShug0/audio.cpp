#pragma once

#include "engine/community_models/whistle_asr/assets.h"
#include "engine/community_models/whistle_asr/encoder.h"
#include "engine/community_models/whistle_asr/frontend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::whistle_asr {

struct WhistleTranscript {
    std::string text;
    std::string language;
};

// One decoder position: the token fed in and the full logits row before argmax.
struct WhistleDecodeStep {
    size_t position = 0;
    int32_t input_token = 0;
    std::vector<float> logits;
};

using WhistleDecodeObserver = std::function<void(const WhistleDecodeStep &)>;

class WhistleWeights;

class WhistleRuntime {
public:
    WhistleRuntime(std::shared_ptr<const WhistleAssets> assets, core::ExecutionContext & execution_context);
    ~WhistleRuntime();
    WhistleRuntime(const WhistleRuntime &) = delete;
    WhistleRuntime & operator=(const WhistleRuntime &) = delete;

    [[nodiscard]] WhistleTranscript transcribe(
        const runtime::AudioBuffer & audio, const std::string & language = "",
        const WhistleDecodeObserver & observer = {});

private:
    std::shared_ptr<const WhistleAssets> assets_;
    WhistleEncoderRuntime encoder_;
    WhistleFrontend frontend_;
    std::unique_ptr<WhistleWeights> weights_;
    int threads_;
};

}  // namespace engine::community_models::whistle_asr
