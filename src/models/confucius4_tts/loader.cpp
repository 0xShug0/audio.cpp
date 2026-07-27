#include "engine/models/confucius4_tts/loader.h"

#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/confucius4_tts/assets.h"
#include "engine/models/confucius4_tts/session.h"

#include <utility>

namespace engine::models::confucius4_tts {
namespace {

constexpr const char * kFamily = "confucius4_tts";

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<ConfuciusAssets> config;
    config.family = kFamily;
    config.load_assets = load_confucius_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const ConfuciusAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<ConfuciusSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::confucius4_tts
