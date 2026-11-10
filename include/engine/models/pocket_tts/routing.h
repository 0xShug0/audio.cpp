#pragma once

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>
#include <string_view>

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::pocket_tts {

struct PocketTTSAssets;

// Package loading and session contracts differ; the acoustic runtime is shared.
struct PocketTTSImplementation {
    const std::string_view family;
    std::shared_ptr<const PocketTTSAssets> (*load_assets)(const runtime::ModelLoadRequest &);
    std::unique_ptr<runtime::IVoiceTaskSession> (*create_session)(
        const runtime::TaskSpec &, const runtime::SessionOptions &,
        std::shared_ptr<const PocketTTSAssets>, const std::filesystem::path &,
        std::shared_ptr<const model_spec::ModelContract>);
};

extern const PocketTTSImplementation kGradiumPhononImplementation;
const PocketTTSImplementation & implementation_for_family(std::string_view family);

}  // namespace engine::models::pocket_tts
