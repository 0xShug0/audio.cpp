#pragma once

#include "engine/framework/runtime/model.h"

#include <filesystem>
#include <memory>
#include <string_view>

namespace engine::model_spec { struct ModelContract; }

namespace engine::models::pocket_tts {

struct PocketTTSAssets;

// Which upstream's prompt punctuation the text conditioner follows.
enum class PromptPunctuation {
    // pocket-tts: a trailing comma, colon or dash becomes a period, any other unfinished prompt
    // gets one after its closing quotes, and those quotes stay with their sentence when it is split.
    // The first letter after any opening quotes or brackets is upper-cased.
    PocketTTS,
    // Gradium Phonon: only a final letter or digit gets a period, and only the first character is
    // upper-cased.
    LetterOrDigit,
};

// Package loading and session contracts differ; the acoustic runtime is shared.
struct PocketTTSImplementation {
    const std::string_view family;
    std::shared_ptr<const PocketTTSAssets> (*load_assets)(const runtime::ModelLoadRequest &);
    std::unique_ptr<runtime::IVoiceTaskSession> (*create_session)(
        const runtime::TaskSpec &, const runtime::SessionOptions &,
        std::shared_ptr<const PocketTTSAssets>, const std::filesystem::path &,
        std::shared_ptr<const model_spec::ModelContract>);
    PromptPunctuation prompt_punctuation = PromptPunctuation::PocketTTS;
};

extern const PocketTTSImplementation kGradiumPhononImplementation;
const PocketTTSImplementation & implementation_for_family(std::string_view family);

}  // namespace engine::models::pocket_tts
