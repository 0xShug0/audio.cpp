#pragma once

#include "engine/models/magpie_tts/assets.h"
#include "engine/models/magpie_tts/types.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::magpie_tts {

class MagpieTextTokenizer {
public:
    explicit MagpieTextTokenizer(std::filesystem::path resource_root);
    ~MagpieTextTokenizer();

    MagpieTextTokenizer(const MagpieTextTokenizer &) = delete;
    MagpieTextTokenizer & operator=(const MagpieTextTokenizer &) = delete;

    MagpieTokenizationResult tokenize(
        const std::string & text,
        const MagpieTTSGenerationOptions & options) const;

    static std::string normalize_language(std::string language);
    static std::vector<std::string> supported_native_languages();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::magpie_tts
