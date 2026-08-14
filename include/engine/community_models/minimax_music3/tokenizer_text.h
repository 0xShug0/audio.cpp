#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::minimax_music3 {

// Builds the checkpoint's special-token prompt from the caption and the lyrics and
// tokenizes it into the conditional/unconditional CFG id pair.
class MiniMaxMusic3TextTokenizer final {
public:
    explicit MiniMaxMusic3TextTokenizer(const assets::ResourceBundle & resources);
    ~MiniMaxMusic3TextTokenizer();

    struct PromptIds {
        std::vector<int32_t> cond_ids;
        std::vector<int32_t> uncond_ids;
    };

    PromptIds encode_prompt(const std::string & caption, const std::string & lyrics) const;

    // Exposed for tests.
    static std::string clean_caption(const std::string & caption);
    static std::string normalize_lyrics(const std::string & lyrics);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
