#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::qwen3_forced_aligner {

// Nagisa's word-segmentation model only; POS tagging does not change boundaries.
// CPU weights and the latest exact-length graph are retained by the session.
class NagisaSegmenter {
public:
    explicit NagisaSegmenter(const assets::ResourceBundle & resources);
    ~NagisaSegmenter();
    std::vector<std::string> segment(const std::string & text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::qwen3_forced_aligner
