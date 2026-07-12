#pragma once

#include <filesystem>

namespace engine::models::chatterbox {

struct ChatterboxAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path voice_encoder_weights;
    std::filesystem::path s3tokenizer_weights;
    std::filesystem::path t3_english_weights;
    std::filesystem::path t3_multilingual_v2_weights;
    std::filesystem::path t3_multilingual_v3_weights;
    std::filesystem::path s3gen_weights;
    std::filesystem::path english_tokenizer;
    std::filesystem::path multilingual_tokenizer;
    std::filesystem::path cangjie_mapping;
    std::filesystem::path builtin_conditionals;
};

ChatterboxAssetPaths resolve_chatterbox_assets(const std::filesystem::path & model_root);
void require_chatterbox_tts_assets(const ChatterboxAssetPaths & assets);
void require_chatterbox_vc_assets(const ChatterboxAssetPaths & assets);

}  // namespace engine::models::chatterbox
