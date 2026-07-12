#include "engine/models/chatterbox/assets.h"

#include "engine/framework/io/filesystem.h"

#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

std::filesystem::path require_existing_file(const std::filesystem::path & path, const char * what) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("missing ") + what + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path optional_existing_file(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        return {};
    }
    return std::filesystem::weakly_canonical(path);
}

void require_resolved_asset(const std::filesystem::path & path, const char * what) {
    if (path.empty()) {
        throw std::runtime_error(std::string("missing ") + what);
    }
}

}  // namespace

ChatterboxAssetPaths resolve_chatterbox_assets(const std::filesystem::path & model_root) {
    const auto root = std::filesystem::weakly_canonical(model_root);
    if (!engine::io::is_existing_directory(root)) {
        throw std::runtime_error("chatterbox model root does not exist: " + model_root.string());
    }
    ChatterboxAssetPaths assets;
    assets.model_root = root;
    assets.voice_encoder_weights = optional_existing_file(root / "ve.safetensors");
    assets.s3tokenizer_weights = require_existing_file(root / "s3gen.safetensors", "S3TokenizerV2 weights");
    assets.t3_english_weights = optional_existing_file(root / "t3_cfg.safetensors");
    assets.t3_multilingual_v2_weights = optional_existing_file(root / "t3_mtl23ls_v2.safetensors");
    assets.t3_multilingual_v3_weights = optional_existing_file(root / "t3_mtl23ls_v3.safetensors");
    assets.s3gen_weights = require_existing_file(root / "s3gen.safetensors", "S3Gen weights");
    assets.english_tokenizer = optional_existing_file(root / "tokenizer.json");
    assets.multilingual_tokenizer = optional_existing_file(root / "grapheme_mtl_merged_expanded_v1.json");
    assets.cangjie_mapping = optional_existing_file(root / "Cangjie5_TC.json");
    assets.builtin_conditionals = optional_existing_file(root / "conds.pt");
    return assets;
}

void require_chatterbox_vc_assets(const ChatterboxAssetPaths & assets) {
    require_resolved_asset(assets.s3tokenizer_weights, "S3TokenizerV2/S3Gen weights: s3gen.safetensors");
    require_resolved_asset(assets.s3gen_weights, "S3Gen weights: s3gen.safetensors");
}

void require_chatterbox_tts_assets(const ChatterboxAssetPaths & assets) {
    require_resolved_asset(assets.voice_encoder_weights, "voice encoder weights: ve.safetensors");
    require_resolved_asset(assets.t3_english_weights, "T3 english weights: t3_cfg.safetensors");
    require_resolved_asset(assets.t3_multilingual_v2_weights, "T3 multilingual v2 weights: t3_mtl23ls_v2.safetensors");
    require_resolved_asset(assets.t3_multilingual_v3_weights, "T3 multilingual v3 weights: t3_mtl23ls_v3.safetensors");
    require_resolved_asset(assets.english_tokenizer, "english tokenizer: tokenizer.json");
    require_resolved_asset(assets.multilingual_tokenizer, "multilingual tokenizer: grapheme_mtl_merged_expanded_v1.json");
    require_resolved_asset(assets.cangjie_mapping, "Cangjie mapping: Cangjie5_TC.json");
    require_chatterbox_vc_assets(assets);
}

}  // namespace engine::models::chatterbox
