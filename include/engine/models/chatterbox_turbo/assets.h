#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::models::chatterbox_turbo {

struct ChatterboxTurboAssets {
    engine::assets::ResourceBundle resources;
    std::shared_ptr<const engine::assets::TensorSource> t3_turbo_weights;
    std::shared_ptr<const engine::assets::TensorSource> builtin_conditionals_turbo;
    // Raw path to the T3 GGUF (not just a TensorSource) so the tokenizer can read the embedded
    // `tokenizer.ggml.tokens`/`tokenizer.ggml.merges` metadata arrays directly, and so the
    // sibling S3Gen GGUF's path can be derived (see s3gen_turbo.cpp: replaces "-t3-" with
    // "-s3gen-" in the filename — the two GGUF files can't both live loosely in one directory
    // under this framework's package loader, so only the T3 file is referenced by model_specs;
    // the S3Gen file is located and opened directly here).
    std::filesystem::path t3_gguf_path;
};

std::shared_ptr<const ChatterboxTurboAssets> load_chatterbox_turbo_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::chatterbox_turbo
