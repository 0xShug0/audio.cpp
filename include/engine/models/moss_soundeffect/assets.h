#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"

#include <memory>
#include <vector>

namespace engine::core { class BackendWeightStore; }

namespace engine::models::moss_soundeffect {

struct DitConfig {
    int64_t dim = 0;
    int64_t ffn_dim = 0;
    int64_t heads = 0;
    int64_t layers = 0;
    int64_t latent_dim = 0;
    int64_t text_dim = 0;
    int64_t freq_dim = 0;
    int64_t patch_size = 0;
    float eps = 0.0F;
};

struct VaeConfig {
    int64_t latent_dim = 0;
    int64_t decoder_dim = 0;
    int64_t hop_length = 0;
    std::vector<int64_t> rates;
};

struct Assets {
    assets::ResourceBundle resources;
    DitConfig dit;
    VaeConfig vae;
    modules::DecoderStackConfig text;
    int64_t vocab_size = 0;
    int64_t text_length = 512;
    int sample_rate = 0;
    int max_duration_sec = 0;
    std::shared_ptr<const assets::TensorSource> text_weights;
    std::shared_ptr<const assets::TensorSource> dit_weights;
    std::shared_ptr<const assets::TensorSource> vae_weights;
};

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path);
modules::CausalDecoderRuntimeWeights load_text_weights(
    const Assets & assets, core::BackendWeightStore & store, assets::TensorStorageType storage);

}  // namespace engine::models::moss_soundeffect
