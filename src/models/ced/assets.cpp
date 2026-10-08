#include "engine/models/ced/assets.h"

#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::ced {

std::shared_ptr<CedAssets> load_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<CedAssets>();
    result->resources = model_spec::load_resource_bundle_for_family(path, "ced");
    const auto model = result->resources.parse_json("config");
    const auto processor = result->resources.parse_json("processor");
    auto & config = result->config;
    config.hidden = model.require("embed_dim").as_i64();
    config.heads = model.require("num_heads").as_i64();
    config.layers = model.require("depth").as_i64();
    config.intermediate = static_cast<int64_t>(config.hidden * model.require("mlp_ratio").as_f32());
    config.patch_size = model.require("patch_size").as_i64();
    config.patch_stride = model.require("patch_stride").as_i64();
    config.target_frames = model.require("target_length").as_i64();
    config.mel_bins = processor.require("feature_size").as_i64();
    config.fft_size = processor.require("n_fft").as_i64();
    config.hop_size = processor.require("hop_size").as_i64();
    config.window_size = processor.require("win_size").as_i64();
    config.sample_rate = static_cast<int>(processor.require("sampling_rate").as_i64());
    config.center = processor.require("center").as_bool();
    config.pad_last = model.require("pad_last").as_bool();
    if (config.hidden <= 0 || config.heads <= 0 || config.hidden % config.heads != 0 ||
        config.layers <= 0 || config.intermediate <= 0 || config.patch_size <= 0 ||
        config.patch_stride <= 0 || config.target_frames < config.patch_size ||
        config.mel_bins < config.patch_size || config.sample_rate != 16000 ||
        model.require("pooling").as_string() != "mean" || !model.require("qkv_bias").as_bool() ||
        processor.require("f_min").as_f32() != 0.0f) {
        throw std::runtime_error("unsupported CED configuration");
    }
    const auto & labels = model.require("id2label").as_object();
    config.labels.resize(labels.size());
    for (const auto & [key, value] : labels) {
        const auto index = static_cast<size_t>(std::stoull(key));
        if (index >= config.labels.size()) throw std::runtime_error("CED label indices must be dense");
        config.labels[index] = value.as_string();
    }
    if (config.labels.size() != static_cast<size_t>(model.require("outputdim").as_i64()) ||
        std::any_of(config.labels.begin(), config.labels.end(), [](const auto & label) { return label.empty(); })) {
        throw std::runtime_error("CED label table does not match the classifier");
    }
    return result;
}

}  // namespace engine::models::ced
