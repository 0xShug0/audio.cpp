#include "engine/models/universr/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::universr {

std::shared_ptr<const UniverSRAssets> load_universr_assets(const std::filesystem::path & path) {
    auto result = std::make_shared<UniverSRAssets>();
    result->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("universr"));
    result->tensors = result->resources.open_tensor_source("weights");
    const auto json = result->resources.parse_json("config");
    if (io::json::require_string(json, "model_type") != "universr") {
        throw std::runtime_error("UniverSR config model_type mismatch");
    }
    const auto & model = json.require("model");
    const auto & transform = json.require("transform");
    auto & config = result->config;
    config.dims = io::json::require_i64_array(model, "dims");
    config.depths = io::json::require_i64_array(model, "depths");
    config.time_dim = io::json::require_i32(model, "time_dim");
    config.cond_dim = io::json::require_i32(model, "cond_dim");
    config.total_freq_bins = io::json::require_i32(model, "total_freq_bins");
    config.hr_freq_bins = io::json::require_i32(model, "hr_freq_bins");
    config.feature_enc_layers = io::json::require_i32(model, "feature_enc_layers");
    // The checkpoint's embedding rows follow ascending sample rates in kHz.
    for (const auto & [rate, bins] : io::json::require_i64_object(model, "sr_to_lr_bins")) {
        size_t consumed = 0;
        const int sr = std::stoi(rate, &consumed);
        if (consumed != rate.size()) {
            throw std::runtime_error("UniverSR sample-rate key must be an integer in kHz");
        }
        config.sr_to_lr_bins.emplace(sr, static_cast<int>(bins));
    }
    config.sample_rate = io::json::require_i32(transform, "sampling_rate");
    config.n_fft = io::json::require_i32(transform, "n_fft");
    config.hop_length = io::json::require_i32(transform, "hop_length");
    config.alpha = io::json::require_f32(transform, "alpha");
    config.beta = io::json::require_f32(transform, "beta");
    config.comp_eps = io::json::require_f32(transform, "comp_eps");
    if (io::json::require_i32(model, "in_channels") != 2 ||
        io::json::require_i32(model, "out_channels") != 2 ||
        config.dims != std::vector<int64_t>{96, 192, 384, 768} ||
        config.depths != std::vector<int64_t>{2, 2, 4, 2} ||
        config.time_dim != 256 || config.cond_dim != 384 || config.feature_enc_layers != 4 ||
        config.total_freq_bins != 512 || config.hr_freq_bins != 432 ||
        config.sr_to_lr_bins != std::map<int, int>{{8, 80}, {12, 128}, {16, 170}, {24, 256}}) {
        throw std::runtime_error("UniverSR requires the released audio/speech U-Net architecture");
    }
    if (config.sample_rate != 48000 || config.n_fft != 1024 || config.hop_length != 512 ||
        io::json::require_string(transform, "window_fn") != "hann" ||
        config.alpha != 0.2f || config.beta != 1.0f || config.comp_eps != 1e-4f) {
        throw std::runtime_error("UniverSR requires the released compressed-complex STFT configuration");
    }
    result->window = result->resources.open_tensor_source("frontend")->require_f32("window", {config.n_fft});
    return result;
}

}  // namespace engine::models::universr
