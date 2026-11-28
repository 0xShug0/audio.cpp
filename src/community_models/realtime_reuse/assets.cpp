#include "engine/community_models/realtime_reuse/assets.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::realtime_reuse {

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<Assets>();
    assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("realtime_reuse"));
    assets->tensors = assets->resources.open_tensor_source("weights");
    const auto config = assets->resources.parse_json("config");
    const auto & model = config.require("model_cfg");
    const auto & stft = config.require("stft_cfg");
    if (io::json::require_i32(model, "hid_feature") != 64 ||
        io::json::require_i32(model, "d_state") != 16 ||
        io::json::require_i32(model, "d_conv") != 4 ||
        io::json::require_i32(model, "expand") != 4 ||
        io::json::require_i32(model, "num_tfmamba") != 12 ||
        io::json::require_string(model, "compress_factor") != "relu_log1p" ||
        io::json::require_i32(stft, "n_fft") != 320 ||
        io::json::require_i32(stft, "hop_size") != 160 ||
        io::json::require_i32(stft, "win_size") != 320 ||
        io::json::require_i32(stft, "sampling_rate") != 8000) {
        throw std::runtime_error("Real-time RE-USE requires the official 12-layer SEMamba configuration");
    }
    return assets;
}

} // namespace engine::models::realtime_reuse
