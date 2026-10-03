#include "engine/models/xtts_v2/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::xtts_v2 {
namespace json = engine::io::json;

std::shared_ptr<const XttsV2Assets> load_xtts_v2_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle(
        model_path, engine::model_spec::default_spec_path("xtts_v2"));
    const auto root = resources.parse_json("config");
    const auto & args = root.require("model_args");

    auto assets = std::make_shared<XttsV2Assets>();
    assets->resources = std::move(resources);
    auto & config = assets->config;
    config.sample_rate = json::require_i64(root.require("audio"), "output_sample_rate");
    config.conditioning_sample_rate = json::require_i64(args, "input_sample_rate");
    config.text_vocab = json::require_i64(args, "gpt_number_text_tokens");
    config.audio_vocab = json::require_i64(args, "gpt_num_audio_tokens");
    config.model_dim = json::require_i64(args, "gpt_n_model_channels");
    config.gpt_layers = json::require_i64(args, "gpt_layers");
    config.gpt_heads = json::require_i64(args, "gpt_n_heads");
    config.max_text_tokens = json::require_i64(args, "gpt_max_text_tokens");
    config.max_audio_tokens = json::require_i64(args, "gpt_max_audio_tokens");
    config.start_audio_token = json::require_i64(args, "gpt_start_audio_token");
    config.stop_audio_token = json::require_i64(args, "gpt_stop_audio_token");
    config.code_stride = json::require_i64(args, "gpt_code_stride_len");
    config.output_hop = json::require_i64(args, "output_hop_length");
    config.speaker_dim = json::require_i64(args, "d_vector_dim");

    if (config.sample_rate != 24000 || config.conditioning_sample_rate != 22050 ||
        config.text_vocab != 6681 || config.audio_vocab != 1026 ||
        config.model_dim != 1024 || config.gpt_layers != 30 || config.gpt_heads != 16 ||
        !json::optional_bool(args, "gpt_use_perceiver_resampler", false)) {
        throw std::runtime_error("unsupported Coqui XTTS v2 checkpoint architecture");
    }
    assets->gpt = assets->resources.open_tensor_source("gpt");
    assets->decoder = assets->resources.open_tensor_source("decoder");
    assets->speaker_encoder = assets->resources.open_tensor_source("speaker_encoder");
    assets->tokenizer_path = assets->resources.require_file("tokenizer");
    (void) assets->resources.require_file("model_license");
    return assets;
}

}  // namespace engine::models::xtts_v2
