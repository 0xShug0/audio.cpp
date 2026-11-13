#include "engine/models/mossformer2/assets.h"
#include "engine/models/mossformer2/routing.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

namespace engine::models::mossformer2 {

std::shared_ptr<const MossFormer2Assets> load_mossformer2_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<MossFormer2Assets>();
    assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("mossformer2"));
    assets->tensors = assets->resources.open_tensor_source("weights");
    const auto json = assets->resources.parse_json("config");
    if (io::json::require_string(json, "model_type") != "mossformer2") {
        throw std::runtime_error("MossFormer2 config model_type mismatch");
    }
    auto & c = assets->config;
    const auto * variant = json.find("variant");
    assets->impl = &implementation_for_variant(variant ? variant->as_string() : "ss");
    c.sample_rate = io::json::require_i32(json, "sampling_rate");
    c.speakers = io::json::require_i32(json, "num_spks");
    if (assets->impl->task == runtime::VoiceTaskKind::SourceSeparation)
        c.kernel = io::json::require_i32(json, "encoder_kernel_size");
    c.channels = io::json::require_i32(json, "encoder_embedding_dim");
    c.hidden = io::json::require_i32(json, "mossformer_sequence_dim");
    c.layers = io::json::require_i32(json, "num_mossformer_layer");
    if (c.sample_rate <= 0 || c.speakers <= 0 || c.kernel < 2 || c.kernel % 2 ||
        c.channels <= 0 || c.hidden <= 0 || c.hidden % 2 || c.layers <= 0) {
        throw std::runtime_error("MossFormer2 invalid architecture dimensions");
    }
    if (assets->impl->task != runtime::VoiceTaskKind::SourceSeparation) {
        c.output_features = io::json::require_i32(json, "output_features");
        c.fft_length = io::json::require_i32(json, "fft_length");
        c.hop_length = io::json::require_i32(json, "hop_length");
        c.mel_bins = io::json::require_i32(json, "mel_bins");
        if (c.sample_rate != 48000 || c.output_features <= 0 || c.fft_length < 2 ||
            c.hop_length <= 0 || c.hop_length > c.fft_length || c.mel_bins <= 3)
            throw std::runtime_error("Invalid MossFormer2 restoration configuration");
    }
    if (assets->impl->use_vocoder) {
        c.mel_low = io::json::require_f32(json, "mel_low");
        c.mel_high = io::json::require_f32(json, "mel_high");
        const auto & v = json.require("vocoder");
        c.vocoder.initial_channels = io::json::require_i32(v, "upsample_initial_channel");
        c.vocoder.upsample_rates = io::json::number_array_as<int64_t>(v.require("upsample_rates"));
        c.vocoder.upsample_kernels = io::json::number_array_as<int64_t>(v.require("upsample_kernel_sizes"));
        c.vocoder.resblock_kernels = io::json::number_array_as<int64_t>(v.require("resblock_kernel_sizes"));
        for (const auto & row : v.require("resblock_dilation_sizes").as_array())
            c.vocoder.resblock_dilations.push_back(io::json::number_array_as<int64_t>(row));
        if (c.vocoder.upsample_rates.size() != c.vocoder.upsample_kernels.size() ||
            c.vocoder.resblock_kernels.size() != c.vocoder.resblock_dilations.size() ||
            c.vocoder.upsample_rates.empty() || c.vocoder.resblock_kernels.empty())
            throw std::runtime_error("Invalid MossFormer2 vocoder stages");
    }
    return assets;
}

}  // namespace engine::models::mossformer2
