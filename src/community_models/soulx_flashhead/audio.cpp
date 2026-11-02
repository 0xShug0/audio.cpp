#include "engine/community_models/soulx_flashhead/audio.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace engine::community_models::soulx_flashhead {
AudioConditioner::AudioConditioner(core::BackendConfig backend, const Assets & assets) {
    modules::Wav2Vec2EncoderConfig config;
    config.hidden_size = 768;
    config.intermediate_size = 3072;
    config.num_hidden_layers = 12;
    config.output_hidden_layer = 12;
    config.num_attention_heads = 12;
    config.feature_extractor_norm = modules::Wav2Vec2FeatureExtractorNorm::FirstLayerGroupNorm;
    config.encoder_layer_norm_order = modules::Wav2Vec2EncoderLayerNormOrder::PostNorm;
    config.apply_encoder_input_layer_norm = true;
    config.apply_final_layer_norm = false;
    config.feature_output_frames = 200;
    modules::Wav2Vec2EncoderWeightBinding binding;
    binding.layer.post_attention_layer_norm = "layer_norm";
    runtime_ = modules::Wav2Vec2EncoderRuntime::load_from_tensor_source(assets.audio, backend, config, binding);
}
std::vector<float> AudioConditioner::encode(const std::vector<float> & window) {
    if (window.size() != 128000)
        throw std::runtime_error("FlashHead audio window must contain eight seconds at 16 kHz");
    auto normalized = window;
    const double mean = std::accumulate(normalized.begin(), normalized.end(), 0.) / normalized.size();
    double variance = 0;
    for (auto value : normalized)
        variance += (value - mean) * (value - mean);
    variance = variance / normalized.size();
    const double stddev = std::sqrt(variance + 1.e-7);
    for (auto & v : normalized)
        v = static_cast<float>((v - mean) / stddev);
    const auto encoded = runtime_.encode_layers(normalized, 1, static_cast<int64_t>(normalized.size()),
                                                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    std::vector<float> context(33 * 5 * 12 * 768);
    for (int f = 0; f < 33; ++f)
        for (int w = 0; w < 5; ++w)
            for (int layer = 0; layer < 12; ++layer) {
                const int pos = std::clamp(200 - 33 + f + w - 2, 0, 199);
                std::copy_n(encoded.hidden_states[static_cast<size_t>(layer)].begin() + pos * 768, 768,
                            context.begin() + ((f * 5 + w) * 12 + layer) * 768);
            }
    return context;
}
} // namespace engine::community_models::soulx_flashhead
