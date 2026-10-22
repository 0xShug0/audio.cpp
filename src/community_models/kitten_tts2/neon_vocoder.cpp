#include "engine/community_models/kitten_tts2/neon_vocoder.h"
#include "engine/framework/neon/blocks.h"

#include <limits>
#include <stdexcept>

namespace engine::community_models::kitten_tts2 {
namespace {

template <typename Conv>
neon::Conv1dWeights bind_conv(const Conv & weights) {
    return {weights.weight, weights.bias};
}

neon::SnakeResidualWeights bind_residual(const modules::HiftVocoderWeights::ResBlockWeights & weights) {
    neon::SnakeResidualWeights out;
    for (const auto & w : weights.convs1) out.first.push_back(bind_conv(w));
    for (const auto & w : weights.convs2) out.second.push_back(bind_conv(w));
    for (const auto & w : weights.activations1) out.alpha_first.push_back(w.alpha);
    for (const auto & w : weights.activations2) out.alpha_second.push_back(w.alpha);
    return out;
}

} // namespace

NeonVocoder::NeonVocoder(const modules::HiftVocoderWeights & weights) {
    const auto & config = weights.config;
    if (!neon::available()) throw std::runtime_error("Kitten NEON vocoder requires an AArch64 NEON CPU");
    if (config.causal_convolutions || config.upsample_mode != modules::HiftVocoderUpsampleMode::ConvTranspose1d)
        throw std::runtime_error("Kitten NEON vocoder requires noncausal transposed-convolution HiFT");
    neon::HiftConfig c;
    c.input_channels = input_channels_ = static_cast<int>(config.in_channels);
    c.initial_channels = static_cast<int>(config.base_channels);
    c.source_channels = source_channels_ = static_cast<int>(config.istft_n_fft + 2);
    c.output_channels = static_cast<int>(weights.conv_post.out_channels);
    c.pre_kernel = static_cast<int>(weights.conv_pre.kernel);
    c.post_kernel = static_cast<int>(weights.conv_post.kernel);
    c.negative_slope = config.lrelu_slope;
    c.upsample_rates.assign(config.upsample_rates.begin(), config.upsample_rates.end());
    c.upsample_kernels.assign(config.upsample_kernel_sizes.begin(), config.upsample_kernel_sizes.end());
    c.residual_kernels.assign(config.resblock_kernel_sizes.begin(), config.resblock_kernel_sizes.end());
    c.source_residual_kernels.assign(config.source_resblock_kernel_sizes.begin(), config.source_resblock_kernel_sizes.end());
    for (const auto & d : config.resblock_dilation_sizes) c.residual_dilations.emplace_back(d.begin(), d.end());
    for (const auto & d : config.source_resblock_dilation_sizes) c.source_residual_dilations.emplace_back(d.begin(), d.end());
    neon::HiftWeights w;
    w.pre = bind_conv(weights.conv_pre);
    w.post = bind_conv(weights.conv_post);
    for (const auto & conv : weights.ups) w.upsample.push_back(bind_conv(conv));
    for (const auto & conv : weights.source_downs) w.source_down.push_back(bind_conv(conv));
    for (const auto & block : weights.source_resblocks) w.source_residual.push_back(bind_residual(block));
    for (const auto & block : weights.resblocks) w.residual.push_back(bind_residual(block));
    auto mel = plan_.input(), source = plan_.input();
    plan_.set_output(neon::build_hift(plan_, mel, source, c, w));
}

std::vector<float> NeonVocoder::run(const std::vector<float> & mel, int64_t frames,
    const std::vector<float> & source_stft, int64_t stft_frames, int threads) {
    if (frames <= 0 || stft_frames <= 0 || frames > std::numeric_limits<int>::max() ||
        stft_frames > std::numeric_limits<int>::max())
        throw std::runtime_error("Kitten NEON vocoder frame count invalid");
    return plan_.run({neon::TensorView(mel, {input_channels_, static_cast<int>(frames)}),
        neon::TensorView(source_stft, {source_channels_, static_cast<int>(stft_frames)})}, threads).values;
}

} // namespace engine::community_models::kitten_tts2
