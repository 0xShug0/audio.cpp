#include "engine/community_models/kitten_tts2/neon_vocoder.h"
#include "engine/framework/neon/blocks.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/sampling/torch_random.h"

#include <functional>
#include <limits>
#include <numeric>
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
    if (config.causal_convolutions || config.upsample_mode != modules::HiftVocoderUpsampleMode::ConvTranspose1d ||
        config.source_mode != modules::HiftVocoderSourceMode::Standard)
        throw std::runtime_error("Kitten NEON vocoder requires noncausal transposed-convolution HiFT");
    scale_ = std::accumulate(config.upsample_rates.begin(), config.upsample_rates.end(),
        config.istft_hop, std::multiplies<int64_t>{});
    n_fft_ = config.istft_n_fft;
    hop_ = config.istft_hop;
    audio_limit_ = config.audio_limit;
    source_config_ = {config.sampling_rate, config.nb_harmonics + 1,
        config.nsf_alpha, config.nsf_sigma, config.nsf_voiced_threshold};
    source_weight_ = weights.source_linear.weight;
    source_bias_ = weights.source_linear.bias[0];
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

std::vector<float> NeonVocoder::synthesize(const std::vector<float> & mel, int64_t frames,
    const std::vector<float> & f0, int threads, uint64_t seed, uint64_t prior_noise_values) {
    if (frames <= 0 || frames > std::numeric_limits<int>::max() ||
        f0.size() != static_cast<size_t>(frames))
        throw std::runtime_error("Kitten NEON vocoder frame count invalid");
    std::vector<float> source;
    {
        const auto phase_count = static_cast<size_t>(source_config_.harmonics);
        const auto phase = sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values);
        const auto gaussian = sampling::generate_torch_cuda_randn(phase_count * static_cast<size_t>(frames * scale_),
            seed, sampling::TorchRandnPrecision::Float32, prior_noise_values + phase_count);
        source = audio::generate_harmonic_source(f0, scale_, source_config_, source_weight_,
            source_bias_, phase, gaussian);
    }
    std::vector<float> window;
    const auto spectrum = audio::stft_real_imag(source, n_fft_, hop_, window);
    const auto stft_frames = spectrum.shape[2];
    if (stft_frames > std::numeric_limits<int>::max())
        throw std::runtime_error("Kitten NEON vocoder STFT frame count invalid");
    const auto post = plan_.run({neon::TensorView(mel, {input_channels_, static_cast<int>(frames)}),
        neon::TensorView(spectrum.values, {source_channels_, static_cast<int>(stft_frames)})}, threads).values;
    return audio::sine_phase_istft(post, stft_frames, n_fft_, hop_, audio_limit_, window);
}

} // namespace engine::community_models::kitten_tts2
