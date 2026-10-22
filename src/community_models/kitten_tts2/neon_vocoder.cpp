#include "engine/community_models/kitten_tts2/neon_vocoder.h"
#include "engine/framework/neon/blocks.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/audio/dsp.h"

#include <algorithm>
#include <cmath>
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
    sample_rate_ = config.sampling_rate;
    harmonics_ = config.nb_harmonics + 1;
    source_amplitude_ = config.nsf_alpha;
    source_noise_std_ = config.nsf_sigma;
    voiced_threshold_ = config.nsf_voiced_threshold;
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
        const int64_t samples = frames * scale_;
        std::vector<float> pitch(static_cast<size_t>(samples));
        for (int64_t t = 0; t < frames; ++t)
            std::fill(pitch.begin() + t * scale_, pitch.begin() + (t + 1) * scale_, f0[t]);
        const auto phase_count = static_cast<size_t>(harmonics_);
        const auto uniform = sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values);
        const auto gaussian = sampling::generate_torch_cuda_randn(phase_count * static_cast<size_t>(samples),
            seed, sampling::TorchRandnPrecision::Float32, prior_noise_values + phase_count);
        constexpr float pi = 3.14159265358979323846F;
        constexpr float two_pi = 2.0F * pi;
        std::vector<float> phase(phase_count), frac(phase_count, 0.0F);
        for (int64_t h = 0; h < harmonics_; ++h) phase[h] = uniform[h] * two_pi - pi;
        phase[0] = 0.0F;
        source.resize(static_cast<size_t>(samples));
        for (int64_t t = 0; t < samples; ++t) {
            const float frequency = pitch[t];
            const float voiced = frequency > voiced_threshold_ ? 1.0F : 0.0F;
            float sum = source_bias_;
            for (int64_t h = 0; h < harmonics_; ++h) {
                frac[h] += frequency * static_cast<float>(h + 1) / static_cast<float>(sample_rate_);
                frac[h] -= std::floor(frac[h]);
                const float theta = two_pi * frac[h];
                float wave = source_amplitude_ * std::sin(theta + phase[h]);
                const float noise = voiced * source_noise_std_ + (1.0F - voiced) * source_amplitude_ / 3.0F;
                // Reference noise is stored harmonic-first, not sample-first.
                wave = wave * voiced + noise * gaussian[h * samples + t];
                sum += wave * source_weight_[h];
            }
            source[t] = std::tanh(sum);
        }
    }
    const int64_t bins = n_fft_ / 2 + 1;
    const auto window = audio::get_cached_stft_window(
        {n_fft_, hop_, n_fft_, true, audio::STFTPadMode::Reflect, audio::STFTFamily::Kokoro});
    std::vector<float> spectrum;
    int64_t stft_frames;
    {
        const auto complex = audio::STFT().compute_complex(source, window, 1,
            static_cast<int64_t>(source.size()), {n_fft_, hop_, n_fft_});
        stft_frames = complex.shape[2];
        spectrum.resize(static_cast<size_t>(2 * bins * stft_frames));
        for (int64_t f = 0; f < bins; ++f) {
            for (int64_t t = 0; t < stft_frames; ++t) {
                spectrum[f * stft_frames + t] = complex.values[(f * stft_frames + t) * 2];
                spectrum[(bins + f) * stft_frames + t] = complex.values[(f * stft_frames + t) * 2 + 1];
            }
        }
    }
    if (stft_frames > std::numeric_limits<int>::max())
        throw std::runtime_error("Kitten NEON vocoder STFT frame count invalid");
    const auto post = plan_.run({neon::TensorView(mel, {input_channels_, static_cast<int>(frames)}),
        neon::TensorView(spectrum, {source_channels_, static_cast<int>(stft_frames)})}, threads).values;
    if (post.size() != static_cast<size_t>(2 * bins * stft_frames))
        throw std::runtime_error("Kitten NEON vocoder post shape mismatch");
    std::vector<float> complex(static_cast<size_t>(bins * stft_frames * 2));
    for (int64_t f = 0; f < bins; ++f) {
        for (int64_t t = 0; t < stft_frames; ++t) {
            const float magnitude = std::min(std::exp(post[f * stft_frames + t]), 1.0e2F);
            const float phase = std::sin(post[(bins + f) * stft_frames + t]);
            complex[(f * stft_frames + t) * 2] = magnitude * std::cos(phase);
            complex[(f * stft_frames + t) * 2 + 1] = magnitude * std::sin(phase);
        }
    }
    auto wav = audio::ISTFT().compute(complex, window, 1, bins, stft_frames,
        hop_ * (stft_frames - 1), {n_fft_, hop_, n_fft_});
    for (float & sample : wav.values) sample = std::clamp(sample, -audio_limit_, audio_limit_);
    return wav.values;
}

} // namespace engine::community_models::kitten_tts2
