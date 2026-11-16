#include "engine/community_models/tiny_voca/source_filter.h"

#include "engine/framework/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>

namespace engine::models::tiny_voca {
namespace {

constexpr double kPi = 3.14159265358979323846;

// 1 / (1 + n / 4) for n from `first`; coefficient 0 is the log gain and needs range.
// Filter frames transformed together.
constexpr int64_t kBlockFrames = 64;

std::vector<float> lifter(int64_t size, int64_t first) {
    std::vector<float> out(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        out[static_cast<size_t>(i)] = 1.0F / (1.0F + static_cast<float>(first + i) / 4.0F);
    }
    if (first == 0 && size > 0) {
        out[0] = 2.0F;
    }
    return out;
}

// The periodic Hann window of torch.hann_window.
std::vector<float> hann(int64_t size) {
    std::vector<float> out(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        out[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(size)));
    }
    return out;
}

// Real and imaginary parts of the transform of a unit coefficient at each of `positions`.
void response_tables(const std::vector<int64_t> & positions, int64_t fft, std::vector<double> & real, std::vector<double> & imaginary) {
    const int64_t bins = fft / 2 + 1;
    real.assign(positions.size() * static_cast<size_t>(bins), 0.0);
    imaginary.assign(positions.size() * static_cast<size_t>(bins), 0.0);
    for (size_t p = 0; p < positions.size(); ++p) {
        for (int64_t k = 0; k < bins; ++k) {
            // exp(-2 pi i k n / fft), with the angle reduced first so that it stays exact
            const int64_t turn = (k * positions[p]) % fft;
            const double angle = -2.0 * kPi * static_cast<double>(turn) / static_cast<double>(fft);
            real[p * static_cast<size_t>(bins) + static_cast<size_t>(k)] = std::cos(angle);
            imaginary[p * static_cast<size_t>(bins) + static_cast<size_t>(k)] = std::sin(angle);
        }
    }
}

}  // namespace

TinyVocaSourceFilter::TinyVocaSourceFilter(
    const TinyVocaConfig & config,
    std::vector<float> harmonic_weight,
    std::vector<float> harmonic_bias,
    std::vector<float> noise_weight,
    std::vector<float> noise_bias)
    : config_(config) {
    harmonic_.fft = config.synth_fft;
    harmonic_.hop = config.synth_hop;
    harmonic_.steps = config.hop / config.synth_hop;
    harmonic_.coefficients = config.cep_causal + config.cep_anti;
    harmonic_.lifter = lifter(config.cep_causal, 0);
    const std::vector<float> anti = lifter(config.cep_anti, 1);
    harmonic_.lifter.insert(harmonic_.lifter.end(), anti.begin(), anti.end());
    std::vector<int64_t> positions;
    for (int64_t i = 0; i < config.cep_causal; ++i) {
        positions.push_back(i);                                 // the minimum-phase part
    }
    for (int64_t i = 0; i < config.cep_anti; ++i) {
        positions.push_back(config.synth_fft - 1 - i);          // the maximum-phase part (the glottal pulse)
    }
    response_tables(positions, harmonic_.fft, harmonic_.real, harmonic_.imaginary);
    harmonic_.window = hann(harmonic_.fft);
    harmonic_.weight = std::move(harmonic_weight);
    harmonic_.bias = std::move(harmonic_bias);

    noise_.fft = config.noise_fft;
    noise_.hop = config.noise_hop;
    noise_.steps = config.hop / config.noise_hop;
    noise_.coefficients = config.noise_cep;
    noise_.lifter = lifter(config.noise_cep, 0);
    positions.clear();
    for (int64_t i = 0; i < config.noise_cep; ++i) {
        positions.push_back(i);
    }
    response_tables(positions, noise_.fft, noise_.real, noise_.imaginary);
    noise_.window = hann(noise_.fft);
    noise_.weight = std::move(noise_weight);
    noise_.bias = std::move(noise_bias);

    for (const Branch * branch : {&harmonic_, &noise_}) {
        const auto outputs = static_cast<size_t>(branch->coefficients + branch->steps);
        if (branch->bias.size() != outputs || branch->weight.size() != outputs * static_cast<size_t>(config.channels)) {
            throw std::runtime_error("tiny-voca waveform head has an unexpected shape");
        }
    }
}

std::vector<float> TinyVocaSourceFilter::pulse_train(const std::vector<float> & f0) const {
    // Frame values sit at samples i * hop and are interpolated in the log domain.
    // The sum of all harmonics below limit * sample_rate has the closed form
    // sin((K + 1/2) x) / (2 sin(x / 2)) - 1/2; the top harmonic fades in and out so
    // the signal stays continuous while the pitch glides. Doubles keep the phase exact.
    const auto frames = static_cast<int64_t>(f0.size());
    const int64_t hop = config_.hop;
    const auto rate = static_cast<double>(config_.sample_rate);
    const auto limit = static_cast<double>(config_.harmonic_limit);
    std::vector<double> knots(static_cast<size_t>(frames + 1));
    for (int64_t i = 0; i < frames; ++i) {
        knots[static_cast<size_t>(i)] = std::log(std::max(static_cast<double>(f0[static_cast<size_t>(i)]), 1.0));
    }
    knots[static_cast<size_t>(frames)] = frames > 0 ? knots[static_cast<size_t>(frames - 1)] : 0.0;
    std::vector<float> out(static_cast<size_t>(frames * hop));
    double total = 0.0;
    for (int64_t n = 0; n < frames * hop; ++n) {
        const double position = static_cast<double>(n) / static_cast<double>(hop);
        const auto low = static_cast<int64_t>(std::floor(position));
        const double weight = position - static_cast<double>(low);
        const double frequency = std::exp(knots[static_cast<size_t>(low)] * (1.0 - weight) + knots[static_cast<size_t>(low + 1)] * weight);
        total += frequency / rate;
        double theta = std::fmod(2.0 * kPi * total, 2.0 * kPi);
        if (theta < 0.0) {
            theta += 2.0 * kPi;
        }
        const double harmonics = std::max(limit * rate / frequency, 1.0);
        const double whole = std::floor(harmonics);
        const double denominator = 2.0 * std::sin(theta / 2.0);
        const double core = std::fabs(denominator) < 1e-7 ? whole : std::sin((whole + 0.5) * theta) / denominator - 0.5;
        const double wave = core + (harmonics - whole) * std::cos((whole + 1.0) * theta);
        out[static_cast<size_t>(n)] = static_cast<float>(wave * std::sqrt(2.0 / harmonics));
    }
    return out;
}

// Time-varying filtering of `source`: its STFT is multiplied by the response
// of each filter frame and transformed back. The result is added to `out`.
void TinyVocaSourceFilter::shape(
    const Branch & branch,
    const std::vector<float> & hidden,
    const std::vector<float> & source,
    int64_t frames,
    bool interpolate,
    std::vector<float> & out) const {
    const int64_t channels = config_.channels;
    const int64_t fft = branch.fft;
    const int64_t hop = branch.hop;
    const int64_t bins = fft / 2 + 1;
    const int64_t steps = branch.steps;
    const int64_t coefficients = branch.coefficients;
    const int64_t outputs = coefficients + steps;
    const int64_t samples = frames * config_.hop;
    // Two more filter frames cover the half window of zeros appended to the source.
    const int64_t filter_frames = steps * frames + 3;

    // The linear head: a cepstrum and one gain for each filter frame, for every acoustic frame.
    std::vector<float> head(static_cast<size_t>(frames * outputs));
    for (int64_t t = 0; t < frames; ++t) {
        const float * row = &hidden[static_cast<size_t>(t * channels)];
        for (int64_t o = 0; o < outputs; ++o) {
            const float * weight = &branch.weight[static_cast<size_t>(o * channels)];
            float sum = 0.0F;
            for (int64_t c = 0; c < channels; ++c) {
                sum += weight[c] * row[c];
            }
            head[static_cast<size_t>(t * outputs + o)] = sum + branch.bias[static_cast<size_t>(o)];
        }
    }

    // The filter frames are worked a block at a time, so the memory in use does
    // not grow with the length of the utterance.
    const int64_t padded = samples + fft / 2;
    const int64_t block = std::min<int64_t>(filter_frames, kBlockFrames);
    std::vector<float> framed(static_cast<size_t>(block * fft));
    std::vector<std::complex<float>> spectrum(static_cast<size_t>(block * bins));
    std::vector<float> sum(static_cast<size_t>(padded + fft), 0.0F);
    std::vector<float> envelope(static_cast<size_t>(padded + fft), 0.0F);
    std::vector<float> cepstrum(static_cast<size_t>(coefficients));
    const auto plan = audio::get_real_fft_plan(static_cast<size_t>(fft));
    const audio::TensorStrideBytes real_strides{static_cast<std::ptrdiff_t>(fft * static_cast<int64_t>(sizeof(float))),
                                                static_cast<std::ptrdiff_t>(sizeof(float))};
    const audio::TensorStrideBytes complex_strides{static_cast<std::ptrdiff_t>(bins * static_cast<int64_t>(sizeof(std::complex<float>))),
                                                   static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    const int64_t last = frames - 1;
    for (int64_t first = 0; first < filter_frames; first += block) {
        const int64_t count = std::min(block, filter_frames - first);
        const audio::TensorShape frame_shape{static_cast<size_t>(count), static_cast<size_t>(fft)};

        // The source in windowed frames, centred, with half a window of zeros after it.
        std::fill(framed.begin(), framed.end(), 0.0F);
        for (int64_t i = 0; i < count; ++i) {
            const int64_t start = (first + i) * hop - fft / 2;
            for (int64_t n = 0; n < fft; ++n) {
                const int64_t at = start + n;
                if (at >= 0 && at < samples) {
                    framed[static_cast<size_t>(i * fft + n)] = source[static_cast<size_t>(at)] * branch.window[static_cast<size_t>(n)];
                }
            }
        }
        plan->forward(frame_shape, real_strides, complex_strides, 1, framed.data(), spectrum.data());

        // Each filter frame: cepstrum -> response exp(FFT(cepstrum)), times the spectrum.
        for (int64_t i = 0; i < count; ++i) {
            const int64_t sub = std::min(first + i, steps * frames - 1);
            const int64_t here = sub / steps;
            const int64_t following = std::min((sub + 1) / steps, last);
            const float * a = &head[static_cast<size_t>(here * outputs)];
            const float * b = &head[static_cast<size_t>(following * outputs)];
            for (int64_t c = 0; c < coefficients; ++c) {
                // the harmonic envelope is interpolated between frames; the noise one is held
                const float value = interpolate ? 0.5F * (a[c] + b[c]) : a[c];
                cepstrum[static_cast<size_t>(c)] = value * branch.lifter[static_cast<size_t>(c)];
            }
            cepstrum[0] += head[static_cast<size_t>(here * outputs + coefficients + sub % steps)];
            std::complex<float> * row = &spectrum[static_cast<size_t>(i * bins)];
            for (int64_t k = 0; k < bins; ++k) {
                double log_magnitude = 0.0;
                double angle = 0.0;
                for (int64_t c = 0; c < coefficients; ++c) {
                    const auto at = static_cast<size_t>(c * bins + k);
                    log_magnitude += static_cast<double>(cepstrum[static_cast<size_t>(c)]) * branch.real[at];
                    angle += static_cast<double>(cepstrum[static_cast<size_t>(c)]) * branch.imaginary[at];
                }
                const double magnitude = std::exp(std::min(std::max(log_magnitude, -14.0), 6.0));
                row[k] *= std::complex<float>(static_cast<float>(magnitude * std::cos(angle)), static_cast<float>(magnitude * std::sin(angle)));
            }
        }

        // Back to samples: overlap-add of the windowed frames over the sum of squared windows.
        plan->inverse(frame_shape, complex_strides, real_strides, 1, spectrum.data(), framed.data(), 1.0F / static_cast<float>(fft));
        for (int64_t i = 0; i < count; ++i) {
            for (int64_t n = 0; n < fft; ++n) {
                const auto at = static_cast<size_t>((first + i) * hop + n);
                const float window = branch.window[static_cast<size_t>(n)];
                sum[at] += framed[static_cast<size_t>(i * fft + n)] * window;
                envelope[at] += window * window;
            }
        }
    }
    for (int64_t n = 0; n < samples; ++n) {
        const auto at = static_cast<size_t>(n + fft / 2);
        out[static_cast<size_t>(n)] += envelope[at] > 1e-11F ? sum[at] / envelope[at] : sum[at];
    }
}

std::vector<float> TinyVocaSourceFilter::synthesize(
    const std::vector<float> & hidden,
    const std::vector<float> & f0,
    std::vector<float> noise,
    int64_t frames) const {
    const int64_t samples = frames * config_.hop;
    if (frames <= 0 || static_cast<int64_t>(hidden.size()) != frames * config_.channels || static_cast<int64_t>(f0.size()) != frames ||
        static_cast<int64_t>(noise.size()) < samples) {
        throw std::runtime_error("tiny-voca waveform head got inputs of the wrong size");
    }
    // The noise goes first and is let go before the pulse train is made, so
    // the two sources are never in memory together.
    std::vector<float> out(static_cast<size_t>(samples), 0.0F);
    shape(noise_, hidden, noise, frames, false, out);
    std::vector<float>().swap(noise);
    const std::vector<float> pulses = pulse_train(f0);
    shape(harmonic_, hidden, pulses, frames, true, out);
    return out;
}

}  // namespace engine::models::tiny_voca
