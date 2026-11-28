#include "engine/community_models/realtime_reuse/frontend.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::realtime_reuse {
namespace {
audio::STFTConfig stft_config(int sample_rate) {
    if (sample_rate < 8000 || sample_rate > 48000) throw std::runtime_error("Real-time RE-USE requires 8-48 kHz audio");
    int64_t fft = 320LL * sample_rate / 8000, hop = 160LL * sample_rate / 8000;
    fft += fft % 2;
    hop += hop % 2;
    return {fft, hop, fft, true, audio::STFTPadMode::Reflect, audio::STFTFamily::PeriodicF32};
}
}

Spectrum analyze_spectrum(const std::vector<float> & waveform, int sample_rate, size_t threads) {
    const auto config = stft_config(sample_rate);
    if (waveform.size() <= static_cast<size_t>(config.n_fft / 2))
        throw std::runtime_error("Real-time RE-USE audio must exceed the STFT reflection pad");
    const auto started = std::chrono::steady_clock::now();
    const int64_t frames = 1 + waveform.size() / config.hop_length, bins = config.n_fft / 2 + 1;
    const auto & window = audio::get_cached_stft_window(config);
    auto spectrum = audio::STFT().compute_complex(waveform, window, 1, waveform.size(), config, threads);
    std::vector<float> features(2 * frames * bins);
    for (int64_t f = 0; f < bins; ++f) for (int64_t t = 0; t < frames; ++t) {
        const auto i = (f * frames + t) * 2;
        features[t * bins + f] = std::log1p(std::hypot(spectrum.values[i], spectrum.values[i + 1]));
        features[(frames + t) * bins + f] = std::atan2(spectrum.values[i + 1], spectrum.values[i]);
    }
    debug::timing_log_scalar("realtime_reuse.frontend.ms", debug::elapsed_ms(started));
    return {config, std::move(spectrum), std::move(features), static_cast<int64_t>(waveform.size()), frames, bins};
}

std::vector<float> restore(Runtime & runtime, const std::vector<float> & waveform, int sample_rate, size_t threads) {
    auto input = analyze_spectrum(waveform, sample_rate, threads);
    const auto frames = input.frames, bins = input.bins;
    const auto & features = input.features;
    const auto network = std::chrono::steady_clock::now();
    // The model is causal after its initial lookahead. Carry state between
    // bounded frame blocks instead of allocating a whole-recording Mamba graph.
    runtime.reset();
    std::array<std::vector<float>, 3> heads;
    constexpr int64_t block_frames = 32;
    const int ahead = runtime.config().lookahead_frames;
    for (int64_t start = 0; start < frames; start += block_frames) {
        const int64_t count = std::min(block_frames, frames - start);
        // Causality permits tail padding, keeping the final block on the same cached graph.
        const int64_t input_frames = block_frames + (start == 0 ? ahead : 0);
        std::vector<float> block(2 * input_frames * bins, 0);
        for (int64_t t = 0; t < input_frames; ++t) {
            const int64_t source = start == 0 ? t : start + ahead + t;
            if (source < frames) {
                std::copy_n(features.data() + source * bins, bins, block.data() + t * bins);
                std::copy_n(features.data() + (frames + source) * bins, bins, block.data() + (input_frames + t) * bins);
            }
        }
        auto inferred = runtime.infer(block, input_frames, bins, true, start == 0);
        const size_t output_bins = inferred[0].size() / block_frames;
        for (size_t i = 0; i < heads.size(); ++i)
            heads[i].insert(heads[i].end(), inferred[i].begin(), inferred[i].begin() + count * output_bins);
    }
    debug::timing_log_scalar("realtime_reuse.network.ms", debug::elapsed_ms(network));
    return synthesize_spectrum(input, heads, threads);
}

std::vector<float> synthesize_spectrum(Spectrum & input, const std::array<std::vector<float>, 3> & heads, size_t threads) {
    const auto & config = input.config;
    const auto & window = audio::get_cached_stft_window(config);
    const auto frames = input.frames, bins = input.bins;
    auto & spectrum = input.spectrum;
    const auto decode = std::chrono::steady_clock::now();
    const int64_t output_bins = heads[0].size() / frames;
    for (int64_t t = 0; t < frames; ++t) {
        int64_t zeros = 0;
        for (int64_t f = 0; f < bins; ++f) zeros += heads[0][t * output_bins + f] <= 0;
        for (int64_t f = 0; f < bins; ++f) {
            const auto index = t * output_bins + f;
            const float magnitude = zeros * 2 > bins ? 0 : std::expm1(std::max(0.0f, heads[0][index]));
            const float phase = std::atan2(heads[2][index], heads[1][index]);
            spectrum.values[(f * frames + t) * 2] = magnitude * std::cos(phase);
            spectrum.values[(f * frames + t) * 2 + 1] = magnitude * std::sin(phase);
        }
    }
    auto output = audio::ISTFT().compute(spectrum.values, window, 1, bins, frames,
        (frames - 1) * config.hop_length, config, threads).values;
    output.resize(input.samples, 1e-8f);
    debug::timing_log_scalar("realtime_reuse.synthesis.ms", debug::elapsed_ms(decode));
    return output;
}

StreamingFrontend::StreamingFrontend(Runtime & runtime, int sample_rate, size_t threads)
    : runtime_(runtime), threads_(threads) {
    const auto config = stft_config(sample_rate);
    fft_ = config.n_fft;
    hop_ = config.hop_length;
    bins_ = fft_ / 2 + 1;
    plan_ = audio::get_real_fft_plan(fft_);
    window_ = audio::get_cached_stft_window(config);
    frame_.resize(fft_);
    spectrum_.resize(bins_);
    overlap_.assign(fft_, 0);
    denominator_.assign(fft_, 0);
    runtime_.reset();
}

std::vector<float> StreamingFrontend::push(const std::vector<float> & samples, bool final) {
    if (finished_) throw std::runtime_error("Real-time RE-USE stream is already finalized");
    input_.insert(input_.end(), samples.begin(), samples.end());
    received_ += samples.size();
    std::vector<float> output;
    if (final && received_ <= fft_ / 2) throw std::runtime_error("Real-time RE-USE audio must exceed the STFT reflection pad");
    analyze(final, output);
    if (final) {
        for (int i = 0; i < runtime_.config().lookahead_frames; ++i) {
            features_.emplace_back(2 * bins_, 0);
        }
        while (!features_.empty()) infer(output, true);
        output.insert(output.end(), received_ - emitted_, 1e-8f);
        emitted_ = received_;
        finished_ = true;
    }
    return output;
}

void StreamingFrontend::analyze(bool final, std::vector<float> & output) {
    if (received_ <= fft_ / 2) return;
    while (final ? analyzed_ <= received_ / hop_ : analyzed_ * hop_ + fft_ / 2 <= received_) {
        const int64_t start = analyzed_ * hop_ - fft_ / 2;
        for (int64_t i = 0; i < fft_; ++i) {
            int64_t index = start + i;
            if (index < 0) index = -index;
            if (index >= received_) index = 2 * received_ - 2 - index;
            frame_[i] = input_.at(static_cast<size_t>(index - input_offset_)) * window_[i];
        }
        plan_->forward({static_cast<size_t>(fft_)}, {sizeof(float)}, {sizeof(std::complex<float>)},
            0, frame_.data(), spectrum_.data(), 1, threads_);
        std::vector<float> feature(2 * bins_);
        for (int64_t f = 0; f < bins_; ++f) {
            feature[f] = std::log1p(std::abs(spectrum_[f]));
            feature[bins_ + f] = std::arg(spectrum_[f]);
        }
        features_.push_back(std::move(feature));
        ++analyzed_;
        infer(output);
        const int64_t keep_from = std::max<int64_t>(0, std::min(analyzed_ * hop_ - fft_ / 2, received_ - fft_));
        while (input_offset_ < keep_from) { input_.pop_front(); ++input_offset_; }
    }
}

void StreamingFrontend::infer(std::vector<float> & output, bool final) {
    const int64_t ahead = synthesized_ == 0 ? runtime_.config().lookahead_frames : 0;
    const int64_t count = batch_frames_ + ahead;
    if (!final && features_.size() < static_cast<size_t>(count)) return;
    const int64_t valid = std::min<int64_t>(batch_frames_, features_.size() - ahead);
    std::vector<float> packed(count * bins_ * 2, 0);
    for (int64_t t = 0; t < count && !features_.empty(); ++t) {
        std::copy_n(features_.front().data(), bins_, packed.data() + t * bins_);
        std::copy_n(features_.front().data() + bins_, bins_, packed.data() + (count + t) * bins_);
        features_.pop_front();
    }
    const auto heads = runtime_.infer(packed, count, bins_, true, synthesized_ == 0);
    const int64_t output_bins = heads[0].size() / batch_frames_;
    for (int64_t t = 0; t < valid; ++t) {
        int64_t zeros = 0;
        for (int64_t f = 0; f < bins_; ++f) zeros += heads[0][t * output_bins + f] <= 0;
        for (int64_t f = 0; f < bins_; ++f) {
            // Upstream's online path applies the sweep filter only after its first frame.
            const float magnitude = synthesized_ > 0 && zeros * 2 > bins_ ? 0 : std::expm1(std::max(0.0f, heads[0][t * output_bins + f]));
            const float phase = std::atan2(heads[2][t * output_bins + f], heads[1][t * output_bins + f]);
            spectrum_[f] = std::complex<float>(magnitude * std::cos(phase), magnitude * std::sin(phase));
        }
        plan_->inverse({static_cast<size_t>(fft_)}, {sizeof(std::complex<float>)}, {sizeof(float)},
            0, spectrum_.data(), frame_.data(), 1.0f / fft_, threads_);
        for (int64_t i = 0; i < fft_; ++i) {
            overlap_[i] += frame_[i] * window_[i];
            denominator_[i] += window_[i] * window_[i];
        }
        for (int64_t i = 0; i < hop_; ++i) {
            if (synthesized_ * hop_ + i >= fft_ / 2) {
                if (denominator_[i] <= 1e-11f) throw std::runtime_error("Real-time RE-USE overlap-add has zero coverage");
                pending_audio_.push_back(overlap_[i] / denominator_[i]);
            }
        }
        // Independent even rounding can make the hop one sample larger than half
        // the FFT (e.g. 22.05 kHz). Retain that sample until the next frame so the
        // centered ISTFT has exactly (frames - 1) * hop samples at finalization.
        while (emitted_ < synthesized_ * hop_) {
            output.push_back(pending_audio_.front());
            pending_audio_.pop_front();
            ++emitted_;
        }
        std::move(overlap_.begin() + hop_, overlap_.end(), overlap_.begin());
        std::fill(overlap_.end() - hop_, overlap_.end(), 0);
        std::move(denominator_.begin() + hop_, denominator_.end(), denominator_.begin());
        std::fill(denominator_.end() - hop_, denominator_.end(), 0);
        ++synthesized_;
    }
    // Keep single-frame processing until the first audio, then grow to 160 ms.
    if (emitted_ > 0) batch_frames_ = std::min<int64_t>(8, batch_frames_ * 2);
}
} // namespace engine::models::realtime_reuse
