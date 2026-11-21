// Voice Isolator signal processing and stream bookkeeping, without weights.
#include "engine/community_models/voice_isolator/attention_window.h"
#include "engine/community_models/voice_isolator/config.h"
#include "engine/community_models/voice_isolator/spectral.h"

#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vi = engine::models::voice_isolator;

namespace {

using engine::test::require;

// The released realtime checkpoint's architecture metadata.
constexpr const char * kRealtimeConfig = R"({"attention_window_frames": 1200, "band_plan": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]], "block_types": ["transformer", "mamba", "transformer", "mamba", "transformer", "mamba", "transformer"], "causal": true, "chunk_seconds": 30.0, "correction_bound": 0.5, "dual_mode": false, "exclusive_axes": "time", "exclusive_self_attention": true, "hop_length": 480, "lookahead_frames": 2, "lookahead_frames_choices": [0, 2, 4], "mamba_conv": 4, "mamba_expand": 2, "mamba_state": 16, "mode_film": false, "n_fft": 2048, "num_heads": 8, "overlap_seconds": 2.0, "sample_rate": 48000, "width": 512})";

std::vector<float> test_signal(int64_t samples) {
    std::vector<float> x(static_cast<size_t>(samples));
    const double pi = 3.14159265358979323846;
    for (int64_t i = 0; i < samples; ++i) {
        const double t = static_cast<double>(i) / 48000.0;
        x[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * pi * 150.0 * t) + 0.1 * std::sin(2.0 * pi * 2200.0 * t + 0.3));
    }
    return x;
}

void test_config_and_bands() {
    const auto config = vi::parse_config(kRealtimeConfig);
    require(config.causal && config.width == 512 && config.head_dim() == 64, "config fields");
    require(config.attention_window_frames == 1200 && config.lookahead_frames == 2, "streaming fields");
    require(config.exclusive_time && !config.exclusive_band, "exclusive axes");
    require(config.dt_rank() == 32 && config.mamba_inner() == 1024, "mamba dimensions");
    const auto layout = vi::make_band_layout(config);
    require(layout.bands() == 46, "46 bands, got " + std::to_string(layout.bands()));
    require(layout.max_band_bins == 85 && layout.encoder_inputs() == 170 && layout.decoder_outputs() == 340, "band widths");
    int64_t next = 0;
    for (const auto & [first, last] : layout.edges) {
        require(first == next && last > first, "bands tile the bins");
        next = last;
    }
    require(next == 1025, "bands cover every bin");
    require(layout.edges.front() == std::make_pair<int64_t, int64_t>(0, 4) &&
        layout.edges[10] == std::make_pair<int64_t, int64_t>(40, 43) &&
        layout.edges[11] == std::make_pair<int64_t, int64_t>(43, 54) &&
        layout.edges.back() == std::make_pair<int64_t, int64_t>(1024, 1025), "band edges");
    bool refused = false;
    try {
        vi::parse_config(R"({"width": 512, "dual_mode": true})");
    } catch (const std::exception &) {
        refused = true;
    }
    require(refused, "dual-mode checkpoints are refused");
}

void test_clip_round_trip() {
    const auto config = vi::parse_config(kRealtimeConfig);
    for (const int64_t samples : {int64_t{700}, int64_t{24000}}) {
        const auto x = test_signal(samples);
        int64_t frames = 0;
        const auto spectra = vi::clip_spectra(config, x, frames);
        require(frames == samples / config.hop_length + 1, "frame count");
        const auto y = vi::clip_waveform(config, spectra, frames, samples);
        double worst = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            worst = std::max(worst, static_cast<double>(std::fabs(x[i] - y[i])));
        }
        require(worst < 1.0e-5, "STFT round trip error " + std::to_string(worst));
    }
}

// Overlap-adding a stream's own frames returns the stream, released in pieces.
void test_stream_synthesis() {
    const auto config = vi::parse_config(kRealtimeConfig);
    const int64_t samples = 9000;
    const auto x = test_signal(samples);
    vi::StreamSynthesizer synthesizer(config);
    const int64_t frames = samples / config.hop_length + 1;
    std::vector<float> y;
    int64_t released = 0;
    for (int64_t first = 0; first < frames; first += 3) {
        const int64_t count = std::min<int64_t>(3, frames - first);
        const auto spectra = vi::stream_spectra(config, x.data(), 0, samples, first, count);
        synthesizer.add(first, count, spectra.data());
        const int64_t upto = std::min(samples, (first + count) * config.hop_length - config.n_fft / 2);
        if (upto > released) {
            std::vector<float> part(static_cast<size_t>(upto - released));
            synthesizer.take(released, upto, part.data());
            y.insert(y.end(), part.begin(), part.end());
            released = upto;
        }
    }
    std::vector<float> rest(static_cast<size_t>(samples - released));
    synthesizer.take(released, samples, rest.data());
    y.insert(y.end(), rest.begin(), rest.end());
    require(static_cast<int64_t>(y.size()) == samples, "stream released every sample");
    double worst = 0.0;
    for (int64_t i = 0; i < samples; ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(x[static_cast<size_t>(i)] - y[static_cast<size_t>(i)])));
    }
    require(worst < 1.0e-5, "stream synthesis error " + std::to_string(worst));
}

void test_attention_window() {
    // A 4-frame window over 4 retained frames, blocks of up to 2: 6 slots,
    // rotary positions rebased once they would pass 8.
    vi::AttentionWindow window(4, 4, 2, 8);
    require(window.capacity() == 6, "ring of retained plus a block");
    const float hidden = -std::numeric_limits<float>::infinity();
    std::vector<float> mask(12);
    auto visible = [&](int64_t q, int64_t slot) { return mask[static_cast<size_t>(q * 6 + slot)] != hidden; };
    auto step = window.advance(2);
    window.fill_mask(step, mask.data());
    require(step.slots == std::vector<int32_t>{0, 1} && step.positions == std::vector<int32_t>{0, 1}, "first block");
    require(visible(0, 0) && !visible(0, 1) && visible(1, 0) && visible(1, 1), "causal within the block");
    window.advance(2);
    step = window.advance(2);
    window.fill_mask(step, mask.data());
    require(step.slots == std::vector<int32_t>{4, 5}, "third block at slots 4 and 5");
    require(!visible(0, 0) && visible(0, 1) && visible(0, 4) && !visible(0, 5), "window of 4 frames");
    step = window.advance(1);
    window.fill_mask(step, mask.data());
    require(step.slots == std::vector<int32_t>{0} && step.positions == std::vector<int32_t>{6} && step.rebase == 0,
        "frame 6 wraps to slot 0");
    require(!visible(0, 1) && !visible(0, 2) && visible(0, 3) && visible(0, 5) && visible(0, 0), "window across the wrap");
    step = window.advance(2);
    require(step.rebase == 0 && step.positions == std::vector<int32_t>{7, 8}, "positions up to the limit");
    step = window.advance(2);
    require(step.rebase == 5 && step.positions == std::vector<int32_t>{4, 5} && step.slots == std::vector<int32_t>{3, 4},
        "rebase to the oldest retained frame");
    window.reset();
    step = window.advance(1);
    require(step.slots == std::vector<int32_t>{0} && step.positions == std::vector<int32_t>{0}, "reset");
}

void test_render_windows() {
    for (const int64_t samples : {int64_t{100}, int64_t{95}, int64_t{30}}) {
        const auto windows = vi::plan_render_windows(samples, 40, 10);
        std::vector<double> total(static_cast<size_t>(samples), 0.0);
        for (const auto & window : windows) {
            const auto weights = window.weights();
            require(window.start >= 0 && window.start + static_cast<int64_t>(weights.size()) <= samples, "window inside the clip");
            for (size_t i = 0; i < weights.size(); ++i) {
                total[static_cast<size_t>(window.start) + i] += weights[i];
            }
        }
        for (const double weight : total) {
            require(std::fabs(weight - 1.0) < 1.0e-6, "crossfades sum to one");
        }
        require(windows.back().start + windows.back().length == samples, "last window ends at the clip's end");
    }
}

}  // namespace

int main() {
    try {
        test_config_and_bands();
        test_clip_round_trip();
        test_stream_synthesis();
        test_attention_window();
        test_render_windows();
        std::cout << "voice_isolator_domain_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "voice_isolator_domain_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
