#include "engine/community_models/sortformer_diar/frontend.h"
#include "engine/community_models/sortformer_diar/stream_schedule.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace engine::community_models::sortformer_diar;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

SortformerV2FeatureExtractorConfig frontend() {
    SortformerV2FeatureExtractorConfig config;
    config.sample_rate = 16000;
    config.n_fft = 512;
    config.hop_length = 160;
    config.win_length = 400;
    config.num_mel_bins = 128;
    return config;
}

std::vector<SortformerV2StreamWindow> run_fragmented(const std::vector<float> & samples) {
    SortformerV2StreamScheduler scheduler(frontend(), {188, 0, 2, 2, 1, 1}, 8);
    std::vector<SortformerV2StreamWindow> windows;
    const std::vector<int64_t> parts = {137, 203, 4096, 701, 173, 1024, 3666};
    int64_t offset = 0;
    for (int64_t part : parts) {
        if (offset >= static_cast<int64_t>(samples.size())) break;
        part = std::min<int64_t>(part, static_cast<int64_t>(samples.size()) - offset);
        engine::runtime::AudioChunk chunk;
        chunk.sample_rate = 16000;
        chunk.channels = 1;
        chunk.start_sample = offset;
        chunk.samples.assign(samples.begin() + offset, samples.begin() + offset + part);
        auto ready = scheduler.push_audio(chunk);
        windows.insert(windows.end(), ready.begin(), ready.end());
        offset += part;
    }
    auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    require(scheduler.finished(), "scheduler did not finish");
    require(scheduler.push_audio({16000, 1, static_cast<int64_t>(samples.size()), {0.0f}}).empty(), "post-finalize audio was accepted");
    return windows;
}

std::vector<SortformerV2StreamWindow> run_single(const std::vector<float> & samples) {
    SortformerV2StreamScheduler scheduler(frontend(), {188, 0, 2, 2, 1, 1}, 8);
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.channels = 1;
    chunk.samples = samples;
    auto windows = scheduler.push_audio(chunk);
    auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    return windows;
}

void compare(const std::vector<SortformerV2StreamWindow> & a, const std::vector<SortformerV2StreamWindow> & b) {
    require(a.size() == b.size(), "chunk boundaries changed window count");
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i].sample_start == b[i].sample_start, "sample start changed with chunk boundaries");
        require(a[i].feature_sample_start == b[i].feature_sample_start, "feature sample start changed with chunk boundaries");
        require(a[i].mel_start == b[i].mel_start && a[i].mel_frames == b[i].mel_frames, "mel window changed with chunk boundaries");
        require(a[i].new_mel_frames == b[i].new_mel_frames, "new mel count changed with chunk boundaries");
        require(a[i].encoder_frame_start == b[i].encoder_frame_start && a[i].encoder_frames == b[i].encoder_frames, "encoder geometry changed with chunk boundaries");
        require(a[i].final_flush == b[i].final_flush, "final flush marker changed with chunk boundaries");
        require(a[i].mono_samples == b[i].mono_samples, "window audio changed with chunk boundaries");
    }
}

void test_stream_features_match_continuous() {
    SortformerV2Assets assets;
    assets.feature_config = frontend();
    std::vector<float> samples(20000);
    for (size_t i = 0; i < samples.size(); ++i) samples[i] = std::sin(static_cast<float>(i) * 0.013f);

    engine::runtime::AudioBuffer full_audio;
    full_audio.sample_rate = 16000;
    full_audio.channels = 1;
    full_audio.samples = samples;
    const auto full = compute_sortformer_v2_features(full_audio, assets, 1);

    SortformerV2StreamScheduler scheduler(frontend(), {188, 0, 2, 2, 1, 1}, 8);
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.channels = 1;
    chunk.samples = samples;
    auto windows = scheduler.push_audio(chunk);
    const auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    require(windows.size() >= 2, "feature parity test did not produce two windows");

    const std::vector<size_t> parity_windows = {0, 1, windows.size() - 1};
    for (const size_t wi : parity_windows) {
        const auto window = compute_sortformer_v2_stream_features(
            windows[wi].mono_samples, assets, 1, windows[wi].mel_start == 0,
            windows[wi].mel_frames);
        require(window.valid_frames == windows[wi].mel_frames, "window feature frame count changed");
        float max_delta = 0.0f;
        for (int64_t t = 0; t < window.valid_frames; ++t) {
            for (int64_t m = 0; m < assets.feature_config.num_mel_bins; ++m) {
                const auto lhs = window.time_major[static_cast<size_t>(t * assets.feature_config.num_mel_bins + m)];
                const auto rhs = full.time_major[static_cast<size_t>(
                    (windows[wi].mel_start + t) * assets.feature_config.num_mel_bins + m)];
                max_delta = std::max(max_delta, std::abs(lhs - rhs));
            }
        }
        require(max_delta < 1.0e-4f, "streaming mel window diverges from continuous mel timeline");
    }
}

void test_exact_duration_tail() {
    SortformerV2StreamScheduler scheduler(frontend(), {188, 0, 188, 188, 1, 1}, 8);
    std::vector<float> samples(static_cast<size_t>(16000 * 120), 0.0f);
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.channels = 1;
    chunk.samples = samples;
    auto windows = scheduler.push_audio(chunk);
    auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    require(scheduler.emitted_frames() == 1500, "exact 120-second input emitted an extra encoder frame");
    require(windows.size() >= 2, "exact duration stream did not produce context windows");
    require(windows[0].mel_frames == 1512, "first mel window has incorrect context framing");
    require(windows[1].mel_start == 1496 && windows[1].sample_start == 1496 * 160 &&
                windows[1].feature_sample_start == 1496 * 160 - 256 &&
                windows[1].mel_frames == 1520,
            "later mel window has incorrect absolute sample framing");
    require(!windows.empty() && windows.back().final_flush, "exact duration tail did not flush");
}

}  // namespace

int main() {
    try {
        std::vector<float> samples(10000);
        for (size_t i = 0; i < samples.size(); ++i) samples[i] = std::sin(static_cast<float>(i) * 0.01f);
        compare(run_single(samples), run_fragmented(samples));
        test_stream_features_match_continuous();
        test_exact_duration_tail();
        SortformerV2StreamScheduler empty(frontend(), {188, 0, 2, 2, 1, 1}, 8);
        require(empty.finalize().empty() && empty.emitted_frames() == 0, "empty stream emitted a phantom frame");
        std::cout << "sortformer_v2_schedule_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "sortformer_v2_schedule_test: " << error.what() << '\n';
        return 1;
    }
}
