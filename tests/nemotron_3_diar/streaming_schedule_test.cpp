#include "engine/models/nemotron_3_diar/streaming.h"

#include "engine/framework/io/safetensors.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace engine::models::nemotron_3_diar;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<StreamWindow> run(int64_t samples, float value = 0.5F) {
    FeatureConfig frontend;
    StreamingConfig streaming;
    streaming.chunk_len = 9;
    streaming.chunk_right_context = 4;
    StreamScheduler scheduler(frontend, streaming, 8);
    engine::runtime::AudioChunk chunk;
    chunk.sample_rate = 16000;
    chunk.samples.assign(static_cast<size_t>(samples), value);
    auto windows = scheduler.push_audio(chunk);
    auto tail = scheduler.finalize();
    windows.insert(windows.end(), tail.begin(), tail.end());
    return windows;
}

// NeMo FilterbankFeatures.get_seq_len: floor(samples / hop) valid mel frames.
void test_final_frame_count() {
    for (const int64_t samples : {16000LL, 16090LL, 16159LL, 16160LL, 23999LL}) {
        const auto windows = run(samples);
        require(!windows.empty(), "no windows");
        const auto & last = windows.back();
        const int64_t end = last.mel_start + last.mel_frames;
        require(end == samples / 160,
            "final mel end " + std::to_string(end) + " != floor(" + std::to_string(samples) + " / 160)");
    }
}

// Right padding must be zero after pre-emphasis, as with torch.stft's
// constant padding: x[n] - 0.97 * x[n - 1] == 0 for every synthetic sample.
void test_tail_is_silent_after_preemphasis() {
    const int64_t samples = 16090;
    const auto windows = run(samples);
    const auto & last = windows.back();
    const int64_t feature_start = last.mel_start == 0 ? 0 : last.mel_start * 160 - 256;
    const int64_t real = samples - feature_start;
    require(real > 0 && real < static_cast<int64_t>(last.mono_samples.size()), "final window does not reach the tail");
    for (size_t i = static_cast<size_t>(real); i < last.mono_samples.size(); ++i) {
        const float emphasized = last.mono_samples[i] - 0.97F * last.mono_samples[i - 1];
        require(std::fabs(emphasized) < 1.0e-6F, "tail sample " + std::to_string(i) + " is not silent after pre-emphasis");
    }
}

// The asr_laN profiles mirror NeMo configure_diar_streaming for an ASR lookahead of N.
void test_asr_profiles() {
    for (const int64_t lookahead : {0LL, 3LL, 6LL, 13LL}) {
        const auto config = streaming_profile(
            StreamingConfig{}, {{"nemotron_3_diar.latency_profile", "asr_la" + std::to_string(lookahead)}});
        require(config.chunk_len == lookahead + 1 && config.chunk_right_context == 0 &&
                    config.spkcache_len == 264 && config.fifo_len == 264 && config.spkcache_update_period == 222,
            "asr_la" + std::to_string(lookahead) + " geometry");
    }
    bool rejected = false;
    try {
        (void) streaming_profile(StreamingConfig{}, {{"nemotron_3_diar.latency_profile", "asr_la1"}});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "asr_la1 must be rejected");
}

// The artifact bytes must read back through the framework safetensors reader.
void test_safetensors_round_trip() {
    const int64_t frames = 5;
    const int64_t speakers = 8;
    std::vector<float> probabilities(static_cast<size_t>(frames * speakers));
    for (size_t i = 0; i < probabilities.size(); ++i) probabilities[i] = static_cast<float>(i) / 64.0F;
    const auto bytes = encode_speaker_probabilities_safetensors(
        probabilities, frames, speakers, {{"format_version", "1"}, {"latency_profile", "asr_la13"}});
    const auto path = std::filesystem::temp_directory_path() / "nemotron_3_diar_probabilities_test.safetensors";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    const auto index = engine::io::load_safetensors_index(path);
    require(index.header_bytes % 8 == 0, "header is not 8-byte aligned");
    require(index.metadata.at("format_version") == "1" && index.metadata.at("latency_profile") == "asr_la13",
        "metadata mismatch");
    const auto & tensor = index.tensors.at("speaker_probabilities");
    require(tensor.dtype == "F32" && tensor.shape == std::vector<int64_t>{frames, speakers}, "tensor header mismatch");
    std::vector<float> read(probabilities.size());
    require(tensor.data_end - tensor.data_begin == read.size() * sizeof(float), "data size mismatch");
    std::memcpy(read.data(), bytes.data() + index.header_bytes + tensor.data_begin, read.size() * sizeof(float));
    require(read == probabilities, "data mismatch");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    try {
        test_final_frame_count();
        test_tail_is_silent_after_preemphasis();
        test_asr_profiles();
        test_safetensors_round_trip();
    } catch (const std::exception & error) {
        std::cerr << "nemotron_3_diar_streaming_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "nemotron_3_diar_streaming_test passed\n";
    return 0;
}
