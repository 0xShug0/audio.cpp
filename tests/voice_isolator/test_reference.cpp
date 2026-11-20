// Voice Isolator against the Python reference. Takes the checkpoint directory and
// the fixtures tests/voice_isolator/make_fixtures.py wrote; see README.md here.
#include "app/cli/args.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/voice_isolator/runtime.h"
#include "engine/framework/io/safetensors.h"

#include "engine/community_models/voice_isolator/config.h"
#include "engine/community_models/voice_isolator/network.h"
#include "engine/community_models/voice_isolator/spectral.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;

constexpr float kMaxDiff = 2.0e-3F;
constexpr double kMeanDiff = 2.0e-4;

std::filesystem::path g_model_dir;
std::filesystem::path g_fixture_dir;
engine::core::BackendConfig g_backend;

// "voice_isolator" is the offline checkpoint, "voice_isolator_realtime" the causal one.
std::filesystem::path weights(const std::string & name) {
    return g_model_dir / (name == "voice_isolator" ? "denoiser.safetensors" : "denoiser_realtime.safetensors");
}

std::shared_ptr<const engine::assets::TensorSource> fixture(const std::string & name) {
    return engine::assets::open_tensor_source(g_fixture_dir / (name + ".safetensors"));
}

// Every model of the run shares one execution context, as a session's models do.
engine::core::ExecutionContext & test_execution() {
    static engine::core::ExecutionContext execution(g_backend);
    return execution;
}

void require_close(const std::vector<float> & actual, const std::vector<float> & expected, const std::string & label) {
    require(actual.size() == expected.size(), label + " size " + std::to_string(actual.size()) + " != " + std::to_string(expected.size()));
    float max_diff = 0.0F;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    mean_diff /= static_cast<double>(std::max<size_t>(1, actual.size()));
    std::cout << label << ": max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
    if (max_diff > kMaxDiff || mean_diff > kMeanDiff) {
        std::ostringstream oss;
        oss << label << " mismatch: max_diff=" << max_diff << " mean_diff=" << mean_diff << " index=" << max_index
            << " expected=" << expected[max_index] << " actual=" << actual[max_index];
        throw std::runtime_error(oss.str());
    }
}

int64_t scalar(const engine::assets::TensorSource & source, const std::string & name) {
    return source.require_i64_scalar(name);
}

// The fixtures' test signal (make_fixtures.py: signal()).
std::vector<float> test_signal(int64_t samples) {
    std::vector<float> x(static_cast<size_t>(samples));
    const double pi = 3.14159265358979323846;
    uint32_t state = 12345U;
    for (int64_t i = 0; i < samples; ++i) {
        const double t = static_cast<double>(i) / 48000.0;
        const double envelope = 0.6 + 0.4 * std::sin(2.0 * pi * 3.0 * t);
        state = state * 1664525U + 1013904223U;
        const double noise = static_cast<double>(state) / 4294967296.0 * 2.0 - 1.0;
        x[static_cast<size_t>(i)] = static_cast<float>(0.3 * std::sin(2.0 * pi * 150.0 * t) * envelope +
            0.15 * std::sin(2.0 * pi * 450.0 * t + 0.5) * envelope + 0.05 * std::sin(2.0 * pi * 3100.0 * t) + 0.08 * noise);
    }
    return x;
}

std::vector<float> stream(engine::models::voice_isolator::StreamingSession & session, const std::vector<float> & input,
    const std::vector<size_t> & pieces) {
    std::vector<float> output;
    size_t start = 0;
    for (size_t i = 0; start < input.size(); ++i) {
        const size_t count = std::min(pieces[i % pieces.size()], input.size() - start);
        const auto part = session.push(input.data() + start, count);
        output.insert(output.end(), part.begin(), part.end());
        start += count;
    }
    const auto tail = session.flush();
    output.insert(output.end(), tail.begin(), tail.end());
    return output;
}

void offline_cases() {
    const auto model = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator"), test_execution());
    require(!model.streams(), "the offline checkpoint does not stream");
    const auto case0 = fixture("offline_case0");
    const auto input = case0->require_f32_tensor("input").values;
    require_close(input, test_signal(static_cast<int64_t>(input.size())), "fixture signal");
    require_close(model.isolate_mono_48k(input).samples, case0->require_f32_tensor("output").values, "offline_case0");
    engine::models::voice_isolator::Options reference;
    reference.weight_storage = engine::assets::TensorStorageType::F32;
    require_close(engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator"), test_execution(), reference)
        .isolate_mono_48k(input).samples, case0->require_f32_tensor("output").values, "offline_case0 F32 weights");

    const auto case1 = fixture("offline_case1");
    engine::models::voice_isolator::Options options;
    options.chunk_seconds = static_cast<double>(scalar(*case1, "chunk_samples")) / 48000.0;
    options.overlap_seconds = static_cast<double>(scalar(*case1, "overlap_samples")) / 48000.0;
    const auto chunked = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator"), test_execution(), options);
    require_close(chunked.isolate_mono_48k(case1->require_f32_tensor("input").values).samples,
        case1->require_f32_tensor("output").values, "offline_case1");
}

void realtime_case(const std::string & name, const std::vector<size_t> & pieces,
    engine::assets::TensorStorageType storage = engine::assets::TensorStorageType::Native) {
    const auto source = fixture(name);
    engine::models::voice_isolator::Options options;
    options.weight_storage = storage;
    options.attention_cache_storage = storage;
    options.lookahead_frames = scalar(*source, "lookahead_frames");
    const auto model = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator_realtime"), test_execution(), options);
    auto session = model.create_streaming_session();
    require(session->latency_samples() == scalar(*source, "latency_samples"), name + " latency");
    require_close(stream(*session, source->require_f32_tensor("input").values, pieces),
        source->require_f32_tensor("output").values, name);
}

void realtime_long() {
    const auto source = fixture("realtime_long");
    engine::models::voice_isolator::Options options;
    options.lookahead_frames = 2;
    const auto model = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator_realtime"), test_execution(), options);
    auto session = model.create_streaming_session();
    const auto output = stream(*session, test_signal(scalar(*source, "num_samples")), {960});
    const auto tail = source->require_f32_tensor("output_tail").values;
    require_close(std::vector<float>(output.end() - static_cast<std::ptrdiff_t>(tail.size()), output.end()), tail, "realtime_long tail");
    std::vector<float> decimated;
    for (size_t i = 0; i < output.size(); i += 100) {
        decimated.push_back(output[i]);
    }
    require_close(decimated, source->require_f32_tensor("output_decimated").values, "realtime_long decimated");
}

// The stream's boundaries: empty pushes, flushing early, reuse after reset,
// clips shorter than one frame, and look-aheads the checkpoint was not trained for.
void edge_cases() {
    const auto source = fixture("realtime_case2");
    engine::models::voice_isolator::Options options;
    options.lookahead_frames = scalar(*source, "lookahead_frames");
    const auto model = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator_realtime"), test_execution(), options);
    auto session = model.create_streaming_session();
    require(session->push(nullptr, 0).empty() && session->flush().empty(), "an empty stream yields nothing");
    bool refused = false;
    try {
        session->push(std::vector<float>(480, 0.0F));
    } catch (const std::exception &) {
        refused = true;
    }
    require(refused, "a flushed stream refuses more audio");
    session->reset();
    const auto input = source->require_f32_tensor("input").values;
    require_close(stream(*session, input, {960}), source->require_f32_tensor("output").values, "realtime_case2 after reset");

    const auto offline = engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator"), test_execution());
    const std::vector<float> clip(input.begin(), input.begin() + 700);
    for (const auto * candidate : {&model, &offline}) {
        const auto output = candidate->isolate_mono_48k(clip).samples;
        require(output.size() == clip.size(), "a sub-frame clip keeps its length");
        for (const float value : output) {
            require(std::isfinite(value), "a sub-frame clip stays finite");
        }
    }

    options.lookahead_frames = 3;
    refused = false;
    try {
        engine::models::voice_isolator::Model::load_from_safetensors(weights("voice_isolator_realtime"), test_execution(), options);
    } catch (const std::exception &) {
        refused = true;
    }
    require(refused, "an untrained look-ahead is refused");
}

// Rotating the cached keys back to a later base (a rebase) leaves a stream unchanged.
void rebase_case() {
    namespace vi = engine::models::voice_isolator;
    const auto path = weights("voice_isolator_realtime");
    const auto config = vi::parse_config(engine::io::load_safetensors_index(path).metadata.at("config"));
    const auto input = test_signal(scalar(*fixture("realtime_long"), "num_samples"));
    auto stream = [&](int64_t rebase_frames) {
        const auto source = engine::assets::open_tensor_source(path);
        vi::NetworkOptions options;
        options.attention_cache_storage = engine::assets::TensorStorageType::F32;
        options.rebase_frames = rebase_frames;
        const vi::Network network(config, *source, test_execution(), options);
        const auto & layout = network.layout();
        int64_t frames = 0;
        const auto spectra = vi::clip_spectra(config, input, frames);
        std::vector<float> packed(static_cast<size_t>(layout.bands() * frames * layout.encoder_inputs()));
        vi::pack_bands(layout, spectra.data(), frames, packed.data());
        std::vector<float> raw(static_cast<size_t>(layout.bands() * frames * layout.decoder_outputs()));
        network.streaming_backbone(2, 4)->run(packed.data(), frames, raw.data());
        return raw;
    };
    // Rebasing once positions pass 1600 frames moves the base back by 400 frames
    // at frame 1600 and again near frame 2000; the default never rebases in 20 s.
    require_close(stream(1600), stream(1 << 20), "rebased stream");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        namespace cli = minitts::cli;
        const auto model = cli::find_arg(argc, argv, "--model");
        const auto fixtures = cli::find_arg(argc, argv, "--fixtures");
        if (!model || !fixtures) {
            throw std::runtime_error("usage: --model <checkpoint dir> --fixtures <fixture dir> [--backend cpu] [--threads 8]");
        }
        g_model_dir = *model;
        g_fixture_dir = *fixtures;
        g_backend.type = cli::parse_backend(cli::find_arg(argc, argv, "--backend").value_or("cpu"));
        g_backend.threads = cli::parse_int_arg(argc, argv, "--threads", 8);
        cli::require_known_args(argc, argv);
        offline_cases();
        realtime_case("realtime_case0", {480, 960, 1440, 333, 2048});
        // F32 weights and cache are the reference path.
        realtime_case("realtime_case0", {480, 960, 1440, 333, 2048}, engine::assets::TensorStorageType::F32);
        realtime_case("realtime_case1", {960});
        realtime_case("realtime_case2", {960});
        realtime_long();
        edge_cases();
        rebase_case();
        std::cout << "voice_isolator reference test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "voice_isolator reference test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
