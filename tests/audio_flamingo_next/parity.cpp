#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/audio_flamingo_next/audio_encoder.h"
#include "engine/models/audio_flamingo_next/decoder.h"
#include "engine/models/audio_flamingo_next/frontend.h"
#include "engine/models/audio_flamingo_next/projector.h"
#include "engine/models/audio_flamingo_next/tokenizer_text.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace engine;
using namespace engine::models::audio_flamingo_next;

bool compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error(name + " shape mismatch");
    double dot = 0, aa = 0, bb = 0, error = 0, max_error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i], d = a - b;
        dot += a * b;
        aa += a * a;
        bb += b * b;
        error += d * d;
        max_error = std::max(max_error, std::abs(d));
    }
    const double cosine = dot / std::sqrt(aa * bb);
    debug::trace_log_scalar(name + ".cosine", cosine);
    debug::trace_log_scalar(name + ".rmse", std::sqrt(error / actual.size()));
    debug::trace_log_scalar(name + ".max_abs", max_error);
    return cosine >= 0.999;
}

int main(int argc, char ** argv) {
    try {
        if ((argc != 6 && argc != 7) || std::string(argv[4]) != "--log") {
            throw std::runtime_error("usage: audio_flamingo_next_parity MODEL FIXTURE native|f32 --log LOG_FILE [AUDIO_WAV]");
        }
        debug::configure_logging({true, std::string(argv[5])});
        const auto started = std::chrono::steady_clock::now();
        auto assets = load_af_next_assets(argv[1]);
        auto fixture = assets::open_tensor_source(argv[2]);
        const auto storage = assets::parse_tensor_storage_type(argv[3]);
        core::ExecutionContext execution({core::BackendType::Cuda, 0, 8});
        auto tensor = fixture->require_f32_tensor("features");
        AFNextAudioFeatures features;
        features.batch = features.windows = tensor.shape.dims[0];
        features.mel_bins = tensor.shape.dims[1];
        features.frames = tensor.shape.dims[2];
        features.values = std::move(tensor.values);
        bool passed = true;
        if (argc == 7) {
            auto wav = audio::read_wav_f32(std::filesystem::path(argv[6]));
            runtime::AudioBuffer input;
            input.sample_rate = wav.sample_rate;
            input.channels = wav.channels;
            input.samples = std::move(wav.samples);
            AFNextFrontend frontend(assets);
            const auto actual = frontend.extract(input);
            passed = compare("frontend", actual.values, features.values);
            const size_t window_size = static_cast<size_t>(features.mel_bins * features.frames);
            for (int64_t window = 0; window < features.batch; ++window) {
                const size_t offset = static_cast<size_t>(window) * window_size;
                compare("frontend.window_" + std::to_string(window),
                    {actual.values.begin() + offset, actual.values.begin() + offset + window_size},
                    {features.values.begin() + offset, features.values.begin() + offset + window_size});
            }
            features.values = actual.values;
        }
        const auto mask = fixture->require_f32("mask");
        features.attention_mask.assign(mask.begin(), mask.end());
        for (int64_t w = 0; w < features.batch; ++w) {
            const auto start = features.attention_mask.begin() + w * features.frames;
            features.post_lengths.push_back(af_next_post_length(std::accumulate(start, start + features.frames, int64_t{0})));
        }
        AFNextTextTokenizer tokenizer(assets);
        const auto prompt = tokenizer.build_prompt("Transcribe the input speech.", features);
        const auto expected_ids = fixture->require_f32("input_ids");
        if (expected_ids.size() != prompt.input_ids.size() ||
            !std::equal(expected_ids.begin(), expected_ids.end(), prompt.input_ids.begin())) {
            throw std::runtime_error("prompt token mismatch");
        }
        {
            AFNextAudioEncoderRuntime encoder(assets, execution, 4 * 1024 * 1024, storage);
            AFNextAudioProjectorRuntime projector(assets, execution, 4 * 1024 * 1024, storage);
            const auto encoded = encoder.encode(features);
            passed = compare("encoder", encoded.values, fixture->require_f32("encoded")) && passed;
            const auto projected = projector.project(encoded, features, prompt);
            passed = compare("projector", projected.values, fixture->require_f32("projected")) && passed;
        }
        {
            AFNextAudioProjectorOutput projected;
            projected.values = fixture->require_f32("projected");
            projected.hidden_size = assets->config.text_decoder.hidden_size;
            projected.tokens = projected.values.size() / projected.hidden_size;
            AFNextQwen2DecoderRuntime decoder(assets, execution, storage);
            AFNextGenerationOptions options;
            options.max_new_tokens = 128;
            const auto output = decoder.generate(prompt, projected, options);
            debug::trace_log_scalar("decoder.text", tokenizer.decode(output.token_ids));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
