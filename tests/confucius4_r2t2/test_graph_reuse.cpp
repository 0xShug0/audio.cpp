#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/audio_encoder.h"
#include "engine/community_models/confucius4_r2t2/thinker.h"
#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"
#include "engine/community_models/confucius4_r2t2/types.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/community_models/confucius4_r2t2/frontend_whisper.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

int main(int argc, char ** argv) {
    namespace model = engine::community_models::confucius4_r2t2;
    std::filesystem::path path = std::filesystem::path(ENGINE_REPO_ROOT) / "models/Confucius4-R2T2";
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Cpu;
    backend.threads = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--model") { path = value; }
        else if (key == "--backend" && value == "metal") { backend.type = engine::core::BackendType::Metal; }
        else if (key != "--backend" || value != "cpu") { return 1; }
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "SKIP: graph reuse parity requires a Confucius4-R2T2 checkpoint\n";
        return 125;
    }
    // Fixture-specific alarms measured independently on CPU and Metal: CPU
    // padding noise is fp32 reduction order and reached about 1.9e-2 relative
    // RMSE at 63 padded tokens, while Metal stays under the 64-token kernel
    // guard and measured at most 7.6e-4. Neither tolerance substitutes for
    // the exact joint token checks below.
    const double encoder_drift_limit = backend.type == engine::core::BackendType::Cpu ? 2.5e-2 : 2e-3;
    try {
        auto assets = model::load_confucius4_r2t2_assets(path);
        engine::core::ExecutionContext execution(backend);
        model::R2T2ASRAudioEncoderRuntime exact(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        model::R2T2ASRAudioEncoderRuntime reused(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        model::R2T2ASRTextTokenizer tokenizer(assets);
        model::R2T2ASRThinkerRuntime thinker(assets, execution, 256ull << 20, 256ull << 20, 64ull << 20,
                                          engine::assets::TensorStorageType::Native);
        auto check_joint = [&](const model::R2T2ASRAudioEmbeddings & expected,
                               const model::R2T2ASRAudioEmbeddings & actual, const std::string & language) {
            const auto prompt = tokenizer.build_prompt("", language, expected.tokens);
            model::R2T2ASRGenerationOptions options;
            options.max_new_tokens = 32;
            const auto reference = thinker.generate(prompt, expected, options);
            options.reuse_graphs = true;
            const auto candidate = thinker.generate(prompt, actual, options);
            if (reference.token_ids.empty()) {
                throw std::runtime_error("real-audio joint regression produced no tokens");
            }
            if (reference.token_ids != candidate.token_ids) {
                throw std::runtime_error("joint encoder/decoder token mismatch: exact=" + tokenizer.decode(reference.token_ids) +
                                         " reused=" + tokenizer.decode(candidate.token_ids));
            }
            std::cout << "joint parity tokens=" << reference.token_ids.size() << " language=" << language << '\n';
        };
        // Exercise partial convolution chunks, capacity boundaries, and shrinking
        // after growth. Nonzero deterministic input exposes padding leakage.
        for (const int64_t frames : {32, 96, 100, 101, 199, 200, 201, 399, 400, 401, 479, 480, 481, 487, 488, 489, 799, 199, 101, 32}) {
            model::R2T2ASRAudioFeatures features;
            features.frames = frames;
            features.mel_bins = assets->config.audio_encoder.num_mel_bins;
            features.encoder_tokens = model::confucius4_r2t2_audio_encoder_token_count(frames);
            features.values.resize(frames * features.mel_bins);
            for (size_t i = 0; i < features.values.size(); ++i) {
                features.values[i] = 0.5f * std::sin(static_cast<float>(i) * 0.037f);
            }
            const auto expected = exact.encode(features);
            const auto actual = reused.encode(features, true);
            if (expected.tokens != actual.tokens || expected.values.size() != actual.values.size()) {
                throw std::runtime_error("encoder output shape changed");
            }
            const int64_t capacity = reused.graph_capacity_frames();
            const bool padded = model::confucius4_r2t2_audio_encoder_token_count(capacity) != features.encoder_tokens;
            if (backend.type == engine::core::BackendType::Metal && features.encoder_tokens < 64 &&
                model::confucius4_r2t2_audio_encoder_token_count(capacity) >= 64) {
                throw std::runtime_error("padding crossed the Metal attention precision boundary");
            }
            float max_error = 0;
            double error2 = 0, reference2 = 0;
            for (size_t i = 0; i < actual.values.size(); ++i) {
                const float error = std::abs(actual.values[i] - expected.values[i]);
                if (!std::isfinite(error)) { throw std::runtime_error("nonfinite encoder output"); }
                max_error = std::max(max_error, error);
                error2 += error * error;
                reference2 += expected.values[i] * expected.values[i];
            }
            const double relative_rmse = std::sqrt(error2 / std::max(reference2, 1e-20));
            std::cout << "frames=" << frames << (padded ? " padded=yes" : " padded=no")
                      << " capacity_frames=" << capacity << " max_error=" << max_error << " relative_rmse=" << relative_rmse << '\n';
            if (!padded) {
                if (max_error != 0.0F) { throw std::runtime_error("unpadded reusable graph differs from exact graph"); }
            } else {
                // This bound is only an embedding drift alarm, not evidence of
                // transcript equivalence or a diagnosis of the numerical cause.
                // Joint encoder/decoder comparisons below check observable tokens.
                if (relative_rmse > encoder_drift_limit) {
                    throw std::runtime_error("padded encoder drift " + std::to_string(relative_rmse) +
                                             " exceeded backend fixture limit " + std::to_string(encoder_drift_limit));
                }
                const auto again = reused.encode(features, true);
                if (again.values != actual.values) { throw std::runtime_error("padded encoder output is not deterministic"); }
            }
        }
        // Real audio prefixes cover both automatic and forced language prompts,
        // then shrink to exercise capacity selection and clearing previous inputs.
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(ENGINE_REPO_ROOT) / "assets/resources/sample_16k.wav");
        model::R2T2ASRWhisperFrontend frontend(assets);
        for (const double seconds : {1.01, 4.01, 4.89, 8.01, 1.99}) {
            engine::runtime::AudioBuffer audio;
            audio.sample_rate = wav.sample_rate;
            audio.channels = wav.channels;
            const size_t count = std::min(wav.samples.size(), static_cast<size_t>(seconds * wav.sample_rate) * wav.channels);
            audio.samples.assign(wav.samples.begin(), wav.samples.begin() + count);
            const auto features = frontend.extract(audio);
            const auto expected = exact.encode(features);
            const auto actual = reused.encode(features, true);
            check_joint(expected, actual, "");
            check_joint(expected, actual, "English");
        }
        // Repeated prompts grow then shrink across prefill blocks and KV buckets.
        for (const int64_t tokens : {4, 65, 129, 7, 65}) {
            const auto prompt = tokenizer.build_prompt("", "English", tokens);
            model::R2T2ASRAudioEmbeddings embeddings;
            embeddings.tokens = tokens;
            embeddings.hidden_size = assets->config.text_decoder.hidden_size;
            embeddings.values.resize(tokens * embeddings.hidden_size);
            for (size_t i = 0; i < embeddings.values.size(); ++i) {
                embeddings.values[i] = 0.1f * std::cos(static_cast<float>(i) * 0.013f);
            }
            model::R2T2ASRGenerationOptions options;
            options.max_new_tokens = 8;
            const auto expected = thinker.generate(prompt, embeddings, options);
            options.reuse_graphs = true;
            const auto actual = thinker.generate(prompt, embeddings, options);
            if (expected.token_ids != actual.token_ids) { throw std::runtime_error("reused decoder token sequence changed"); }
            std::cout << "injection_tokens=" << tokens << " decoder parity passed\n";
        }
        std::cout << "PASS: bounded encoder drift and joint token parity after growth and shrink\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
