#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/audio_encoder.h"
#include "engine/community_models/confucius4_r2t2/thinker.h"
#include "engine/community_models/confucius4_r2t2/tokenizer_text.h"
#include "engine/community_models/confucius4_r2t2/types.h"
#include "engine/framework/core/execution_context.h"

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
    try {
        auto assets = model::load_confucius4_r2t2_assets(path);
        engine::core::ExecutionContext execution(backend);
        model::R2T2ASRAudioEncoderRuntime exact(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        model::R2T2ASRAudioEncoderRuntime reused(assets, execution, 128ull << 20, engine::assets::TensorStorageType::Native);
        // Exercise partial convolution chunks, capacity boundaries, and shrinking
        // after growth. Nonzero deterministic input exposes padding leakage.
        for (const int64_t frames : {32, 96, 100, 101, 199, 200, 201, 399, 400, 401, 799, 101, 32}) {
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
            // Mirror the runtime's bucketing to know whether this run computed
            // padded tokens at all. Without padded tokens the reusable graph
            // must match the exact graph bit for bit.
            const int64_t chunk_frames = assets->config.audio_encoder.n_window * 2;
            const int64_t chunks = (frames + chunk_frames - 1) / chunk_frames;
            const int64_t bucket_chunks = chunks <= 2 ? chunks : (chunks + 3) / 4 * 4;
            const bool padded = frames >= chunk_frames &&
                model::confucius4_r2t2_audio_encoder_token_count(bucket_chunks * chunk_frames) != features.encoder_tokens;
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
                      << " max_error=" << max_error << " relative_rmse=" << relative_rmse << '\n';
            if (!padded) {
                if (max_error != 0.0F) { throw std::runtime_error("unpadded reusable graph differs from exact graph"); }
            } else {
                // Padded tokens change the reduction order inside softmax and
                // attention value sums, so bit equality is impossible. Deep
                // stacks amplify the kernel-order noise; the observed ceiling
                // is about 2e-2 relative RMSE at 63 padded tokens. The real
                // correctness bar is greedy decoder parity below plus the
                // end-to-end streaming transcript check.
                if (relative_rmse > 2e-2) { throw std::runtime_error("padded encoder drift exceeded the noise budget"); }
                const auto again = reused.encode(features, true);
                if (again.values != actual.values) { throw std::runtime_error("padded encoder output is not deterministic"); }
            }
        }
        model::R2T2ASRTextTokenizer tokenizer(assets);
        model::R2T2ASRThinkerRuntime thinker(assets, execution, 256ull << 20, 256ull << 20, 64ull << 20,
                                          engine::assets::TensorStorageType::Native);
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
        std::cout << "PASS: graph reuse matches exact encoder and decoder after growth and shrink\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
