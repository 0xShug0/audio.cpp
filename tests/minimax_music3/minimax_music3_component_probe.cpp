// Component-level probe for MiniMax-Music3 parity testing.
//
// Reads raw float32 inputs, runs one pipeline component, and writes raw float32
// output, so the Python reference (tests/minimax_music3/reference_dump.py) can
// compare intermediates without going through the full pipeline.

#include "engine/community_models/minimax_music3/assets.h"
#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/dit.h"
#include "engine/community_models/minimax_music3/lm.h"
#include "engine/community_models/minimax_music3/tokenizer_text.h"
#include "engine/community_models/minimax_music3/types.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/core/backend_weight_store.h"

#include <algorithm>
#include <chrono>
#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    const auto bytes = static_cast<size_t>(input.tellg());
    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error("input file size is not a multiple of float32: " + path);
    }
    std::vector<float> values(bytes / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return values;
}

void write_f32_file(const std::string & path, const std::vector<float> & values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    output.write(
        reinterpret_cast<const char *>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

struct Args {
    std::string model;
    std::string component;
    std::string input;
    std::string input2;
    std::string output;
    std::string backend = "cuda";
    int64_t length = 0;
};

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto next = [&]() -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            return argv[++index];
        };
        if (key == "--model") {
            args.model = next();
        } else if (key == "--component") {
            args.component = next();
        } else if (key == "--input") {
            args.input = next();
        } else if (key == "--input2") {
            args.input2 = next();
        } else if (key == "--output") {
            args.output = next();
        } else if (key == "--backend") {
            args.backend = next();
        } else if (key == "--length") {
            args.length = std::stoll(next());
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model.empty() || args.component.empty() || args.input.empty() || args.output.empty()) {
        throw std::runtime_error(
            "usage: minimax_music3_component_probe --model <package-or-lm-gguf> --component vocoder "
            "--input <in.f32> --output <out.f32> [--backend cuda|cpu] [--length N]");
    }
    return args;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        namespace core = engine::core;
        namespace music3 = engine::models::minimax_music3;

        auto assets = music3::load_minimax_music3_assets(args.model);
        core::BackendConfig backend_config;
        backend_config.type = args.backend == "cpu" ? core::BackendType::Cpu : core::BackendType::Cuda;
        backend_config.device = 0;
        backend_config.threads = 8;
        core::ExecutionContext execution(backend_config);

        const auto input = args.component == "tokenizer" ? std::vector<float>() : read_f32_file(args.input);
        if (args.component == "vocoder") {
            const auto & config = assets->config;
            int64_t length = args.length;
            if (length == 0) {
                length = static_cast<int64_t>(input.size()) / config.vocoder_latent_channels;
            }
            music3::MiniMaxMusic3VocoderRuntime vocoder(
                execution,
                assets->vocoder_weights,
                config,
                512ull * 1024ull * 1024ull);
            write_f32_file(args.output, vocoder.decode(input, length));
        } else if (args.component == "condition_encoder") {
            const auto & config = assets->config;
            const int64_t row = config.cond_layers * config.cond_hidden;
            int64_t frames = args.length;
            if (frames == 0) {
                frames = static_cast<int64_t>(input.size()) / row;
            }
            music3::MiniMaxMusic3ConditionEncoderRuntime encoder(
                execution,
                assets->condition_encoder_weights,
                config,
                512ull * 1024ull * 1024ull);
            write_f32_file(args.output, encoder.encode(input, frames));
        } else if (args.component == "depth_decoder") {
            // Deterministic frame decode: input = last_hidden [2, hidden], the semantic
            // code comes from --length, codes picked by argmax over the CFG-guided logits.
            const auto & config = assets->config;
            engine::core::BackendWeightStore embed_store(
                execution.backend(),
                execution.backend_type(),
                "minimax_music3.probe.lm_embed",
                2048ull * 1024ull * 1024ull);
            auto lm_embedding = embed_store.load_tensor(
                *assets->lm_weights,
                "model.embed_tokens.weight",
                engine::assets::TensorStorageType::Native,
                {config.lm_vocab_size, config.lm_hidden});
            embed_store.upload();
            music3::MiniMaxMusic3DepthDecoderRuntime depth(
                execution,
                assets->depth_decoder_weights,
                lm_embedding,
                config,
                2048ull * 1024ull * 1024ull);
            // Empty Gumbel noise selects greedy decoding, matching the reference argmax rollout.
            const auto frame = depth.decode_frame(input, static_cast<int32_t>(args.length), {});
            {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < 20; ++iteration) {
                    (void)depth.decode_frame(input, static_cast<int32_t>(args.length), {});
                }
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
                std::cerr << "depth frame avg ms: " << (static_cast<double>(elapsed) / 20.0) << "\n";
            }
            std::vector<float> packed;
            for (const int32_t code : frame.codes) {
                packed.push_back(static_cast<float>(code));
            }
            packed.insert(packed.end(), frame.depth_hidden.begin(), frame.depth_hidden.end());
            packed.insert(packed.end(), frame.feedback_embedding.begin(), frame.feedback_embedding.end());
            write_f32_file(args.output, packed);
        } else if (args.component == "tokenizer") {
            // input = caption text file, input2 = lyrics text file; output = cond and
            // uncond ids as float32.
            std::ifstream caption_file(args.input), lyrics_file(args.input2);
            std::stringstream caption, lyrics;
            caption << caption_file.rdbuf();
            lyrics << lyrics_file.rdbuf();
            music3::MiniMaxMusic3TextTokenizer tokenizer(assets->resources);
            const auto prompt = tokenizer.encode_prompt(caption.str(), lyrics.str());
            std::vector<float> packed;
            packed.push_back(static_cast<float>(prompt.cond_ids.size()));
            for (const int32_t id : prompt.cond_ids) {
                packed.push_back(static_cast<float>(id));
            }
            for (const int32_t id : prompt.uncond_ids) {
                packed.push_back(static_cast<float>(id));
            }
            write_f32_file(args.output, packed);
        } else if (args.component == "dit") {
            // input = latent [128, L] then condition [2048, L] channel-major concatenated;
            // --length = L. Guidance 1.0 reduces the CFG mix to the conditional branch.
            const auto & config = assets->config;
            const int64_t length = args.length;
            const size_t latent_size = static_cast<size_t>(config.dit_in_channels * length);
            const size_t cond_size = static_cast<size_t>(config.dit_condition_dim * length);
            if (input.size() != latent_size + cond_size) {
                throw std::runtime_error("dit probe input size mismatch");
            }
            std::vector<float> latent(input.begin(), input.begin() + static_cast<int64_t>(latent_size));
            std::vector<float> condition(input.begin() + static_cast<int64_t>(latent_size), input.end());
            music3::MiniMaxMusic3DitRuntime dit(
                execution, assets->dit_weights, config, 512ull * 1024ull * 1024ull);
            dit.begin_chunk(condition, length);
            write_f32_file(args.output, dit.guided_velocity(latent, 0.5F, 1.0F));
            {
                const auto start = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < 10; ++iteration) {
                    (void)dit.guided_velocity(latent, 0.5F, 1.0F);
                }
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
                std::cerr << "dit forward avg ms: " << (static_cast<double>(elapsed) / 10.0) << "\n";
            }
        } else if (args.component == "lm_prefill") {
            // input = token ids as float32 (cond then uncond, equal length).
            const auto & config = assets->config;
            const size_t half = input.size() / 2;
            std::vector<int32_t> cond_ids, uncond_ids;
            for (size_t index = 0; index < half; ++index) {
                cond_ids.push_back(static_cast<int32_t>(input[index]));
                uncond_ids.push_back(static_cast<int32_t>(input[half + index]));
            }
            music3::MiniMaxMusic3LmRuntime lm(
                execution, assets->lm_weights, config, 512ull * 1024ull * 1024ull);
            auto step = lm.prefill(cond_ids, uncond_ids, static_cast<int64_t>(half) + 8);
            std::vector<float> packed;
            packed.insert(packed.end(), step.cond_logits.begin(), step.cond_logits.end());
            packed.insert(packed.end(), step.uncond_logits.begin(), step.uncond_logits.end());
            packed.insert(packed.end(), step.last_hidden.begin(), step.last_hidden.end());
            write_f32_file(args.output, packed);
        } else if (args.component == "lm_decode") {
            // input = embedding [hidden] then cond ids then uncond ids (equal length).
            const auto & config = assets->config;
            const size_t hidden = static_cast<size_t>(config.lm_hidden);
            const size_t half = (input.size() - hidden) / 2;
            std::vector<float> embedding(input.begin(), input.begin() + static_cast<int64_t>(hidden));
            std::vector<int32_t> cond_ids, uncond_ids;
            for (size_t index = 0; index < half; ++index) {
                cond_ids.push_back(static_cast<int32_t>(input[hidden + index]));
                uncond_ids.push_back(static_cast<int32_t>(input[hidden + half + index]));
            }
            music3::MiniMaxMusic3LmRuntime lm(
                execution, assets->lm_weights, config, 512ull * 1024ull * 1024ull);
            (void)lm.prefill(cond_ids, uncond_ids, static_cast<int64_t>(half) + 8);
            const auto step = lm.decode_embedding(embedding);
            std::vector<float> packed;
            packed.insert(packed.end(), step.cond_logits.begin(), step.cond_logits.end());
            packed.insert(packed.end(), step.uncond_logits.begin(), step.uncond_logits.end());
            packed.insert(packed.end(), step.last_hidden.begin(), step.last_hidden.end());
            write_f32_file(args.output, packed);
        } else {
            throw std::runtime_error("unknown component: " + args.component);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "minimax_music3_component_probe: " << error.what() << "\n";
        return 1;
    }
}
