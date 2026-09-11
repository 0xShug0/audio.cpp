#include "engine/community_models/niagara_asr/assets.h"
#include "engine/community_models/niagara_asr/weights.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::filesystem::path model;
    int threads = 8;
};

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (key == "--model") {
            args.model = require_value("--model");
        } else if (key == "--threads") {
            args.threads = std::stoi(require_value("--threads"));
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model.empty()) {
        throw std::runtime_error("--model is required");
    }
    return args;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = parse_args(argc, argv);
        auto assets = engine::community_models::niagara_asr::load_niagara_asr_assets(args.model);
        engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, args.threads});
        const auto weights = engine::community_models::niagara_asr::load_niagara_weights(
            *assets,
            execution,
            engine::assets::TensorStorageType::F32,
            512ull * 1024ull * 1024ull);
        const auto & encoder = assets->config.encoder;
        std::cout << "family=niagara_asr\n";
        std::cout << "hidden_size=" << encoder.hidden_size << "\n";
        std::cout << "intermediate_size=" << encoder.intermediate_size << "\n";
        std::cout << "state_channels=" << encoder.state_channels << "\n";
        std::cout << "state_size=" << encoder.state_size << "\n";
        std::cout << "num_layers=" << encoder.num_layers << "\n";
        std::cout << "vocab_size=" << assets->config.vocab_size << "\n";
        std::cout << "loaded_layers=" << weights->layers.size() << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "niagara_asr_weight_load_probe failed: " << ex.what() << "\n";
        return 1;
    }
}
