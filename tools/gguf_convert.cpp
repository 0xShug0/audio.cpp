#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout
        << "Usage: audiocpp_gguf --input <weights.safetensors|model.safetensors.index.json> --output <weights.gguf> "
           "--type <f16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> [--overwrite] [--no-sidecars]\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::filesystem::path input;
        std::filesystem::path output;
        std::string type;
        bool overwrite = false;
        bool embed_sidecars = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--overwrite") {
                overwrite = true;
                continue;
            }
            if (arg == "--no-sidecars") {
                embed_sidecars = false;
                continue;
            }
            if ((arg == "--input" || arg == "--output" || arg == "--type") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--input") input = value;
                else if (arg == "--output") output = value;
                else type = value;
                continue;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        if (input.empty() || output.empty() || type.empty()) {
            print_usage();
            return 2;
        }
        if (!std::filesystem::is_regular_file(input)) {
            throw std::runtime_error("input tensor file does not exist: " + input.string());
        }
        if (std::filesystem::exists(output)) {
            if (!overwrite) {
                throw std::runtime_error("output already exists; pass --overwrite to replace it: " + output.string());
            }
        }
        const auto storage_type = engine::assets::parse_tensor_storage_type(type);
        engine::assets::convert_tensor_source_to_gguf(input, output, storage_type, overwrite, embed_sidecars);
        std::cout << "gguf=" << std::filesystem::weakly_canonical(output).string() << "\n";
        std::cout << "weight_type=" << type << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
