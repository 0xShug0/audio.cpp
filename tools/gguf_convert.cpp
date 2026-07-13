#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout
        << "Usage: audiocpp_gguf --input [namespace=]<weights> [--input namespace=<weights> ...] "
           "--output <weights.gguf> --type <f16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> "
           "[--root <model-dir>] [--sidecar <source>=<destination>] [--overwrite] [--no-sidecars]\n"
        << "       audiocpp_gguf --inspect <model.gguf>\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::vector<engine::assets::TensorSourceInput> inputs;
        std::vector<engine::assets::GgufEmbeddedFile> sidecars;
        std::filesystem::path output;
        std::filesystem::path inspect_path;
        std::filesystem::path sidecar_root;
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
            if ((arg == "--input" || arg == "--output" || arg == "--type" || arg == "--inspect" ||
                 arg == "--root" || arg == "--sidecar") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--input") {
                    const auto separator = value.find('=');
                    if (separator == std::string::npos) inputs.push_back({value, ""});
                    else inputs.push_back({value.substr(separator + 1), value.substr(0, separator)});
                }
                else if (arg == "--output") output = value;
                else if (arg == "--type") type = value;
                else if (arg == "--inspect") inspect_path = value;
                else if (arg == "--root") sidecar_root = value;
                else {
                    const auto separator = value.find('=');
                    if (separator == std::string::npos || separator == 0 || separator + 1 == value.size()) {
                        throw std::runtime_error("--sidecar requires <source>=<destination>");
                    }
                    sidecars.push_back({value.substr(0, separator), value.substr(separator + 1)});
                }
                continue;
            }
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
        if (!inspect_path.empty()) {
            const auto source = engine::assets::open_tensor_source(inspect_path);
            const auto tensors = source->tensors();
            size_t scalar_count = 0;
            std::set<std::string> namespaces;
            for (const auto & tensor : tensors) {
                if (tensor.shape.empty()) ++scalar_count;
                const auto separator = tensor.name.find('/');
                if (separator != std::string::npos) namespaces.insert(tensor.name.substr(0, separator));
            }
            std::cout << "gguf=" << std::filesystem::weakly_canonical(inspect_path).string() << "\n";
            std::cout << "tensors=" << tensors.size() << "\n";
            std::cout << "rank0_scalars=" << scalar_count << "\n";
            std::cout << "embedded_sidecars=" << (engine::assets::gguf_has_embedded_sidecars(inspect_path) ? "true" : "false") << "\n";
            for (const auto & value : namespaces) std::cout << "namespace=" << value << "\n";
            return 0;
        }
        if (inputs.empty() || output.empty() || type.empty()) {
            print_usage();
            return 2;
        }
        for (const auto & input : inputs) {
            if (!std::filesystem::is_regular_file(input.path)) {
                throw std::runtime_error("input tensor file does not exist: " + input.path.string());
            }
        }
        if (std::filesystem::exists(output)) {
            if (!overwrite) {
                throw std::runtime_error("output already exists; pass --overwrite to replace it: " + output.string());
            }
        }
        const auto storage_type = engine::assets::parse_tensor_storage_type(type);
        engine::assets::convert_tensor_sources_to_gguf(
            inputs, output, storage_type, overwrite, embed_sidecars, sidecar_root, sidecars);
        std::cout << "gguf=" << std::filesystem::weakly_canonical(output).string() << "\n";
        std::cout << "weight_type=" << type << "\n";
        std::cout << "tensor_sources=" << inputs.size() << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
