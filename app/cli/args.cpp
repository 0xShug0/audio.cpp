#include "args.h"

#include <stdexcept>

namespace minitts::cli {

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool has_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> collect_args(int argc, char ** argv, const std::string & name) {
    std::vector<std::string> values;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            values.emplace_back(argv[i + 1]);
        }
    }
    return values;
}

std::unordered_map<std::string, std::string> collect_key_value_args(
    int argc,
    char ** argv,
    const std::string & name) {
    std::unordered_map<std::string, std::string> values;
    for (const auto & item : collect_args(argc, argv, name)) {
        const auto pos = item.find('=');
        if (pos == std::string::npos || pos == 0 || pos + 1 >= item.size()) {
            throw std::runtime_error("expected " + name + " key=value, got: " + item);
        }
        values[item.substr(0, pos)] = item.substr(pos + 1);
    }
    return values;
}

void set_option(
    std::unordered_map<std::string, std::string> & options,
    const std::string & key,
    const std::string & value) {
    const auto [it, inserted] = options.emplace(key, value);
    if (!inserted && it->second != value) {
        throw std::runtime_error("conflicting request option value for " + key);
    }
}

void set_option_from_arg(
    int argc,
    char ** argv,
    const std::string & arg_name,
    const std::string & option_key,
    std::unordered_map<std::string, std::string> & options) {
    if (const auto value = find_arg(argc, argv, arg_name)) {
        set_option(options, option_key, *value);
    }
}

int parse_int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    if (const auto value = find_arg(argc, argv, name)) {
        return std::stoi(*value);
    }
    return fallback;
}

std::optional<float> parse_optional_float_arg(int argc, char ** argv, const std::string & name) {
    if (const auto value = find_arg(argc, argv, name)) {
        return std::stof(*value);
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> optional_path_arg(int argc, char ** argv, const std::string & name) {
    if (const auto value = find_arg(argc, argv, name)) {
        return std::filesystem::path(*value);
    }
    return std::nullopt;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "hip" || value == "rocm") {
        return engine::core::BackendType::Hip;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

}  // namespace minitts::cli
