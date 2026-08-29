#include "args.h"

#include <iostream>
#include <set>
#include <stdexcept>

namespace minitts::cli {

namespace {

// Every option name the CLI looked for, split by whether the lookup consumes the following
// argument. Populated by the lookups below, so a new flag is recognised the moment its lookup
// is added and there is no second list to keep in sync.
std::set<std::string> & value_names() {
    static std::set<std::string> names;
    return names;
}

std::set<std::string> & flag_names() {
    static std::set<std::string> names;
    return names;
}

void record_value_query(const std::string & name) {
    value_names().insert(name);
}

void record_flag_query(const std::string & name) {
    flag_names().insert(name);
}

}  // namespace

void report_unused_args(int argc, char ** argv) {
    const auto & values = value_names();
    const auto & flags = flag_names();
    std::vector<std::string> unused;
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token.rfind("--", 0) != 0 || token == "--") {
            continue;
        }
        // Only an option that takes a value consumes the next argument, so only then can the
        // next argument be something that merely looks like an option.
        if (values.count(token) != 0) {
            ++i;
            continue;
        }
        if (flags.count(token) != 0) {
            continue;
        }
        unused.push_back(token);
    }
    if (unused.empty()) {
        return;
    }
    std::cerr << "audiocpp_cli warning: ignored unknown option";
    if (unused.size() > 1) {
        std::cerr << "s";
    }
    std::cerr << ":";
    for (const auto & token : unused) {
        std::cerr << " " << token;
    }
    std::cerr << "\n";
}

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name) {
    record_value_query(name);
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

bool has_arg(int argc, char ** argv, const std::string & name) {
    record_flag_query(name);
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> collect_args(int argc, char ** argv, const std::string & name) {
    record_value_query(name);
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
