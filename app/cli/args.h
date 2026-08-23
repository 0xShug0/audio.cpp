#pragma once

#include "engine/framework/core/backend.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace minitts::cli {

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name);
bool has_arg(int argc, char ** argv, const std::string & name);
std::vector<std::string> collect_args(int argc, char ** argv, const std::string & name);
std::unordered_map<std::string, std::string> collect_key_value_args(
    int argc,
    char ** argv,
    const std::string & name);
void set_option(
    std::unordered_map<std::string, std::string> & options,
    const std::string & key,
    const std::string & value);
void set_option_from_arg(
    int argc,
    char ** argv,
    const std::string & arg_name,
    const std::string & option_key,
    std::unordered_map<std::string, std::string> & options);
int parse_int_arg(int argc, char ** argv, const std::string & name, int fallback);
std::optional<float> parse_optional_float_arg(int argc, char ** argv, const std::string & name);
std::optional<std::filesystem::path> optional_path_arg(int argc, char ** argv, const std::string & name);
engine::core::BackendType parse_backend(const std::string & value);

}  // namespace minitts::cli
