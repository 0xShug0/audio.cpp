#pragma once

#include "engine/framework/core/backend.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace minitts::cli {

// Throws if the command line carries a --flag that no lookup ever asked for, so a typo like
// --bakend is refused instead of being silently discarded. Call once every option has been
// read, which on the run paths is before the model does any work.
void require_known_args(int argc, char ** argv);

// The same check reported as a warning, for the informational commands. Those return before the
// rest of the options are read, so all that can honestly be said there is that an option was
// ignored, not that it was misspelled. Does nothing if require_known_args already ran.
void warn_ignored_args(int argc, char ** argv);

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
