#include "args.h"

#include "engine/framework/io/yaml.h"

#include <charconv>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace minitts::cli {

namespace {

const std::set<std::string> & boolean_flags() {
    static const std::set<std::string> flags = {
        "--help",
        "--inspect",
        "--json",
        "--list-devices",
        "--list-loaders",
        "--list-pipelines",
        "--list-recipes",
        "--log",
        "--metrics",
        "--version",
    };
    return flags;
}

const std::set<std::string> & value_options() {
    // expand_cli_args runs before runtime lookups populate value_names().
    static const std::set<std::string> options = {
        "--audio", "--audio-chunk-mode", "--audio-chunk-seconds", "--audio-converter", "--backend",
        "--backend-impl", "--batch-audio-dir", "--batch-audio-role", "--batch-manifest-out",
        "--batch-merge-audio", "--batch-text-dir", "--batch-text-file", "--config", "--device",
        "--do-sample", "--duration-seconds", "--emotion", "--energy-scale", "--family",
        "--guidance-scale", "--input-channels", "--input-format", "--input-rate", "--instruct",
        "--language", "--load-option", "--load-progress", "--log-file", "--lyrics", "--max-steps",
        "--max-tokens", "--mode", "--model", "--model-spec-override", "--moe-cpu-share",
        "--moe-disk-experts", "--moe-pin-experts", "--moe-expert-cache-size",
        "--moe-disk-expert-read-threads", "--moe-cpu-expert-layers",
        "--num-beams", "--num-inference-steps", "--out", "--out-dir", "--out-format",
        "--pitch-shift", "--pipeline", "--predict-target-prosody", "--prosody-ref",
        "--prosody-shift-steps", "--repaint-end", "--repaint-mode", "--repaint-start",
        "--repaint-strength", "--repetition-penalty", "--registry-config", "--reference-duration-seconds",
        "--reference-text", "--recipe", "--recipe-family", "--recipe-model", "--request-option",
        "--request-sequence", "--resample-kernel-cache-mb", "--seed", "--segments-out",
        "--session-option", "--source-audio", "--source-shift-steps", "--speaker", "--speaking-rate",
        "--style-language", "--style-ref", "--style-ref-text", "--style-shift-steps", "--style-tag",
        "--target-duration-seconds", "--target-text", "--target-voice", "--task", "--task-route",
        "--temperature", "--text", "--text-chunk-mode", "--text-chunk-size", "--text-out",
        "--threads", "--top-k", "--top-p", "--track-name", "--turns-out", "--use-pitch-shift",
        "--use-prosody-code", "--vad-chunk-max-seconds", "--vad-chunk-merge-gap-seconds",
        "--vad-chunk-padding-seconds", "--vad-chunks-out", "--voice-id", "--voice-ref",
        "--voice-state-out", "--weight", "--workflow-input", "--words-out",
    };
    return options;
}

std::string normalize_config_option(std::string key) {
    if (key.rfind("--", 0) == 0) {
        key.erase(0, 2);
    }
    if (key.empty() || key.find_first_of(" \t\r\n=") != std::string::npos) {
        throw std::runtime_error("invalid CLI option key in YAML: " + key);
    }
    const std::string option = "--" + key;
    if (option == "--config-file") {
        throw std::runtime_error("CLI YAML config cannot include --config-file");
    }
    return option;
}

std::set<std::string> explicit_key_value_overrides(
    const std::vector<std::string> & arguments,
    const std::string & option) {
    std::set<std::string> keys;
    for (size_t i = 1; i + 1 < arguments.size(); ++i) {
        if (arguments[i] != option) {
            if (value_options().count(arguments[i]) != 0 && i + 1 < arguments.size()) {
                ++i;
            }
            continue;
        }
        const auto separator = arguments[i + 1].find('=');
        if (separator != std::string::npos && separator > 0) {
            keys.insert(arguments[i + 1].substr(0, separator));
        }
        ++i;
    }
    return keys;
}

const std::unordered_map<std::string, std::string> & request_option_aliases() {
    static const std::unordered_map<std::string, std::string> option_keys = {
        {"--language", "language"},
        {"--task-route", "route"},
        {"--source-audio", "source_audio"},
        {"--target-voice", "target_voice"},
        {"--prosody-ref", "prosody_ref"},
        {"--style-ref", "style_ref"},
        {"--target-text", "target_text"},
        {"--style-ref-text", "style_ref_text"},
        {"--lyrics", "lyrics"},
        {"--track-name", "track_name"},
        {"--speaker", "speaker"},
        {"--duration-seconds", "duration_sec"},
        {"--repaint-start", "repainting_start"},
        {"--repaint-end", "repainting_end"},
        {"--repaint-mode", "repaint_mode"},
        {"--repaint-strength", "repaint_strength"},
        {"--seed", "seed"},
        {"--max-tokens", "max_tokens"},
        {"--max-steps", "max_steps"},
        {"--temperature", "temperature"},
        {"--top-k", "top_k"},
        {"--top-p", "top_p"},
        {"--repetition-penalty", "repetition_penalty"},
        {"--do-sample", "do_sample"},
        {"--num-beams", "num_beams"},
        {"--guidance-scale", "guidance_scale"},
        {"--num-inference-steps", "num_inference_steps"},
        {"--text-chunk-size", "text_chunk_size"},
        {"--text-chunk-mode", "text_chunk_mode"},
        {"--audio-chunk-seconds", "audio_chunk_duration_sec"},
        {"--audio-chunk-mode", "audio_chunk_mode"},
        {"--use-prosody-code", "use_prosody_code"},
        {"--predict-target-prosody", "predict_target_prosody"},
        {"--use-pitch-shift", "use_pitch_shift"},
        {"--source-shift-steps", "source_shift_steps"},
        {"--prosody-shift-steps", "prosody_shift_steps"},
        {"--style-shift-steps", "style_shift_steps"},
        {"--target-duration-seconds", "target_duration_seconds"},
        {"--reference-duration-seconds", "reference_duration_seconds"},
        {"--reference-text", "reference_text"},
        {"--instruct", "instruct"},
    };
    return option_keys;
}

const std::unordered_map<std::string, std::string> & moe_session_option_aliases() {
    static const std::unordered_map<std::string, std::string> option_keys = {
        {"--moe-cpu-share", "moe_cpu_share"},
        {"--moe-disk-experts", "moe_disk_experts"},
        {"--moe-pin-experts", "moe_pin_experts"},
        {"--moe-expert-cache-size", "moe_expert_cache_size"},
        {"--moe-disk-expert-read-threads", "moe_disk_expert_read_threads"},
        {"--moe-cpu-expert-layers", "moe_cpu_expert_layers"},
    };
    return option_keys;
}

std::set<std::string> explicit_request_option_overrides(const std::vector<std::string> & arguments) {
    const auto & option_keys = request_option_aliases();
    std::set<std::string> keys = explicit_key_value_overrides(arguments, "--request-option");
    for (size_t i = 1; i < arguments.size(); ++i) {
        const auto option = option_keys.find(arguments[i]);
        if (option != option_keys.end()) {
            keys.insert(option->second);
            ++i;
        } else if (value_options().count(arguments[i]) != 0 && i + 1 < arguments.size()) {
            ++i;
        }
    }
    return keys;
}

std::set<std::string> explicit_session_option_overrides(const std::vector<std::string> & arguments) {
    std::set<std::string> keys = explicit_key_value_overrides(arguments, "--session-option");
    const auto & option_keys = moe_session_option_aliases();
    for (size_t i = 1; i < arguments.size(); ++i) {
        const auto option = option_keys.find(arguments[i]);
        if (option != option_keys.end()) {
            keys.insert(option->second);
            ++i;
        } else if (value_options().count(arguments[i]) != 0 && i + 1 < arguments.size()) {
            ++i;
        }
    }
    return keys;
}

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

std::vector<std::string> args_nobody_asked_for(int argc, char ** argv) {
    const auto & values = value_names();
    const auto & flags = flag_names();
    std::vector<std::string> unknown;
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
        unknown.push_back(token);
    }
    return unknown;
}

// Set once the strict check has spoken for this run, so the warning below does not repeat it.
bool strict_check_ran = false;

}  // namespace

void require_known_args(int argc, char ** argv) {
    strict_check_ran = true;
    const auto unknown = args_nobody_asked_for(argc, argv);
    if (unknown.empty()) {
        return;
    }
    std::string message = unknown.size() > 1 ? "unknown options:" : "unknown option:";
    for (const auto & token : unknown) {
        message += " " + token;
    }
    throw std::runtime_error(message);
}

void warn_ignored_args(int argc, char ** argv) {
    if (strict_check_ran) {
        return;
    }
    const auto ignored = args_nobody_asked_for(argc, argv);
    if (ignored.empty()) {
        return;
    }
    std::cerr << "audiocpp_cli warning: ignored option";
    if (ignored.size() > 1) {
        std::cerr << "s";
    }
    std::cerr << ":";
    for (const auto & token : ignored) {
        std::cerr << " " << token;
    }
    std::cerr << "\n";
}

std::vector<std::string> expand_cli_args(int argc, char ** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc));
    arguments.emplace_back(argv[0]);

    std::optional<std::string> config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        if (token == "--config-file") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--config-file requires a YAML file path");
            }
            if (config_path.has_value()) {
                throw std::runtime_error("--config-file may only be specified once");
            }
            config_path = argv[++i];
        } else {
            arguments.push_back(token);
            if (value_options().count(token) != 0 && i + 1 < argc) {
                arguments.emplace_back(argv[++i]);
            }
        }
    }
    if (!config_path.has_value()) {
        return arguments;
    }

    std::ifstream input(std::filesystem::u8path(*config_path), std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open CLI YAML config: " + *config_path);
    }
    const auto input_begin = std::istreambuf_iterator<char>(input);
    const auto input_end = std::istreambuf_iterator<char>();
    const std::string contents(input_begin, input_end);
    if (input.bad()) {
        throw std::runtime_error("failed to read CLI YAML config: " + *config_path);
    }

    const auto config = engine::io::yaml::parse_flattened_document(contents);
    std::set<std::string> keys;
    for (const auto & [key, value] : config.scalars) {
        (void)value;
        keys.insert(key);
    }
    for (const auto & [key, values] : config.lists) {
        (void)values;
        if (!keys.insert(key).second) {
            throw std::runtime_error("CLI YAML option has both scalar and list values: " + key);
        }
    }

    const auto session_overrides = explicit_session_option_overrides(arguments);
    const auto request_overrides = explicit_request_option_overrides(arguments);
    const auto load_overrides = explicit_key_value_overrides(arguments, "--load-option");
    for (const auto & key : keys) {
        const auto option = normalize_config_option(key);
        const std::set<std::string> * key_value_overrides = nullptr;
        if (option == "--session-option") {
            key_value_overrides = &session_overrides;
        } else if (option == "--load-option") {
            key_value_overrides = &load_overrides;
        } else if (option == "--request-option") {
            key_value_overrides = &request_overrides;
        } else {
            const auto moe_alias = moe_session_option_aliases().find(option);
            if (moe_alias != moe_session_option_aliases().end() &&
                session_overrides.count(moe_alias->second) != 0) {
                continue;
            }
            const auto alias = request_option_aliases().find(option);
            if (alias != request_option_aliases().end() && request_overrides.count(alias->second) != 0) {
                continue;
            }
        }
        const auto scalar = config.scalars.find(key);
        if (scalar != config.scalars.end()) {
            if (boolean_flags().count(option) != 0) {
                if (engine::io::yaml::parse_bool_scalar(scalar->second, key)) {
                    arguments.push_back(option);
                }
            } else if (key_value_overrides == nullptr) {
                arguments.push_back(option);
                arguments.push_back(scalar->second);
            } else {
                const auto separator = scalar->second.find('=');
                if (separator == std::string::npos || separator == 0 ||
                    key_value_overrides->count(scalar->second.substr(0, separator)) == 0) {
                    arguments.push_back(option);
                    arguments.push_back(scalar->second);
                }
            }
            continue;
        }

        if (boolean_flags().count(option) != 0) {
            throw std::runtime_error("CLI YAML boolean flag cannot be a list: " + key);
        }
        const auto & values = config.lists.at(key);
        for (const auto & value : values) {
            if (key_value_overrides != nullptr) {
                const auto separator = value.find('=');
                if (separator != std::string::npos && separator > 0 &&
                    key_value_overrides->count(value.substr(0, separator)) != 0) {
                    continue;
                }
            }
            arguments.push_back(option);
            arguments.push_back(value);
        }
    }
    return arguments;
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

int parse_non_negative_int(const std::string & value, const std::string & name) {
    int parsed = -1;
    const auto * last = value.data() + value.size();
    const auto [ptr, ec] = std::from_chars(value.data(), last, parsed);
    if (ec != std::errc{} || ptr != last || parsed < 0) {
        throw std::runtime_error(name + " expects a non-negative integer, got: " + value);
    }
    return parsed;
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
    if (value == "webgpu") {
        return engine::core::BackendType::WebGPU;
    }
    if (value == "sycl") {
        return engine::core::BackendType::Sycl;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

engine::audio::WavSampleFormat parse_wav_sample_format(const std::string & value) {
    if (value == "pcm16") {
        return engine::audio::WavSampleFormat::Pcm16;
    }
    if (value == "pcm24") {
        return engine::audio::WavSampleFormat::Pcm24;
    }
    if (value == "float32") {
        return engine::audio::WavSampleFormat::Float32;
    }
    throw std::runtime_error("unsupported --out-format: " + value + " (expected pcm16, pcm24, or float32)");
}

engine::audio::WavWriteOptions wav_write_options_from_cli(int argc, char ** argv) {
    engine::audio::WavWriteOptions options;
    if (const auto value = find_arg(argc, argv, "--out-format")) {
        options.format = parse_wav_sample_format(*value);
    }
    return options;
}

}  // namespace minitts::cli
