#include "workflow.h"

#include "file_sink.h"
#include "../cli/args.h"
#include "../cli/request.h"

#include "engine/framework/audio/activity.h"
#include "engine/framework/audio/mixing.h"
#include "engine/framework/audio/output.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include "engine/framework/text/subtitle.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <process.h>
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char ** environ;
#endif

namespace minitts::app {
namespace {

struct WorkflowContext {
    std::filesystem::path workflow_dir;
    std::filesystem::path output_dir;
    std::unordered_map<std::string, std::string> values;
    std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::string>>> batches;
};

std::string workflow_string(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        throw std::runtime_error("workflow object missing required string field: " + key);
    }
    return value->as_string();
}

std::istringstream workflow_buffer(const engine::io::json::Value & object, const std::string & key) {
    const auto str = workflow_string(object, key);
    return std::istringstream(str);
}

std::string workflow_string_or(
    const engine::io::json::Value & object,
    const std::string & key,
    std::string fallback) {
    const auto * value = object.find(key);
    return value == nullptr || value->is_null() ? std::move(fallback) : value->as_string();
}

bool workflow_has_field(const engine::io::json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value != nullptr && !value->is_null();
}

std::optional<std::string> workflow_optional_string(
    const engine::io::json::Value & object,
    const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return std::nullopt;
    }
    return value->as_string();
}

std::optional<double> workflow_optional_number(
    const engine::io::json::Value & object,
    const std::string & key) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return std::nullopt;
    }
    return value->as_number();
}

bool workflow_optional_bool(
    const engine::io::json::Value & object,
    const std::string & key,
    bool fallback) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    if (value->is_bool()) {
        return value->as_bool();
    }
    if (value->is_string()) {
        const std::string & text = value->as_string();
        if (text == "true") {
            return true;
        }
        if (text == "false") {
            return false;
        }
    }
    throw std::runtime_error("workflow field must be a bool: " + key);
}

std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

std::string expand_value(std::string value, const WorkflowContext & context) {
    value = replace_all(value, "${out_dir}", context.output_dir.string());
    value = replace_all(value, "${workflow_dir}", context.workflow_dir.string());
    for (const auto & [key, item] : context.values) {
        value = replace_all(value, "${" + key + "}", item);
    }
    return value;
}

std::string safe_workflow_id(std::string value) {
    if (value.empty()) {
        throw std::runtime_error("workflow batch item id must not be empty");
    }
    for (char & ch : value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) == 0 && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return value;
}

std::filesystem::path resolve_workflow_path(const std::string & value, const WorkflowContext & context) {
    std::filesystem::path path(expand_value(value, context));
    if (path.is_absolute()) {
        return path;
    }
    if (value.find("${out_dir}") != std::string::npos ||
        value.find("${workflow_dir}") != std::string::npos ||
        value.find("${") != std::string::npos) {
        throw std::runtime_error(
            "workflow path inputs expanded from placeholders must be absolute: " + value +
            " -> " + path.string());
    }
    return context.workflow_dir / path;
}

std::string lowercase_extension(const std::filesystem::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

bool is_wav_path(const std::filesystem::path & path) {
    return lowercase_extension(path) == ".wav";
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open workflow batch text file: " + path.string());
    }
    std::ostringstream raw;
    raw << input.rdbuf();
    return raw.str();
}

std::string normalize_text_as_paragraph(const std::string & text) {
    std::string normalized;
    bool in_space = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            in_space = !normalized.empty();
            continue;
        }
        if (in_space) {
            normalized.push_back(' ');
            in_space = false;
        }
        normalized.push_back(static_cast<char>(ch));
    }
    return normalized;
}

std::string read_batch_text_payload(const std::filesystem::path & path) {
    const auto ext = lowercase_extension(path);
    if (ext == ".txt" || ext == ".md") {
        return normalize_text_as_paragraph(read_text_file(path));
    }
    if (ext == ".json") {
        const auto root = engine::io::json::parse_file(path);
        if (root.is_string()) {
            return normalize_text_as_paragraph(root.as_string());
        }
        if (root.is_object()) {
            if (const auto * input = root.find("input"); input != nullptr && input->is_string()) {
                return normalize_text_as_paragraph(input->as_string());
            }
            if (const auto * text = root.find("text"); text != nullptr && text->is_string()) {
                return normalize_text_as_paragraph(text->as_string());
            }
        }
        throw std::runtime_error("workflow batch JSON text file requires a string root, input, or text: " + path.string());
    }
    throw std::runtime_error("unsupported workflow batch text extension: " + ext);
}

bool is_text_batch_path(const std::filesystem::path & path) {
    const auto ext = lowercase_extension(path);
    return ext == ".txt" || ext == ".md" || ext == ".json";
}

void validate_unique_batch_ids(
    const std::vector<std::unordered_map<std::string, std::string>> & items,
    const std::string & step_id) {
    std::unordered_set<std::string> ids;
    for (const auto & item : items) {
        const auto it = item.find("id");
        if (it == item.end()) {
            throw std::runtime_error("workflow batch item missing id: " + step_id);
        }
        if (!ids.insert(it->second).second) {
            throw std::runtime_error("workflow batch item id collision in " + step_id + ": " + it->second);
        }
    }
}

void load_workflow_inputs(
    const engine::io::json::Value & root,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const auto * inputs = root.find("inputs");
    if (inputs != nullptr && !inputs->is_null()) {
        if (!inputs->is_object()) {
            throw std::runtime_error("workflow inputs must be an object");
        }
        for (const auto & [key, value] : inputs->as_object()) {
            if (!value.is_string()) {
                throw std::runtime_error("workflow input default must be a string: " + key);
            }
            context.values[key] = value.as_string();
        }
    }
    for (const auto & [key, value] : options.workflow_inputs) {
        context.values[key] = value;
    }
    for (auto & [_, value] : context.values) {
        value = expand_value(value, context);
    }
}

void run_batch_inputs_step(
    const engine::io::json::Value & step,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    if (workflow_has_field(step, "foreach")) {
        throw std::runtime_error("batch_inputs cannot use foreach: " + id);
    }
    const bool has_audio_dir = workflow_has_field(step, "audio_dir");
    const bool has_text_dir = workflow_has_field(step, "text_dir");
    if ((has_audio_dir ? 1 : 0) + (has_text_dir ? 1 : 0) != 1) {
        throw std::runtime_error("batch_inputs requires exactly one of audio_dir or text_dir: " + id);
    }

    std::vector<std::filesystem::path> paths;
    std::vector<std::unordered_map<std::string, std::string>> items;
    if (has_audio_dir) {
        const auto dir = resolve_workflow_path(workflow_string(step, "audio_dir"), context);
        if (!std::filesystem::is_directory(dir)) {
            throw std::runtime_error("batch_inputs audio_dir must be an existing directory: " + dir.string());
        }
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file() && is_wav_path(entry.path())) {
                paths.push_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) {
            throw std::runtime_error("batch_inputs audio_dir contains no .wav files: " + dir.string());
        }
        items.reserve(paths.size());
        for (const auto & path : paths) {
            items.push_back({
                {"id", safe_workflow_id(path.stem().string())},
                {"audio_path", path.string()},
            });
        }
    } else {
        const auto dir = resolve_workflow_path(workflow_string(step, "text_dir"), context);
        if (!std::filesystem::is_directory(dir)) {
            throw std::runtime_error("batch_inputs text_dir must be an existing directory: " + dir.string());
        }
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file() && is_text_batch_path(entry.path())) {
                paths.push_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) {
            throw std::runtime_error("batch_inputs text_dir contains no .txt, .md, or .json files: " + dir.string());
        }
        items.reserve(paths.size());
        for (const auto & path : paths) {
            const auto text = read_batch_text_payload(path);
            if (text.empty()) {
                throw std::runtime_error("batch_inputs text file contains no non-whitespace text: " + path.string());
            }
            items.push_back({
                {"id", safe_workflow_id(path.stem().string())},
                {"text_path", path.string()},
                {"text", text},
            });
        }
    }

    validate_unique_batch_ids(items, id);
    context.values[id + ".count"] = std::to_string(items.size());
    context.batches[id] = std::move(items);
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "batch_items=" << context.values[id + ".count"] << "\n";
}

std::string path_arg(const std::filesystem::path & path) {
    return path.string();
}

std::string format_process_args(const std::string & program, const std::vector<std::string> & args) {
    std::ostringstream out;
    out << program;
    for (const std::string & arg : args) {
        out << ' ' << arg;
    }
    return out.str();
}

int run_process(const std::string & program, const std::vector<std::string> & args) {
#if defined(_WIN32)
    std::vector<std::wstring> wide_args;
    wide_args.reserve(args.size() + 1);
    wide_args.push_back(std::filesystem::path(program).wstring());
    for (const std::string & arg : args) {
        wide_args.push_back(std::filesystem::path(arg).wstring());
    }

    std::vector<const wchar_t *> argv;
    argv.reserve(wide_args.size() + 1);
    for (const std::wstring & arg : wide_args) {
        argv.push_back(arg.c_str());
    }
    argv.push_back(nullptr);

    const intptr_t rc = _wspawnvp(_P_WAIT, wide_args.front().c_str(), argv.data());
    if (rc == -1) {
        throw std::runtime_error(
            "failed to launch process: " + format_process_args(program, args) + ": " + std::strerror(errno));
    }
    return static_cast<int>(rc);
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char *>(program.c_str()));
    for (const std::string & arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = 0;
    const int spawn_rc = posix_spawnp(&pid, program.c_str(), nullptr, nullptr, argv.data(), environ);
    if (spawn_rc != 0) {
        throw std::runtime_error(
            "failed to launch process: " + format_process_args(program, args) + ": " + std::strerror(spawn_rc));
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error("failed waiting for process: " + program + ": " + std::strerror(errno));
        }
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    throw std::runtime_error("process exited unexpectedly: " + program);
#endif
}

void run_convert_audio_step(
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const auto input = resolve_workflow_path(workflow_string(step, "input"), context);
    const auto output = resolve_workflow_path(workflow_string(step, "output"), context);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    const int sample_rate = static_cast<int>(workflow_optional_number(step, "sample_rate").value_or(48000.0));
    const int channels = static_cast<int>(workflow_optional_number(step, "channels").value_or(2.0));
    if (sample_rate <= 0 || channels <= 0) {
        throw std::runtime_error("convert_audio step requires positive sample_rate and channels");
    }

    std::vector<std::string> args{
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    };
    if (const auto start = workflow_optional_number(step, "start_seconds")) {
        args.push_back("-ss");
        args.push_back(std::to_string(*start));
    }
    if (const auto duration = workflow_optional_number(step, "duration_seconds")) {
        if (!(*duration > 0.0)) {
            throw std::runtime_error("convert_audio duration_seconds must be positive");
        }
        args.push_back("-t");
        args.push_back(std::to_string(*duration));
    }
    args.push_back("-i");
    args.push_back(path_arg(input));
    args.push_back("-ac");
    args.push_back(std::to_string(channels));
    args.push_back("-ar");
    args.push_back(std::to_string(sample_rate));
    args.push_back(path_arg(output));

    const int rc = run_process(options.audio_converter, args);
    if (rc != 0) {
        throw std::runtime_error("convert_audio step failed: " + id);
    }
    context.values[id + ".audio_path"] = output.string();
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "audio_path=" << output.string() << "\n";
}

void write_runtime_audio(
    const std::filesystem::path & path,
    const engine::runtime::AudioBuffer & audio) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    engine::audio::WavPcm16Sink().write(path, engine::audio::AudioBuffer{
        audio.sample_rate,
        audio.channels,
        audio.samples,
    });
}

engine::io::json::Value expanded_json(const engine::io::json::Value & value, const WorkflowContext & context) {
    if (value.is_string()) {
        return engine::io::json::Value::make_string(expand_value(value.as_string(), context));
    }
    if (value.is_array()) {
        engine::io::json::Value::Array out;
        for (const auto & item : value.as_array()) {
            out.push_back(expanded_json(item, context));
        }
        return engine::io::json::Value::make_array(std::move(out));
    }
    if (value.is_object()) {
        engine::io::json::Value::Object out;
        for (const auto & [key, item] : value.as_object()) {
            out.emplace(key, expanded_json(item, context));
        }
        return engine::io::json::Value::make_object(std::move(out));
    }
    return value;
}

std::unordered_map<std::string, std::string> merged_options(
    const std::unordered_map<std::string, std::string> & base,
    const engine::io::json::Value * value,
    const WorkflowContext & context) {
    auto out = base;
    if (value == nullptr || value->is_null()) {
        return out;
    }
    for (const auto & [key, child] : value->as_object()) {
        out[key] = expand_value(minitts::cli::json_option_string(child), context);
    }
    return out;
}

void record_result_paths(
    const std::string & id,
    const engine::runtime::TaskResult & result,
    const std::optional<std::filesystem::path> & audio_out,
    const std::optional<std::filesystem::path> & text_out,
    const std::optional<std::filesystem::path> & words_out,
    const std::filesystem::path & step_dir,
    WorkflowContext & context) {
    if (result.audio_output.has_value() && audio_out.has_value()) {
        context.values[id + ".audio_path"] = audio_out->string();
    }
    for (const auto & output : result.named_audio_outputs) {
        const auto path = step_dir / (output.id + ".wav");
        context.values[id + "." + output.id + "_path"] = path.string();
    }
    if (result.text_output.has_value()) {
        context.values[id + ".text"] = result.text_output->text;
    }
    if (text_out.has_value()) {
        context.values[id + ".text_path"] = text_out->string();
    }
    if (words_out.has_value()) {
        context.values[id + ".words_path"] = words_out->string();
    }
}

void write_text_result(
    const engine::runtime::TaskResult & result,
    const std::filesystem::path & path,
    const std::string & id) {
    if (!result.text_output.has_value()) {
        throw std::runtime_error("workflow model step requested text_out but produced no text: " + id);
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    out << result.text_output->text << "\n";
    std::cout << "text_out=" << path.string() << "\n";
}

void write_words_result(
    const engine::runtime::TaskResult & result,
    const std::filesystem::path & path,
    const std::string & id) {
    if (result.word_timestamps.empty()) {
        throw std::runtime_error("workflow model step requested words_out but produced no word timestamps: " + id);
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    out << word_timestamps_to_json(result.word_timestamps) << "\n";
    std::cout << "words_out=" << path.string() << "\n";
}

engine::core::BackendConfig parse_step_backend(
    const engine::io::json::Value & step,
    engine::core::BackendConfig fallback) {
    if (const auto backend = workflow_optional_string(step, "backend")) {
        fallback.type = minitts::cli::parse_backend(*backend);
    }
    if (const auto device = workflow_optional_number(step, "device")) {
        fallback.device = static_cast<int>(*device);
    }
    if (const auto threads = workflow_optional_number(step, "threads")) {
        fallback.threads = static_cast<int>(*threads);
    }
    if (fallback.threads <= 0) {
        throw std::runtime_error("workflow step threads must be positive");
    }
    return fallback;
}

void run_model_step_impl(
    const engine::runtime::ModelRegistry & registry,
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const auto request_value = step.find("request");
    if (request_value == nullptr || request_value->is_null()) {
        throw std::runtime_error("model step requires request object: " + id);
    }

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = resolve_workflow_path(workflow_string(step, "model"), context);
    load_request.model_spec_override = options.model_spec_override;
    if (const auto family = workflow_optional_string(step, "family")) {
        load_request.family_hint = *family;
    }
    if (const auto config = workflow_optional_string(step, "config")) {
        load_request.config_id = *config;
    }
    if (const auto weight = workflow_optional_string(step, "weight")) {
        load_request.weight_id = *weight;
    }
    load_request.options = merged_options(options.load_options, step.find("load_options"), context);

    engine::runtime::SessionOptions session_options;
    session_options.backend = parse_step_backend(step, options.backend);
    session_options.options = merged_options(options.session_options, step.find("session_options"), context);

    const engine::runtime::TaskSpec task_spec{
        engine::runtime::parse_voice_task_kind(workflow_string(step, "task")),
        engine::runtime::parse_run_mode(workflow_string_or(step, "mode", "offline")),
    };
    if (task_spec.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("workflow model steps currently require offline mode: " + id);
    }

    auto model = registry.load(load_request);
    auto session = model->create_task_session(task_spec, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
        throw std::runtime_error("workflow model step does not support offline execution: " + id);
    }

    const auto expanded_request = expanded_json(*request_value, context);
    auto request = minitts::cli::build_request_from_json(expanded_request, context.workflow_dir);
    const auto text_out_value = workflow_optional_string(step, "text_out");
    const auto words_out_value = workflow_optional_string(step, "words_out");
    std::optional<std::filesystem::path> text_out;
    std::optional<std::filesystem::path> words_out;
    if (text_out_value.has_value()) {
        text_out = resolve_workflow_path(*text_out_value, context);
    }
    if (words_out_value.has_value()) {
        words_out = resolve_workflow_path(*words_out_value, context);
        request.options["return_timestamps"] = "true";
    }
    session->prepare(engine::runtime::build_preparation_request(request));
    const auto result = offline->run(request);

    std::filesystem::path step_dir = context.output_dir / id;
    if (const auto out_dir = workflow_optional_string(step, "out_dir")) {
        step_dir = resolve_workflow_path(*out_dir, context);
    }
    std::optional<std::filesystem::path> audio_out;
    if (const auto audio = workflow_optional_string(step, "audio_out")) {
        audio_out = resolve_workflow_path(*audio, context);
    } else if (result.audio_output.has_value() || result.named_audio_outputs.size() == 1) {
        audio_out = step_dir / (id + ".wav");
    }
    emit_task_result(result, audio_out, step_dir, step_dir, std::nullopt, std::nullopt, std::nullopt);
    if (text_out.has_value()) {
        write_text_result(result, *text_out, id);
    }
    if (words_out.has_value()) {
        write_words_result(result, *words_out, id);
    }
    record_result_paths(id, result, audio_out, text_out, words_out, step_dir, context);

    std::cout << "workflow_step=" << id << "\n";
}

std::vector<engine::runtime::WordTimestamp> read_word_timestamps_json(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    if (!root.is_array()) {
        throw std::runtime_error("word timestamp input must be a JSON array: " + path.string());
    }
    std::vector<engine::runtime::WordTimestamp> words;
    for (const auto & item : root.as_array()) {
        engine::runtime::WordTimestamp word;
        word.span.start_sample = engine::io::json::require_i64(item, "start_sample");
        word.span.end_sample = engine::io::json::require_i64(item, "end_sample");
        word.word = engine::io::json::require_string(item, "word");
        const auto * confidence = item.find("confidence");
        if (confidence != nullptr && !confidence->is_null()) {
            word.confidence = confidence->as_f32();
        }
        words.push_back(std::move(word));
    }
    return words;
}

void run_format_subtitles_step(
    const engine::io::json::Value & step,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const std::string format = workflow_string(step, "format");
    if (format != "srt") {
        throw std::runtime_error("format_subtitles currently supports only format=srt: " + id);
    }
    const auto words_path = resolve_workflow_path(workflow_string(step, "words"), context);
    const auto output = resolve_workflow_path(workflow_string(step, "output"), context);
    const auto sample_rate_value = workflow_optional_number(step, "sample_rate");
    if (!sample_rate_value.has_value()) {
        throw std::runtime_error("format_subtitles requires sample_rate: " + id);
    }
    const int sample_rate = static_cast<int>(*sample_rate_value);
    engine::text::SubtitleFormatOptions format_options;
    format_options.sample_rate = sample_rate;
    format_options.max_chars_per_line =
        static_cast<int>(workflow_optional_number(step, "max_chars_per_line").value_or(42.0));
    format_options.max_lines =
        static_cast<int>(workflow_optional_number(step, "max_lines").value_or(2.0));
    format_options.max_block_seconds = workflow_optional_number(step, "max_block_seconds").value_or(6.0);
    format_options.max_gap_seconds = workflow_optional_number(step, "max_gap_seconds").value_or(1.0);

    const auto words = read_word_timestamps_json(words_path);
    const auto srt = engine::text::format_srt(words, format_options);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream out(output);
    out << srt;
    context.values[id + ".srt_path"] = output.string();
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "srt_out=" << output.string() << "\n";
}

void run_chunked_model_step(
    const engine::runtime::ModelRegistry & registry,
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const auto request_value = step.find("request");
    if (request_value == nullptr || request_value->is_null()) {
        throw std::runtime_error("chunked_model step requires request object: " + id);
    }
    const std::string audio_key = workflow_string_or(step, "audio_key", "audio");
    engine::audio::WavData input_wav;
    if (workflow_has_field(*request_value, "audio_data")) {
        auto input_audio_data = workflow_buffer(*request_value, "audio_data");
        input_wav = engine::audio::read_wav_f32(input_audio_data);
    } else {
        const auto input_audio_path = resolve_workflow_path(workflow_string(*request_value, audio_key), context);
        input_wav = engine::audio::read_wav_f32(input_audio_path);
    }
    if (input_wav.channels <= 0 || input_wav.sample_rate <= 0 || input_wav.samples.empty()) {
        throw std::runtime_error("chunked_model input audio is invalid: " + id);
    }
    const int64_t total_frames = static_cast<int64_t>(input_wav.samples.size() / static_cast<size_t>(input_wav.channels));
    const double chunk_seconds = workflow_optional_number(step, "chunk_seconds").value_or(30.0);
    const double overlap_seconds = workflow_optional_number(step, "overlap_seconds").value_or(0.0);
    if (!(chunk_seconds > 0.0) || overlap_seconds != 0.0) {
        throw std::runtime_error("chunked_model requires chunk_seconds > 0 and overlap_seconds == 0");
    }
    const int64_t chunk_frames = static_cast<int64_t>(chunk_seconds * static_cast<double>(input_wav.sample_rate));
    const int64_t step_frames = chunk_frames;
    if (chunk_frames <= 0 || step_frames <= 0) {
        throw std::runtime_error("chunked_model computed invalid chunk frame count");
    }
    std::optional<engine::audio::WavData> timing_wav;
    if (const auto timing_audio = workflow_optional_string(step, "timing_audio")) {
        timing_wav = engine::audio::read_wav_f32(resolve_workflow_path(*timing_audio, context));
        if (timing_wav->sample_rate != input_wav.sample_rate || timing_wav->channels != input_wav.channels) {
            throw std::runtime_error("chunked_model timing_audio must match the source sample rate and channel count");
        }
        const int64_t timing_frames =
            static_cast<int64_t>(timing_wav->samples.size() / static_cast<size_t>(timing_wav->channels));
        if (timing_frames < total_frames) {
            throw std::runtime_error("chunked_model timing_audio is shorter than the source audio");
        }
    }
    const bool use_timing_repaint_window = workflow_optional_bool(step, "set_repaint_window_from_timing", false);
    if (use_timing_repaint_window && !timing_wav.has_value()) {
        throw std::runtime_error("chunked_model set_repaint_window_from_timing requires timing_audio");
    }
    const float timing_threshold_dbfs =
        static_cast<float>(workflow_optional_number(step, "timing_threshold_dbfs").value_or(-40.0));
    const double timing_window_seconds = workflow_optional_number(step, "timing_window_seconds").value_or(0.1);
    const double timing_margin_seconds = workflow_optional_number(step, "timing_margin_seconds").value_or(0.25);

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = resolve_workflow_path(workflow_string(step, "model"), context);
    load_request.model_spec_override = options.model_spec_override;
    if (const auto family = workflow_optional_string(step, "family")) {
        load_request.family_hint = *family;
    }
    load_request.options = merged_options(options.load_options, step.find("load_options"), context);

    engine::runtime::SessionOptions session_options;
    session_options.backend = parse_step_backend(step, options.backend);
    session_options.options = merged_options(options.session_options, step.find("session_options"), context);
    const engine::runtime::TaskSpec task_spec{
        engine::runtime::parse_voice_task_kind(workflow_string(step, "task")),
        engine::runtime::parse_run_mode(workflow_string_or(step, "mode", "offline")),
    };
    if (task_spec.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("chunked_model steps require offline mode: " + id);
    }
    auto model = registry.load(load_request);
    auto session = model->create_task_session(task_spec, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
        throw std::runtime_error("chunked_model step does not support offline execution: " + id);
    }

    const auto step_dir = context.output_dir / id;
    const auto chunk_dir = step_dir / "chunks";
    std::filesystem::create_directories(chunk_dir);
    engine::runtime::AudioBuffer merged;
    std::vector<std::string> text_chunks;
    std::string text_language;
    int64_t chunk_index = 0;
    for (int64_t start = 0; start < total_frames; start += step_frames) {
        const int64_t frames = std::min(chunk_frames, total_frames - start);
        if (frames <= 0) {
            break;
        }
        engine::runtime::AudioBuffer chunk_audio;
        chunk_audio.sample_rate = input_wav.sample_rate;
        chunk_audio.channels = input_wav.channels;
        chunk_audio.samples.resize(static_cast<size_t>(frames * input_wav.channels), 0.0F);
        const auto copy_begin = input_wav.samples.begin() + static_cast<std::ptrdiff_t>(start * input_wav.channels);
        const auto copy_end = copy_begin + static_cast<std::ptrdiff_t>(frames * input_wav.channels);
        std::copy(copy_begin, copy_end, chunk_audio.samples.begin());
        const auto chunk_path = chunk_dir / ("chunk_" + std::to_string(chunk_index) + ".wav");
        write_runtime_audio(chunk_path, chunk_audio);

        auto request_object = expanded_json(*request_value, context).as_object();
        request_object[audio_key] = engine::io::json::Value::make_string(chunk_path.string());
        request_object["duration_seconds"] =
            engine::io::json::Value::make_number(static_cast<double>(frames) / static_cast<double>(input_wav.sample_rate));
        if (use_timing_repaint_window) {
            const auto active = engine::audio::find_interleaved_audio_activity_region(
                timing_wav->samples,
                timing_wav->channels,
                timing_wav->sample_rate,
                start,
                frames,
                timing_threshold_dbfs,
                timing_window_seconds,
                timing_margin_seconds);
            const double start_seconds = active.has_activity
                ? static_cast<double>(active.start_frame - start) / static_cast<double>(input_wav.sample_rate)
                : static_cast<double>(frames) / static_cast<double>(input_wav.sample_rate);
            const double end_seconds = active.has_activity
                ? static_cast<double>(active.end_frame - start) / static_cast<double>(input_wav.sample_rate)
                : static_cast<double>(frames) / static_cast<double>(input_wav.sample_rate);
            request_object["repaint_start"] = engine::io::json::Value::make_number(start_seconds);
            request_object["repaint_end"] = engine::io::json::Value::make_number(end_seconds);
        } else if (workflow_optional_bool(step, "set_repaint_window", false)) {
            request_object["repaint_start"] = engine::io::json::Value::make_number(0.0);
            request_object["repaint_end"] = request_object["duration_seconds"];
        }
        auto request = minitts::cli::build_request_from_json(
            engine::io::json::Value::make_object(std::move(request_object)),
            context.workflow_dir);
        if (chunk_index == 0) {
            session->prepare(engine::runtime::build_preparation_request(request));
        }
        auto result = offline->run(request);
        if (result.text_output.has_value()) {
            text_chunks.push_back(result.text_output->text);
            if (text_language.empty()) {
                text_language = result.text_output->language;
            }
        }
        if (result.audio_output.has_value()) {
            const auto & audio = *result.audio_output;
            if (merged.sample_rate == 0) {
                merged.sample_rate = audio.sample_rate;
                merged.channels = audio.channels;
            }
            if (merged.sample_rate != audio.sample_rate || merged.channels != audio.channels) {
                throw std::runtime_error("chunked_model output sample rate/channel count changed across chunks: " + id);
            }
            merged.samples.insert(merged.samples.end(), audio.samples.begin(), audio.samples.end());
        }
        ++chunk_index;
    }
    if (merged.samples.empty() && text_chunks.empty()) {
        throw std::runtime_error("chunked_model produced no audio or text: " + id);
    }
    if (!merged.samples.empty()) {
        const auto * audio_out_value = step.find("audio_out");
        if (audio_out_value == nullptr || audio_out_value->is_null()) {
            throw std::runtime_error("chunked_model audio output requires audio_out: " + id);
        }
        const auto audio_out = resolve_workflow_path(audio_out_value->as_string(), context);
        write_runtime_audio(audio_out, merged);
        context.values[id + ".audio_path"] = audio_out.string();
        std::cout << "audio_out=" << audio_out.string() << "\n";
    }
    if (!text_chunks.empty()) {
        std::ostringstream joined;
        for (size_t index = 0; index < text_chunks.size(); ++index) {
            if (index != 0) {
                joined << ' ';
            }
            joined << text_chunks[index];
        }
        const std::string text = joined.str();
        context.values[id + ".text"] = text;
        if (!text_language.empty()) {
            context.values[id + ".language"] = text_language;
        }
        const auto text_out_option = workflow_optional_string(step, "text_out");
        const auto text_out = text_out_option.has_value()
            ? resolve_workflow_path(*text_out_option, context)
            : step_dir / (id + ".txt");
        if (!text_out.parent_path().empty()) {
            std::filesystem::create_directories(text_out.parent_path());
        }
        std::ofstream out(text_out);
        out << text << "\n";
        context.values[id + ".text_path"] = text_out.string();
        std::cout << "text_out=" << text_out.string() << "\n";
    }
    std::cout << "workflow_step=" << id << "\n";
}

void run_mix_audio_step(
    const engine::io::json::Value & step,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const auto * inputs_value = step.find("inputs");
    if (inputs_value == nullptr || !inputs_value->is_array()) {
        throw std::runtime_error("mix_audio step requires inputs array: " + id);
    }
    std::vector<engine::audio::AudioMixInput> inputs;
    for (const auto & item : inputs_value->as_array()) {
        engine::audio::AudioMixInput input;
        input.audio = engine::audio::read_wav_f32(resolve_workflow_path(workflow_string(item, "path"), context));
        input.gain = static_cast<float>(workflow_optional_number(item, "gain").value_or(1.0));
        if (const auto activity_ref = workflow_optional_string(item, "activity_ref")) {
            input.activity_reference = engine::audio::read_wav_f32(resolve_workflow_path(*activity_ref, context));
            input.activity_threshold_dbfs =
                static_cast<float>(workflow_optional_number(item, "activity_threshold_dbfs").value_or(-40.0));
            input.activity_window_seconds = workflow_optional_number(item, "activity_window_seconds").value_or(0.1);
            input.activity_margin_seconds = workflow_optional_number(item, "activity_margin_seconds").value_or(0.35);
            input.activity_fade_seconds = workflow_optional_number(item, "activity_fade_seconds").value_or(0.03);
        }
        inputs.push_back(std::move(input));
    }
    const bool normalize = workflow_optional_bool(step, "normalize_peak", true);
    const auto mixed = engine::audio::mix_audio_to_reference_shape(inputs, normalize);
    const auto output = resolve_workflow_path(workflow_string(step, "output"), context);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    engine::audio::WavPcm16Sink().write(output, engine::audio::AudioBuffer{
        mixed.sample_rate,
        mixed.channels,
        mixed.samples,
    });
    context.values[id + ".audio_path"] = output.string();
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "audio_path=" << output.string() << "\n";
}

std::string parse_foreach_source(const engine::io::json::Value & step) {
    std::string value = workflow_string(step, "foreach");
    if (value.size() >= 3 && value.front() == '$' && value[1] == '{' && value.back() == '}') {
        value = value.substr(2, value.size() - 3);
    }
    const std::string suffix = ".items";
    if (value.size() <= suffix.size() ||
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("workflow foreach must reference ${step_id.items}: " + value);
    }
    return value.substr(0, value.size() - suffix.size());
}

void run_model_step_foreach(
    const engine::runtime::ModelRegistry & registry,
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const std::string id = workflow_string(step, "id");
    const std::string source_id = parse_foreach_source(step);
    if (source_id == id) {
        throw std::runtime_error("workflow foreach cannot reference the current step: " + id);
    }
    const auto source_it = context.batches.find(source_id);
    if (source_it == context.batches.end()) {
        throw std::runtime_error("workflow foreach source is not a batch: " + source_id);
    }
    const auto source_items = source_it->second;
    const auto * request_value = step.find("request");
    if (request_value == nullptr || request_value->is_null()) {
        throw std::runtime_error("model step requires request object: " + id);
    }

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = resolve_workflow_path(workflow_string(step, "model"), context);
    load_request.model_spec_override = options.model_spec_override;
    if (const auto family = workflow_optional_string(step, "family")) {
        load_request.family_hint = *family;
    }
    if (const auto config = workflow_optional_string(step, "config")) {
        load_request.config_id = *config;
    }
    if (const auto weight = workflow_optional_string(step, "weight")) {
        load_request.weight_id = *weight;
    }
    load_request.options = merged_options(options.load_options, step.find("load_options"), context);

    engine::runtime::SessionOptions session_options;
    session_options.backend = parse_step_backend(step, options.backend);
    session_options.options = merged_options(options.session_options, step.find("session_options"), context);

    const engine::runtime::TaskSpec task_spec{
        engine::runtime::parse_voice_task_kind(workflow_string(step, "task")),
        engine::runtime::parse_run_mode(workflow_string_or(step, "mode", "offline")),
    };
    if (task_spec.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("workflow model steps currently require offline mode: " + id);
    }

    auto model = registry.load(load_request);
    auto session = model->create_task_session(task_spec, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
        throw std::runtime_error("workflow model step does not support offline execution: " + id);
    }

    const auto base_values = context.values;
    std::vector<std::unordered_map<std::string, std::string>> output_items;
    output_items.reserve(source_items.size());
    bool prepared = false;
    for (const auto & item : source_items) {
        context.values = base_values;
        for (const auto & [key, value] : item) {
            context.values["item." + key] = value;
        }

        const auto expanded_request = expanded_json(*request_value, context);
        auto request = minitts::cli::build_request_from_json(expanded_request, context.workflow_dir);
        const auto text_out_value = workflow_optional_string(step, "text_out");
        const auto words_out_value = workflow_optional_string(step, "words_out");
        std::optional<std::filesystem::path> text_out;
        std::optional<std::filesystem::path> words_out;
        if (text_out_value.has_value()) {
            text_out = resolve_workflow_path(*text_out_value, context);
        }
        if (words_out_value.has_value()) {
            words_out = resolve_workflow_path(*words_out_value, context);
            request.options["return_timestamps"] = "true";
        }
        if (!prepared) {
            session->prepare(engine::runtime::build_preparation_request(request));
            prepared = true;
        }
        const auto result = offline->run(request);

        std::filesystem::path step_dir = context.output_dir / id / item.at("id");
        if (const auto out_dir = workflow_optional_string(step, "out_dir")) {
            step_dir = resolve_workflow_path(*out_dir, context);
        }
        std::optional<std::filesystem::path> audio_out;
        if (const auto audio = workflow_optional_string(step, "audio_out")) {
            audio_out = resolve_workflow_path(*audio, context);
        } else if (result.audio_output.has_value() || result.named_audio_outputs.size() == 1) {
            audio_out = step_dir / (id + ".wav");
        }
        emit_task_result(result, audio_out, step_dir, step_dir, std::nullopt, std::nullopt, std::nullopt);
        if (text_out.has_value()) {
            write_text_result(result, *text_out, id);
        }
        if (words_out.has_value()) {
            write_words_result(result, *words_out, id);
        }
        record_result_paths(id, result, audio_out, text_out, words_out, step_dir, context);

        auto output_item = item;
        const std::string prefix = id + ".";
        for (const auto & [key, value] : context.values) {
            if (key.rfind(prefix, 0) == 0) {
                output_item[key.substr(prefix.size())] = value;
            }
        }
        output_items.push_back(std::move(output_item));
        std::cout << "workflow_step=" << id << "\n";
    }

    context.values = base_values;
    context.values[id + ".count"] = std::to_string(output_items.size());
    context.batches[id] = std::move(output_items);
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "batch_items=" << context.values[id + ".count"] << "\n";
}

void run_workflow_step_once(
    const engine::runtime::ModelRegistry & registry,
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    const std::string type = workflow_string(step, "type");
    if (type == "batch_inputs") {
        run_batch_inputs_step(step, context);
    } else if (type == "convert_audio") {
        run_convert_audio_step(step, options, context);
    } else if (type == "model") {
        run_model_step_impl(registry, step, options, context);
    } else if (type == "format_subtitles") {
        run_format_subtitles_step(step, context);
    } else if (type == "chunked_model") {
        run_chunked_model_step(registry, step, options, context);
    } else if (type == "mix_audio") {
        run_mix_audio_step(step, context);
    } else {
        throw std::runtime_error("unsupported workflow step type: " + type);
    }
}

void run_workflow_step(
    const engine::runtime::ModelRegistry & registry,
    const engine::io::json::Value & step,
    const WorkflowRunOptions & options,
    WorkflowContext & context) {
    if (!workflow_has_field(step, "foreach")) {
        run_workflow_step_once(registry, step, options, context);
        return;
    }

    const std::string id = workflow_string(step, "id");
    const std::string type = workflow_string(step, "type");
    if (type == "batch_inputs") {
        throw std::runtime_error("batch_inputs cannot use foreach: " + id);
    }
    if (type == "model") {
        run_model_step_foreach(registry, step, options, context);
        return;
    }
    const std::string source_id = parse_foreach_source(step);
    if (source_id == id) {
        throw std::runtime_error("workflow foreach cannot reference the current step: " + id);
    }
    const auto source_it = context.batches.find(source_id);
    if (source_it == context.batches.end()) {
        throw std::runtime_error("workflow foreach source is not a batch: " + source_id);
    }

    const auto base_values = context.values;
    std::vector<std::unordered_map<std::string, std::string>> output_items;
    output_items.reserve(source_it->second.size());
    for (const auto & item : source_it->second) {
        context.values = base_values;
        for (const auto & [key, value] : item) {
            context.values["item." + key] = value;
        }
        run_workflow_step_once(registry, step, options, context);

        auto output_item = item;
        const std::string prefix = id + ".";
        for (const auto & [key, value] : context.values) {
            if (key.rfind(prefix, 0) == 0) {
                output_item[key.substr(prefix.size())] = value;
            }
        }
        output_items.push_back(std::move(output_item));
    }

    context.values = base_values;
    context.values[id + ".count"] = std::to_string(output_items.size());
    context.batches[id] = std::move(output_items);
    std::cout << "workflow_step=" << id << "\n";
    std::cout << "batch_items=" << context.values[id + ".count"] << "\n";
}

void write_workflow_manifest(const WorkflowContext & context) {
    const auto path = context.output_dir / "workflow_manifest.json";
    std::filesystem::create_directories(context.output_dir);
    std::ofstream out(path);
    out << "{";
    bool first = true;
    for (const auto & [key, value] : context.values) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\n  " << engine::io::json::stringify_string(key)
            << ": " << engine::io::json::stringify_string(value);
    }
    out << "\n}\n";
    std::cout << "workflow_manifest_out=" << path.string() << "\n";
}

}  // namespace

void run_json_workflow(
    const engine::runtime::ModelRegistry & registry,
    const WorkflowRunOptions & options) {
    if (options.workflow_path.empty()) {
        throw std::runtime_error("workflow path is required");
    }
    if (options.output_dir.empty()) {
        throw std::runtime_error("workflow --out-dir is required");
    }
    const auto root = engine::io::json::parse_file(options.workflow_path);
    const auto * steps = root.find("steps");
    if (steps == nullptr || !steps->is_array()) {
        throw std::runtime_error("workflow JSON requires a steps array");
    }
    WorkflowContext context{
        options.workflow_path.parent_path().empty()
            ? std::filesystem::current_path()
            : std::filesystem::absolute(options.workflow_path.parent_path()),
        std::filesystem::absolute(options.output_dir),
        {},
        {},
    };
    load_workflow_inputs(root, options, context);
    std::filesystem::create_directories(context.output_dir);
    for (const auto & step : steps->as_array()) {
        run_workflow_step(registry, step, options, context);
    }
    if (options.final_audio_out.has_value()) {
        const auto * final_key = root.find("final_audio");
        if (final_key == nullptr || final_key->is_null()) {
            throw std::runtime_error("--out with workflow requires final_audio in workflow JSON");
        }
        const auto src = resolve_workflow_path(final_key->as_string(), context);
        if (!options.final_audio_out->parent_path().empty()) {
            std::filesystem::create_directories(options.final_audio_out->parent_path());
        }
        std::filesystem::copy_file(src, *options.final_audio_out, std::filesystem::copy_options::overwrite_existing);
        std::cout << "audio_out=" << options.final_audio_out->string() << "\n";
    }
    write_workflow_manifest(context);
}

}  // namespace minitts::app
