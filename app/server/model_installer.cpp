#include "model_installer.h"

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace minitts::server {
namespace {

std::string json_quote(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (ch < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    result += "\\u00";
                    result.push_back(hex[(ch >> 4) & 0xf]);
                    result.push_back(hex[ch & 0xf]);
                } else {
                    result.push_back(static_cast<char>(ch));
                }
        }
    }
    result.push_back('"');
    return result;
}

bool valid_package_id(const std::string & value) {
    if (value.empty() || value.size() > 128) {
        return false;
    }
    for (const unsigned char ch : value) {
        if (!(std::isalnum(ch) || ch == '_' || ch == '-')) {
            return false;
        }
    }
    return true;
}

void validate_argument(const std::string & value, const char * name) {
    if (value.find_first_of("\r\n") != std::string::npos) {
        throw std::runtime_error(std::string(name) + " contains an invalid newline");
    }
#ifdef _WIN32
    if (value.find_first_of("\"&|<>^") != std::string::npos) {
        throw std::runtime_error(std::string(name) + " contains an unsupported command character");
    }
#endif
}

std::string shell_quote(const std::string & value) {
#ifdef _WIN32
    return "\"" + value + "\"";
#else
    std::string result = "'";
    for (const char ch : value) {
        if (ch == '\'') {
            result += "'\\''";
        } else {
            result.push_back(ch);
        }
    }
    return result + "'";
#endif
}

std::string python_command() {
    if (const char * configured = std::getenv("AUDIOCPP_PYTHON")) {
        const std::string value(configured);
        validate_argument(value, "AUDIOCPP_PYTHON");
        return shell_quote(value);
    }
#ifdef _WIN32
    return "python";
#else
    return "python3";
#endif
}

std::string read_log_tail(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    constexpr std::streamoff kTailBytes = 8192;
    if (size > kTailBytes) {
        input.seekg(size - kTailBytes);
    } else {
        input.seekg(0);
    }
    std::ostringstream content;
    content << input.rdbuf();
    std::string text = content.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    const auto newline = text.find_last_of("\r\n");
    return newline == std::string::npos ? text : text.substr(newline + 1);
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

struct ModelInstaller::State {
    struct Job {
        std::string package_id;
        std::string state = "queued";
        std::string message;
        std::filesystem::path log_path;
        int exit_code = -1;
        int64_t started_at_ms = 0;
        int64_t finished_at_ms = 0;
    };

    std::filesystem::path repository_root;
    std::filesystem::path models_root;
    std::filesystem::path job_root;
    mutable std::mutex mutex;
    std::map<std::string, Job> jobs;
};

ModelInstaller::ModelInstaller(
    std::filesystem::path repository_root,
    std::filesystem::path models_root)
    : state_(std::make_shared<State>()) {
    state_->repository_root = std::filesystem::absolute(std::move(repository_root)).lexically_normal();
    state_->models_root = std::filesystem::absolute(std::move(models_root)).lexically_normal();
    state_->job_root = std::filesystem::temp_directory_path() / "audiocpp-model-installer";
    std::filesystem::create_directories(state_->job_root);
    std::filesystem::create_directories(state_->models_root);
}

ModelInstaller::~ModelInstaller() = default;

std::string ModelInstaller::start(
    const std::string & package_id,
    const std::string & source_file,
    const std::string & output_file,
    const std::string & source_directory,
    const std::string & variant,
    bool overwrite) {
    if (!valid_package_id(package_id)) {
        throw std::runtime_error("invalid model-manager package id");
    }
    validate_argument(source_directory, "source directory");
    validate_argument(source_file, "source file");
    validate_argument(output_file, "output file");
    validate_argument(variant, "variant");

    const bool legacy_conversion =
        !source_directory.empty() || !source_file.empty() || !output_file.empty() || !variant.empty();
    const auto script = state_->repository_root / "tools" /
        (legacy_conversion ? "model_manager_deprecated.py" : "model_manager_v2.py");
    if (!std::filesystem::is_regular_file(script)) {
        throw std::runtime_error(
            "model preparation helper was not found at " + script.string() +
            "; run the server from an updated audio.cpp source or portable bundle root");
    }

    std::filesystem::path source_path;
    if (!source_directory.empty()) {
        source_path = std::filesystem::path(source_directory);
        if (source_path.is_relative()) {
            source_path = state_->repository_root / source_path;
        }
        source_path = std::filesystem::absolute(source_path).lexically_normal();
    }
    auto resolve_optional_path = [this](const std::string & value) {
        if (value.empty()) {
            return std::filesystem::path{};
        }
        auto path = std::filesystem::path(value);
        if (path.is_relative()) {
            path = state_->repository_root / path;
        }
        return std::filesystem::absolute(path).lexically_normal();
    };
    const auto source_file_path = resolve_optional_path(source_file);
    const auto output_file_path = resolve_optional_path(output_file);

    const auto log_path = state_->job_root / (package_id + ".log");
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto existing = state_->jobs.find(package_id);
        if (existing != state_->jobs.end() &&
            (existing->second.state == "queued" || existing->second.state == "running")) {
            throw std::runtime_error("installation is already running for " + package_id);
        }
        State::Job job;
        job.package_id = package_id;
        job.message = "Waiting for model preparation worker";
        job.log_path = log_path;
        state_->jobs[package_id] = std::move(job);
    }

    const auto shared = state_;
    std::thread([shared, package_id, source_file_path, output_file_path, source_path, variant, overwrite,
                 legacy_conversion, script, log_path]() {
        try {
            {
                std::lock_guard<std::mutex> lock(shared->mutex);
                auto & job = shared->jobs.at(package_id);
                job.state = "running";
                job.message = "Downloading and preparing model files";
                job.started_at_ms = now_ms();
            }

            std::string command = python_command() + " " + shell_quote(script.string()) +
                " install " + shell_quote(package_id) +
                " --models-root " + shell_quote(shared->models_root.string());
            if (overwrite) {
                command += " --overwrite";
            }
            if (legacy_conversion && !source_path.empty()) {
                command += " --source-dir " + shell_quote(source_path.string());
            }
            if (legacy_conversion && !source_file_path.empty()) {
                command += " --source-file " + shell_quote(source_file_path.string());
            }
            if (legacy_conversion && !output_file_path.empty()) {
                command += " --output-file " + shell_quote(output_file_path.string());
            }
            if (legacy_conversion && !variant.empty()) {
                command += " --variant " + shell_quote(variant);
            }
            command += " > " + shell_quote(log_path.string()) + " 2>&1";

            const int result = std::system(command.c_str());
            const std::string last_line = read_log_tail(log_path);
            std::lock_guard<std::mutex> lock(shared->mutex);
            auto & job = shared->jobs.at(package_id);
            job.exit_code = result;
            job.finished_at_ms = now_ms();
            job.state = result == 0 ? "complete" : "failed";
            job.message = !last_line.empty()
                ? last_line
                : (result == 0 ? "Model installation completed" : "Model installation failed");
        } catch (const std::exception & error) {
            std::lock_guard<std::mutex> lock(shared->mutex);
            auto & job = shared->jobs.at(package_id);
            job.state = "failed";
            job.message = error.what();
            job.exit_code = -1;
            job.finished_at_ms = now_ms();
        }
    }).detach();

    return status(package_id);
}

std::string ModelInstaller::status(const std::string & package_id) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto job_json = [](const State::Job & job) {
        std::string message = job.message;
        if (job.state == "running") {
            const auto last_line = read_log_tail(job.log_path);
            if (!last_line.empty()) {
                message = last_line;
            }
        }
        return std::string("{\"id\":") + json_quote(job.package_id) +
            ",\"state\":" + json_quote(job.state) +
            ",\"message\":" + json_quote(message) +
            ",\"exit_code\":" + std::to_string(job.exit_code) +
            ",\"started_at_ms\":" + std::to_string(job.started_at_ms) +
            ",\"finished_at_ms\":" + std::to_string(job.finished_at_ms) + "}";
    };

    if (!package_id.empty()) {
        const auto found = state_->jobs.find(package_id);
        if (found == state_->jobs.end()) {
            return "{\"id\":" + json_quote(package_id) +
                ",\"state\":\"idle\",\"message\":\"Not started\",\"exit_code\":-1,"
                "\"started_at_ms\":0,\"finished_at_ms\":0}";
        }
        return job_json(found->second);
    }

    std::string result = "{\"data\":[";
    bool first = true;
    for (const auto & item : state_->jobs) {
        if (!first) {
            result += ",";
        }
        first = false;
        result += job_json(item.second);
    }
    return result + "]}";
}

}  // namespace minitts::server
