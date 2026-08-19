#include "model_installer.h"
#include "engine/framework/package_manager/manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
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

std::string read_log_tail_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    constexpr std::streamoff kTailBytes = 65536;
    if (size > kTailBytes) {
        input.seekg(size - kTailBytes);
    } else {
        input.seekg(0);
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void write_text_atomic(const std::filesystem::path & path, const std::string & text) {
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not write package inventory: " + temporary);
        }
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path);
}

struct LogSnapshot {
    std::string message;
    uint64_t downloaded_bytes = 0;
    uint64_t total_bytes = 0;
    bool has_progress = false;
};

LogSnapshot inspect_log(const std::filesystem::path & path) {
    LogSnapshot snapshot;
    std::istringstream lines(read_log_tail_text(path));
    std::string line;
    constexpr std::string_view marker = "AUDIOCPP_PROGRESS ";
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line.rfind(marker.data(), 0) != 0) {
            snapshot.message = line;
            continue;
        }
        uint64_t downloaded = 0;
        uint64_t total = 0;
        bool parsed_downloaded = false;
        std::istringstream fields(line.substr(marker.size()));
        std::string field;
        while (fields >> field) {
            const auto separator = field.find('=');
            if (separator == std::string::npos) {
                continue;
            }
            try {
                const auto value = std::stoull(field.substr(separator + 1));
                const auto name = field.substr(0, separator);
                if (name == "downloaded") {
                    downloaded = value;
                    parsed_downloaded = true;
                } else if (name == "total") {
                    total = value;
                }
            } catch (const std::exception &) {
                // A partially written progress line is expected while the worker is
                // appending to the log. Keep the previous complete marker.
            }
        }
        if (parsed_downloaded) {
            snapshot.downloaded_bytes = downloaded;
            snapshot.total_bytes = total;
            snapshot.has_progress = true;
        }
    }
    return snapshot;
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::filesystem::path make_job_root() {
    const auto temporary = std::filesystem::temp_directory_path();
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const auto candidate = temporary /
            ("audiocpp-model-installer-" + std::to_string(ticks) + "-" +
             std::to_string(static_cast<unsigned long long>(random())));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error)) {
            return candidate;
        }
        if (error) {
            throw std::runtime_error(
                "failed to create model installer temporary directory: " + error.message());
        }
    }
    throw std::runtime_error("failed to allocate a unique model installer temporary directory");
}

}  // namespace

struct ModelInstaller::State {
    struct Job {
        std::string package_id;
        std::string state = "queued";
        std::string message;
        std::filesystem::path log_path;
        std::filesystem::path cancel_path;
        bool cancel_requested = false;
        std::shared_ptr<std::atomic_bool> cancel_flag = std::make_shared<std::atomic_bool>(false);
        int exit_code = -1;
        int64_t started_at_ms = 0;
        int64_t finished_at_ms = 0;
        uint64_t downloaded_bytes = 0;
        uint64_t total_bytes = 0;
    };

    std::filesystem::path repository_root;
    std::filesystem::path models_root;
    std::filesystem::path job_root;
    mutable std::mutex mutex;
    std::map<std::string, Job> jobs;
    std::string size_state = "idle";
    std::string size_message = "Package sizes have not been checked";
    std::filesystem::path size_output_path;
    std::filesystem::path size_error_path;
    std::filesystem::path installed_output_path;
    uint64_t size_generation = 0;
    std::shared_ptr<engine::package_manager::PackageManager> native_manager;

    ~State() {
        std::error_code error;
        std::filesystem::remove_all(job_root, error);
    }
};

ModelInstaller::ModelInstaller(
    std::filesystem::path repository_root,
    std::filesystem::path models_root)
    : state_(std::make_shared<State>()) {
    state_->repository_root = std::filesystem::absolute(std::move(repository_root)).lexically_normal();
    state_->models_root = std::filesystem::absolute(std::move(models_root)).lexically_normal();
    state_->job_root = make_job_root();
    std::filesystem::create_directories(state_->models_root);
    state_->size_output_path = state_->job_root / "package-sizes.json";
    state_->size_error_path = state_->job_root / "package-sizes.log";
    state_->installed_output_path = state_->job_root / "installed-packages.json";
    state_->native_manager = std::make_shared<engine::package_manager::PackageManager>(
        state_->repository_root, state_->models_root);
}

ModelInstaller::~ModelInstaller() = default;

bool ModelInstaller::has_active_jobs() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto & [id, job] : state_->jobs) {
        (void) id;
        if (job.state == "queued" || job.state == "running" || job.state == "cancelling") {
            return true;
        }
    }
    return false;
}

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
    // The native WebUI does not send converter inputs, so normal UI installs
    // always use the C++ package manager below. This deprecated Python branch
    // is retained only for old direct API callers that still pass conversion
    // fields, and should not be treated as part of the UI download path.
    const auto script = state_->repository_root / "tools" / "model_manager_deprecated.py";
    if (legacy_conversion && !std::filesystem::is_regular_file(script)) {
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
    const auto cancel_path = state_->job_root / (package_id + ".cancel");
    {
        std::error_code error;
        std::filesystem::remove(cancel_path, error);
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto existing = state_->jobs.find(package_id);
        if (existing != state_->jobs.end() &&
            (existing->second.state == "queued" || existing->second.state == "running" ||
             existing->second.state == "cancelling")) {
            throw std::runtime_error("installation is already running for " + package_id);
        }
        State::Job job;
        job.package_id = package_id;
        job.message = "Waiting for model preparation worker";
        job.log_path = log_path;
        job.cancel_path = cancel_path;
        state_->jobs[package_id] = std::move(job);
    }

    const auto shared = state_;
    std::thread([shared, package_id, source_file_path, output_file_path, source_path, variant, overwrite,
                 legacy_conversion, script, log_path, cancel_path]() {
        try {
            {
                std::lock_guard<std::mutex> lock(shared->mutex);
                auto & job = shared->jobs.at(package_id);
                if (!job.cancel_requested) {
                    job.state = "running";
                    job.message = "Downloading and preparing model files";
                }
                job.started_at_ms = now_ms();
            }

            if (!legacy_conversion) {
                std::shared_ptr<std::atomic_bool> cancel_flag;
                {
                    std::lock_guard<std::mutex> lock(shared->mutex);
                    cancel_flag = shared->jobs.at(package_id).cancel_flag;
                }
                const auto message = shared->native_manager->install(
                    package_id, overwrite, cancel_flag,
                    [shared, package_id](const engine::package_manager::PackageProgress & progress) {
                        std::lock_guard<std::mutex> lock(shared->mutex);
                        auto & job = shared->jobs.at(package_id);
                        job.downloaded_bytes = progress.downloaded_bytes;
                        job.total_bytes = progress.total_bytes;
                        if (!progress.message.empty()) job.message = progress.message;
                    });
                std::lock_guard<std::mutex> lock(shared->mutex);
                auto & job = shared->jobs.at(package_id);
                job.finished_at_ms = now_ms();
                job.exit_code = 0;
                job.state = job.cancel_requested ? "cancelled" : "complete";
                job.message = job.cancel_requested
                    ? "Download stopped; completed staging files were removed" : message;
                if (job.state == "complete") {
                    job.downloaded_bytes = job.total_bytes;
                    ++shared->size_generation;
                    shared->size_state = "idle";
                    shared->size_message = "Package inventory will be refreshed";
                }
                return;
            }

            std::string command = python_command() + " -u " + shell_quote(script.string()) +
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
            const auto log = inspect_log(log_path);
            std::lock_guard<std::mutex> lock(shared->mutex);
            auto & job = shared->jobs.at(package_id);
            job.exit_code = result;
            job.finished_at_ms = now_ms();
            const bool cancelled = job.cancel_requested || std::filesystem::is_regular_file(cancel_path);
            job.state = cancelled ? "cancelled" : result == 0 ? "complete" : "failed";
            job.downloaded_bytes = log.downloaded_bytes;
            job.total_bytes = log.total_bytes;
            job.message = !log.message.empty()
                ? log.message
                : (cancelled ? "Download stopped; completed staging files were removed"
                             : result == 0 ? "Model installation completed" : "Model installation failed");
            if (result == 0) {
                ++shared->size_generation;
                shared->size_state = "idle";
                shared->size_message = "Package inventory will be refreshed";
            }
        } catch (const std::exception & error) {
            std::lock_guard<std::mutex> lock(shared->mutex);
            auto & job = shared->jobs.at(package_id);
            job.state = job.cancel_requested ? "cancelled" : "failed";
            job.message = job.cancel_requested
                ? "Download stopped; completed staging files were removed"
                : error.what();
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
        uint64_t downloaded_bytes = job.downloaded_bytes;
        uint64_t total_bytes = job.total_bytes;
        if (job.state == "running" || job.state == "cancelling") {
            const auto log = inspect_log(job.log_path);
            if (!log.message.empty()) {
                message = log.message;
            }
            if (log.has_progress) {
                downloaded_bytes = log.downloaded_bytes;
                total_bytes = log.total_bytes;
            }
        }
        int progress_percent = -1;
        if (job.state == "queued") {
            progress_percent = 0;
        } else if (job.state == "complete") {
            progress_percent = 100;
        } else if (total_bytes > 0) {
            progress_percent = static_cast<int>(std::min<uint64_t>(100, downloaded_bytes * 100 / total_bytes));
        }
        return std::string("{\"id\":") + json_quote(job.package_id) +
            ",\"state\":" + json_quote(job.state) +
            ",\"message\":" + json_quote(message) +
            ",\"exit_code\":" + std::to_string(job.exit_code) +
            ",\"downloaded_bytes\":" + std::to_string(downloaded_bytes) +
            ",\"total_bytes\":" + std::to_string(total_bytes) +
            ",\"progress_percent\":" + std::to_string(progress_percent) +
            ",\"started_at_ms\":" + std::to_string(job.started_at_ms) +
            ",\"finished_at_ms\":" + std::to_string(job.finished_at_ms) + "}";
    };

    if (!package_id.empty()) {
        const auto found = state_->jobs.find(package_id);
        if (found == state_->jobs.end()) {
            return "{\"id\":" + json_quote(package_id) +
                ",\"state\":\"idle\",\"message\":\"Not started\",\"exit_code\":-1,"
                "\"downloaded_bytes\":0,\"total_bytes\":0,\"progress_percent\":-1,"
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

std::string ModelInstaller::stop(const std::string & package_id) {
    if (!valid_package_id(package_id)) {
        throw std::runtime_error("invalid model-manager package id");
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->jobs.find(package_id);
        if (found == state_->jobs.end()) {
            throw std::runtime_error("no installation job exists for " + package_id);
        }
        auto & job = found->second;
        if (job.state != "queued" && job.state != "running" && job.state != "cancelling") {
            throw std::runtime_error("installation is not running for " + package_id);
        }
        std::ofstream marker(job.cancel_path, std::ios::binary | std::ios::trunc);
        if (!marker) {
            throw std::runtime_error("could not create download cancellation marker");
        }
        marker << "cancel\n";
        job.cancel_requested = true;
        job.cancel_flag->store(true);
        job.state = "cancelling";
        job.message = "Stopping download and removing its staging files";
    }
    return status(package_id);
}

std::string ModelInstaller::clean_partial(const std::string & package_id) {
    if (!valid_package_id(package_id)) {
        throw std::runtime_error("invalid model-manager package id");
    }
    std::filesystem::path cancel_path;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto found = state_->jobs.find(package_id);
        if (found != state_->jobs.end() &&
            (found->second.state == "queued" || found->second.state == "running" ||
             found->second.state == "cancelling")) {
            throw std::runtime_error("stop the active installation before cleaning partial files for " + package_id);
        }
        if (found != state_->jobs.end()) {
            cancel_path = found->second.cancel_path;
        }
    }

    const auto message = state_->native_manager->clean_partial(package_id);
    if (!cancel_path.empty()) {
        std::error_code error;
        std::filesystem::remove(cancel_path, error);
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->jobs.erase(package_id);
    }
    return "{\"id\":" + json_quote(package_id) +
        ",\"cleaned\":true,\"message\":" +
        json_quote(message.empty() ? "Partial download files cleaned" : message) + "}";
}

std::string ModelInstaller::remove(const std::string & package_id) {
    if (!valid_package_id(package_id)) {
        throw std::runtime_error("invalid model-manager package id");
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        const auto job = state_->jobs.find(package_id);
        if (job != state_->jobs.end() &&
            (job->second.state == "queued" || job->second.state == "running" ||
             job->second.state == "cancelling")) {
            throw std::runtime_error("installation is still running for " + package_id);
        }
    }

    const auto message = state_->native_manager->remove(package_id);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->jobs.erase(package_id);
        ++state_->size_generation;
        state_->size_state = "idle";
        state_->size_message = "Package inventory will be refreshed";
    }
    return "{\"id\":" + json_quote(package_id) +
        ",\"removed\":true,\"message\":" +
        json_quote(message.empty() ? "Model package removed" : message) + "}";
}

std::string ModelInstaller::package_sizes() {
    bool start_scan = false;
    uint64_t scan_generation = 0;
    std::filesystem::path size_output_path;
    std::filesystem::path size_error_path;
    std::filesystem::path installed_output_path;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->size_state == "idle") {
            scan_generation = ++state_->size_generation;
            const auto suffix = std::to_string(scan_generation);
            state_->size_output_path = state_->job_root / ("package-sizes-" + suffix + ".json");
            state_->size_error_path = state_->job_root / ("package-sizes-" + suffix + ".log");
            state_->installed_output_path = state_->job_root / ("installed-packages-" + suffix + ".json");
            size_output_path = state_->size_output_path;
            size_error_path = state_->size_error_path;
            installed_output_path = state_->installed_output_path;
            state_->size_state = "running";
            state_->size_message = "Checking package sizes";
            start_scan = true;
            std::ofstream(state_->size_output_path, std::ios::trunc);
            std::ofstream(state_->size_error_path, std::ios::trunc);
            write_text_atomic(state_->installed_output_path, state_->native_manager->inventory(false));
        }
    }

    if (start_scan) {
        const auto shared = state_;
        std::thread([shared, scan_generation, size_output_path, size_error_path,
                         installed_output_path]() {
                try {
                    const auto output = shared->native_manager->inventory(true);
                    write_text_atomic(size_output_path, output);
                    std::lock_guard<std::mutex> lock(shared->mutex);
                    if (shared->size_generation != scan_generation) {
                        return;
                    }
                    shared->size_state = "complete";
                    shared->size_message = "Package sizes are ready";
                } catch (const std::exception & error) {
                    std::lock_guard<std::mutex> lock(shared->mutex);
                    if (shared->size_generation == scan_generation) {
                        shared->size_state = "failed";
                        shared->size_message = error.what();
                    }
                }
        }).detach();
    }

    std::lock_guard<std::mutex> lock(state_->mutex);
    std::string data = "[]";
    if (state_->size_state == "complete" || state_->size_state == "running") {
        const auto & source = state_->size_state == "complete"
            ? state_->size_output_path
            : state_->installed_output_path;
        data = read_log_tail_text(source);
        while (!data.empty() && std::isspace(static_cast<unsigned char>(data.back()))) {
            data.pop_back();
        }
        if (data.empty() || data.front() != '[' || data.back() != ']') {
            data = "[]";
        }
    }
    return "{\"state\":" + json_quote(state_->size_state) +
        ",\"message\":" + json_quote(state_->size_message) +
        ",\"data\":" + data + "}";
}

}  // namespace minitts::server
