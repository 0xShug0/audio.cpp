#include "engine/framework/package_manager/manager.h"

#include "engine/framework/io/json.h"

#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::package_manager {
namespace {

namespace json = engine::io::json;

struct RemoteFileInfo {
    std::optional<uint64_t> size;
    std::string revision;
    std::string etag;
};

struct Package {
    std::string family;
    std::string id;
    std::string display_name;
    std::filesystem::path target_directory;
    std::string format;
    std::string precision;
    std::vector<std::string> files;
    std::string strip_prefix;
    std::string download_kind;
    std::string repo;
    std::string revision = "main";
    bool gated = false;
};

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("download cancelled by user") {}
};

std::string trim_quotes(std::string value) {
    while (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }
    return value;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string getenv_text(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

std::string huggingface_token() {
    std::string token = getenv_text("HF_TOKEN");
    if (token.empty()) {
        token = getenv_text("HUGGING_FACE_HUB_TOKEN");
    }
    if (!token.empty()) {
        return token;
    }
#ifdef _WIN32
    const auto home = getenv_text("USERPROFILE");
#else
    const auto home = getenv_text("HOME");
#endif
    if (!home.empty()) {
        std::ifstream input(std::filesystem::path(home) / ".cache" / "huggingface" / "token");
        if (input) {
            std::getline(input, token);
        }
    }
    return token;
}

bool unreserved(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.' || ch == '~';
}

std::string url_encode(std::string_view value, bool keep_slash) {
    constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char ch : value) {
        if (unreserved(ch) || (keep_slash && ch == '/')) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('%');
            out.push_back(hex[ch >> 4]);
            out.push_back(hex[ch & 0xf]);
        }
    }
    return out;
}

std::string hf_url(const Package & package, const std::string & remote_path) {
    auto base = getenv_text("AUDIOCPP_HF_BASE_URL");
    if (base.empty()) base = "https://huggingface.co";
    while (!base.empty() && base.back() == '/') base.pop_back();
    return base + "/" + package.repo + "/resolve/" +
        url_encode(package.revision, false) + "/" + url_encode(remote_path, true);
}

std::filesystem::path validate_relative_path(const std::filesystem::path & path, const char * label) {
    if (path.empty() || path == "." || path.is_absolute()) {
        throw std::runtime_error(std::string(label) + " must be a non-empty relative path: " + path.string());
    }
    for (const auto & part : path) {
        if (part == "..") {
            throw std::runtime_error(std::string(label) + " must not escape its root: " + path.string());
        }
    }
    return path;
}

std::filesystem::path stripped_path(const Package & package, const std::string & remote) {
    std::string result = remote;
    if (!package.strip_prefix.empty() && package.strip_prefix != ".") {
        std::string prefix = package.strip_prefix;
        while (!prefix.empty() && prefix.back() == '/') prefix.pop_back();
        if (result.size() <= prefix.size() || result.compare(0, prefix.size(), prefix) != 0 ||
            result[prefix.size()] != '/') {
            throw std::runtime_error("file path does not start with strip_prefix '" + prefix + "': " + remote);
        }
        result.erase(0, prefix.size() + 1);
    }
    return validate_relative_path(std::filesystem::path(result), "local file path");
}

std::filesystem::path manifest_path(const Package & package, const std::filesystem::path & models_root) {
    return models_root / package.target_directory / (".audiocpp-package-" + package.id + ".json");
}

bool package_installed(const Package & package, const std::filesystem::path & models_root) {
    if (package.files.empty()) return false;
    const auto root = models_root / package.target_directory;
    return std::all_of(package.files.begin(), package.files.end(), [&](const std::string & remote) {
        return std::filesystem::is_regular_file(root / stripped_path(package, remote));
    });
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not read " + path.string());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void write_text_atomic(const std::filesystem::path & path, const std::string & text) {
    std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("could not write " + temporary);
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
    std::error_code error;
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path);
}

json::Value string_value(const std::string & value) { return json::Value::make_string(value); }
json::Value bool_value(bool value) { return json::Value::make_bool(value); }
json::Value number_value(uint64_t value) { return json::Value::make_number(static_cast<double>(value)); }

std::vector<Package> parse_specs(const std::vector<std::pair<std::string, std::string>> & specs) {
    std::vector<Package> packages;
    std::set<std::string> ids;
    for (const auto & [source, text] : specs) {
        const auto root = json::parse(text);
        const auto family = json::require_string(root, "family");
        std::string default_kind;
        std::string default_repo;
        std::string default_revision = "main";
        bool default_gated = false;
        if (const auto * defaults = root.find("package_defaults")) {
            if (const auto * download = defaults->find("download")) {
                default_kind = json::optional_string(*download, "kind", "");
                default_repo = json::optional_string(*download, "repo", "");
                default_revision = json::optional_string(*download, "revision", "main");
                default_gated = json::optional_bool(*download, "gated", false);
            }
        }
        const auto * values = root.find("packages");
        if (values == nullptr) continue;
        for (const auto & item : values->as_array()) {
            Package package;
            package.family = family;
            package.id = json::require_string(item, "id");
            package.display_name = json::require_string(item, "display_name");
            package.target_directory = validate_relative_path(
                json::require_string(item, "target_directory"), "target_directory");
            package.format = json::require_string(item, "format");
            package.precision = json::require_string(item, "precision");
            package.files = json::require_string_array(item, "files");
            package.strip_prefix = json::optional_string(item, "strip_prefix", "");
            package.download_kind = default_kind;
            package.repo = default_repo;
            package.revision = default_revision;
            package.gated = default_gated;
            if (const auto * download = item.find("download")) {
                package.download_kind = json::optional_string(*download, "kind", package.download_kind);
                package.repo = json::optional_string(*download, "repo", package.repo);
                package.revision = json::optional_string(*download, "revision", package.revision);
                package.gated = json::optional_bool(*download, "gated", package.gated);
            }
            if (package.download_kind != "huggingface_snapshot") continue;
            if (package.repo.empty()) {
                throw std::runtime_error(source + ": package " + package.id + " has no Hugging Face repo");
            }
            if (!ids.insert(package.id).second) {
                throw std::runtime_error("duplicate package id: " + package.id);
            }
            for (const auto & remote : package.files) (void) stripped_path(package, remote);
            packages.push_back(std::move(package));
        }
    }
    return packages;
}

std::vector<std::pair<std::string, std::string>> load_specs(const std::filesystem::path & repository_root) {
    std::vector<std::pair<std::string, std::string>> specs;
    const auto directory = repository_root / "model_specs";
    if (std::filesystem::is_directory(directory)) {
        for (const auto & entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                specs.emplace_back(entry.path().string(), read_text(entry.path()));
            }
        }
        std::sort(specs.begin(), specs.end());
    }
    if (!specs.empty()) return specs;

    static const std::pair<const char *, const char *> embedded[] = {
#include "native_package_specs.inc"
    };
    for (const auto & item : embedded) specs.emplace_back(item.first, item.second);
    if (specs.empty()) {
        throw std::runtime_error("no model package specifications are available");
    }
    return specs;
}

struct HttpResult {
    long status = 0;
    std::unordered_map<std::string, std::string> headers;
};

struct HttpUrl {
    std::string scheme;
    std::string host;
    int port = 0;
    std::string path;
};

HttpUrl parse_http_url(const std::string & url) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid model URL: no scheme");
    }

    HttpUrl result;
    result.scheme = lower(url.substr(0, scheme_end));
    if (result.scheme != "http" && result.scheme != "https") {
        throw std::runtime_error("unsupported model URL scheme: " + result.scheme);
    }

    const auto authority_start = scheme_end + 3;
    const auto path_start = url.find('/', authority_start);
    auto authority = url.substr(authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    result.path = path_start == std::string::npos ? "/" : url.substr(path_start);
    if (authority.empty()) throw std::runtime_error("invalid model URL: no host");

    std::string port;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string::npos) throw std::runtime_error("invalid IPv6 model URL");
        result.host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') throw std::runtime_error("invalid model URL authority");
            port = authority.substr(close + 2);
        }
    } else {
        const auto colon = authority.rfind(':');
        if (colon != std::string::npos && authority.find(':') == colon) {
            result.host = authority.substr(0, colon);
            port = authority.substr(colon + 1);
        } else {
            result.host = authority;
        }
    }
    if (result.host.empty()) throw std::runtime_error("invalid model URL: no host");

    try {
        result.port = port.empty() ? (result.scheme == "https" ? 443 : 80) : std::stoi(port);
    } catch (...) {
        throw std::runtime_error("invalid model URL port");
    }
    if (result.port <= 0 || result.port > 65535) throw std::runtime_error("invalid model URL port");
    return result;
}

std::string http_origin(const HttpUrl & url) {
    const auto host = url.host.find(':') == std::string::npos ? url.host : "[" + url.host + "]";
    return url.scheme + "://" + host + ":" + std::to_string(url.port);
}

HttpResult http_request(
    const std::string & url,
    bool head,
    std::ofstream * output,
    const std::shared_ptr<std::atomic_bool> & cancelled,
    const std::function<void(uint64_t)> & progress) {
    const auto parsed = parse_http_url(url);
    httplib::Client client(http_origin(parsed));
    client.set_follow_location(true);
    client.set_connection_timeout(60, 0);
    client.set_read_timeout(head ? 60 : 300, 0);
    client.set_write_timeout(60, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    client.enable_server_certificate_verification(true);
#endif

    httplib::Headers request_headers{{"User-Agent", "audio.cpp native model manager/1.0"}};
    const auto token = huggingface_token();
    if (!token.empty()) {
        request_headers.emplace("Authorization", "Bearer " + token);
    }

    uint64_t downloaded = 0;
    bool write_failed = false;
    const httplib::ContentReceiver receiver = [&](const char * data, size_t size) {
        if (cancelled && cancelled->load()) return false;
        if (output != nullptr) {
            output->write(data, static_cast<std::streamsize>(size));
            if (!*output) {
                write_failed = true;
                return false;
            }
        }
        downloaded += size;
        if (progress) progress(downloaded);
        return true;
    };
    const httplib::DownloadProgress keep_downloading = [&](size_t, size_t) {
        return !(cancelled && cancelled->load());
    };

    httplib::Result response = head
        ? client.Head(parsed.path, request_headers)
        : client.Get(parsed.path, request_headers, receiver, keep_downloading);
    if (!response) {
        if (cancelled && cancelled->load()) throw Cancelled();
        if (write_failed) throw std::runtime_error("could not write downloaded model file");
        throw std::runtime_error("model host request failed: " + httplib::to_string(response.error()));
    }

    HttpResult result;
    result.status = response->status;
    for (const auto & [name, value] : response->headers) {
        result.headers[lower(name)] = value;
    }
    return result;
}

RemoteFileInfo remote_info(const Package & package, const std::string & remote) {
    const auto response = http_request(hf_url(package, remote), true, nullptr, {}, {});
    if (response.status == 401 || response.status == 403) {
        if (package.gated) return {};
    }
    if (response.status < 200 || response.status >= 300) {
        throw std::runtime_error("remote file is not accessible: " + package.repo + "/" + remote +
            " (HTTP " + std::to_string(response.status) + ")");
    }
    RemoteFileInfo result;
    const auto size = response.headers.find("content-length");
    if (size != response.headers.end() && !size->second.empty()) {
        try { result.size = std::stoull(size->second); } catch (...) {}
    }
    const auto revision = response.headers.find("x-repo-commit");
    if (revision != response.headers.end()) result.revision = revision->second;
    const auto etag = response.headers.find("etag");
    if (etag != response.headers.end()) result.etag = trim_quotes(etag->second);
    return result;
}

void download_file(
    const Package & package,
    const std::string & remote,
    const std::filesystem::path & destination,
    const std::shared_ptr<std::atomic_bool> & cancelled,
    const std::function<void(uint64_t)> & progress) {
    std::filesystem::create_directories(destination.parent_path());
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not create " + destination.string());
    const auto response = http_request(hf_url(package, remote), false, &output, cancelled, progress);
    output.close();
    if (response.status == 401 || response.status == 403) {
        throw std::runtime_error(package.repo + "/" + remote +
            " requires accepted Hugging Face access and a valid HF token");
    }
    if (response.status < 200 || response.status >= 300) {
        throw std::runtime_error("failed to download " + package.repo + "/" + remote +
            " (HTTP " + std::to_string(response.status) + ")");
    }
}

std::filesystem::path make_staging(const std::filesystem::path & models_root, const Package & package) {
    std::filesystem::create_directories(models_root);
    std::string prefix = package.target_directory.generic_string();
    std::replace(prefix.begin(), prefix.end(), '/', '_');
    std::random_device random;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const auto path = models_root / ("." + prefix + "." + std::to_string(random()));
        std::error_code error;
        if (std::filesystem::create_directory(path, error)) return path;
    }
    throw std::runtime_error("could not create package staging directory");
}

}  // namespace

struct PackageManager::Impl {
    std::filesystem::path repository_root;
    std::filesystem::path models_root;
    std::vector<Package> packages;

    const Package & require_package(const std::string & id) const {
        const auto found = std::find_if(packages.begin(), packages.end(), [&](const Package & item) {
            return item.id == id;
        });
        if (found == packages.end()) throw std::runtime_error("unknown package: " + id);
        return *found;
    }

    std::set<std::filesystem::path> shared_outputs(const Package & package) const {
        std::set<std::filesystem::path> shared;
        const auto root = models_root / package.target_directory;
        for (const auto & other : packages) {
            if (other.id == package.id || other.repo != package.repo || other.revision != package.revision ||
                !package_installed(other, models_root)) continue;
            const auto other_root = models_root / other.target_directory;
            for (const auto & remote : package.files) {
                const auto output = root / stripped_path(package, remote);
                for (const auto & other_remote : other.files) {
                    if (remote == other_remote && output == other_root / stripped_path(other, other_remote)) {
                        shared.insert(output);
                    }
                }
            }
        }
        return shared;
    }
};

PackageManager::PackageManager(
    std::filesystem::path repository_root,
    std::filesystem::path models_root)
    : impl_(std::make_unique<Impl>()) {
    impl_->repository_root = std::filesystem::absolute(std::move(repository_root)).lexically_normal();
    impl_->models_root = std::filesystem::absolute(std::move(models_root)).lexically_normal();
    impl_->packages = parse_specs(load_specs(impl_->repository_root));
    std::filesystem::create_directories(impl_->models_root);
}

PackageManager::~PackageManager() = default;

std::string PackageManager::install(
    const std::string & package_id,
    bool overwrite,
    const std::shared_ptr<std::atomic_bool> & cancelled,
    const std::function<void(const PackageProgress &)> & progress) {
    const auto & package = impl_->require_package(package_id);
    const auto final_root = impl_->models_root / package.target_directory;
    std::vector<std::pair<std::string, std::filesystem::path>> plan;
    for (const auto & remote : package.files) plan.emplace_back(remote, final_root / stripped_path(package, remote));

    const auto shared = impl_->shared_outputs(package);
    std::vector<std::pair<std::string, std::filesystem::path>> downloads;
    for (const auto & item : plan) {
        if (std::filesystem::exists(item.second) && !overwrite) {
            if (shared.count(item.second) != 0) continue;
            if (std::all_of(plan.begin(), plan.end(), [](const auto & entry) {
                    return std::filesystem::is_regular_file(entry.second);
                })) {
                return "Already installed " + package.id;
            }
            throw std::runtime_error("some package files already exist in: " + final_root.string() + " (use overwrite)");
        }
        downloads.push_back(item);
    }

    std::map<std::string, RemoteFileInfo> remote_files;
    uint64_t total = 0;
    bool known_total = true;
    for (const auto & [remote, output] : downloads) {
        (void) output;
        auto info = remote_info(package, remote);
        if (!info.size) known_total = false; else total += *info.size;
        remote_files.emplace(remote, std::move(info));
    }
    if (!known_total) total = 0;
    if (progress) progress({0, total, "Downloading and preparing model files"});

    const auto staging = make_staging(impl_->models_root, package);
    uint64_t completed = 0;
    try {
        for (const auto & [remote, output] : downloads) {
            if (cancelled && cancelled->load()) throw Cancelled();
            const auto destination = staging / output.lexically_relative(final_root);
            download_file(package, remote, destination, cancelled, [&](uint64_t file_bytes) {
                if (progress) progress({completed + file_bytes, total, "Downloading " + remote});
            });
            const auto file_size = std::filesystem::file_size(destination);
            const auto expected = remote_files.at(remote).size;
            if (expected && file_size != *expected) {
                throw std::runtime_error(
                    "downloaded file size mismatch for " + remote + ": expected " +
                    std::to_string(*expected) + ", got " + std::to_string(file_size));
            }
            completed += file_size;
            if (progress) progress({completed, total, "Downloaded " + remote});
        }
        if (cancelled && cancelled->load()) throw Cancelled();
        std::filesystem::create_directories(final_root);
        std::vector<std::filesystem::path> staged_files;
        for (const auto & entry : std::filesystem::recursive_directory_iterator(staging)) {
            if (entry.is_regular_file()) staged_files.push_back(entry.path());
        }
        std::sort(staged_files.begin(), staged_files.end());
        for (const auto & staged_file : staged_files) {
            const auto relative = staged_file.lexically_relative(staging);
            const auto destination = final_root / relative;
            std::filesystem::create_directories(destination.parent_path());
            std::error_code error;
            if (std::filesystem::exists(destination)) {
                if (!overwrite) throw std::runtime_error("package file already exists: " + destination.string());
                std::filesystem::remove(destination, error);
                if (error) throw std::runtime_error("could not replace " + destination.string());
            }
            std::filesystem::rename(staged_file, destination);
        }
        std::filesystem::remove_all(staging);

        // Include reused sidecars in the per-package manifest as well.
        for (const auto & [remote, output] : plan) {
            (void) output;
            if (remote_files.count(remote) == 0) remote_files.emplace(remote, remote_info(package, remote));
        }
        json::Value::Object manifest;
        manifest["schema_version"] = number_value(1);
        manifest["package_id"] = string_value(package.id);
        manifest["repo"] = string_value(package.repo);
        manifest["requested_revision"] = string_value(package.revision);
        std::string resolved = package.revision;
        for (const auto & [remote, info] : remote_files) {
            (void) remote;
            if (!info.revision.empty()) { resolved = info.revision; break; }
        }
        manifest["resolved_revision"] = string_value(resolved);
        manifest["installed_at_unix"] = number_value(static_cast<uint64_t>(std::time(nullptr)));
        json::Value::Object files;
        for (const auto & [remote, info] : remote_files) {
            json::Value::Object file;
            file["size"] = info.size ? number_value(*info.size) : json::Value::make_null();
            file["etag"] = string_value(info.etag);
            files[remote] = json::Value::make_object(std::move(file));
        }
        manifest["files"] = json::Value::make_object(std::move(files));
        write_text_atomic(manifest_path(package, impl_->models_root),
            json::stringify(json::Value::make_object(std::move(manifest))) + "\n");
    } catch (...) {
        std::error_code error;
        std::filesystem::remove_all(staging, error);
        throw;
    }
    return "Installed " + package.id + " -> " + final_root.string();
}

std::string PackageManager::clean_partial(const std::string & package_id) {
    const auto & package = impl_->require_package(package_id);
    std::string prefix = "." + package.target_directory.generic_string() + ".";
    std::replace(prefix.begin(), prefix.end(), '/', '_');
    size_t removed = 0;
    for (const auto & entry : std::filesystem::directory_iterator(impl_->models_root)) {
        const auto name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind(prefix, 0) == 0) {
            removed += std::filesystem::remove_all(entry.path()) > 0 ? 1 : 0;
        }
    }
    return "Cleaned " + std::to_string(removed) + " partial download " +
        (removed == 1 ? "directory" : "directories") + " for " + package.id;
}

std::string PackageManager::remove(const std::string & package_id) {
    const auto & package = impl_->require_package(package_id);
    if (!package_installed(package, impl_->models_root)) {
        throw std::runtime_error("package is not installed: " + package.id);
    }
    const auto root = impl_->models_root / package.target_directory;
    const auto shared = impl_->shared_outputs(package);
    std::vector<std::filesystem::path> outputs;
    for (const auto & remote : package.files) outputs.push_back(root / stripped_path(package, remote));
    for (const auto & output : outputs) {
        if (shared.count(output) == 0) std::filesystem::remove(output);
    }
    std::filesystem::remove(manifest_path(package, impl_->models_root));
    std::set<std::filesystem::path> directories;
    for (auto output : outputs) {
        for (auto parent = output.parent_path(); parent != root.parent_path(); parent = parent.parent_path()) {
            directories.insert(parent);
        }
    }
    std::vector<std::filesystem::path> ordered(directories.begin(), directories.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) {
        return a.native().size() > b.native().size();
    });
    for (const auto & directory : ordered) {
        std::error_code error;
        std::filesystem::remove(directory, error);
    }
    return "Removed " + package.id + " -> " + root.string();
}

std::string PackageManager::inventory(bool query_remote, const std::string & package_id) const {
    std::vector<const Package *> selected;
    selected.reserve(impl_->packages.size());
    for (const auto & package : impl_->packages) {
        if (package_id.empty() || package.id == package_id) selected.push_back(&package);
    }
    if (!package_id.empty() && selected.empty()) throw std::runtime_error("unknown package id: " + package_id);
    std::vector<json::Value> rows(selected.size());
    std::atomic_size_t next{0};
    const auto worker_count = std::min<size_t>(12, std::max<size_t>(1, selected.size()));
    std::vector<std::future<void>> workers;
    workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        workers.push_back(std::async(std::launch::async, [this, query_remote, &selected, &rows, &next] {
          for (;;) {
            const auto index = next.fetch_add(1);
            if (index >= selected.size()) return;
            const auto & package = *selected[index];
            const bool installed = package_installed(package, impl_->models_root);
            json::Value::Object row;
            row["id"] = string_value(package.id);
            row["installed"] = bool_value(installed);
            row["state"] = string_value(query_remote ? "ok" : "pending");
            row["message"] = string_value("");
            row["size_bytes"] = json::Value::make_null();
            row["version_state"] = string_value(installed ? "unknown" : "not_installed");
            row["local_revision"] = string_value("");
            row["remote_revision"] = string_value("");
            if (!query_remote) {
                rows[index] = json::Value::make_object(std::move(row));
                continue;
            }
            try {
                uint64_t total = 0;
                bool size_known = true;
                std::map<std::string, RemoteFileInfo> remote;
                std::string revision;
                for (const auto & file : package.files) {
                    auto info = remote_info(package, file);
                    if (!info.size) size_known = false; else total += *info.size;
                    if (revision.empty()) revision = info.revision;
                    remote.emplace(file, std::move(info));
                }
                if (size_known) row["size_bytes"] = number_value(total);
                row["remote_revision"] = string_value(revision);
                if (!size_known && package.gated) {
                    row["state"] = string_value("gated");
                    row["message"] = string_value("Hugging Face access and a valid token are required");
                }
                if (installed) {
                    try {
                        const auto manifest = json::parse_file(manifest_path(package, impl_->models_root));
                        row["local_revision"] = string_value(json::optional_string(manifest, "resolved_revision", ""));
                        size_t compared = 0;
                        bool changed = false;
                        if (const auto * files = manifest.find("files")) {
                            for (const auto & [name, stored] : files->as_object()) {
                                const auto found = remote.find(name);
                                const auto local_etag = trim_quotes(json::optional_string(stored, "etag", ""));
                                if (found != remote.end() && !local_etag.empty() && !found->second.etag.empty()) {
                                    ++compared;
                                    changed = changed || local_etag != found->second.etag;
                                }
                            }
                        }
                        row["version_state"] = string_value(
                            changed ? "update_available" : compared ? "up_to_date" : "unknown");
                    } catch (...) {
                        row["version_state"] = string_value("unknown");
                    }
                }
            } catch (const std::exception & error) {
                row["state"] = string_value("error");
                row["message"] = string_value(error.what());
            }
            rows[index] = json::Value::make_object(std::move(row));
          }
        }));
    }
    for (auto & worker : workers) worker.get();
    return json::stringify(json::Value::make_array(std::move(rows)));
}

}  // namespace engine::package_manager
