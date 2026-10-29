#include "server_ui.h"

#include "server_protocol.h"
#include "ui_assets.h"
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
#include "model_installer.h"
#include "engine/framework/runtime/registry.h"
#endif

#include "engine/framework/io/json.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace minitts::server {

std::filesystem::path resolve_ui_model_path(
    const std::filesystem::path &path,
    const std::filesystem::path &request_base,
    const std::filesystem::path &repository_root) {
    if (path.is_absolute()) {
        return path;
    }

    const auto relative = path.lexically_normal();
    const auto generic = relative.generic_string();
    if (!repository_root.empty() &&
        (generic == "assets/framework" || generic.rfind("assets/framework/", 0) == 0)) {
        return (repository_root / relative).lexically_normal();
    }
    return (request_base / relative).lexically_normal();
}

HttpResponse ui_path_status_response(
    const std::string &body_text,
    bool ui_management,
    const std::filesystem::path &request_base,
    const std::filesystem::path &repository_root) {
    if (!ui_management) {
        return error_response(403, "UI path inspection is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto path = resolve_ui_model_path(
        engine::io::json::require_string(body, "path"), request_base, repository_root);
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    const bool directory = exists && std::filesystem::is_directory(path, ec);
    const bool regular_file = exists && std::filesystem::is_regular_file(path, ec);
    return json_response(
        "{\"path\":" + json_quote(path.string()) +
        ",\"exists\":" + (exists ? "true" : "false") +
        ",\"directory\":" + (directory ? "true" : "false") +
        ",\"file\":" + (regular_file ? "true" : "false") + "}");
}

HttpResponse ui_upload_response(
    const HttpRequest &request,
    bool enabled,
    const std::function<std::filesystem::path(std::string_view)> &make_path) {
    if (!enabled) {
        return error_response(403, "UI uploads are disabled", "forbidden");
    }
    if (request.body.empty()) {
        return error_response(400, "upload body is empty", "invalid_request_error");
    }
    constexpr size_t kMaxUploadBytes = size_t{2} * 1024 * 1024 * 1024;
    if (request.body.size() > kMaxUploadBytes) {
        return error_response(413, "upload exceeds the 2 GiB limit", "invalid_request_error");
    }
    std::string filename = "audio.wav";
    if (const auto it = request.headers.find("x-audiocpp-filename"); it != request.headers.end()) {
        filename = safe_upload_name(it->second);
    }
    const auto path = make_path(filename);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not create temporary upload: " + path.string());
    }
    out.write(request.body.data(), static_cast<std::streamsize>(request.body.size()));
    if (!out) {
        throw std::runtime_error("could not write temporary upload: " + path.string());
    }
    return json_response(
        "{\"path\":" + json_quote(path.string()) +
        ",\"bytes\":" + std::to_string(request.body.size()) + "}");
}

HttpResponse ui_asset_response(bool ui_enabled) {
    if (!ui_enabled) {
        return error_response(404, "WebUI is disabled", "not_found");
    }
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/html; charset=utf-8";
    const auto html = embedded_ui_html();
    response.body.assign(html.data(), html.size());
    // The WebUI shares the server origin (usually localhost:8080) with any app
    // that previously occupied that port. Never let an old shell survive a
    // server upgrade, and ask the browser to discard only cached resources.
    // Deliberately omit the Clear-Site-Data "storage" directive: saved voices,
    // the selected models folder, and UI preferences live in local storage.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0";
    response.headers["Clear-Site-Data"] = "\"cache\"";
    response.headers["Expires"] = "0";
    response.headers["Pragma"] = "no-cache";
    response.headers["Content-Security-Policy"] =
        "default-src 'self' 'unsafe-inline' blob: data:; connect-src 'self'; media-src 'self' blob: data:";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
}

HttpResponse ui_voice_preview_response(
    const HttpRequest &request,
    bool ui_enabled,
    const std::optional<std::filesystem::path> &voice_dir) {
    if (!ui_enabled) {
        return error_response(404, "WebUI is disabled", "not_found");
    }
    if (!voice_dir.has_value()) {
        return error_response(404, "voice library is not configured", "not_found");
    }
    const std::string voice = decoded_query_param(request.query, "voice");
    const auto wav = resolve_voice_library_wav(*voice_dir, voice);
    if (!wav.has_value()) {
        return error_response(404, "voice preview is not available", "not_found");
    }
    const auto file_size = std::filesystem::file_size(*wav);
    std::ifstream file(*wav, std::ios::binary);
    if (!file) {
        return error_response(404, "voice preview is not available", "not_found");
    }
    std::uintmax_t offset = 0;
    std::uintmax_t count = file_size;
    bool partial = false;
    if (const auto range = request.headers.find("range"); range != request.headers.end() &&
        range->second.rfind("bytes=", 0) == 0) {
        const std::string spec = range->second.substr(6);
        const size_t dash = spec.find('-');
        if (dash != std::string::npos && dash > 0) {
            const std::uintmax_t begin = std::stoull(spec.substr(0, dash));
            const std::uintmax_t end = dash + 1 < spec.size()
                ? std::stoull(spec.substr(dash + 1))
                : file_size - 1;
            if (begin >= file_size || end < begin) {
                HttpResponse response;
                response.status = 416;
                response.content_type = "text/plain";
                response.headers["Content-Range"] = "bytes */" + std::to_string(file_size);
                response.headers["Accept-Ranges"] = "bytes";
                return response;
            }
            offset = begin;
            count = std::min(end, file_size - 1) - begin + 1;
            partial = true;
        }
    }
    HttpResponse response;
    response.status = partial ? 206 : 200;
    response.content_type = "audio/wav";
    if (partial) {
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        response.body.resize(static_cast<size_t>(count));
        file.read(response.body.data(), static_cast<std::streamsize>(response.body.size()));
        response.body.resize(static_cast<size_t>(file.gcount()));
        response.headers["Content-Range"] =
            "bytes " + std::to_string(offset) + "-" +
            std::to_string(offset + response.body.size() - 1) + "/" +
            std::to_string(file_size);
    } else {
        response.body.assign(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());
    }
    response.headers["Accept-Ranges"] = "bytes";
    response.headers["Cache-Control"] = "no-store";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
}

#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
UiModelInstallRequest parse_ui_model_install_request(const std::string &body_text) {
    const auto body = engine::io::json::parse(body_text);
    UiModelInstallRequest request;
    request.package_id = engine::io::json::require_string(body, "id");
    request.source_directory =
        engine::io::json::optional_string(body, "source_directory", "");
    request.source_file =
        engine::io::json::optional_string(body, "source_file", "");
    request.output_file =
        engine::io::json::optional_string(body, "output_file", "");
    request.variant =
        engine::io::json::optional_string(body, "variant", "");
    request.overwrite =
        engine::io::json::optional_bool(body, "overwrite", false);
    return request;
}

std::string parse_ui_model_id(const std::string &body_text) {
    return engine::io::json::require_string(engine::io::json::parse(body_text), "id");
}

HttpResponse ui_model_install_response(
    const UiModelInstallRequest &request,
    ModelInstaller &installer) {
    return json_response(installer.start(
        request.package_id,
        request.source_file,
        request.output_file,
        request.source_directory,
        request.variant,
        request.overwrite));
}

HttpResponse ui_model_remove_response(
    const std::string &package_id,
    ModelInstaller &installer) {
    return json_response(installer.remove(package_id));
}

HttpResponse ui_model_install_stop_response(
    const std::string &package_id,
    ModelInstaller &installer) {
    return json_response(installer.stop(package_id));
}

HttpResponse ui_model_clean_partial_response(
    const std::string &package_id,
    ModelInstaller &installer) {
    return json_response(installer.clean_partial(package_id));
}

HttpResponse ui_model_install_status_response(
    const HttpRequest &request,
    const ModelInstaller &installer) {
    return json_response(installer.status(query_param(request.query, "id")));
}

HttpResponse ui_model_package_sizes_response(ModelInstaller &installer) {
    namespace json = engine::io::json;
    auto inventory = json::parse(installer.package_sizes()).as_object();
    const auto registry = engine::runtime::make_default_registry();
    auto rows = inventory.at("data").as_array();
    for (auto & value : rows) {
        auto row = value.as_object();
        const auto & metadata = row.at("package");
        engine::assets::ModelPackage package;
        package.root = json::require_string(metadata, "root");
        package.format = json::require_string(metadata, "format");
        for (const auto & file : metadata.require("files").as_array()) {
            package.files.emplace_back(file.as_string());
        }
        try {
            const auto target = registry.resolve_package_load(json::require_string(metadata, "family"), package);
            json::Value::Object load_options, session_options;
            for (const auto & option : target.load_options) {
                load_options[option.first] = json::Value::make_string(option.second);
            }
            for (const auto & option : target.session_options) {
                session_options[option.first] = json::Value::make_string(option.second);
            }
            json::Value::Object resolved;
            resolved["family"] = json::Value::make_string(target.family);
            resolved["model_path"] = json::Value::make_string(target.model_path.generic_string());
            resolved["load_options"] = json::Value::make_object(std::move(load_options));
            resolved["session_options"] = json::Value::make_object(std::move(session_options));
            row["load_target"] = json::Value::make_object(std::move(resolved));
        } catch (const std::exception & error) {
            row["load_target_error"] = json::Value::make_string(error.what());
        }
        value = json::Value::make_object(std::move(row));
    }
    inventory["data"] = json::Value::make_array(std::move(rows));
    return json_response(json::stringify(json::Value::make_object(std::move(inventory))));
}

HttpResponse ui_models_root_response(
    const std::filesystem::path &models_root,
    const std::filesystem::path &default_models_root) {
    return json_response(
        "{\"models_root\":" + json_quote(models_root.string()) +
        ",\"default_models_root\":" + json_quote(default_models_root.string()) +
        ",\"is_default\":" + (models_root == default_models_root ? "true" : "false") + "}");
}

HttpResponse ui_directory_browser_response(
    const std::string &requested_path,
    const std::filesystem::path &models_root,
    const std::filesystem::path &request_base) {
    auto current = requested_path.empty() ? models_root : std::filesystem::path(requested_path);
    if (current.is_relative()) {
        current = request_base / current;
    }
    current = std::filesystem::absolute(current).lexically_normal();

    std::error_code error;
    if (!std::filesystem::is_directory(current, error)) {
        return error_response(
            400, "folder is not accessible: " + current.string(), "invalid_request_error");
    }

    std::vector<std::filesystem::path> roots;
#ifdef _WIN32
    for (char drive = 'A'; drive <= 'Z'; ++drive) {
        const auto root = std::filesystem::path(std::string(1, drive) + ":\\");
        error.clear();
        if (std::filesystem::is_directory(root, error)) {
            roots.push_back(root);
        }
    }
#else
    roots.emplace_back("/");
#endif

    struct DirectoryEntry {
        std::string name;
        std::filesystem::path path;
    };
    std::vector<DirectoryEntry> directories;
    error.clear();
    std::filesystem::directory_iterator iterator(
        current, std::filesystem::directory_options::skip_permission_denied, error);
    const std::filesystem::directory_iterator end;
    while (!error && iterator != end) {
        std::error_code entry_error;
        if (iterator->is_directory(entry_error)) {
            directories.push_back(DirectoryEntry{
                iterator->path().filename().string(),
                iterator->path().lexically_normal(),
            });
        }
        iterator.increment(error);
    }
    std::sort(directories.begin(), directories.end(), [](const auto &left, const auto &right) {
        return left.name < right.name;
    });

    auto parent = current.parent_path();
    if (parent == current) {
        parent.clear();
    }
    std::string response =
        "{\"current\":" + json_quote(current.string()) +
        ",\"parent\":" + json_quote(parent.string()) + ",\"roots\":[";
    for (size_t index = 0; index < roots.size(); ++index) {
        if (index > 0) response += ",";
        response += json_quote(roots[index].string());
    }
    response += "],\"directories\":[";
    for (size_t index = 0; index < directories.size(); ++index) {
        if (index > 0) response += ",";
        response += "{\"name\":" + json_quote(directories[index].name) +
            ",\"path\":" + json_quote(directories[index].path.string()) + "}";
    }
    return json_response(response + "]}");
}
#endif

} // namespace minitts::server
