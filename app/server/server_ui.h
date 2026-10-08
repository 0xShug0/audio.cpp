#pragma once

#include "http.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace minitts::server {

std::filesystem::path resolve_ui_model_path(
    const std::filesystem::path &path,
    const std::filesystem::path &request_base,
    const std::filesystem::path &repository_root);
HttpResponse ui_path_status_response(
    const std::string &body_text,
    bool ui_management,
    const std::filesystem::path &request_base,
    const std::filesystem::path &repository_root);
HttpResponse ui_upload_response(
    const HttpRequest &request,
    bool enabled,
    const std::function<std::filesystem::path(std::string_view)> &make_path);
HttpResponse ui_asset_response(bool ui_enabled);
HttpResponse ui_voice_preview_response(
    const HttpRequest &request,
    bool ui_enabled,
    const std::optional<std::filesystem::path> &voice_dir);

#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
class ModelInstaller;

struct UiModelInstallRequest {
    std::string package_id;
    std::string source_directory;
    std::string source_file;
    std::string output_file;
    std::string variant;
    bool overwrite = false;
};

UiModelInstallRequest parse_ui_model_install_request(const std::string &body_text);
std::string parse_ui_model_id(const std::string &body_text);
HttpResponse ui_model_install_response(
    const UiModelInstallRequest &request,
    ModelInstaller &installer);
HttpResponse ui_model_remove_response(
    const std::string &package_id,
    ModelInstaller &installer);
HttpResponse ui_model_install_stop_response(
    const std::string &package_id,
    ModelInstaller &installer);
HttpResponse ui_model_clean_partial_response(
    const std::string &package_id,
    ModelInstaller &installer);
HttpResponse ui_model_install_status_response(
    const HttpRequest &request,
    const ModelInstaller &installer);
HttpResponse ui_model_package_sizes_response(ModelInstaller &installer);
HttpResponse ui_models_root_response(
    const std::filesystem::path &models_root,
    const std::filesystem::path &default_models_root);
HttpResponse ui_directory_browser_response(
    const std::string &requested_path,
    const std::filesystem::path &models_root,
    const std::filesystem::path &request_base);
#endif

} // namespace minitts::server
