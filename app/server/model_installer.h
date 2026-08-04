#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace minitts::server {

// Runs the repository model manager outside the request thread.  The native UI
// owns the workflow and status display. Normal packages use model_manager_v2;
// requests with converter inputs use the deprecated manager until those legacy
// preparation workflows have migrated to package specs.
class ModelInstaller {
public:
    ModelInstaller(std::filesystem::path repository_root, std::filesystem::path models_root);
    ~ModelInstaller();

    ModelInstaller(const ModelInstaller &) = delete;
    ModelInstaller & operator=(const ModelInstaller &) = delete;

    std::string start(
        const std::string & package_id,
        const std::string & source_file,
        const std::string & output_file,
        const std::string & source_directory,
        const std::string & variant,
        bool overwrite);
    std::string status(const std::string & package_id = {}) const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

}  // namespace minitts::server
