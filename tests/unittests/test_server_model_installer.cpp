#include "model_installer.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path make_root() {
    const auto suffix = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto root = std::filesystem::temp_directory_path() /
        ("audiocpp-model-installer-test-" + std::to_string(suffix));
    std::filesystem::create_directories(root);
    return root;
}

void test_idle_status_and_validation() {
    const auto root = make_root();
    try {
        minitts::server::ModelInstaller installer(root, root / "models");
        const auto idle = installer.status("qwen3_asr_0_6b");
        require(idle.find("\"state\":\"idle\"") != std::string::npos, "unknown jobs report idle");

        bool invalid_rejected = false;
        try {
            (void) installer.start("bad & package", "", "", "", "", false);
        } catch (const std::runtime_error &) {
            invalid_rejected = true;
        }
        require(invalid_rejected, "unsafe package ids are rejected");

        bool missing_helper_reported = false;
        try {
            (void) installer.start("qwen3_asr_0_6b", "", "", "", "", false);
        } catch (const std::runtime_error & error) {
            missing_helper_reported =
                std::string(error.what()).find("model_manager_v2.py") != std::string::npos;
        }
        require(missing_helper_reported, "a missing v2 preparation helper has a useful error");

        bool missing_legacy_helper_reported = false;
        try {
            (void) installer.start("qwen3_asr_0_6b", "checkpoint.bin", "", "", "", false);
        } catch (const std::runtime_error & error) {
            missing_legacy_helper_reported =
                std::string(error.what()).find("model_manager_deprecated.py") != std::string::npos;
        }
        require(missing_legacy_helper_reported, "converter inputs select the legacy preparation helper");
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        throw;
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
}

}  // namespace

int main() {
    try {
        test_idle_status_and_validation();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_model_installer_test passed\n";
    return 0;
}
