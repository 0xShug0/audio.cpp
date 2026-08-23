#include "model_installer.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <future>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path make_root() {
    const auto suffix = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const auto root = std::filesystem::temp_directory_path() /
        ("audiocpp-native-model-manager-test-" + std::to_string(suffix));
    std::filesystem::create_directories(root);
    return root;
}

void write(const std::filesystem::path & path, const std::string & value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
}

void set_base_url(const std::string & value) {
#ifdef _WIN32
    _putenv_s("AUDIOCPP_HF_BASE_URL", value.c_str());
#else
    setenv("AUDIOCPP_HF_BASE_URL", value.c_str(), 1);
#endif
}

std::string wait_for(minitts::server::ModelInstaller & installer, const std::string & id,
    const std::string & state, int attempts = 200) {
    for (int attempt = 0; attempt < attempts; ++attempt) {
        const auto status = installer.status(id);
        if (status.find("\"state\":\"" + state + "\"") != std::string::npos) return status;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("timed out waiting for " + id + " to become " + state);
}

void test_native_package_lifecycle() {
    const auto root = make_root();
    try {
        write(root / "model_specs" / "demo.json", R"JSON({
          "family":"demo","display_name":"Demo","description":"","category":"tts",
          "status":"supported","tasks":["tts"],"modes":["offline"],"languages":[],
          "capabilities":{},"runtime":{},
          "package_defaults":{"download":{"kind":"huggingface_snapshot","repo":"org/repo","revision":"main"}},
          "packages":[
            {"id":"demo_q8","display_name":"Demo Q8","default":true,"format":"gguf","precision":"q8_0",
             "target_directory":"Demo","files":["Demo/model-q8.gguf","Demo/shared.json"],"strip_prefix":"Demo"},
            {"id":"demo_f16","display_name":"Demo F16","format":"gguf","precision":"f16",
             "target_directory":"Demo","files":["Demo/model-f16.gguf","Demo/shared.json"],"strip_prefix":"Demo"},
            {"id":"demo_slow","display_name":"Demo Slow","format":"gguf","precision":"q8_0",
             "target_directory":"Slow","files":["Slow/model.gguf"],"strip_prefix":"Slow"}
          ],"sources":[]
        })JSON");
        set_base_url("http://127.0.0.1:18991");
        auto fixture = std::async(std::launch::async, [] {
#ifdef _WIN32
            return std::system("python \"" AUDIOCPP_NATIVE_MANAGER_FIXTURE "\" --requests 14");
#else
            return std::system("python3 \"" AUDIOCPP_NATIVE_MANAGER_FIXTURE "\" --requests 14");
#endif
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        minitts::server::ModelInstaller installer(root, root / "models");
        require(installer.status("missing").find("\"state\":\"idle\"") != std::string::npos,
            "unknown jobs are idle");
        bool rejected = false;
        try { (void) installer.start("bad & package", "", "", "", "", false); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "unsafe package ids are rejected");

        auto started = installer.start("demo_q8", "", "", "", "", false);
        require(started.find("\"state\":\"queued\"") != std::string::npos ||
                started.find("\"state\":\"running\"") != std::string::npos,
            "native install starts asynchronously");
        auto complete = wait_for(installer, "demo_q8", "complete");
        require(complete.find("\"progress_percent\":100") != std::string::npos,
            "completed install reports 100 percent");
        require(std::filesystem::is_regular_file(root / "models" / "Demo" / "model-q8.gguf"),
            "model payload is installed");
        require(std::filesystem::is_regular_file(root / "models" / "Demo" / "shared.json"),
            "shared sidecar is installed");
        require(std::filesystem::is_regular_file(root / "models" / "Demo" / ".audiocpp-package-demo_q8.json"),
            "native install writes a version manifest");

        (void) installer.start("demo_f16", "", "", "", "", false);
        (void) wait_for(installer, "demo_f16", "complete");
        require(std::filesystem::is_regular_file(root / "models" / "Demo" / "model-f16.gguf"),
            "sibling precision installs beside Q8");
        auto removed = installer.remove("demo_q8");
        require(removed.find("\"removed\":true") != std::string::npos, "Q8 removal succeeds");
        require(!std::filesystem::exists(root / "models" / "Demo" / "model-q8.gguf"),
            "Q8-owned file is removed");
        require(std::filesystem::exists(root / "models" / "Demo" / "shared.json"),
            "sidecar shared by installed F16 is retained");

        (void) installer.start("demo_slow", "", "", "", "", false);
        (void) wait_for(installer, "demo_slow", "running");
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        const auto stopping = installer.stop("demo_slow");
        require(stopping.find("\"state\":\"cancelling\"") != std::string::npos,
            "native download accepts cancellation");
        (void) wait_for(installer, "demo_slow", "cancelled");
        require(!std::filesystem::exists(root / "models" / "Slow" / "model.gguf"),
            "cancelled package is never published to its final path");

        const auto partial = root / "models" / ".Demo.abandoned";
        write(partial / "model.tmp", "partial");
        const auto cleaned = installer.clean_partial("demo_q8");
        require(cleaned.find("\"cleaned\":true") != std::string::npos, "partial cleanup succeeds");
        require(!std::filesystem::exists(partial), "partial staging directory is removed");

        const auto initial = installer.package_sizes();
        require(initial.find("\"state\":\"running\"") != std::string::npos,
            "remote inventory starts in the background");
        bool inventory_complete = false;
        for (int attempt = 0; attempt < 200; ++attempt) {
            const auto sizes = installer.package_sizes();
            if (sizes.find("\"state\":\"complete\"") != std::string::npos) {
                require(sizes.find("\"id\":\"demo_f16\"") != std::string::npos,
                    "inventory includes package ids");
                require(sizes.find("\"installed\":true") != std::string::npos,
                    "inventory reports installed packages");
                require(sizes.find("\"version_state\":\"up_to_date\"") != std::string::npos,
                    "per-file ETag version check reports up to date");
                inventory_complete = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        require(inventory_complete, "remote package inventory completes");

        bool legacy_missing = false;
        try { (void) installer.start("demo_q8", "checkpoint.bin", "", "", "", false); }
        catch (const std::runtime_error & error) {
            legacy_missing = std::string(error.what()).find("model_manager_deprecated.py") != std::string::npos;
        }
        require(legacy_missing, "explicit converter inputs retain the legacy converter boundary");
        require(fixture.get() == 0, "local HTTP fixture completed normally");
    } catch (...) {
        std::error_code error;
        std::filesystem::remove_all(root, error);
        throw;
    }
    std::error_code error;
    std::filesystem::remove_all(root, error);
}

}  // namespace

int main() {
    try { test_native_package_lifecycle(); }
    catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_model_installer_test passed\n";
    return 0;
}
