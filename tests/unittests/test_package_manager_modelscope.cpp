#include "engine/framework/package_manager/manager.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
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
        ("audiocpp-modelscope-package-manager-test-" + std::to_string(suffix));
    std::filesystem::create_directories(root);
    return root;
}

void write(const std::filesystem::path & path, const std::string & value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
}

std::string read(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void set_base_url(const std::string & value) {
#ifdef _WIN32
    _putenv_s("AUDIOCPP_MS_BASE_URL", value.c_str());
#else
    setenv("AUDIOCPP_MS_BASE_URL", value.c_str(), 1);
#endif
}

void test_modelscope_package_lifecycle() {
    const auto root = make_root();
    try {
        // No explicit revision: modelscope_snapshot must default to master.
        write(root / "model_specs" / "demo_ms.json", R"JSON({
          "family":"demo_ms","display_name":"Demo MS","description":"","category":"tts",
          "status":"supported","tasks":["tts"],"modes":["offline"],"languages":[],
          "capabilities":{},"runtime":{},
          "package_defaults":{"download":{"kind":"modelscope_snapshot","repo":"ms/repo"}},
          "packages":[
            {"id":"demo_ms_q8","display_name":"Demo MS Q8","default":true,"format":"gguf","precision":"q8_0",
             "target_directory":"DemoMS","files":["Demo/model-q8.gguf","Demo/shared.json"],"strip_prefix":"Demo"}
          ],"sources":[]
        })JSON");
        set_base_url("http://127.0.0.1:18992");
        auto fixture = std::async(std::launch::async, [] {
#ifdef _WIN32
            return std::system("python \"" AUDIOCPP_NATIVE_MANAGER_FIXTURE "\" --port 18992 --requests 4");
#else
            return std::system("python3 \"" AUDIOCPP_NATIVE_MANAGER_FIXTURE "\" --port 18992 --requests 4");
#endif
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        engine::package_manager::PackageManager manager(root, root / "models");
        uint64_t last_total = 0;
        const auto installed = manager.install("demo_ms_q8", false, nullptr,
            [&](const engine::package_manager::PackageProgress & progress) {
                last_total = progress.total_bytes;
            });
        require(installed.find("Installed demo_ms_q8") != std::string::npos,
            "modelscope package installs through the fixture");
        require(last_total == 16 + 16, "listing sizes drive the download total");
        require(std::filesystem::is_regular_file(root / "models" / "DemoMS" / "model-q8.gguf"),
            "model payload is installed");
        require(std::filesystem::is_regular_file(root / "models" / "DemoMS" / "shared.json"),
            "sidecar is installed");

        const auto manifest_path =
            root / "models" / "DemoMS" / ".audiocpp-package-demo_ms_q8.json";
        require(std::filesystem::is_regular_file(manifest_path),
            "native install writes a version manifest");
        const auto manifest = read(manifest_path);
        require(manifest.find("\"requested_revision\":\"master\"") != std::string::npos,
            "modelscope packages default to the master revision");
        require(manifest.find("\"resolved_revision\":\"master\"") != std::string::npos,
            "resolved revision falls back to the requested revision");
        require(manifest.find("\"etag\":\"1a53a6a47a59980589f5c699aa3da20ee9502d67224708a2bf5113852e1e1fa6\"")
                != std::string::npos,
            "manifest records the ModelScope sha256 as the etag");

        const auto again = manager.install("demo_ms_q8", false, nullptr, nullptr);
        require(again.find("Already installed demo_ms_q8") != std::string::npos,
            "re-install without overwrite is a no-op");

        const auto inventory = manager.inventory(true);
        require(inventory.find("\"id\":\"demo_ms_q8\"") != std::string::npos,
            "inventory includes the modelscope package");
        require(inventory.find("\"version_state\":\"up_to_date\"") != std::string::npos,
            "sha256 etag version check reports up to date");
        require(inventory.find("\"size_bytes\":32") != std::string::npos,
            "inventory reports the listing size total");

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
    try { test_modelscope_package_lifecycle(); }
    catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "package_manager_modelscope_test passed\n";
    return 0;
}
