#include "engine/framework/package_manager/manager.h"
#include "engine/framework/io/json.h"
#include "httplib.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

int main() {
    namespace fs = std::filesystem;
    namespace json = engine::io::json;
    const auto root = fs::temp_directory_path() / ("audiocpp-http-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    httplib::Server server;
    std::thread worker;
    std::mutex mutex;
    std::set<int> connections;
    int cdn_heads = 0;
    int downloads = 0;
    auto require = [](bool condition, const char *message) {
        if (!condition) throw std::runtime_error(message);
    };
    int result = 0;
    try {
        fs::create_directories(root / "model_specs");
        std::ofstream(root / "model_specs" / "demo.json") << R"({
          "family":"demo","display_name":"Demo","category":"tts",
          "status":"supported","tasks":["tts"],"modes":["offline"],
          "capabilities":{},"runtime":{},
          "package_defaults":{"download":{"kind":"huggingface_snapshot","repo":"org/repo"}},
          "packages":[{"id":"demo","display_name":"Demo","format":"gguf",
            "precision":"f32","target_directory":"Demo",
            "files":["a.gguf","b.gguf","fallback.gguf"]}]
        })";
        server.Get(R"(/org/repo/resolve/main/(.*))", [&](const auto &request, auto &response) {
            std::lock_guard<std::mutex> lock(mutex);
            connections.insert(request.remote_port);
            const auto file = request.matches[1].str();
            response.status = 302;
            response.set_header("Location", "/cdn/" + file);
            if (file != "fallback.gguf") {
                response.set_header("X-Linked-Size", "7");
                response.set_header("X-Linked-ETag", "\"same-etag\"");
                response.set_header("X-Repo-Commit", "revision");
                if (file == "a.gguf") {
                    response.set_header("X-Linked-ETag", "\"different-lfs-sha\"");
                    response.set_header("X-Xet-Hash", "same-etag");
                }
            }
        });
        server.Get(R"(/cdn/(.*))", [&](const auto &request, auto &response) {
            std::lock_guard<std::mutex> lock(mutex);
            if (request.method == "HEAD") ++cdn_heads;
            else ++downloads;
            response.set_header("ETag", "\"same-etag\"");
            response.set_header("X-Repo-Commit", "revision");
            response.set_content("payload", "application/octet-stream");
        });
        const auto port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "fixture bind failed");
        const auto url = "http://127.0.0.1:" + std::to_string(port);
#ifdef _WIN32
        _putenv_s("AUDIOCPP_HF_BASE_URL", url.c_str());
#else
        setenv("AUDIOCPP_HF_BASE_URL", url.c_str(), 1);
#endif
        worker = std::thread([&] { server.listen_after_bind(); });
        while (!server.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        engine::package_manager::PackageManager manager(root, root / "models");
        const auto inventory = json::parse(manager.inventory(true));
        const auto &row = inventory.as_array().at(0);
        require(json::require_string(row, "state") == "ok", "inventory failed");
        require(row.find("size_bytes")->as_f32() == 21, "redirect body size used instead of file size");
        require(json::require_string(row, "remote_revision") == "revision", "revision lost");
        {
            std::lock_guard<std::mutex> lock(mutex);
            require(connections.size() == 1, "inventory did not reuse its origin connection");
            require(cdn_heads == 1, "metadata redirects should skip CDN; missing metadata must fall back");
        }
        manager.install("demo", false, nullptr, nullptr);
        {
            std::lock_guard<std::mutex> lock(mutex);
            require(downloads == 3, "downloads must still follow redirects");
        }
        require(fs::file_size(root / "models" / "Demo" / "a.gguf") == 7, "bad installed payload");
        const auto manifest = json::parse_file(root / "models" / "Demo" / ".audiocpp-package-demo.json");
        require(manifest.require("files").require("a.gguf").require("etag").as_string() == "same-etag",
            "Xet metadata must preserve the original CDN ETag for existing manifests");
        const auto installed = json::parse(manager.inventory(true));
        require(json::require_string(installed.as_array().at(0), "version_state") == "up_to_date",
            "metadata/manifest comparison changed");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    server.stop();
    if (worker.joinable()) worker.join();
    std::error_code error;
    fs::remove_all(root, error);
    if (result == 0) std::cout << "package_manager_http_test passed\n";
    return result;
}
