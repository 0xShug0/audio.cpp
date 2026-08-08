#include "model_installer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

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
        require(idle.find("\"progress_percent\":-1") != std::string::npos, "idle jobs have no progress");

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

        std::filesystem::create_directories(root / "tools");
        {
            std::ofstream script(root / "tools" / "model_manager_v2.py", std::ios::binary);
            script
                << "import sys, time\n"
                << "if len(sys.argv) > 1 and sys.argv[1] == 'sizes':\n"
                << " time.sleep(0.15)\n"
                << " print('[{\\\"id\\\":\\\"demo_q8_0\\\",\\\"size_bytes\\\":25,\\\"state\\\":\\\"ok\\\",\\\"message\\\":\\\"\\\",\\\"installed\\\":true}]', flush=True)\n"
                << " raise SystemExit(0)\n"
                << "if len(sys.argv) > 1 and sys.argv[1] == 'installed':\n"
                << " print('[{\\\"id\\\":\\\"demo_q8_0\\\",\\\"size_bytes\\\":null,\\\"state\\\":\\\"pending\\\",\\\"message\\\":\\\"\\\",\\\"installed\\\":true}]', flush=True)\n"
                << " raise SystemExit(0)\n"
                << "if len(sys.argv) > 1 and sys.argv[1] == 'uninstall':\n"
                << " print('removed test package', flush=True)\n"
                << " raise SystemExit(0)\n"
                << "if len(sys.argv) > 1 and sys.argv[1] == 'clean-partial':\n"
                << " print('cleaned partial test package', flush=True)\n"
                << " raise SystemExit(0)\n"
                << "print('preparing package', flush=True)\n"
                << "print('AUDIOCPP_PROGRESS downloaded=25 total=100', flush=True)\n"
                << "time.sleep(0.35)\n"
                << "print('AUDIOCPP_PROGRESS downloaded=100 total=100', flush=True)\n"
                << "print('installed test package', flush=True)\n";
        }
        const auto started = installer.start("qwen3_asr_0_6b", "", "", "", "", false);
        require(started.find("\"state\":\"queued\"") != std::string::npos ||
                started.find("\"state\":\"running\"") != std::string::npos,
                "a valid install starts in the background");
        bool observed_progress = false;
        bool completed = false;
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto current = installer.status("qwen3_asr_0_6b");
            observed_progress = observed_progress ||
                current.find("\"downloaded_bytes\":25") != std::string::npos;
            if (current.find("\"state\":\"complete\"") != std::string::npos) {
                require(current.find("\"downloaded_bytes\":100") != std::string::npos,
                        "completed jobs preserve downloaded bytes");
                require(current.find("\"total_bytes\":100") != std::string::npos,
                        "completed jobs preserve total bytes");
                require(current.find("\"progress_percent\":100") != std::string::npos,
                        "completed jobs report 100 percent");
                completed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(observed_progress, "running progress markers are exposed through status JSON");
        require(completed, "the background test installation completes");

        (void) installer.start("cancel_demo", "", "", "", "", false);
        const auto stopping = installer.stop("cancel_demo");
        require(stopping.find("\"state\":\"cancelling\"") != std::string::npos ||
                stopping.find("\"state\":\"cancelled\"") != std::string::npos,
                "an active package download can be stopped");
        bool cancelled = false;
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto current = installer.status("cancel_demo");
            if (current.find("\"state\":\"cancelled\"") != std::string::npos) {
                cancelled = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(cancelled, "a stopped package reaches the cancelled state");
        const auto cleaned = installer.clean_partial("cancel_demo");
        require(cleaned.find("\"cleaned\":true") != std::string::npos,
                "cancelled package staging files can be cleaned explicitly");
        require(cleaned.find("cleaned partial test package") != std::string::npos,
                "partial cleanup reports the manager result");

        const auto initial_sizes = installer.package_sizes();
        require(initial_sizes.find("\"state\":\"running\"") != std::string::npos,
                "package size scan starts in the background");
        const auto removed = installer.remove("qwen3_asr_0_6b");
        require(removed.find("\"removed\":true") != std::string::npos,
                "a package can be removed while an older metadata scan is running");
        require(removed.find("removed test package") != std::string::npos,
                "package removal reports the manager result");

        bool sizes_completed = false;
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto sizes = installer.package_sizes();
            if (sizes.find("\"state\":\"complete\"") != std::string::npos) {
                require(sizes.find("\"id\":\"demo_q8_0\"") != std::string::npos,
                        "package size metadata includes its package id");
                require(sizes.find("\"size_bytes\":25") != std::string::npos,
                        "package size metadata includes the summed byte count");
                require(sizes.find("\"installed\":true") != std::string::npos,
                        "package metadata includes its installed state");
                sizes_completed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(sizes_completed, "the background package size scan completes");

        const auto other_root = make_root();
        try {
            std::filesystem::create_directories(other_root / "tools");
            {
                std::ofstream script(other_root / "tools" / "model_manager_v2.py", std::ios::binary);
                script
                    << "import sys\n"
                    << "row = '[{\\\"id\\\":\\\"other_q8_0\\\",\\\"size_bytes\\\":50,"
                       "\\\"state\\\":\\\"ok\\\",\\\"message\\\":\\\"\\\","
                       "\\\"installed\\\":false}]'\n"
                    << "print(row, flush=True)\n";
            }
            {
                minitts::server::ModelInstaller other(other_root, other_root / "models");
                (void) other.package_sizes();
                bool other_completed = false;
                for (int attempt = 0; attempt < 100; ++attempt) {
                    const auto sizes = other.package_sizes();
                    if (sizes.find("\"state\":\"complete\"") != std::string::npos) {
                        require(sizes.find("\"id\":\"other_q8_0\"") != std::string::npos,
                                "a second installer reads its own metadata output");
                        other_completed = true;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                require(other_completed, "the second package size scan completes");
            }
            const auto original_sizes = installer.package_sizes();
            require(original_sizes.find("\"id\":\"demo_q8_0\"") != std::string::npos,
                    "installer instances keep isolated metadata output files");
            require(original_sizes.find("\"id\":\"other_q8_0\"") == std::string::npos,
                    "another installer cannot overwrite a live inventory snapshot");
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove_all(other_root, ec);
            throw;
        }
        {
            std::error_code ec;
            std::filesystem::remove_all(other_root, ec);
        }

        const auto restarted = installer.start("qwen3_asr_0_6b", "", "", "", "", false);
        require(restarted.find("\"state\":\"queued\"") != std::string::npos ||
                restarted.find("\"state\":\"running\"") != std::string::npos,
                "a completed package can be prepared again");
        bool repeated_completed = false;
        for (int attempt = 0; attempt < 100; ++attempt) {
            const auto current = installer.status("qwen3_asr_0_6b");
            if (current.find("\"state\":\"complete\"") != std::string::npos) {
                repeated_completed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        require(repeated_completed, "the repeated package preparation completes");
        const auto refreshed_sizes = installer.package_sizes();
        require(refreshed_sizes.find("\"state\":\"running\"") != std::string::npos,
                "a successful install invalidates the cached package inventory");

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
