#include "engine/framework/package_manager/manager.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Arguments {
    std::string command;
    std::string package_id;
    std::filesystem::path models_root = "models";
    std::filesystem::path repository_root = ".";
    bool overwrite = false;
    bool query_remote = false;
};

void print_usage(std::ostream & output) {
    output <<
        "audio.cpp native model package manager\n\n"
        "Usage:\n"
        "  audiocpp_model_manager list [--remote] [--models-dir PATH] [--repository-root PATH]\n"
        "  audiocpp_model_manager info PACKAGE [--remote] [--models-dir PATH] [--repository-root PATH]\n"
        "  audiocpp_model_manager install PACKAGE [--overwrite] [--models-dir PATH] [--repository-root PATH]\n"
        "  audiocpp_model_manager clean PACKAGE [--models-dir PATH] [--repository-root PATH]\n"
        "  audiocpp_model_manager remove PACKAGE [--models-dir PATH] [--repository-root PATH]\n\n"
        "The active model_specs catalog is compiled into this executable. An external\n"
        "model_specs directory below --repository-root overrides the embedded catalog.\n";
}

std::string require_value(int & index, int argc, char ** argv, std::string_view option) {
    if (++index >= argc) throw std::runtime_error(std::string(option) + " requires a value");
    return argv[index];
}

Arguments parse_arguments(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("missing command");
    Arguments result;
    result.command = argv[1];
    if (result.command == "--help" || result.command == "-h" || result.command == "help") return result;
    const bool requires_package = result.command == "info" || result.command == "install" ||
                                  result.command == "clean" || result.command == "remove";
    int index = 2;
    if (requires_package) {
        if (index >= argc || argv[index][0] == '-') throw std::runtime_error(result.command + " requires PACKAGE");
        result.package_id = argv[index++];
    } else if (result.command != "list") {
        throw std::runtime_error("unknown command: " + result.command);
    }
    for (; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--models-dir") result.models_root = require_value(index, argc, argv, option);
        else if (option == "--repository-root") result.repository_root = require_value(index, argc, argv, option);
        else if (option == "--overwrite") result.overwrite = true;
        else if (option == "--remote") result.query_remote = true;
        else throw std::runtime_error("unknown option: " + option);
    }
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.command == "--help" || arguments.command == "-h" || arguments.command == "help") {
            print_usage(std::cout);
            return 0;
        }

        engine::package_manager::PackageManager manager(
            arguments.repository_root, arguments.models_root);
        if (arguments.command == "list" || arguments.command == "info") {
            // Inventory JSON is the shared machine-readable contract used by the
            // server, scripts, and Docker workflows.
            std::cout << manager.inventory(
                arguments.query_remote,
                arguments.command == "info" ? arguments.package_id : std::string{}) << '\n';
            return 0;
        }
        if (arguments.command == "clean") {
            std::cout << manager.clean_partial(arguments.package_id) << '\n';
            return 0;
        }
        if (arguments.command == "remove") {
            std::cout << manager.remove(arguments.package_id) << '\n';
            return 0;
        }

        auto cancelled = std::make_shared<std::atomic_bool>(false);
        uint64_t last_reported = 0;
        const auto result = manager.install(
            arguments.package_id, arguments.overwrite, cancelled,
            [&last_reported](const engine::package_manager::PackageProgress & progress) {
                if (progress.downloaded_bytes == last_reported && progress.message.empty()) return;
                last_reported = progress.downloaded_bytes;
                std::cerr << "downloaded=" << progress.downloaded_bytes
                          << " total=" << progress.total_bytes;
                if (!progress.message.empty()) std::cerr << " " << progress.message;
                std::cerr << '\n';
            });
        std::cout << result << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "audiocpp_model_manager failed: " << error.what() << '\n';
        print_usage(std::cerr);
        return 1;
    }
}
