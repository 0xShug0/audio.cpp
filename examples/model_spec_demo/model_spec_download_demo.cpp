#include "engine/framework/model_spec/package.h"

#include <exception>
#include <iostream>
#include <string>

namespace {

namespace json = engine::io::json;

std::string optional_string(const json::Value & object, const std::string & key) {
    const auto * value = object.find(key);
    return value != nullptr && value->is_string() ? value->as_string() : std::string();
}

std::string merged_string(const json::Value * package_download, const json::Value * default_download, const std::string & key) {
    if (package_download != nullptr) {
        const auto value = optional_string(*package_download, key);
        if (!value.empty()) {
            return value;
        }
    }
    if (default_download != nullptr) {
        return optional_string(*default_download, key);
    }
    return {};
}

void print_string_array(const json::Value * value, const char * label) {
    if (value == nullptr) {
        return;
    }
    std::cout << "  " << label << "=";
    bool first = true;
    for (const auto & item : value->as_array()) {
        if (!first) {
            std::cout << ",";
        }
        std::cout << item.as_string();
        first = false;
    }
    std::cout << "\n";
}

void print_download_plan(const json::Value & package, const json::Value * default_download) {
    const auto * package_download = package.find("download");
    const auto kind = merged_string(package_download, default_download, "kind");
    std::cout << "package=" << package.require("id").as_string() << "\n";
    std::cout << "  target=" << package.require("target_directory").as_string() << "\n";
    std::cout << "  format=" << package.require("format").as_string() << "\n";
    std::cout << "  precision=" << package.require("precision").as_string() << "\n";
    std::cout << "  download_kind=" << kind << "\n";
    print_string_array(package.find("files"), "files");
    const auto strip_prefix = optional_string(package, "strip_prefix");
    if (!strip_prefix.empty()) {
        std::cout << "  strip_prefix=" << strip_prefix << "\n";
    }
    if (kind == "huggingface_snapshot") {
        std::cout << "  repo=" << merged_string(package_download, default_download, "repo") << "\n";
        const auto revision = merged_string(package_download, default_download, "revision");
        if (!revision.empty()) {
            std::cout << "  revision=" << revision << "\n";
        }
        print_string_array(package_download != nullptr ? package_download->find("exclude") : nullptr, "exclude");
    } else if (kind == "local_snapshot") {
        std::cout << "  path=" << merged_string(package_download, default_download, "path") << "\n";
    } else if (kind == "converter") {
        std::cout << "  converter=" << merged_string(package_download, default_download, "converter") << "\n";
        std::cout << "  description=" << merged_string(package_download, default_download, "description") << "\n";
    } else if (kind == "unsupported") {
        std::cout << "  reason=" << merged_string(package_download, default_download, "reason") << "\n";
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: model_spec_download_demo <spec.json> [package-id]\n";
        return 2;
    }
    try {
        const auto spec = engine::model_spec::load_spec(argv[1]);
        const std::string selected = argc == 3 ? argv[2] : std::string();
        bool found = false;
        std::cout << "family=" << spec.require("family").as_string() << "\n";
        const json::Value * default_download = nullptr;
        if (const auto * defaults = spec.find("package_defaults")) {
            default_download = defaults->find("download");
        }
        for (const auto & package : spec.require("packages").as_array()) {
            if (!selected.empty() && package.require("id").as_string() != selected) {
                continue;
            }
            print_download_plan(package, default_download);
            found = true;
        }
        if (!selected.empty() && !found) {
            std::cerr << "package not found: " << selected << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "model_spec_download_demo failed: " << error.what() << "\n";
        return 1;
    }
}
