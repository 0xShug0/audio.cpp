#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace engine::assets {

struct ModelPackage {
    std::filesystem::path root;
    // Installed paths relative to root, after applying the download's strip_prefix.
    std::vector<std::filesystem::path> files;
    std::string format;
};

}  // namespace engine::assets
