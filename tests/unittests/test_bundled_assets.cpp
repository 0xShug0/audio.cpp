// resolve_bundled_asset(): a relative asset path is used as given when it exists from the
// working directory, and is otherwise found beside the module that links audio.cpp (here,
// this executable) or up to three directories above it.
//
// Usage: bundled_assets_test <directory of this executable>

#include "engine/framework/assets/bundled_assets.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

void touch(const fs::path & path) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << "probe";
    engine::test::require(fs::is_regular_file(path), "could not create " + path.string());
}

bool same_file(const fs::path & a, const fs::path & b) {
    std::error_code ec;
    return fs::equivalent(a, b, ec) && !ec;
}

// Removes the probe trees and restores the working directory, also when a check throws.
struct Cleanup {
    fs::path original_cwd;
    std::vector<fs::path> trees;
    ~Cleanup() {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        for (const auto & tree : trees) {
            fs::remove_all(tree, ec);
        }
    }
};

}  // namespace

int main(int argc, char ** argv) try {
    using engine::assets::resolve_bundled_asset;
    using engine::test::require;
    require(argc == 2, "usage: bundled_assets_test <directory of this executable>");

    const fs::path module_dir = fs::absolute(argv[1]);
    const fs::path parent_dir = module_dir.parent_path();
    require(fs::is_directory(module_dir), "not a directory: " + module_dir.string());

    std::random_device random;
    const std::string unique = "audiocpp-bundled-asset-test-" + std::to_string(random());
    const fs::path work = fs::temp_directory_path() / unique;
    Cleanup cleanup{fs::current_path(), {work, module_dir / unique, parent_dir / unique}};
    fs::create_directories(work / "cwd");
    fs::current_path(work / "cwd");

    // Unchanged: empty, rooted, and missing everywhere.
    require(resolve_bundled_asset({}).empty(), "empty path");
    const fs::path rooted = work / "missing.txt";
    require(resolve_bundled_asset(rooted) == rooted, "a rooted path must be returned unchanged");
    const fs::path nowhere = fs::path(unique) / "nowhere.txt";
    require(resolve_bundled_asset(nowhere) == nowhere, "a path found nowhere must be returned unchanged");

    // Beside the module, and one directory above it.
    const fs::path beside = fs::path(unique) / "beside.txt";
    touch(module_dir / beside);
    require(same_file(resolve_bundled_asset(beside), module_dir / beside), "asset beside the module");
    const fs::path above = fs::path(unique) / "above.txt";
    touch(parent_dir / above);
    require(same_file(resolve_bundled_asset(above), parent_dir / above), "asset in the module's parent");

    // The working directory comes first, and the path then stays as given.
    touch(work / "cwd" / beside);
    require(resolve_bundled_asset(beside) == beside, "an asset under the working directory must win");

    std::cout << "bundled asset resolution tests passed\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
