// load_safetensors_index() accepts a non-object `__metadata__` value (MLX writes
// `null`) by handing it to skip_json_value(), which recurses once per nested
// array/object with no depth limit. The header length is read straight from the
// file and is not bounded either, so a header such as
//
//   {"__metadata__":[[[[ ... ]]]]}
//
// recurses until the native stack is exhausted and the process dies with
// SIGSEGV instead of the loader reporting a malformed file. The reference
// safetensors loader parses the header with serde_json, which stops at a fixed
// recursion limit and returns an error.
//
// A malformed header must be rejected with an exception, like every other
// header defect load_safetensors_index() already reports.

#include "engine/framework/io/safetensors.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Minimal safetensors container: u64 little-endian header length, then the
// JSON header. No tensor data is needed to reach the metadata parser.
std::filesystem::path write_safetensors(const std::string & header_json) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("safetensors_metadata_depth_" + std::to_string(counter++) + ".safetensors");
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open temp safetensors file");
    const uint64_t header_len = header_json.size();
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
    out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
    require(out.good(), "failed to write temp safetensors file");
    return path;
}

void test_deeply_nested_metadata_is_rejected() {
    // 1M levels is a 2 MB header: far below any realistic file size, and deep
    // enough to exhaust an 8 MB main-thread stack with one frame per level.
    constexpr size_t kDepth = 1000000;
    const std::string header =
        "{\"__metadata__\":" + std::string(kDepth, '[') + std::string(kDepth, ']') + "}";
    const auto path = write_safetensors(header);
    bool threw = false;
    try {
        (void) engine::io::load_safetensors_index(path);
    } catch (const std::exception &) {
        threw = true;
    }
    std::filesystem::remove(path);
    require(threw, "deeply nested __metadata__ must be rejected as a malformed header");
}

void test_null_metadata_still_parses() {
    // The non-object path exists for MLX, which writes "__metadata__": null.
    const auto path = write_safetensors(
        R"({"__metadata__":null,"t":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})");
    const auto index = engine::io::load_safetensors_index(path);
    std::filesystem::remove(path);
    require(index.tensors.size() == 1, "null __metadata__ must still be accepted");
}

void test_shallow_nested_metadata_still_parses() {
    // Nesting well inside the limit is skipped as before.
    constexpr size_t kDepth = 64;
    const auto path = write_safetensors(
        "{\"__metadata__\":" + std::string(kDepth, '[') + std::string(kDepth, ']') +
        R"(,"t":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})");
    const auto index = engine::io::load_safetensors_index(path);
    std::filesystem::remove(path);
    require(index.tensors.size() == 1, "shallow nested __metadata__ must still be accepted");
}

}  // namespace

int main() {
    try {
        test_null_metadata_still_parses();
        test_shallow_nested_metadata_still_parses();
        test_deeply_nested_metadata_is_rejected();
    } catch (const std::exception & error) {
        std::cerr << "safetensors metadata depth test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "safetensors metadata depth test passed\n";
    return 0;
}
