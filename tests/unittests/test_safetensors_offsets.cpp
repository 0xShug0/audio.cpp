// A safetensors header carries data_offsets straight from the file. Casting a
// negative value to size_t yields ~SIZE_MAX, and data_end < data_begin makes the
// (data_end - data_begin) that every consumer computes underflow to the same.
// Either then wraps the caller's `offset + size > blob.size()` bounds check back
// under the limit, producing an enormous span over out-of-bounds memory from a
// malicious model file.
//
// These offsets are now rejected where they are parsed, so no consumer can
// inherit the underflow.

#include "engine/framework/io/safetensors.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Minimal safetensors container: u64 little-endian header length, then the
// JSON header, then the tensor data region.
std::filesystem::path write_safetensors(const std::string & header_json, size_t data_bytes) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("safetensors_offsets_" + std::to_string(counter++) + ".safetensors");
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open temp safetensors file");

    const uint64_t header_len = header_json.size();
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
    out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
    const std::vector<char> data(data_bytes, 0);
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    require(out.good(), "failed to write temp safetensors file");
    return path;
}

bool rejects(const std::string & header_json) {
    const auto path = write_safetensors(header_json, 64);
    bool threw = false;
    try {
        (void) engine::io::load_safetensors_index(path);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    std::filesystem::remove(path);
    return threw;
}

void test_inverted_offsets_are_rejected() {
    // data_end < data_begin: the subtraction underflows to ~SIZE_MAX.
    require(rejects(R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[32,0]}})"),
            "inverted data_offsets must be rejected");
}

void test_negative_offsets_are_rejected() {
    require(rejects(R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[-1,16]}})"),
            "negative data_begin must be rejected");
    require(rejects(R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[0,-1]}})"),
            "negative data_end must be rejected");
}

void test_valid_offsets_still_parse() {
    // The guard must not reject legitimate headers, including empty tensors
    // where data_begin == data_end.
    const auto path = write_safetensors(
        R"({"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},)"
        R"("b":{"dtype":"F32","shape":[0],"data_offsets":[16,16]}})", 64);
    const auto index = engine::io::load_safetensors_index(path);
    std::filesystem::remove(path);

    require(index.tensors.size() == 2, "both tensors should parse");
    const auto & a = index.tensors.at("a");
    require(a.data_begin == 0 && a.data_end == 16, "tensor a offsets should round-trip");
    const auto & b = index.tensors.at("b");
    require(b.data_begin == b.data_end, "an empty tensor should remain valid");
}

}  // namespace

int main() {
    try {
        test_inverted_offsets_are_rejected();
        test_negative_offsets_are_rejected();
        test_valid_offsets_still_parse();
    } catch (const std::exception & error) {
        std::cerr << "safetensors offsets test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "safetensors offsets test passed\n";
    return 0;
}
