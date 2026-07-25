// Minimal writer for the NumPy .npy format (version 1.0, float32, row-major/
// C-contiguous only) — just enough for the parity dump tools to hand off
// tensors to compare_parity.py via np.load(), without pulling in a real npy
// library dependency for this one C++ test binary.
#pragma once

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::parakeet_tdt::parity {

inline void write_npy_f32(
    const std::string & path,
    const std::vector<float> & data,
    const std::vector<int64_t> & shape) {
    int64_t expected = 1;
    for (int64_t dim : shape) {
        expected *= dim;
    }
    if (expected != static_cast<int64_t>(data.size())) {
        throw std::runtime_error("write_npy_f32: shape does not match data size for " + path);
    }

    std::ostringstream shape_str;
    shape_str << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) {
            shape_str << ",";
        }
        if (i + 1 < shape.size()) {
            shape_str << " ";
        }
    }
    shape_str << ")";

    std::string header =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str.str() + ", }";
    // Total header (magic + version + header-len field + header + padding)
    // must be a multiple of 64 bytes, per the .npy spec, ending in '\n'.
    const size_t prefix_len = 6 /* magic */ + 2 /* version */ + 2 /* header len field */;
    size_t total_len = prefix_len + header.size() + 1;
    size_t padded_len = ((total_len + 63) / 64) * 64;
    header.append(padded_len - total_len, ' ');
    header.push_back('\n');

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("write_npy_f32: failed to open " + path);
    }
    out.write("\x93NUMPY", 6);
    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char *>(version), 2);
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("write_npy_f32: failed to write " + path);
    }
}

// Minimal reader counterpart: float32, version 1.0 or 2.0, C-contiguous only
// (exactly what write_npy_f32 above produces, and what numpy's default
// np.save emits for a float32 ndarray). Returns the flat row-major data and
// the shape parsed out of the header's Python-literal-ish "shape: (...)"
// field.
inline std::vector<float> read_npy_f32(const std::string & path, std::vector<int64_t> * shape_out = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("read_npy_f32: failed to open " + path);
    }
    char magic[6];
    in.read(magic, 6);
    if (!in || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("read_npy_f32: not an .npy file: " + path);
    }
    unsigned char version[2];
    in.read(reinterpret_cast<char *>(version), 2);
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char *>(&len16), sizeof(len16));
        header_len = len16;
    } else {
        in.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
    }
    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (header.find("<f4") == std::string::npos) {
        throw std::runtime_error("read_npy_f32: only little-endian float32 .npy files are supported: " + path);
    }

    std::vector<int64_t> shape;
    const auto shape_key = header.find("'shape':");
    if (shape_key == std::string::npos) {
        throw std::runtime_error("read_npy_f32: missing shape field in " + path);
    }
    const auto open_paren = header.find('(', shape_key);
    const auto close_paren = header.find(')', open_paren);
    std::string shape_body = header.substr(open_paren + 1, close_paren - open_paren - 1);
    std::string token;
    std::istringstream shape_stream(shape_body);
    while (std::getline(shape_stream, token, ',')) {
        // strip whitespace
        size_t start = token.find_first_not_of(" \t");
        if (start == std::string::npos) {
            continue;
        }
        size_t end = token.find_last_not_of(" \t");
        shape.push_back(std::stoll(token.substr(start, end - start + 1)));
    }

    int64_t total = 1;
    for (int64_t dim : shape) {
        total *= dim;
    }
    std::vector<float> data(static_cast<size_t>(total));
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("read_npy_f32: failed to read data from " + path);
    }
    if (shape_out != nullptr) {
        *shape_out = shape;
    }
    return data;
}

}  // namespace engine::community_models::parakeet_tdt::parity
