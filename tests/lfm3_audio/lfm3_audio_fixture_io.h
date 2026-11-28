#pragma once

// NumPy fixture reader, kept out of production inference.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm3_model_test::fixtures {
inline int64_t element_count(const std::vector<int64_t> & shape) {
    int64_t count = 1;
    for (const auto dim : shape) {
        if (dim < 0) throw std::runtime_error("LFM3-Audio npy shape has a negative dimension");
        count *= dim;
    }

    return count;
}

inline std::vector<char> read_npy(const std::filesystem::path & path, const char * descr, std::vector<int64_t> * shape) {
    std::ifstream in(path, std::ios::binary);
    char magic[8] = {};
    if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, "\x93NUMPY", 6) != 0 || (magic[6] != 1 && magic[6] != 2)) {
        throw std::runtime_error("LFM3-Audio cannot read npy file " + path.string());
    }

    unsigned char length[4] = {};
    in.read(reinterpret_cast<char *>(length), magic[6] == 1 ? 2 : 4);
    const size_t header_bytes = magic[6] == 1 ? (length[0] | length[1] << 8)
                                              : (length[0] | length[1] << 8 | length[2] << 16 | static_cast<size_t>(length[3]) << 24);
    std::string header(header_bytes, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_bytes));
    if (!in || header.find(std::string("'descr': '") + descr + "'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("LFM3-Audio npy file " + path.string() + " is not C-order " + descr + ": " + header);
    }

    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("LFM3-Audio npy file " + path.string() + " has no shape");
    }

    std::vector<int64_t> dims;
    for (size_t pos = open + 1; pos < close;) {
        const auto end = std::min(header.find(',', pos), close);
        const auto item = header.substr(pos, end - pos);
        if (item.find_first_not_of(' ') != std::string::npos) dims.push_back(std::stoll(item));
        pos = end + 1;
    }

    std::vector<char> data(static_cast<size_t>(element_count(dims)) * 4);
    if (!in.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("LFM3-Audio npy file " + path.string() + " is truncated");
    }

    if (shape != nullptr) *shape = std::move(dims);
    return data;
}

template <typename T>
std::vector<T> read_npy_as(const std::filesystem::path & path, const char * descr, std::vector<int64_t> * shape) {
    static_assert(sizeof(T) == 4);
    const auto bytes = read_npy(path, descr, shape);
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

inline std::vector<int32_t> read_i32(const std::filesystem::path & path, std::vector<int64_t> * shape) {
    return read_npy_as<int32_t>(path, "<i4", shape);
}
inline std::vector<float> read_f32(const std::filesystem::path & path, std::vector<int64_t> * shape) {
    return read_npy_as<float>(path, "<f4", shape);
}

} // namespace lfm3_model_test::fixtures
