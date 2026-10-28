#include "engine/community_models/lfm3_audio/debug.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

const char * env_value(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

int64_t element_count(const std::vector<int64_t> & shape) {
    int64_t count = 1;
    for (const auto dim : shape) {
        if (dim < 0) throw std::runtime_error("LFM3-Audio npy shape has a negative dimension");
        count *= dim;
    }

    return count;
}

// NumPy format v1.0: magic, version, little-endian u16 header length, then a
// Python dict literal padded with spaces to a 64-byte boundary and ended by '\n'.
void write_npy(const std::string & name, const char * descr, const void * data, size_t element_bytes,
               const std::vector<int64_t> & shape) {
    const char * dir = env_value("LFM3_AUDIO_DUMP_DIR");
    if (dir == nullptr) {
        return;
    }

    std::string dims;
    for (const auto dim : shape) {
        dims += std::to_string(dim) + ",";
    }

    if (shape.size() > 1) dims.pop_back();  // only a 1-tuple keeps its trailing comma
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    header.append(63 - (10 + header.size()) % 64, ' ');
    header.push_back('\n');

    const auto path = std::filesystem::path(dir) / (name + ".npy");
    std::ofstream out(path, std::ios::binary);
    const uint16_t header_bytes = static_cast<uint16_t>(header.size());
    const char preamble[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0,
                             static_cast<char>(header_bytes & 0xff), static_cast<char>(header_bytes >> 8)};
    out.write(preamble, sizeof(preamble));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(element_count(shape) * element_bytes));
    if (!out) {
        throw std::runtime_error("LFM3-Audio failed to write dump " + path.string());
    }
}

std::vector<char> read_npy(const std::filesystem::path & path, const char * descr, std::vector<int64_t> * shape) {
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

}  // namespace

bool lfm3_dump_enabled() {
    return env_value("LFM3_AUDIO_DUMP_DIR") != nullptr;
}

void lfm3_dump(const std::string & name, const float * data, const std::vector<int64_t> & shape) {
    write_npy(name, "<f4", data, sizeof(float), shape);
}

void lfm3_dump(const std::string & name, const int32_t * data, const std::vector<int64_t> & shape) {
    write_npy(name, "<i4", data, sizeof(int32_t), shape);
}

int64_t lfm3_dump_frames() {
    const char * value = env_value("LFM3_AUDIO_DUMP_FRAMES");
    if (value == nullptr) {
        return 4;
    }

    char * end = nullptr;
    const long long frames = std::strtoll(value, &end, 10);
    if (*end != '\0' || frames < 0) {
        throw std::runtime_error(std::string("LFM3_AUDIO_DUMP_FRAMES must be a non-negative integer: ") + value);
    }

    return frames;
}

std::vector<Lfm3Codes> lfm3_injected_orders() {
    const char * path = env_value("LFM3_AUDIO_MASKGIT_ORDERS");
    if (path == nullptr) {
        return {};
    }

    std::vector<int64_t> shape;
    const auto values = lfm3_read_npy_i32(path, &shape);
    if (shape.size() != 2 || shape[1] != 12) {
        throw std::runtime_error(std::string("LFM3_AUDIO_MASKGIT_ORDERS must hold int32 [frames, 12]: ") + path);
    }

    std::vector<Lfm3Codes> orders(static_cast<size_t>(shape[0]));
    for (size_t f = 0; f < orders.size(); ++f) {
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(f * 12), 12, orders[f].begin());
        auto sorted = orders[f];
        std::sort(sorted.begin(), sorted.end());
        for (int32_t i = 0; i < 12; ++i) {
            if (sorted[static_cast<size_t>(i)] != i) {
                throw std::runtime_error("LFM3_AUDIO_MASKGIT_ORDERS row " + std::to_string(f) + " is not a permutation of 0..11");
            }
        }
    }

    return orders;
}

std::vector<int32_t> lfm3_read_npy_i32(const std::filesystem::path & path, std::vector<int64_t> * shape) {
    return read_npy_as<int32_t>(path, "<i4", shape);
}

std::vector<float> lfm3_read_npy_f32(const std::filesystem::path & path, std::vector<int64_t> * shape) {
    return read_npy_as<float>(path, "<f4", shape);
}

}  // namespace engine::community_models::lfm3_audio
