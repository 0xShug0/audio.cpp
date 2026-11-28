#include "engine/community_models/zonos3/npy.h"

#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::zonos3 {
namespace {

struct NpyMessages {
    const char * open;
    const char * header;
    const char * metadata;
    const char * order;
};

struct NpyFile {
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<unsigned char> bytes;
    size_t data = 0;
};

[[noreturn]] void reject(const std::string & message) {
    throw runtime::InvalidRequestError("zonos3: " + message);
}

// The NPY v1/v2/v3 framing: magic, header dict (descr, fortran_order, shape); data follows.
NpyFile read_npy(const std::filesystem::path & path, const NpyMessages & messages) {
    std::ifstream in(path, std::ios::binary);
    if (!in) reject(std::string(messages.open) + path.string());
    NpyFile file;
    file.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    const auto & b = file.bytes;
    if (b.size() < 8 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0 || b[7] != 0 || (b[6] != 1 && b[6] != 2 && b[6] != 3)) {
        reject(messages.header);
    }
    const size_t length_bytes = b[6] == 1 ? 2 : 4;
    if (b.size() < 8 + length_bytes) reject("truncated NPY header");
    uint32_t header_size = 0;
    for (size_t i = 0; i < length_bytes; ++i) header_size |= static_cast<uint32_t>(b[8 + i]) << (8 * i);
    if (header_size > 65536) reject("oversized NPY header");
    const size_t begin = 8 + length_bytes;
    if (b.size() < begin + header_size) reject("truncated NPY header");
    const std::string header(reinterpret_cast<const char *>(b.data() + begin), header_size);
    file.data = begin + header_size;
    const size_t order = header.find("fortran_order");
    const size_t shape = header.find("shape");
    const size_t descr = header.find("descr");
    if (order == std::string::npos || shape == std::string::npos || descr == std::string::npos) reject(messages.metadata);
    size_t p = header.find(':', order);
    if (p == std::string::npos) reject("invalid NPY array order");
    p = header.find_first_not_of(" \t", p + 1);
    if (p == std::string::npos || header.compare(p, 5, "False") != 0) reject(messages.order);
    p = header.find(':', descr);
    if (p == std::string::npos) reject("invalid NPY dtype");
    p = header.find_first_of("'\"", p + 1);
    if (p == std::string::npos) reject("invalid NPY dtype");
    const size_t end = header.find(header[p], p + 1);
    if (end == std::string::npos) reject("invalid NPY dtype");
    file.dtype = header.substr(p + 1, end - p - 1);
    p = header.find('(', shape);
    const size_t close = p == std::string::npos ? p : header.find(')', p);
    if (close == std::string::npos) reject("invalid NPY shape");
    for (size_t i = p + 1; i < close;) {
        if (header[i] < '0' || header[i] > '9') {
            ++i;
            continue;
        }
        int64_t value = 0;
        while (i < close && header[i] >= '0' && header[i] <= '9') {
            if (value > std::numeric_limits<int64_t>::max() / 16) reject("invalid NPY shape");
            value = value * 10 + (header[i++] - '0');
        }
        file.shape.push_back(value);
    }
    return file;
}

// One little-endian element of `dtype` (<f4, <f8, <i4, <i8, <u4, <u8) as double.
double element(const NpyFile & file, size_t index) {
    const int bytes = file.dtype[2] - '0';
    const unsigned char * raw = file.bytes.data() + file.data + index * static_cast<size_t>(bytes);
    uint64_t bits = 0;
    for (int i = 0; i < bytes; ++i) bits |= static_cast<uint64_t>(raw[i]) << (8 * i);
    if (file.dtype[1] == 'f') {
        if (bytes == 4) {
            const auto u = static_cast<uint32_t>(bits);
            float v = 0.0f;
            std::memcpy(&v, &u, 4);
            return v;
        }
        double v = 0.0;
        std::memcpy(&v, &bits, 8);
        return v;
    }
    if (file.dtype[1] == 'i') {
        return bytes == 4 ? static_cast<double>(static_cast<int32_t>(bits)) : static_cast<double>(static_cast<int64_t>(bits));
    }
    return static_cast<double>(bits);
}

void require_payload(const NpyFile & file, size_t count, const char * truncated, const char * trailing) {
    const size_t need = count * static_cast<size_t>(file.dtype[2] - '0');
    const size_t have = file.bytes.size() - file.data;
    if (have < need) reject(truncated);
    if (have > need) reject(trailing);
}

std::vector<std::byte> npy_bytes(const std::string & dict, const void * data, size_t bytes) {
    std::string header = dict;
    header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
    header += '\n';
    std::vector<std::byte> out(10 + header.size() + bytes);
    const unsigned char prefix[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
                                      static_cast<unsigned char>(header.size()),
                                      static_cast<unsigned char>(header.size() >> 8)};
    std::memcpy(out.data(), prefix, 10);
    std::memcpy(out.data() + 10, header.data(), header.size());
    // Little-endian hosts only, as ZONOS3's writers.
    std::memcpy(out.data() + 10 + header.size(), data, bytes);
    return out;
}

}  // namespace

std::vector<float> read_speaker_embedding_npy(const std::filesystem::path & path) {
    const auto file = read_npy(path, {"cannot open speaker embedding: ", "invalid NPY speaker embedding header",
                                      "invalid NPY speaker embedding metadata", "speaker embedding NPY must be C-order"});
    if (file.dtype != "<f4" && file.dtype != "<f8") reject("speaker embedding NPY needs little-endian f4 or f8 dtype");
    if (!(file.shape == std::vector<int64_t>{192} || file.shape == std::vector<int64_t>{1, 192})) {
        reject("speaker embedding must have shape [192] or [1,192]");
    }
    require_payload(file, 192, "truncated speaker embedding", "unexpected trailing data in speaker embedding NPY");
    std::vector<float> out(192);
    for (size_t i = 0; i < out.size(); ++i) {
        const double value = element(file, i);
        if (!std::isfinite(value) || std::fabs(value) > static_cast<double>(std::numeric_limits<float>::max())) {
            reject("speaker embedding contains a non-finite value");
        }
        out[i] = static_cast<float>(value);
    }
    return out;
}

std::vector<int32_t> read_reference_codes_npy(const std::filesystem::path & path, int max_frames) {
    const auto file = read_npy(path, {"cannot open reference codes: ", "invalid NPY reference header",
                                      "invalid NPY reference metadata", "reference NPY must be C-order"});
    const auto & d = file.dtype;
    if (d != "<i4" && d != "<i8" && d != "<u4" && d != "<u8" && d != "<f4" && d != "<f8") {
        reject("reference NPY needs little-endian i4/i8/u4/u8/f4/f8 dtype");
    }
    if (file.shape.size() < 2 || file.shape[0] < 1 || file.shape[1] != 10) {
        reject("reference codes must have shape [T,10], T >= 1");
    }
    if (file.shape.size() != 2) reject("reference codes must be two-dimensional");
    const size_t count = static_cast<size_t>(file.shape[0]) * 10;
    require_payload(file, count, "truncated reference codes", "unexpected trailing data in reference NPY");
    const size_t kept = static_cast<size_t>(std::min<int64_t>(file.shape[0], std::max(0, max_frames))) * 10;
    std::vector<int32_t> out(kept);
    for (size_t i = 0; i < count; ++i) {
        const double value = element(file, i);
        if (!std::isfinite(value) || value < 0 || value > 4095 || value != std::floor(value)) {
            reject("reference codes must be ordinary integer codec IDs 0..4095");
        }
        if (i < kept) out[i] = static_cast<int32_t>(value);
    }
    return out;
}

std::vector<std::byte> speaker_embedding_npy(const std::vector<float> & embedding) {
    if (embedding.size() != 192) throw std::runtime_error("zonos3: speaker embedding must have 192 values");
    return npy_bytes("{'descr': '<f4', 'fortran_order': False, 'shape': (192,), }", embedding.data(),
                     embedding.size() * sizeof(float));
}

std::vector<std::byte> reference_codes_npy(const std::vector<int32_t> & codes) {
    if (codes.empty() || codes.size() % 10 != 0) throw std::runtime_error("zonos3: reference codes must be [T,10]");
    return npy_bytes("{'descr': '<i4', 'fortran_order': False, 'shape': (" + std::to_string(codes.size() / 10) +
                         ", 10), }",
                     codes.data(), codes.size() * sizeof(int32_t));
}

Zonos3NpyF32 read_f32_npy(const std::filesystem::path & path) {
    const auto file = read_npy(path, {"cannot open ", "invalid NPY header", "invalid NPY metadata", "NPY must be C-order"});
    const auto & d = file.dtype;
    if (d != "<i4" && d != "<i8" && d != "<u4" && d != "<u8" && d != "<f4" && d != "<f8") {
        reject("unsupported NPY dtype " + d + ": " + path.string());
    }
    size_t count = 1;
    for (const auto dim : file.shape) count *= static_cast<size_t>(dim);
    require_payload(file, count, "truncated NPY data", "unexpected trailing NPY data");
    Zonos3NpyF32 out;
    out.shape = file.shape;
    out.data.resize(count);
    for (size_t i = 0; i < count; ++i) out.data[i] = static_cast<float>(element(file, i));
    return out;
}

}  // namespace engine::models::zonos3
