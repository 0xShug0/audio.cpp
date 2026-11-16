// Reads the built data files from a directory, as the Python package does.
// g2p.npz is a ZIP archive of NumPy arrays stored without compression; the
// little that is needed of both formats is read here.
#include "dizione.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace dizione {

namespace {

bool read_file(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    out = buffer.str();
    return true;
}

uint32_t u16(const std::string & bytes, size_t at) {
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[at])) | (static_cast<uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 8);
}

uint32_t u32(const std::string & bytes, size_t at) {
    return u16(bytes, at) | (u16(bytes, at + 2) << 16);
}

// One array in NumPy's .npy format: little-endian float32, C order.
Tensor npy(const std::string & bytes, size_t start, size_t size, const std::string & name) {
    const auto bad = [&](const char * what) { return ValueError("g2p.npz: " + name + ": " + what); };
    if (size < 12 || bytes.compare(start, 6, "\x93NUMPY") != 0) {
        throw bad("not a NumPy array");
    }
    const int format = static_cast<unsigned char>(bytes[start + 6]);
    const size_t header_size = format == 1 ? u16(bytes, start + 8) : u32(bytes, start + 8);
    const size_t header_start = start + (format == 1 ? 10 : 12);
    if (header_start + header_size > start + size) {
        throw bad("truncated header");
    }
    const std::string header = bytes.substr(header_start, header_size);
    if (header.find("'<f4'") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) {
        throw bad("expected little-endian float32 in C order");
    }
    const size_t open = header.find('(', header.find("'shape'"));
    const size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw bad("no shape");
    }
    Tensor tensor;
    size_t count = 1;
    size_t value = 0;
    bool digits = false;
    for (size_t i = open + 1; i <= close; ++i) {
        const char c = header[i];
        if (c >= '0' && c <= '9') {
            value = value * 10 + static_cast<size_t>(c - '0');
            digits = true;
        } else if (digits) {
            tensor.shape.push_back(value);
            count *= value;
            value = 0;
            digits = false;
        }
    }
    const size_t data = header_start + header_size;
    if (data + count * 4 > start + size) {
        throw bad("truncated data");
    }
    tensor.values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t bits = u32(bytes, data + 4 * i);
        float number;
        std::memcpy(&number, &bits, sizeof(number));
        tensor.values[i] = number;
    }
    return tensor;
}

Tensors npz(const std::string & bytes) {
    // The directory at the end of the archive lists every file.
    if (bytes.size() < 22) {
        throw ValueError("g2p.npz: not a ZIP archive");
    }
    size_t end = bytes.size() - 22;
    while (u32(bytes, end) != 0x06054b50) {
        if (end == 0) {
            throw ValueError("g2p.npz: not a ZIP archive");
        }
        --end;
    }
    const size_t entries = u16(bytes, end + 10);
    size_t at = u32(bytes, end + 16);
    Tensors out;
    for (size_t k = 0; k < entries; ++k) {
        if (at + 46 > bytes.size() || u32(bytes, at) != 0x02014b50) {
            throw ValueError("g2p.npz: damaged directory");
        }
        const uint32_t method = u16(bytes, at + 10);
        const size_t size = u32(bytes, at + 20);
        const size_t name_size = u16(bytes, at + 28);
        const size_t extra_size = u16(bytes, at + 30);
        const size_t comment_size = u16(bytes, at + 32);
        const size_t local = u32(bytes, at + 42);
        std::string name = bytes.substr(at + 46, name_size);
        at += 46 + name_size + extra_size + comment_size;
        if (method != 0) {
            throw ValueError("g2p.npz: " + name + " is compressed; save the arrays with numpy.savez");
        }
        if (local + 30 > bytes.size() || u32(bytes, local) != 0x04034b50) {
            throw ValueError("g2p.npz: damaged entry " + name);
        }
        const size_t start = local + 30 + u16(bytes, local + 26) + u16(bytes, local + 28);
        if (start + size > bytes.size()) {
            throw ValueError("g2p.npz: truncated entry " + name);
        }
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) {
            name.resize(name.size() - 4);
        }
        out.emplace(name, npy(bytes, start, size, name));
    }
    return out;
}

}  // namespace

Data load_directory(const std::string & directory) {
    Data data;
    if (!read_file(directory + "/lexicon.tsv", data.lexicon)) {
        throw ValueError("no phonemizer data in " + directory + ": lexicon.tsv is missing");
    }
    if (!read_file(directory + "/align.json", data.align)) {
        throw ValueError("no phonemizer data in " + directory + ": align.json is missing");
    }
    std::string archive;
    if (read_file(directory + "/g2p.npz", archive)) {
        if (!read_file(directory + "/g2p.json", data.g2p)) {
            throw ValueError("no phonemizer data in " + directory + ": g2p.json is missing");
        }
        data.g2p_weights = npz(archive);
    }
    read_file(directory + "/homographs.json", data.homographs);
    return data;
}

}  // namespace dizione
