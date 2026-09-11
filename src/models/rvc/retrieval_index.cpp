#include "retrieval_index.h"

#include "engine/framework/assets/tensor_source.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::rvc {
namespace {

int64_t rounded_nonnegative_index(float value, const std::string & name) {
    if (value < 0.0F) {
        throw std::runtime_error("RVC retrieval sidecar contains negative " + name);
    }
    return static_cast<int64_t>(std::llround(static_cast<double>(value)));
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("failed to open RVC retrieval index: " + path.string());
    }
    const std::streamsize size = stream.tellg();
    if (size < 0) {
        throw std::runtime_error("RVC retrieval index has invalid size: " + path.string());
    }
    stream.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(bytes.data()), size)) {
        throw std::runtime_error("failed to read RVC retrieval index: " + path.string());
    }
    return bytes;
}

RvcRetrievalIndex load_retrieval_sidecar(
    const engine::assets::TensorSource & source,
    int64_t dim,
    const std::string & label) {
    int64_t nlist = 0;
    int64_t vector_count = 0;
    for (const auto & tensor : source.tensors()) {
        if (tensor.name == "centroids") {
            if (tensor.shape.size() != 2 || tensor.shape[1] != dim) {
                throw std::runtime_error("RVC retrieval centroids shape mismatch: " + label);
            }
            nlist = tensor.shape[0];
        }
        if (tensor.name == "vectors") {
            if (tensor.shape.size() != 2 || tensor.shape[1] != dim) {
                throw std::runtime_error("RVC retrieval vectors shape mismatch: " + label);
            }
            vector_count = tensor.shape[0];
        }
    }
    if (nlist <= 0 || vector_count <= 0) {
        throw std::runtime_error("RVC retrieval sidecar is missing centroids or vectors: " + label);
    }
    RvcRetrievalIndex index;
    index.dim = dim;
    index.nlist = nlist;
    index.centroids = source.require_f32("centroids", {nlist, dim});
    index.vectors = source.require_f32("vectors", {vector_count, dim});
    const auto offsets = source.require_f32("list_offsets", {nlist});
    const auto lengths = source.require_f32("list_lengths", {nlist});
    index.list_offsets.resize(static_cast<size_t>(nlist));
    index.list_lengths.resize(static_cast<size_t>(nlist));
    int64_t total = 0;
    for (int64_t list = 0; list < nlist; ++list) {
        index.list_offsets[static_cast<size_t>(list)] = rounded_nonnegative_index(offsets[static_cast<size_t>(list)], "offset");
        index.list_lengths[static_cast<size_t>(list)] = rounded_nonnegative_index(lengths[static_cast<size_t>(list)], "length");
        total += index.list_lengths[static_cast<size_t>(list)];
    }
    if (total != vector_count) {
        throw std::runtime_error("RVC retrieval sidecar list lengths do not match vector count: " + label);
    }
    return index;
}

uint32_t faiss_fourcc(const char (&tag)[5]) {
    return static_cast<uint32_t>(static_cast<uint8_t>(tag[0])) |
        static_cast<uint32_t>(static_cast<uint8_t>(tag[1])) << 8 |
        static_cast<uint32_t>(static_cast<uint8_t>(tag[2])) << 16 |
        static_cast<uint32_t>(static_cast<uint8_t>(tag[3])) << 24;
}

class FaissIndexReader {
public:
    explicit FaissIndexReader(std::filesystem::path path) : path_(std::move(path)), bytes_(read_binary_file(path_)) {}

    RvcRetrievalIndex read_ivf_flat(int64_t expected_dim) {
        if (read_u32() != faiss_fourcc("IwFl")) {
            throw std::runtime_error("RVC retrieval_index_path must be a FAISS IndexIVFFlat: " + path_.string());
        }
        const auto header = read_index_header();
        if (header.d != expected_dim) {
            throw std::runtime_error("RVC retrieval_index_path dimension mismatch: " + path_.string());
        }
        const size_t nlist = read_size();
        const size_t nprobe = read_size();
        if (nlist == 0 || nprobe == 0) {
            throw std::runtime_error("RVC retrieval_index_path has invalid IVF header: " + path_.string());
        }
        const auto centroids = read_flat_l2(expected_dim, static_cast<int64_t>(nlist));
        read_direct_map();
        RvcRetrievalIndex out = read_array_inverted_lists(expected_dim, nlist);
        out.dim = expected_dim;
        out.nlist = static_cast<int64_t>(nlist);
        out.centroids = centroids;
        if (static_cast<int64_t>(out.vectors.size()) != header.ntotal * expected_dim) {
            throw std::runtime_error("RVC retrieval_index_path vector count mismatch: " + path_.string());
        }
        if (cursor_ != bytes_.size()) {
            throw std::runtime_error("RVC retrieval_index_path has trailing unsupported data: " + path_.string());
        }
        return out;
    }

private:
    struct IndexHeader {
        int32_t d = 0;
        int64_t ntotal = 0;
    };

    void expect(size_t count) const {
        if (cursor_ + count > bytes_.size()) {
            throw std::runtime_error("RVC retrieval_index_path is truncated: " + path_.string());
        }
    }
    uint8_t read_u8() {
        expect(1);
        return bytes_[cursor_++];
    }
    uint32_t read_u32() {
        expect(4);
        uint32_t value = 0;
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(value));
        cursor_ += sizeof(value);
        return value;
    }
    int32_t read_i32() {
        return static_cast<int32_t>(read_u32());
    }
    int64_t read_i64() {
        expect(8);
        int64_t value = 0;
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(value));
        cursor_ += sizeof(value);
        return value;
    }
    size_t read_size() {
        const int64_t value = read_i64();
        if (value < 0) {
            throw std::runtime_error("RVC retrieval_index_path contains a negative size: " + path_.string());
        }
        return static_cast<size_t>(value);
    }
    float read_f32() {
        expect(4);
        float value = 0.0F;
        std::memcpy(&value, bytes_.data() + cursor_, sizeof(value));
        cursor_ += sizeof(value);
        return value;
    }
    void skip(size_t count) {
        expect(count);
        cursor_ += count;
    }
    IndexHeader read_index_header() {
        IndexHeader header;
        header.d = read_i32();
        header.ntotal = read_i64();
        (void)read_i64();
        (void)read_i64();
        const uint8_t is_trained = read_u8();
        const int32_t metric_type = read_i32();
        if (is_trained == 0 || metric_type != 1) {
            throw std::runtime_error("RVC retrieval_index_path must be a trained L2 FAISS index: " + path_.string());
        }
        if (header.d <= 0 || header.ntotal < 0) {
            throw std::runtime_error("RVC retrieval_index_path has invalid index header: " + path_.string());
        }
        return header;
    }
    std::vector<float> read_flat_l2(int64_t expected_dim, int64_t expected_count) {
        if (read_u32() != faiss_fourcc("IxF2")) {
            throw std::runtime_error("RVC retrieval_index_path IVF quantizer must be IndexFlatL2: " + path_.string());
        }
        const auto header = read_index_header();
        if (header.d != expected_dim || header.ntotal != expected_count) {
            throw std::runtime_error("RVC retrieval_index_path quantizer shape mismatch: " + path_.string());
        }
        const size_t vector_floats = read_size();
        const size_t expected_floats = static_cast<size_t>(expected_count * expected_dim);
        if (vector_floats != expected_floats) {
            throw std::runtime_error("RVC retrieval_index_path quantizer vector size mismatch: " + path_.string());
        }
        std::vector<float> values(vector_floats);
        for (float & value : values) {
            value = read_f32();
        }
        return values;
    }
    void read_direct_map() {
        const uint8_t direct_map_type = read_u8();
        const size_t array_size = read_size();
        skip(array_size * sizeof(int64_t));
        if (direct_map_type == 2) {
            const size_t hashtable_size = read_size();
            skip(hashtable_size * 2 * sizeof(int64_t));
        } else if (direct_map_type != 0 && direct_map_type != 1) {
            throw std::runtime_error("RVC retrieval_index_path has unsupported FAISS direct map type: " + path_.string());
        }
    }
    std::vector<size_t> read_full_list_sizes(size_t nlist) {
        if (read_u32() != faiss_fourcc("full")) {
            throw std::runtime_error("RVC retrieval_index_path must store full IVF list sizes: " + path_.string());
        }
        const size_t count = read_size();
        if (count != nlist) {
            throw std::runtime_error("RVC retrieval_index_path IVF list count mismatch: " + path_.string());
        }
        std::vector<size_t> sizes(count);
        for (auto & size : sizes) {
            size = read_size();
        }
        return sizes;
    }
    RvcRetrievalIndex read_array_inverted_lists(int64_t dim, size_t nlist) {
        if (read_u32() != faiss_fourcc("ilar")) {
            throw std::runtime_error("RVC retrieval_index_path must store ArrayInvertedLists: " + path_.string());
        }
        const size_t stored_nlist = read_size();
        const size_t code_size = read_size();
        if (stored_nlist != nlist || code_size != static_cast<size_t>(dim * sizeof(float))) {
            throw std::runtime_error("RVC retrieval_index_path inverted-list header mismatch: " + path_.string());
        }
        const auto sizes = read_full_list_sizes(nlist);
        RvcRetrievalIndex out;
        out.list_offsets.resize(nlist);
        out.list_lengths.resize(nlist);
        int64_t total = 0;
        for (size_t list = 0; list < nlist; ++list) {
            out.list_offsets[list] = total;
            out.list_lengths[list] = static_cast<int64_t>(sizes[list]);
            total += static_cast<int64_t>(sizes[list]);
        }
        out.vectors.resize(static_cast<size_t>(total * dim));
        for (size_t list = 0; list < nlist; ++list) {
            const size_t list_size = sizes[list];
            float * dst = out.vectors.data() + static_cast<size_t>(out.list_offsets[list] * dim);
            const size_t floats = list_size * static_cast<size_t>(dim);
            for (size_t i = 0; i < floats; ++i) {
                dst[i] = read_f32();
            }
            skip(list_size * sizeof(int64_t));
        }
        return out;
    }

    std::filesystem::path path_;
    std::vector<uint8_t> bytes_;
    size_t cursor_ = 0;
};

}  // namespace

RvcRetrievalIndex load_rvc_retrieval_index(
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    int64_t dim,
    const std::string & label) {
    if (source == nullptr) {
        throw std::runtime_error("missing RVC retrieval tensor source: " + label);
    }
    return load_retrieval_sidecar(*source, dim, label);
}

RvcRetrievalIndex load_rvc_retrieval_index(const std::filesystem::path & path, int64_t dim) {
    const auto extension = path.extension().string();
    if (extension == ".safetensors" || extension == ".gguf") {
        return load_retrieval_sidecar(*engine::assets::open_tensor_source(path), dim, path.string());
    }
    if (extension == ".index") {
        return FaissIndexReader(path).read_ivf_flat(dim);
    }
    throw std::runtime_error("unsupported RVC retrieval index format: " + path.string());
}

}  // namespace engine::models::rvc
