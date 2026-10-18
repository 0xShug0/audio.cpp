// torch.save() keeps view relationships: tensors that share one storage are
// written once, and each tensor records where it starts inside that storage.
// In the pickle every tensor is rebuilt as
//
//   torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, ...)
//
// and torch.load() applies storage_offset (in elements) and stride with
// tensor.set_(storage, storage_offset, size, stride).
//
// A reader that keeps only the storage and the size drops storage_offset
// (argument 1) and stride (argument 3), so every tensor is read from element 0 of
// its storage. A checkpoint saved as
//
//   x = torch.arange(8.)
//   torch.save({"head": x[:4], "tail": x[4:]}, "model.bin")
//
// loads "tail" as [0, 1, 2, 3] instead of torch.load()'s [4, 5, 6, 7]. No error is
// raised, so the wrong weights are used without any warning.
//
// This test writes that exact checkpoint (STORED zip + protocol 2 pickle, the
// layout torch.save produces) and checks the values torch.load() would return.
// Non-contiguous tensors and views past the end of their storage must be rejected
// rather than read in storage order.

#include "engine/framework/assets/torch_bin.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void put_u16(std::vector<uint8_t> & out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xffU));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xffU));
    }
}

void put_text(std::vector<uint8_t> & out, const std::string & text) {
    out.insert(out.end(), text.begin(), text.end());
}

// Uncompressed (STORED) zip archive, which is what torch.save writes.
std::vector<uint8_t> stored_zip(const std::vector<std::pair<std::string, std::vector<uint8_t>>> & files) {
    std::vector<uint8_t> out;
    std::vector<uint8_t> central;
    for (const auto & [name, data] : files) {
        const auto local_offset = static_cast<uint32_t>(out.size());
        const auto size = static_cast<uint32_t>(data.size());
        put_u32(out, 0x04034b50);  // local file header
        put_u16(out, 20);          // version needed
        put_u16(out, 0);           // flags
        put_u16(out, 0);           // method: stored
        put_u16(out, 0);           // mod time
        put_u16(out, 0);           // mod date
        put_u32(out, 0);           // crc32 (not checked by the reader)
        put_u32(out, size);
        put_u32(out, size);
        put_u16(out, static_cast<uint16_t>(name.size()));
        put_u16(out, 0);           // extra length
        put_text(out, name);
        out.insert(out.end(), data.begin(), data.end());

        put_u32(central, 0x02014b50);  // central directory header
        put_u16(central, 20);          // version made by
        put_u16(central, 20);          // version needed
        put_u16(central, 0);           // flags
        put_u16(central, 0);           // method: stored
        put_u16(central, 0);           // mod time
        put_u16(central, 0);           // mod date
        put_u32(central, 0);           // crc32
        put_u32(central, size);
        put_u32(central, size);
        put_u16(central, static_cast<uint16_t>(name.size()));
        put_u16(central, 0);           // extra length
        put_u16(central, 0);           // comment length
        put_u16(central, 0);           // disk number
        put_u16(central, 0);           // internal attributes
        put_u32(central, 0);           // external attributes
        put_u32(central, local_offset);
        put_text(central, name);
    }
    const auto central_offset = static_cast<uint32_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());
    put_u32(out, 0x06054b50);  // end of central directory
    put_u16(out, 0);
    put_u16(out, 0);
    put_u16(out, static_cast<uint16_t>(files.size()));
    put_u16(out, static_cast<uint16_t>(files.size()));
    put_u32(out, static_cast<uint32_t>(central.size()));
    put_u32(out, central_offset);
    put_u16(out, 0);
    return out;
}

void put_binunicode(std::vector<uint8_t> & pickle, const std::string & text) {
    pickle.push_back('X');
    put_u32(pickle, static_cast<uint32_t>(text.size()));
    put_text(pickle, text);
}

void put_int_tuple(std::vector<uint8_t> & pickle, const std::vector<uint8_t> & values) {
    pickle.push_back('(');  // MARK
    for (const uint8_t value : values) {
        pickle.push_back('K');
        pickle.push_back(value);
    }
    pickle.push_back('t');  // TUPLE
}

// Appends `key: _rebuild_tensor_v2(<FloatStorage "0" of storage_numel>,
// storage_offset, shape, stride, False, OrderedDict())`, the same opcodes
// torch.save emits for a float tensor.
void put_tensor_entry(
    std::vector<uint8_t> & pickle,
    const std::string & key,
    uint8_t storage_offset,
    const std::vector<uint8_t> & shape,
    const std::vector<uint8_t> & stride,
    uint8_t storage_numel) {
    put_binunicode(pickle, key);
    put_text(pickle, "ctorch._utils\n_rebuild_tensor_v2\n");
    pickle.push_back('(');  // MARK: _rebuild_tensor_v2 arguments
    pickle.push_back('(');  // MARK: persistent id
    put_binunicode(pickle, "storage");
    put_text(pickle, "ctorch\nFloatStorage\n");
    put_binunicode(pickle, "0");
    put_binunicode(pickle, "cpu");
    pickle.push_back('K');
    pickle.push_back(storage_numel);
    pickle.push_back('t');   // TUPLE
    pickle.push_back('Q');   // BINPERSID
    pickle.push_back('K');
    pickle.push_back(storage_offset);
    put_int_tuple(pickle, shape);
    put_int_tuple(pickle, stride);
    pickle.push_back(0x89);  // NEWFALSE: requires_grad
    put_text(pickle, "ccollections\nOrderedDict\n");
    pickle.push_back(')');   // EMPTY_TUPLE
    pickle.push_back('R');   // REDUCE: backward_hooks
    pickle.push_back('t');   // TUPLE: arguments
    pickle.push_back('R');   // REDUCE: _rebuild_tensor_v2
}

struct TensorEntry {
    std::string key;
    uint8_t storage_offset;
    std::vector<uint8_t> shape;
    std::vector<uint8_t> stride;
};

// Writes one checkpoint whose tensors all view a single float storage holding
// 0, 1, ..., storage_numel - 1.
std::filesystem::path write_checkpoint(const std::vector<TensorEntry> & tensors, uint8_t storage_numel) {
    std::vector<uint8_t> pickle = {0x80, 0x02, '}', '('};  // PROTO 2, EMPTY_DICT, MARK
    for (const auto & tensor : tensors) {
        put_tensor_entry(pickle, tensor.key, tensor.storage_offset, tensor.shape, tensor.stride, storage_numel);
    }
    pickle.push_back('u');  // SETITEMS
    pickle.push_back('.');  // STOP

    std::vector<uint8_t> storage(static_cast<size_t>(storage_numel) * sizeof(float));
    for (int i = 0; i < storage_numel; ++i) {
        const float value = static_cast<float>(i);
        std::memcpy(storage.data() + static_cast<size_t>(i) * sizeof(float), &value, sizeof(float));
    }

    const auto archive = stored_zip({{"archive/data.pkl", pickle}, {"archive/data/0", storage}});
    const auto path = std::filesystem::temp_directory_path() / "torch_bin_storage_offset.bin";
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open temp torch .bin file");
    out.write(reinterpret_cast<const char *>(archive.data()), static_cast<std::streamsize>(archive.size()));
    require(out.good(), "failed to write temp torch .bin file");
    return path;
}

std::vector<float> iota(int first, int count) {
    std::vector<float> values;
    for (int i = 0; i < count; ++i) {
        values.push_back(static_cast<float>(first + i));
    }
    return values;
}

std::string format(const std::vector<float> & values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        out += (i == 0 ? "" : ", ") + std::to_string(static_cast<int>(values[i]));
    }
    return out + "]";
}

void test_views_honour_storage_offset() {
    // x = torch.arange(8.); torch.save({"head": x[:4], "tail": x[4:]}, path)
    const auto path = write_checkpoint({{"head", 0, {4}, {1}}, {"tail", 4, {4}, {1}}}, 8);
    const auto source = engine::assets::open_torch_bin_tensor_source(path);
    const auto head = source->require_f32("head");
    const auto tail = source->require_f32("tail");
    std::filesystem::remove(path);

    const std::vector<float> expected_head = iota(0, 4);
    const std::vector<float> expected_tail = iota(4, 4);
    require(head == expected_head, "head should be " + format(expected_head) + ", got " + format(head));
    require(tail == expected_tail,
            "tail (storage_offset 4) should be " + format(expected_tail) + " as torch.load returns, got " +
                format(tail));
}

void test_chunked_matrix_honours_storage_offset() {
    // w = torch.arange(18.).reshape(6, 3); q, k = w.chunk(2, 0); torch.save({"q": q, "k": k}, path)
    // k is w[3:]: storage_offset 9, shape (3, 3), stride (3, 1).
    const auto path = write_checkpoint({{"q", 0, {3, 3}, {3, 1}}, {"k", 9, {3, 3}, {3, 1}}}, 18);
    const auto source = engine::assets::open_torch_bin_tensor_source(path);
    const auto q = source->require_f32("q", std::vector<int64_t>{3, 3});
    const auto k = source->require_f32("k", std::vector<int64_t>{3, 3});
    std::filesystem::remove(path);

    require(q == iota(0, 9), "q should be " + format(iota(0, 9)) + ", got " + format(q));
    require(k == iota(9, 9), "k (storage_offset 9) should be " + format(iota(9, 9)) + ", got " + format(k));
}

void test_size_one_dims_ignore_stride() {
    // torch.arange(4.).unsqueeze(0) is contiguous; torch may record any stride for a size-1 dim.
    const auto path = write_checkpoint({{"row", 0, {1, 4}, {1, 1}}}, 4);
    const auto source = engine::assets::open_torch_bin_tensor_source(path);
    const auto row = source->require_f32("row", std::vector<int64_t>{1, 4});
    std::filesystem::remove(path);
    require(row == iota(0, 4), "row should be " + format(iota(0, 4)) + ", got " + format(row));
}

void expect_rejected(const std::vector<TensorEntry> & tensors, uint8_t storage_numel, const std::string & what) {
    const auto path = write_checkpoint(tensors, storage_numel);
    bool threw = false;
    try {
        (void) engine::assets::open_torch_bin_tensor_source(path);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    std::filesystem::remove(path);
    require(threw, what + " should be rejected instead of loading the wrong values");
}

void test_unsupported_views_are_rejected() {
    // torch.arange(6.).reshape(2, 3).t(): shape (3, 2), stride (1, 3). Reading it in storage
    // order would return a silently transposed matrix.
    expect_rejected({{"wt", 0, {3, 2}, {1, 3}}}, 6, "a non-contiguous (transposed) tensor");
    // A view whose range runs past the end of its storage.
    expect_rejected({{"tail", 6, {4}, {1}}}, 8, "a view past the end of its storage");
}

}  // namespace

int main() {
    try {
        test_views_honour_storage_offset();
        test_chunked_matrix_honours_storage_offset();
        test_size_one_dims_ignore_stride();
        test_unsupported_views_are_rejected();
    } catch (const std::exception & error) {
        std::cerr << "torch .bin storage offset test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "torch .bin storage offset test passed\n";
    return 0;
}
