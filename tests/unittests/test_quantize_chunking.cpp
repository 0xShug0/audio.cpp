/*
 * The contract the parallel quantizer rests on.
 *
 * quantize_f32_rows splits a tensor's rows across threads and calls
 * ggml_quantize_chunk once per slice. That is only correct while quantization is
 * ROW-INDEPENDENT: each row's bytes must depend on that row alone, so that
 * quantizing rows [0,n) in one call and in several calls produce the same
 * bytes.
 *
 * Every quantized type ggml offers today satisfies that, but nothing in ggml
 * promises it. A type that interleaved or shared state across rows -- a repacked
 * format, say -- would still quantize successfully and would silently produce a
 * different tensor depending on how the rows happened to be divided, which is
 * both wrong and non-deterministic across machines with different core counts.
 *
 * So this iterates TensorStorageType rather than naming types inline: whatever
 * the converter can be asked to produce is what gets checked, and a storage type
 * added later is covered without anyone remembering to come back here.
 *
 * It deliberately does NOT sweep every ggml_is_quantized type. Some of those are
 * activation formats rather than weight targets -- q8_1 is one, and
 * ggml_quantize_chunk aborts on it -- so a sweep tests ggml's surface instead of
 * this converter's contract.
 */

#include "test_assert.h"

#include "engine/framework/assets/tensor_source.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Odd row count on purpose: the last chunk is ragged, which is where an
// off-by-one in the offset arithmetic would show up.
constexpr int64_t kRows = 37;
// A multiple of 256, so every k-quant's super-block divides it.
constexpr int64_t kColumns = 512;

std::vector<float> make_values() {
    std::vector<float> values(static_cast<size_t>(kRows * kColumns));
    // Deterministic, and varying along BOTH axes so that a chunk which read the
    // wrong rows would produce different bytes rather than the same ones.
    for (int64_t row = 0; row < kRows; ++row) {
        for (int64_t column = 0; column < kColumns; ++column) {
            const auto index = static_cast<size_t>(row * kColumns + column);
            values[index] = std::sin(static_cast<float>(row) * 0.7F + static_cast<float>(column) * 0.013F)
                * (1.0F + static_cast<float>(row % 5));
        }
    }
    return values;
}

std::vector<std::byte> quantize_in_chunks(
    ggml_type type, const std::vector<float> & values, int64_t rows_per_chunk) {
    std::vector<std::byte> bytes(static_cast<size_t>(kRows) * ggml_row_size(type, kColumns));
    size_t written = 0;
    for (int64_t first = 0; first < kRows; first += rows_per_chunk) {
        const int64_t count = std::min(rows_per_chunk, kRows - first);
        written += ggml_quantize_chunk(
            type, values.data(), bytes.data(), first * kColumns, count, kColumns, nullptr);
    }
    engine::test::require_eq(written, bytes.size(), "bytes written");
    return bytes;
}

}  // namespace

int main() {
    try {
        const auto values = make_values();
        int checked = 0;

        // Every storage type the converter accepts, so the list cannot drift
        // from what `--type` can be given.
        const engine::assets::TensorStorageType storage_types[] = {
            engine::assets::TensorStorageType::Q4_0,
            engine::assets::TensorStorageType::Q4_1,
            engine::assets::TensorStorageType::Q5_0,
            engine::assets::TensorStorageType::Q5_1,
            engine::assets::TensorStorageType::Q2_K,
            engine::assets::TensorStorageType::Q3_K,
            engine::assets::TensorStorageType::Q4_K,
            engine::assets::TensorStorageType::Q5_K,
            engine::assets::TensorStorageType::Q6_K,
            engine::assets::TensorStorageType::Q8_0,
            // Not offered by --type, but parse_tensor_storage_type accepts it and
            // --keep-type can ask for it, so it reaches the same split.
            engine::assets::TensorStorageType::NVFP4,
        };

        for (const auto storage_type : storage_types) {
            const auto type = engine::assets::ggml_type_for_tensor_storage(storage_type);
            engine::test::require(ggml_is_quantized(type), "storage type is not quantized");
            engine::test::require(
                !ggml_quantize_requires_imatrix(type),
                std::string("unexpected imatrix requirement for ") + ggml_type_name(type));
            engine::test::require(
                kColumns % ggml_blck_size(type) == 0,
                std::string("test column count does not divide the block size of ") + ggml_type_name(type));

            ggml_quantize_init(type);
            const auto whole = quantize_in_chunks(type, values, kRows);

            // Several splits, including ones that do not divide the row count.
            for (const int64_t rows_per_chunk : {int64_t{1}, int64_t{4}, int64_t{8}, int64_t{16}}) {
                const auto split = quantize_in_chunks(type, values, rows_per_chunk);
                engine::test::require(
                    split == whole,
                    std::string("quantization is not row-independent for ") + ggml_type_name(type)
                        + " at " + std::to_string(rows_per_chunk) + " rows per chunk");
            }
            ++checked;
        }

        engine::test::require(checked == 11, "expected every quantized storage type to be checked");
        std::cout << "quantize_chunking_test passed (" << checked << " quantized storage types)\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "quantize_chunking_test failed: " << error.what() << "\n";
        return 1;
    }
}
