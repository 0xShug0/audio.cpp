#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/filesystem.h"
#include "test_assert.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<unsigned char> bytes_for(const std::vector<T> & values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

void test_safetensors_to_gguf_roundtrip() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_gguf_tensor_source_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";

    std::vector<float> matrix(64);
    for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = static_cast<float>(i) / 64.0F - 0.5F;
    const std::string long_embedding_name =
        "model.language_model.layers.really_long_component_name.embed_tokens.weight";
    engine::io::write_safetensors_file(safetensors, {
        {"projection.weight", "F32", {2, 32}, bytes_for(matrix)},
        {long_embedding_name, "F32", {2, 32}, bytes_for(matrix)},
        {"projection.bias", "F32", {2}, bytes_for(std::vector<float>{1.25F, -2.5F})},
        {"step", "I64", {1}, bytes_for(std::vector<int64_t>{42})},
        {"singleton.conv.weight", "F32", {2, 32, 1}, bytes_for(matrix)},
    });
    std::filesystem::create_directories(root / "tokenizer");
    const std::string binary_sidecar("abc\0def", 7);
    {
        std::ofstream output(root / "tokenizer" / "tokenizer.model", std::ios::binary);
        output.write(binary_sidecar.data(), static_cast<std::streamsize>(binary_sidecar.size()));
    }

    engine::assets::convert_tensor_source_to_gguf(
        safetensors,
        gguf,
        engine::assets::TensorStorageType::Q8_0);
    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require(source->has_tensor("projection.weight"), "GGUF is missing projection.weight");
    engine::test::require(source->has_tensor(long_embedding_name), "GGUF lost the long logical tensor name");
    engine::test::require_eq(source->require_metadata("projection.weight").dtype, std::string("q8_0"), "matrix dtype");
    engine::test::require_eq(source->require_metadata(long_embedding_name).dtype, std::string("f16"), "embedding dtype");
    engine::test::require_eq(source->require_i64_scalar("step"), int64_t{42}, "I64 scalar");
    engine::test::require(
        source->require_metadata("singleton.conv.weight").shape == std::vector<int64_t>({2, 32, 1}),
        "GGUF lost an exact singleton tensor dimension");

    const auto bias = source->require_f32("projection.bias", {2});
    engine::test::require_close(bias[0], 1.25F, 0.0F, "bias[0]");
    engine::test::require_close(bias[1], -2.5F, 0.0F, "bias[1]");
    const auto quantized = source->require_f32("projection.weight", {2, 32});
    for (size_t i = 0; i < matrix.size(); ++i) {
        engine::test::require_close(quantized[i], matrix[i], 0.01F, "quantized matrix value");
    }
    const auto prepared = engine::assets::prepare_model_directory(gguf);
    engine::test::require(prepared.standalone_gguf.has_value(), "GGUF sidecars were not detected");
    engine::test::require_eq(
        engine::io::read_text_file(prepared.model_root / "tokenizer" / "tokenizer.model"),
        binary_sidecar,
        "binary nested GGUF sidecar");

    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_safetensors_to_gguf_roundtrip();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "gguf_tensor_source_test passed\n";
    return 0;
}
