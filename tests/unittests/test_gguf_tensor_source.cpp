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

void write_rank0_safetensors_with_null_metadata(
    const std::filesystem::path & path,
    std::string_view name,
    int64_t value) {
    std::string header = "{\"__metadata__\":null,\"" + std::string(name) +
        "\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[0,8]}}";
    while (header.size() % 8 != 0) header.push_back(' ');
    const uint64_t header_size = static_cast<uint64_t>(header.size());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
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
        {"scalar.scale", "F32", {}, bytes_for(std::vector<float>{0.5F})},
        {"step", "I64", {1}, bytes_for(std::vector<int64_t>{42})},
        {"num_batches_tracked", "I64", {}, bytes_for(std::vector<int64_t>{7})},
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
    engine::test::require(
        source->require_f32("scalar.scale", std::vector<int64_t>{}) == std::vector<float>{0.5F},
        "rank-0 F32 scalar value");
    engine::test::require(source->require_metadata("scalar.scale").shape.empty(), "rank-0 F32 scalar shape");
    engine::test::require_eq(source->require_i64_scalar("step"), int64_t{42}, "I64 scalar");
    engine::test::require_eq(source->require_i64_scalar("num_batches_tracked"), int64_t{7}, "rank-0 I64 scalar");
    engine::test::require(
        source->require_metadata("num_batches_tracked").shape.empty(),
        "GGUF did not preserve rank-0 scalar shape");
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

void test_packed_multi_source_gguf() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_packed_gguf_test";
    std::filesystem::remove_all(root);
    const auto model_root = root / "model";
    const auto shared_root = root / "shared";
    std::filesystem::create_directories(model_root);
    std::filesystem::create_directories(shared_root);
    engine::io::write_safetensors_file(model_root / "gpt.safetensors", {
        {"layer.weight", "F32", {2, 2}, bytes_for(std::vector<float>{1, 2, 3, 4})},
    });
    write_rank0_safetensors_with_null_metadata(
        model_root / "campplus.safetensors", "head.bn1.num_batches_tracked", 11);
    {
        std::ofstream output(shared_root / "preprocessor_config.json", std::ios::binary);
        output << "{\"feature_size\":128}";
    }

    const auto gguf = model_root / "packed.gguf";
    engine::assets::convert_tensor_sources_to_gguf(
        {
            {model_root / "gpt.safetensors", "gpt"},
            {model_root / "campplus.safetensors", "campplus"},
        },
        gguf,
        engine::assets::TensorStorageType::F16,
        false,
        true,
        model_root,
        {{shared_root / "preprocessor_config.json", "preprocessor_config.json"}});

    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require(source->has_tensor("gpt/layer.weight"), "packed GGUF lost GPT namespace");
    engine::test::require(
        source->require_metadata("campplus/head.bn1.num_batches_tracked").shape.empty(),
        "packed GGUF lost rank-0 component scalar");
    engine::test::require_eq(
        source->require_i64_scalar("campplus/head.bn1.num_batches_tracked"),
        int64_t{11},
        "packed GGUF rank-0 scalar value");
    const auto campplus = engine::assets::open_tensor_source(gguf, "campplus");
    engine::test::require(
        campplus->has_tensor("head.bn1.num_batches_tracked"),
        "packed GGUF namespace view did not strip its component prefix");
    engine::test::require_eq(
        campplus->require_i64_scalar("head.bn1.num_batches_tracked"),
        int64_t{11},
        "packed GGUF namespace scalar value");
    const auto prepared = engine::assets::prepare_model_directory(gguf);
    engine::test::require_eq(
        engine::io::read_text_file(prepared.model_root / "preprocessor_config.json"),
        std::string("{\"feature_size\":128}"),
        "explicit packed GGUF sidecar");

    bool duplicate_destination_rejected = false;
    try {
        engine::assets::convert_tensor_sources_to_gguf(
            {{model_root / "gpt.safetensors", "gpt"}},
            model_root / "duplicate-sidecar.gguf",
            engine::assets::TensorStorageType::F16,
            false,
            true,
            model_root,
            {
                {shared_root / "preprocessor_config.json", "preprocessor_config.json"},
                {shared_root / "preprocessor_config.json", "preprocessor_config.json"},
            });
    } catch (const std::runtime_error &) {
        duplicate_destination_rejected = true;
    }
    engine::test::require(
        duplicate_destination_rejected,
        "packed GGUF accepted duplicate embedded sidecar destinations");
    std::filesystem::remove_all(root);
}

void test_all_rank0_gguf() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_rank0_only_gguf_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "scalar.safetensors";
    const auto gguf = root / "scalar.gguf";
    write_rank0_safetensors_with_null_metadata(safetensors, "num_batches_tracked", 23);
    engine::assets::convert_tensor_source_to_gguf(
        safetensors,
        gguf,
        engine::assets::TensorStorageType::F16);
    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require(source->tensors().front().shape.empty(), "rank-0-only GGUF lost scalar rank");
    engine::test::require_eq(
        source->require_i64_scalar("num_batches_tracked"),
        int64_t{23},
        "rank-0-only GGUF scalar value");
    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_safetensors_to_gguf_roundtrip();
        test_packed_multi_source_gguf();
        test_all_rank0_gguf();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "gguf_tensor_source_test passed\n";
    return 0;
}
