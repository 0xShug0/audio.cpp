#include "engine/framework/assets/model_package.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/safetensors.h"
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
    engine::assets::convert_tensor_source_to_gguf(safetensors, gguf, engine::assets::TensorStorageType::F16, false,
                                                  false);
    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require(source->tensors().front().shape.empty(), "rank-0-only GGUF lost scalar rank");
    engine::test::require_eq(source->require_i64_scalar("num_batches_tracked"), int64_t{23},
        "rank-0-only GGUF scalar value");
    std::filesystem::remove_all(root);
}

void test_embedded_model_spec_roundtrip_and_precedence() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_embedded_model_spec_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";
    const auto override_spec = root / "override.json";
    engine::io::write_safetensors_file(safetensors, {
                                                        {"weight", "F32", {1}, bytes_for(std::vector<float>{1.0F})},
                                                    });
    const std::string spec_json = R"json({
        "family":"embedded_test",
        "sources":[{
            "format":"gguf",
            "roots":{"weights":"$gguf"},
            "tensors":{"weights":"weights:"}
        }]
    })json";
    engine::assets::convert_tensor_sources_to_gguf({{safetensors, ""}}, gguf, engine::assets::TensorStorageType::F16,
                                                   false, false, {}, {},
                                                   engine::assets::GgufEmbeddedModelSpec{"embedded_test", spec_json});

    const auto embedded = engine::assets::read_gguf_embedded_model_spec(gguf);
    engine::test::require(embedded.has_value(), "GGUF lost its embedded model package spec");
    engine::test::require_eq(embedded->family, std::string("embedded_test"), "embedded spec family");
    engine::test::require_eq(embedded->json, spec_json, "embedded spec JSON");

    {
        engine::assets::ScopedModelPackageSpecOverride scope(std::nullopt, gguf);
        engine::test::require_eq(engine::assets::default_model_package_spec_path("embedded_test"),
                                 std::filesystem::path("@gguf") / "embedded_test.json",
                                 "embedded GGUF spec precedence");
        bool family_mismatch_rejected = false;
        try {
            (void)engine::assets::default_model_package_spec_path("another_family");
        } catch (const std::runtime_error &) {
            family_mismatch_rejected = true;
        }
        engine::test::require(family_mismatch_rejected, "embedded GGUF spec accepted a different family");
    }

    const auto uppercase_gguf = root / "RENAMED.GGUF";
    std::filesystem::copy_file(gguf, uppercase_gguf);
    {
        engine::assets::ScopedModelPackageSpecOverride scope(std::nullopt, uppercase_gguf);
        const auto spec_path = engine::assets::default_model_package_spec_path("embedded_test");
        const auto resources = engine::assets::load_resource_bundle_from_package_spec(uppercase_gguf, spec_path);
        engine::test::require(resources.open_tensor_source("weights")->has_tensor("weight"),
                              "uppercase standalone GGUF did not use its embedded package spec");
    }

    {
        std::ofstream output(override_spec, std::ios::binary);
        output << spec_json;
    }
    {
        engine::assets::ScopedModelPackageSpecOverride scope(override_spec, gguf);
        engine::test::require_eq(engine::assets::default_model_package_spec_path("embedded_test"),
                                 std::filesystem::weakly_canonical(override_spec),
                                 "explicit package spec override precedence");
    }
    std::filesystem::remove_all(root);
}

void test_package_spec_errors_name_selected_spec() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_package_spec_error_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model");
    const auto malformed_spec = root / "malformed.json";
    const auto missing_file_spec = root / "missing_file.json";

    {
        std::ofstream output(malformed_spec, std::ios::binary);
        output << R"json({"family":"broken","sources":[)json";
    }
    {
        std::ofstream output(missing_file_spec, std::ios::binary);
        output << R"json({
            "family":"missing_file_test",
            "sources":[{
                "format":"safetensors",
                "roots":{"model":"."},
                "files":{"config":"model:config.json"}
            }]
        })json";
    }

    bool malformed_named = false;
    try {
        (void)engine::assets::load_resource_bundle_from_package_spec(root / "model", malformed_spec);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        malformed_named = message.find("failed to parse model package spec") != std::string::npos &&
            message.find(malformed_spec.string()) != std::string::npos;
    }
    engine::test::require(malformed_named, "malformed package spec error did not name the selected spec");

    bool selected_source_named = false;
    try {
        (void)engine::assets::load_resource_bundle_from_package_spec(root / "model", missing_file_spec);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        selected_source_named = message.find("using model package spec") != std::string::npos &&
            message.find(missing_file_spec.string()) != std::string::npos &&
            message.find("source 'safetensors'") != std::string::npos &&
            message.find("missing model package file 'config'") != std::string::npos;
    }
    engine::test::require(selected_source_named, "resource error did not name the selected spec and source");

    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_safetensors_to_gguf_roundtrip();
        test_packed_multi_source_gguf();
        test_all_rank0_gguf();
        test_embedded_model_spec_roundtrip_and_precedence();
        test_package_spec_errors_name_selected_spec();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "gguf_tensor_source_test passed\n";
    return 0;
}
