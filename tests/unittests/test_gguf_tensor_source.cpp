#include "engine/framework/model_spec/package.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/weight_adapter.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "gguf.h"
#include "test_assert.h"

#include <cstring>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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

void test_source_policy_cache_and_views() {
    using namespace engine::assets;
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_source_policy_test";
    std::filesystem::create_directories(root);
    const auto file = root / "model.safetensors";
    engine::io::write_safetensors_file(file, {
        {"block/weight", "F32", {2, 2}, bytes_for(std::vector<float>{1, 2, 3, 4})},
        {"block/step", "I64", {}, bytes_for(std::vector<int64_t>{42})},
        {"norm.weight_g", "F32", {2, 1}, bytes_for(std::vector<float>{1, 2})},
        {"norm.weight_v", "F32", {2, 2}, bytes_for(std::vector<float>{3, 4, 3, 4})},
    });
    const auto index = root / "model.safetensors.index.json";
    std::ofstream(index) << R"({"weight_map":{"block/weight":"model.safetensors","block/step":"model.safetensors","norm.weight_g":"model.safetensors","norm.weight_v":"model.safetensors"}})";
    const auto gguf = root / "model.gguf";
    convert_tensor_source_to_gguf(file, gguf, TensorStorageType::F32, true, false);
    // Bundle cache synchronization and source storage synchronization are separate opt-ins.
    ResourceBundle bundle(root, ResourceBundleOptions{true});
    bundle.add_file("primary", file);
    bundle.add_file("alias", file);
    bundle.add_tensor_source("prefix", file, "block");
    const auto original = bundle.open_tensor_source("primary");
    const auto shared = bundle.open_tensor_source("primary", TensorSourceOptions{true});
    engine::test::require(original != shared, "synchronized resource reused a legacy cached source");
    engine::test::require(bundle.open_tensor_source("alias") == original,
        "legacy canonical-path cache behavior changed");
    engine::test::require(bundle.open_tensor_source("alias", TensorSourceOptions{true}) == shared,
        "synchronized canonical-path cache did not retain source identity");
    std::vector<std::future<void>> jobs;
    for (int worker = 0; worker < 8; ++worker) {
        jobs.push_back(std::async(std::launch::async, [&] {
            for (int wave = 0; wave < 50; ++wave) {
                const auto source = bundle.open_tensor_source("prefix", TensorSourceOptions{true});
                engine::test::require(source->require_i64_scalar("step") == 42, "prefixed source policy lost data");
                shared->release_storage();
            }
        }));
    }
    for (auto & job : jobs) { job.get(); }
    // Indexed shards and both view types retain the underlying source's storage lock.
    for (const auto & path : {file, gguf, index}) {
        const auto source = open_tensor_source(path, TensorSourceOptions{true});
        const auto prefix = make_prefixed_tensor_source(source, "block");
        const auto folded = make_weight_norm_folded_tensor_source(source, {"norm.weight"});
        const auto expected = folded->require_f32("norm.weight");
        auto release = std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i) { source->release_storage(); }
        });
        for (int i = 0; i < 100; ++i) {
            engine::test::require(prefix->require_i64_scalar("step") == 42, "indexed/prefixed scalar changed");
            engine::test::require(folded->require_f32("norm.weight") == expected, "folded view lost source synchronization");
        }
        release.get();
    }
    std::filesystem::remove_all(root);
}

void test_concurrent_storage_release() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_concurrent_tensor_source_test";
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";
    std::vector<float> values(262144);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i % 128);
    const auto expected = bytes_for(values);
    engine::io::write_safetensors_file(safetensors, {
        {"weight", "F32", {512, 512}, expected},
        {"step", "I64", {}, bytes_for(std::vector<int64_t>{42})},
    });
    engine::assets::convert_tensor_source_to_gguf(
        safetensors, gguf, engine::assets::TensorStorageType::F32, true, false);
    for (const auto & path : {safetensors, gguf}) {
        const auto source = engine::assets::open_tensor_source(path, engine::assets::TensorSourceOptions{true});
        source->release_storage();
        std::atomic<bool> start{false};
        std::atomic<bool> failed{false};
        auto read = [&] {
            while (!start.load()) std::this_thread::yield();
            try {
                for (int i = 0; i < 100; ++i) {
                    const auto tensor = source->require_tensor_data("weight");
                    if (tensor.bytes.size() != expected.size() ||
                        std::memcmp(tensor.bytes.data(), expected.data(), expected.size()) != 0 ||
                        source->require_i64_scalar("step") != 42) failed.store(true);
                }
            } catch (...) { failed.store(true); }
        };
        std::thread first(read);
        std::thread second(read);
        std::thread release([&] {
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < 1000; ++i) {
                source->release_storage();
                std::this_thread::yield();
            }
        });
        start.store(true);
        first.join();
        second.join();
        release.join();
        engine::test::require(!failed.load(), "concurrent read/release changed tensor data");
    }
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
    engine::test::require(embedded.has_value(), "GGUF lost its embedded model spec");
    engine::test::require_eq(embedded->family, std::string("embedded_test"), "embedded spec family");
    engine::test::require_eq(embedded->json, spec_json, "embedded spec JSON");

    {
        engine::model_spec::ScopedSpecOverride scope(std::nullopt, gguf);
        engine::test::require_eq(engine::model_spec::default_spec_path("embedded_test"),
                                 std::filesystem::path("@gguf") / "embedded_test.json",
                                 "embedded GGUF spec precedence");
        bool family_mismatch_rejected = false;
        try {
            (void)engine::model_spec::default_spec_path("another_family");
        } catch (const std::runtime_error &) {
            family_mismatch_rejected = true;
        }
        engine::test::require(family_mismatch_rejected, "embedded GGUF spec accepted a different family");
    }

    const auto uppercase_gguf = root / "RENAMED.GGUF";
    std::filesystem::copy_file(gguf, uppercase_gguf);
    {
        engine::model_spec::ScopedSpecOverride scope(std::nullopt, uppercase_gguf);
        const auto spec_path = engine::model_spec::default_spec_path("embedded_test");
        const auto resources = engine::model_spec::load_resource_bundle(uppercase_gguf, spec_path);
        engine::test::require(resources.open_tensor_source("weights")->has_tensor("weight"),
                              "uppercase standalone GGUF did not use its embedded package spec");
    }

    {
        std::ofstream output(override_spec, std::ios::binary);
        output << spec_json;
    }
    {
        engine::model_spec::ScopedSpecOverride scope(override_spec, gguf);
        engine::test::require_eq(engine::model_spec::default_spec_path("embedded_test"),
                                 std::filesystem::weakly_canonical(override_spec),
                                 "explicit package spec override precedence");
    }
    std::filesystem::remove_all(root);
}

// Published GGUF packages install under their release name (vevo2-q8_0.gguf, not model.gguf)
// into a directory that the WebUI and the server config hand over as the model path. That
// directory has to resolve to its GGUF, or the spec falls back to the safetensors source and
// fails on config files the GGUF package never ships (issue #113).
void test_release_named_gguf_directory_selects_the_gguf_source() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_named_gguf_directory_test";
    std::filesystem::remove_all(root);
    const auto model_dir = root / "Vevo2-GGUF";
    std::filesystem::create_directories(model_dir);
    const auto safetensors = root / "weights.safetensors";
    const auto gguf = model_dir / "vevo2-q8_0.gguf";
    engine::io::write_safetensors_file(safetensors, {
                                                        {"weight", "F32", {1}, bytes_for(std::vector<float>{1.0F})},
                                                    });
    const std::string spec_json = R"json({
        "family":"named_gguf_test",
        "sources":[{
            "format":"gguf",
            "roots":{"weights":"$gguf"},
            "tensors":{"weights":"weights:"}
        },{
            "format":"safetensors",
            "roots":{"model":"."},
            "files":{"config":"model:config.json"},
            "tensors":{"weights":"model:model.safetensors"}
        }]
    })json";
    engine::assets::convert_tensor_sources_to_gguf({{safetensors, ""}}, gguf, engine::assets::TensorStorageType::F16,
                                                   false, false, {}, {},
                                                   engine::assets::GgufEmbeddedModelSpec{"named_gguf_test", spec_json});

    engine::test::require_eq(engine::assets::find_directory_gguf(model_dir).value_or(""), gguf,
                             "sole release-named GGUF in a model directory");
    {
        engine::model_spec::ScopedSpecOverride scope(std::nullopt, model_dir);
        engine::test::require_eq(engine::model_spec::default_spec_path("named_gguf_test"),
                                 std::filesystem::path("@gguf") / "named_gguf_test.json",
                                 "embedded spec of a release-named GGUF in a model directory");
        const auto resources = engine::model_spec::load_resource_bundle(
            model_dir, engine::model_spec::default_spec_path("named_gguf_test"));
        engine::test::require(resources.open_tensor_source("weights")->has_tensor("weight"),
                              "release-named GGUF directory did not select the GGUF source");
    }

    // model.gguf keeps winning, so a directory carrying both stays on its documented entry point.
    const auto default_named = model_dir / "model.gguf";
    std::filesystem::copy_file(gguf, default_named);
    engine::test::require_eq(engine::assets::find_directory_gguf(model_dir).value_or(""), default_named,
                             "model.gguf precedence over a release-named sibling");
    std::filesystem::remove(default_named);

    // Two release-named GGUFs cannot be narrowed down to one; that must be said plainly rather
    // than silently falling back to the safetensors source.
    std::filesystem::copy_file(gguf, model_dir / "vevo2-f16.gguf");
    engine::test::require(!engine::assets::find_directory_gguf(model_dir).has_value(),
                          "ambiguous GGUF directory resolved to a single file");
    bool ambiguity_reported = false;
    try {
        engine::model_spec::ScopedSpecOverride scope(std::nullopt, model_dir);
        (void)engine::model_spec::load_resource_bundle_for_family(model_dir, "named_gguf_test");
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        ambiguity_reported = message.find("contains 2 GGUF files") != std::string::npos &&
            message.find("vevo2-f16.gguf") != std::string::npos &&
            message.find("vevo2-q8_0.gguf") != std::string::npos;
    }
    engine::test::require(ambiguity_reported, "ambiguous GGUF directory did not report its candidates");

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
        (void)engine::model_spec::load_resource_bundle(root / "model", malformed_spec);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        malformed_named = message.find("failed to parse model spec") != std::string::npos &&
            message.find(malformed_spec.string()) != std::string::npos;
    }
    engine::test::require(malformed_named, "malformed package spec error did not name the selected spec");

    bool selected_source_named = false;
    try {
        (void)engine::model_spec::load_resource_bundle(root / "model", missing_file_spec);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        selected_source_named = message.find("using model spec") != std::string::npos &&
            message.find(missing_file_spec.string()) != std::string::npos &&
            message.find("source 'safetensors'") != std::string::npos &&
            message.find("missing model package file 'config'") != std::string::npos;
    }
    engine::test::require(selected_source_named, "resource error did not name the selected spec and source");

    std::filesystem::remove_all(root);
}

// The name reader is a shortcut past the content copy, not past the validation: a
// package whose byte ranges do not fit its blob is malformed whether the caller
// wants the names or the bytes. --inspect reads the names, and used to report a
// count for a package no loader could ever read (PR #579 review).
void test_malformed_embedded_sidecars_are_rejected_by_the_name_reader() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_malformed_sidecar_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";
    engine::io::write_safetensors_file(safetensors, {
                                                        {"weight", "F32", {1}, bytes_for(std::vector<float>{1.0F})},
                                                    });
    {
        std::ofstream output(root / "config.json", std::ios::binary);
        output << "{}";
    }
    engine::assets::convert_tensor_sources_to_gguf({{safetensors, ""}}, gguf, engine::assets::TensorStorageType::F16,
                                                   false, true, root, {{root / "config.json", "config.json"}});
    engine::test::require_eq(engine::assets::gguf_embedded_sidecar_names(gguf).size(), size_t{1},
                             "intact package sidecar names");

    auto * ctx = gguf_init_from_file(gguf.string().c_str(), {true, nullptr});
    engine::test::require(ctx != nullptr, "open fixture GGUF");
    const uint64_t offsets[] = {0, 999};
    gguf_set_arr_data(ctx, "audiocpp.embedded_files.offsets", GGUF_TYPE_UINT64, offsets, 2);
    engine::test::require(gguf_write_to_file(ctx, gguf.string().c_str(), true), "write malformed fixture");
    gguf_free(ctx);

    bool names_rejected = false;
    try {
        (void)engine::assets::gguf_embedded_sidecar_names(gguf);
    } catch (const std::runtime_error &) {
        names_rejected = true;
    }
    engine::test::require(names_rejected, "name reader accepted an out-of-range sidecar byte range");

    bool presence_rejected = false;
    try {
        (void)engine::assets::gguf_has_embedded_sidecars(gguf);
    } catch (const std::runtime_error &) {
        presence_rejected = true;
    }
    engine::test::require(presence_rejected, "presence check accepted an out-of-range sidecar byte range");

    std::filesystem::remove_all(root);
}

void test_weight_adapter_conversion() {
    using namespace engine;
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_gguf_adapter_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto original = root / "original.safetensors";
    io::write_safetensors_file(original, {
        {"projection.weight", "F32", {2, 32}, bytes_for(std::vector<float>(64, 0.25F))},
        {"untouched.weight", "F32", {2, 32}, bytes_for(std::vector<float>(64, 0.5F))},
        {"step", "F32", {}, bytes_for(std::vector<float>{42.0F})},
    });
    const auto base_path = root / "base.gguf";
    assets::convert_tensor_sources_to_gguf({{original, ""}}, base_path, assets::TensorStorageType::F16,
        false, false);
    const auto base = assets::open_tensor_source(base_path);
    const auto original_bytes = base->require_tensor_data("projection.weight").bytes;
    auto adapter = std::make_shared<assets::WeightAdapter>();
    adapter->updates.emplace("projection.weight", assets::WeightUpdate{
        assets::LowRankWeightUpdate{std::vector<float>(32, 0.5F), {1.0F, -1.0F}, 1, 32, 2},
        0.5F, assets::WeightMergeMode::DeltaF32});
    const auto overlay = assets::make_weight_adapter_source(base, *adapter);
    int index = 0;
    for (auto type : {assets::TensorStorageType::Native, assets::TensorStorageType::F16,
                     assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0}) {
        assets::GgufConversionOptions options;
        options.weight_adapter = adapter;
        const auto output = root / (std::to_string(index++) + ".gguf");
        assets::convert_tensor_sources_to_gguf({{base_path, ""}}, output, type, false, false,
            {}, {}, std::nullopt, options);
        const auto actual = assets::open_tensor_source(output);
        test::require(actual->require_tensor_data("step").bytes == base->require_tensor_data("step").bytes,
            "Adapter conversion must preserve untouched scalar metadata");
        for (const auto & name : {"projection.weight", "untouched.weight"}) {
            const auto expected = overlay->require_tensor(name, type, {2, 32});
            test::require(actual->require_tensor_data(name).bytes == expected.bytes,
                "GGUF adapter must merge before dtype conversion, including orig");
        }
    }
    test::require(base->require_tensor_data("projection.weight").bytes == original_bytes,
        "GGUF adapter conversion changed base weights");
    auto replacement = std::make_shared<assets::WeightAdapter>();
    replacement->replacements.emplace("projection.weight",
        assets::WeightReplacement{{2, 32}, std::vector<float>(64, 0.1001F)});
    assets::GgufConversionOptions replacement_options;
    replacement_options.weight_adapter = replacement;
    replacement_options.weight_adapter_merge_mode = assets::WeightMergeMode::RoundedBF16Delta;
    const auto replacement_path = root / "replacement.gguf";
    assets::convert_tensor_sources_to_gguf({{base_path, ""}}, replacement_path,
        assets::TensorStorageType::F32, false, false, {}, {}, std::nullopt, replacement_options);
    const auto rounded = assets::open_tensor_source(replacement_path)->require_f32("projection.weight");
    const float want = ggml_bf16_to_fp32(ggml_fp32_to_bf16(0.1001F));
    test::require(std::all_of(rounded.begin(), rounded.end(), [want](float value) { return value == want; }),
        "Replacement-only export must retain the selected BF16 merge policy");
    std::filesystem::remove_all(root);
}

void test_deferred_packed_weights() {
    using namespace engine;
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_deferred_packed_test";
    std::filesystem::create_directories(root);
    const auto path = root / "weights.safetensors";
    std::vector<float> a(64), b(32);
    for (size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<float>(i) - 21) / 32;
    for (size_t i = 0; i < b.size(); ++i) b[i] = (static_cast<float>(i) + 3) / 64;
    io::write_safetensors_file(path, {
        {"a.weight", "F32", {2, 32}, bytes_for(a)},
        {"b.weight", "F32", {1, 32}, bytes_for(b)},
        {"a.bias", "F32", {2}, bytes_for(std::vector<float>{1, -2})},
        {"b.bias", "F32", {1}, bytes_for(std::vector<float>{3})},
    });
    const auto source = assets::open_tensor_source(path);
    core::BackendConfig backend;
    backend.type = core::BackendType::Cpu;
    core::ExecutionContext execution(backend);
    for (const auto type : {assets::TensorStorageType::Native, assets::TensorStorageType::F32, assets::TensorStorageType::F16,
                            assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0}) {
        core::BackendWeightStore eager(execution.backend(), backend.type, "test.eager", 1024 * 1024);
        core::BackendWeightStore deferred(execution.backend(), backend.type, "test.deferred", 1024 * 1024);
        modules::PackedLinearWeightsConfig config{32,
            {{"a.weight", "a.bias", 2}, {"b.weight", "b.bias", 1}}, true};
        const auto expected = modules::PackedLinearWeightsBuilder(config).build(eager, *source, type);
        config.defer_upload = true;
        const auto actual = modules::PackedLinearWeightsBuilder(config).build(deferred, *source, type);
        eager.upload();
        deferred.upload();
        for (const auto & pair : {std::make_pair(expected.weight.tensor, actual.weight.tensor),
                                 std::make_pair(expected.bias->tensor, actual.bias->tensor)}) {
            const size_t bytes = ggml_nbytes(pair.first);
            test::require(bytes == ggml_nbytes(pair.second), "deferred packed tensor size");
            std::vector<std::byte> lhs(bytes), rhs(bytes);
            ggml_backend_tensor_get(pair.first, lhs.data(), 0, bytes);
            ggml_backend_tensor_get(pair.second, rhs.data(), 0, bytes);
            test::require(lhs == rhs, "deferred packed weights must preserve exact bytes");
        }
    }
    for (const auto & parts : std::vector<std::vector<core::BackendWeightStore::TensorPart>>{
             {}, {{"a.weight", {3, 32}}},
             {{"a.weight", {2, 32}}, {"b.weight", {1, 64}}}}) {
        core::BackendWeightStore invalid(execution.backend(), backend.type, "test.invalid", 1024 * 1024);
        bool rejected = false;
        try {
            invalid.load_tensor(*source, parts, assets::TensorStorageType::F32);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        test::require(rejected, "invalid packed tensor sources must be rejected before upload");
    }
    std::filesystem::remove_all(root);
}

class CountingTensorSource final : public engine::assets::TensorSource {
public:
    std::filesystem::path path{"counting://weight_norm"};
    mutable int direct_uploads = 0;
    const std::filesystem::path & source_path() const noexcept override { return path; }
    bool has_tensor(std::string_view name) const noexcept override {
        return name == "plain.weight" || name == "norm.weight_v" || name == "norm.weight_g";
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        if (name == "plain.weight") return {"plain.weight", "F32", {2}};
        if (name == "norm.weight_v") return {"norm.weight_v", "F32", {2, 2}};
        if (name == "norm.weight_g") return {"norm.weight_g", "F32", {2}};
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        return {require_metadata("plain.weight"), require_metadata("norm.weight_v"), require_metadata("norm.weight_g")};
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto values = require_f32(name, std::nullopt);
        engine::assets::RawTensorData data;
        data.metadata = require_metadata(name);
        data.bytes.resize(values.size() * sizeof(float));
        std::memcpy(data.bytes.data(), values.data(), data.bytes.size());
        return data;
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto metadata = require_metadata(name);
        if (expected_shape && *expected_shape != metadata.shape) throw std::runtime_error("shape mismatch");
        if (name == "plain.weight") return {3.0F, 4.0F};
        if (name == "norm.weight_v") return {3.0F, 4.0F, 0.0F, 2.0F};
        if (name == "norm.weight_g") return {10.0F, 6.0F};
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view) const override { throw std::runtime_error("no scalar"); }
    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        engine::assets::TensorStorageType type,
        const std::vector<int64_t> & shape) const override {
        ++direct_uploads;
        const auto metadata = require_metadata(name);
        if (type != engine::assets::TensorStorageType::F32 || shape != metadata.shape)
            throw std::runtime_error("unexpected direct upload type or shape");
        const auto values = require_f32(name, shape);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
};

void test_weight_norm_folded_source_uses_direct_backend_upload() {
    using namespace engine;
    using namespace engine::assets;
    auto backing = std::make_shared<CountingTensorSource>();
    const auto folded = make_weight_norm_folded_tensor_source(backing, {"norm.weight"});
    core::BackendConfig config;
    config.type = core::BackendType::Cpu;
    config.threads = 2;
    core::ExecutionContext execution(config);
    core::BackendWeightStore store(execution.backend(), config.type, "test.weight_norm", 1024 * 1024);
    auto plain = store.load_tensor(*folded, "plain.weight", TensorStorageType::F32, {2});
    auto norm = store.load_tensor(*folded, "norm.weight", TensorStorageType::F32, {2, 2});
    store.upload();
    ggml_backend_synchronize(execution.backend());
    test::require_eq(backing->direct_uploads, 1, "unfolded tensor must delegate its direct backend upload");
    std::vector<float> plain_values(2), norm_values(4);
    ggml_backend_tensor_get(plain.tensor, plain_values.data(), 0, plain_values.size() * sizeof(float));
    ggml_backend_tensor_get(norm.tensor, norm_values.data(), 0, norm_values.size() * sizeof(float));
    test::require(plain_values == std::vector<float>({3.0F, 4.0F}), "unfolded direct upload values");
    const std::vector<float> expected{6.0F, 8.0F, 0.0F, 6.0F};
    for (size_t i = 0; i < expected.size(); ++i)
        test::require_close(norm_values[i], expected[i], 1e-6F, "folded backend upload value");
    bool mismatch_rejected = false;
    try {
        (void)store.load_tensor(*folded, "norm.weight", TensorStorageType::F32, {4, 1});
    } catch (const std::runtime_error &) {
        mismatch_rejected = true;
    }
    test::require(mismatch_rejected, "folded backend upload must preserve shape validation");
}

}  // namespace

int main() {
    try {
        engine::io::json::enable_serialized_json_parsing();
        test_safetensors_to_gguf_roundtrip();
        test_concurrent_storage_release();
        test_source_policy_cache_and_views();
        test_packed_multi_source_gguf();
        test_all_rank0_gguf();
        test_embedded_model_spec_roundtrip_and_precedence();
        test_release_named_gguf_directory_selects_the_gguf_source();
        test_package_spec_errors_name_selected_spec();
        test_malformed_embedded_sidecars_are_rejected_by_the_name_reader();
        test_weight_adapter_conversion();
        test_deferred_packed_weights();
        test_weight_norm_folded_source_uses_direct_backend_upload();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "gguf_tensor_source_test passed\n";
    return 0;
}
