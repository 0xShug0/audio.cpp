#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/model_package.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/safetensors.h"
#include "engine/models/citrinet_asr/assets.h"
#include "engine/models/hviske_asr/assets.h"
#include "test_assert.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<unsigned char> float_bytes(float value) {
    std::vector<unsigned char> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
}

void write_dummy_weights(const std::filesystem::path & path) {
    engine::io::write_safetensors_file(path, {
        {"dummy.weight", "F32", {1}, float_bytes(1.0F)},
    });
}

void write_quantizable_matrix(const std::filesystem::path & path) {
    std::vector<float> values(32 * 256, 0.25F);
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    engine::io::write_safetensors_file(path, {
        {"conv.weight", "F32", {32, 256}, std::move(bytes)},
    });
}

void test_hviske_standalone_gguf() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_hviske_standalone_test";
    std::filesystem::remove_all(root);
    const auto native = root / "native";
    const auto packed = root / "unrelated";
    std::filesystem::create_directories(native);
    std::filesystem::create_directories(packed);
    write_dummy_weights(native / "model.safetensors");
    write_text(native / "config.json", R"json({
        "model_type":"cohere_asr","architectures":["CohereAsrForConditionalGeneration"],
        "max_audio_clip_s":35,"overlap_chunk_second":5,"min_energy_window_samples":1600,
        "supported_languages":["da"],"vocab_size":10,
        "preprocessor":{"sample_rate":16000,"features":128,"n_fft":512,"window_size":0.025,"window_stride":0.01,"pad_to":16,"dither":0.0},
        "encoder":{"feat_in":128,"d_model":8,"ff_expansion_factor":4,"n_layers":1,"n_heads":1,"conv_kernel_size":3,"subsampling_conv_channels":8,"subsampling_factor":1,"pos_emb_max_len":16},
        "transf_decoder":{"config_dict":{"hidden_size":8,"inner_size":16,"num_layers":1,"num_attention_heads":1,"max_sequence_length":16}},
        "head":{"log_softmax":true}
    })json");
    write_text(native / "generation_config.json", R"json({
        "pad_token_id":2,"eos_token_id":3,"bos_token_id":4,"decoder_start_token_id":5
    })json");
    std::filesystem::copy_file(
        std::filesystem::path(ENGINE_TEST_ASSET_ROOT) / "tokenizers" / "tokenizer-1.model",
        native / "tokenizer.model");

    const auto gguf = packed / "renamed-hviske.gguf";
    engine::assets::convert_tensor_source_to_gguf(
        native / "model.safetensors",
        gguf,
        engine::assets::TensorStorageType::F16);
    const auto assets = engine::models::hviske_asr::load_hviske_assets(gguf);
    engine::test::require_eq(assets->config.model_type, std::string("cohere_asr"), "Hviske model type");
    engine::test::require(assets->model_weights->has_tensor("dummy.weight"), "Hviske standalone GGUF weights");
    engine::test::require(!assets->tokenizer_pieces.empty(), "Hviske embedded tokenizer");

    std::filesystem::copy_file(gguf, native / "model.gguf");
    const auto explicitly_native = engine::models::hviske_asr::load_hviske_assets(native / "model.safetensors");
    engine::test::require_eq(
        explicitly_native->model_weights->require_metadata("dummy.weight").dtype,
        std::string("F32"),
        "Hviske explicit safetensors selection");
    std::filesystem::remove_all(root);
}

void test_citrinet_standalone_gguf() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_citrinet_standalone_test";
    std::filesystem::remove_all(root);
    const auto native = root / "native";
    const auto packed = root / "unrelated";
    std::filesystem::create_directories(native);
    std::filesystem::create_directories(packed);
    write_dummy_weights(native / "citrinet_256.safetensors");
    write_text(native / "citrinet_256_config.json", R"json({"vocab_file":"citrinet_256_vocab.txt"})json");
    write_text(native / "citrinet_256_tokenizer.model", "test tokenizer sidecar");
    // Legacy sidecar: the spec no longer references vocab.txt (CTC ids decode
    // through tokenizer.model), but previously converted model directories
    // still contain it — loading must ignore the stray file, not choke on it.
    write_text(native / "citrinet_256_vocab.txt", "a\nb\n");

    const auto gguf = packed / "renamed-citrinet.gguf";
    engine::assets::convert_tensor_source_to_gguf(
        native / "citrinet_256.safetensors",
        gguf,
        engine::assets::TensorStorageType::F16);
    const auto assets = engine::assets::load_resource_bundle_from_package_spec(
        gguf,
        engine::assets::default_model_package_spec_path("citrinet_asr"));
    engine::test::require_eq(
        assets.require_file("weights"),
        std::filesystem::weakly_canonical(gguf),
        "Citrinet standalone GGUF checkpoint");
    engine::test::require_eq(
        assets.require_file("config").filename().string(),
        std::string("citrinet_256_config.json"),
        "Citrinet embedded config");
    engine::test::require_eq(
        assets.require_file("tokenizer").filename().string(),
        std::string("citrinet_256_tokenizer.model"),
        "Citrinet embedded tokenizer");

    const auto native_assets = engine::assets::load_resource_bundle_from_package_spec(
        native / "citrinet_256.safetensors",
        engine::assets::default_model_package_spec_path("citrinet_asr"));
    engine::test::require_eq(
        native_assets.open_tensor_source("weights")->require_metadata("dummy.weight").dtype,
        std::string("F32"),
        "Citrinet explicit safetensors selection");
    std::filesystem::remove_all(root);
}

void test_native_q8_matrix_reshape_falls_back_to_f32() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_q8_conv_reshape_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";
    write_quantizable_matrix(safetensors);
    engine::assets::convert_tensor_source_to_gguf(
        safetensors,
        gguf,
        engine::assets::TensorStorageType::Q8_0,
        false,
        false);
    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require_eq(
        source->require_metadata("conv.weight").dtype,
        std::string("q8_0"),
        "quantized reshape source type");
    auto * backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    {
        engine::core::BackendWeightStore store(
            backend,
            engine::core::BackendType::Cpu,
            "q8_conv_reshape_test",
            1024 * 1024);
        const auto weight = store.load_tensor_as_shape(
            *source,
            "conv.weight",
            engine::assets::TensorStorageType::Native,
            {32, 256},
            engine::core::TensorShape::from_dims({32, 256, 1}));
        engine::test::require_eq(weight.tensor->type, GGML_TYPE_F32, "quantized reshape backend type");
        store.upload();
    }
    ggml_backend_free(backend);
    std::filesystem::remove_all(root);
}

void test_native_q8_matrix_reshape_preserves_q8_when_block_aligned() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_q8_block_aligned_reshape_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto safetensors = root / "model.safetensors";
    const auto gguf = root / "model.gguf";
    write_quantizable_matrix(safetensors);
    engine::assets::convert_tensor_source_to_gguf(
        safetensors,
        gguf,
        engine::assets::TensorStorageType::Q8_0,
        false,
        false);
    const auto source = engine::assets::open_tensor_source(gguf);
    engine::test::require_eq(
        source->require_metadata("conv.weight").dtype,
        std::string("q8_0"),
        "quantized block-aligned source type");
    auto * backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    {
        engine::core::BackendWeightStore store(
            backend,
            engine::core::BackendType::Cpu,
            "q8_block_aligned_reshape_test",
            1024 * 1024);
        const auto weight = store.load_tensor_as_shape(
            *source,
            "conv.weight",
            engine::assets::TensorStorageType::Native,
            {32, 256},
            engine::core::TensorShape::from_dims({32, 256}));
        engine::test::require_eq(weight.tensor->type, GGML_TYPE_Q8_0, "block-aligned native q8 backend type");
        store.upload();
    }
    ggml_backend_free(backend);
    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_hviske_standalone_gguf();
        test_citrinet_standalone_gguf();
        test_native_q8_matrix_reshape_falls_back_to_f32();
        test_native_q8_matrix_reshape_preserves_q8_when_block_aligned();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "asr_standalone_gguf_test passed\n";
    return 0;
}
