#include "engine/community_models/mms_forced_aligner/assets.h"
#include "test_assert.h"

#include "engine/framework/io/json.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using engine::test::require;
using engine::test::require_eq;

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() / "mms_assets_test";
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::filesystem::remove_all(path);
    }
    std::filesystem::path write(const char * name, const std::string & content) const {
        const auto file = path / name;
        std::ofstream out(file);
        out << content;
        return file;
    }
};

const char * kValidConfig = R"json({
  "model_type": "wav2vec2",
  "architectures": ["Wav2Vec2ForCTC"],
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "conv_dim": [512, 512, 512, 512, 512, 512, 512],
  "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
  "conv_stride": [5, 2, 2, 2, 2, 2, 2],
  "layer_norm_eps": 1.0e-5,
  "num_conv_pos_embeddings": 128,
  "num_conv_pos_embedding_groups": 16,
  "do_stable_layer_norm": true,
  "conv_bias": true,
  "feat_extract_norm": "layer",
  "hidden_act": "gelu",
  "vocab_size": 31,
  "pad_token_id": 0
})json";

const char * kValidVocab = R"json({
  "<blank>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
  "a": 4, "i": 5, "e": 6, "n": 7, "o": 8, "u": 9,
  "t": 10, "s": 11, "r": 12, "m": 13, "k": 14, "l": 15,
  "d": 16, "g": 17, "h": 18, "y": 19, "b": 20, "p": 21,
  "w": 22, "c": 23, "v": 24, "j": 25, "z": 26, "f": 27,
  "'": 28, "q": 29, "x": 30
})json";

engine::assets::ResourceBundle make_bundle(
    const TempDir & dir,
    const char * config_json,
    const char * vocab_json = kValidVocab) {
    engine::assets::ResourceBundle bundle(dir.path);
    bundle.add_file("config", dir.write("config.json", config_json));
    bundle.add_file("vocab", dir.write("vocab.json", vocab_json));
    return bundle;
}

void test_valid_config() {
    TempDir dir;
    const auto bundle = make_bundle(dir, kValidConfig);
    const auto parsed = engine::community_models::mms_forced_aligner::parse_mms_forced_aligner_configs(bundle);
    require_eq(parsed.model_config.hidden_size, int64_t{1024}, "hidden_size");
    require_eq(parsed.model_config.num_hidden_layers, int64_t{24}, "num_hidden_layers");
    require_eq(parsed.preprocessor_config.sampling_rate, int64_t{16000}, "sampling_rate");
    require(parsed.preprocessor_config.do_normalize, "preprocessor do_normalize must default to true");
    require_eq(parsed.vocabulary.blank_id, int32_t{0}, "blank_id");
    require_eq(parsed.vocabulary.token_to_id.at("'"), int32_t{28}, "apostrophe id");
    require_eq(parsed.vocabulary.id_to_token[static_cast<size_t>(30)], std::string("x"), "id 30 token");
}

void expect_rejected(const engine::assets::ResourceBundle & bundle) {
    bool threw = false;
    try {
        (void) engine::community_models::mms_forced_aligner::parse_mms_forced_aligner_configs(bundle);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "expected config validation to reject the input");
}

void test_wrong_model_type() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, "{\"model_type\": \"whisper\", \"architectures\": [\"Wav2Vec2ForCTC\"],"
             "\"hidden_size\": 1024, \"intermediate_size\": 4096, \"num_hidden_layers\": 24,"
             "\"num_attention_heads\": 16, \"conv_dim\": [512], \"conv_kernel\": [10],"
             "\"conv_stride\": [5], \"layer_norm_eps\": 1e-5, \"num_conv_pos_embeddings\": 128,"
             "\"num_conv_pos_embedding_groups\": 16, \"do_stable_layer_norm\": true,"
             "\"conv_bias\": true, \"feat_extract_norm\": \"layer\", \"hidden_act\": \"gelu\","
             "\"vocab_size\": 31, \"pad_token_id\": 0}");
    expect_rejected(bundle);
}

void test_missing_ctc_arch() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, "{\"model_type\": \"wav2vec2\", \"architectures\": [\"Wav2Vec2Model\"],"
             "\"hidden_size\": 1024, \"intermediate_size\": 4096, \"num_hidden_layers\": 24,"
             "\"num_attention_heads\": 16, \"conv_dim\": [512], \"conv_kernel\": [10],"
             "\"conv_stride\": [5], \"layer_norm_eps\": 1e-5, \"num_conv_pos_embeddings\": 128,"
             "\"num_conv_pos_embedding_groups\": 16, \"do_stable_layer_norm\": true,"
             "\"conv_bias\": true, \"feat_extract_norm\": \"layer\", \"hidden_act\": \"gelu\","
             "\"vocab_size\": 31, \"pad_token_id\": 0}");
    expect_rejected(bundle);
}

void test_unstable_layer_norm() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, "{\"model_type\": \"wav2vec2\", \"architectures\": [\"Wav2Vec2ForCTC\"],"
             "\"hidden_size\": 1024, \"intermediate_size\": 4096, \"num_hidden_layers\": 24,"
             "\"num_attention_heads\": 16, \"conv_dim\": [512], \"conv_kernel\": [10],"
             "\"conv_stride\": [5], \"layer_norm_eps\": 1e-5, \"num_conv_pos_embeddings\": 128,"
             "\"num_conv_pos_embedding_groups\": 16, \"do_stable_layer_norm\": false,"
             "\"conv_bias\": true, \"feat_extract_norm\": \"layer\", \"hidden_act\": \"gelu\","
             "\"vocab_size\": 31, \"pad_token_id\": 0}");
    expect_rejected(bundle);
}

void test_wrong_vocab_size() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, "{\"model_type\": \"wav2vec2\", \"architectures\": [\"Wav2Vec2ForCTC\"],"
             "\"hidden_size\": 1024, \"intermediate_size\": 4096, \"num_hidden_layers\": 24,"
             "\"num_attention_heads\": 16, \"conv_dim\": [512], \"conv_kernel\": [10],"
             "\"conv_stride\": [5], \"layer_norm_eps\": 1e-5, \"num_conv_pos_embeddings\": 128,"
             "\"num_conv_pos_embedding_groups\": 16, \"do_stable_layer_norm\": true,"
             "\"conv_bias\": true, \"feat_extract_norm\": \"layer\", \"hidden_act\": \"gelu\","
             "\"vocab_size\": 32, \"pad_token_id\": 0}");
    expect_rejected(bundle);
}

void test_wrong_sampling_rate() {
    TempDir dir;
    auto bundle = make_bundle(dir, kValidConfig);
    bundle.add_file(
        "preprocessor_config",
        dir.write("preprocessor_config.json", "{\"sampling_rate\": 8000, \"do_normalize\": true}"));
    expect_rejected(bundle);
}

void test_bad_stride_product() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, "{\"model_type\": \"wav2vec2\", \"architectures\": [\"Wav2Vec2ForCTC\"],"
             "\"hidden_size\": 1024, \"intermediate_size\": 4096, \"num_hidden_layers\": 24,"
             "\"num_attention_heads\": 16, \"conv_dim\": [512], \"conv_kernel\": [10],"
             "\"conv_stride\": [4], \"layer_norm_eps\": 1e-5, \"num_conv_pos_embeddings\": 128,"
             "\"num_conv_pos_embedding_groups\": 16, \"do_stable_layer_norm\": true,"
             "\"conv_bias\": true, \"feat_extract_norm\": \"layer\", \"hidden_act\": \"gelu\","
             "\"vocab_size\": 31, \"pad_token_id\": 0}");
    expect_rejected(bundle);
}

void test_duplicate_vocab_id() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, kValidConfig,
        "{\"<blank>\": 0, \"<pad>\": 1, \"</s>\": 2, \"<unk>\": 3, \"a\": 4, \"i\": 5,"
        " \"e\": 6, \"n\": 7, \"o\": 8, \"u\": 9, \"t\": 10, \"s\": 11, \"r\": 12,"
        " \"m\": 13, \"k\": 14, \"l\": 15, \"d\": 16, \"g\": 17, \"h\": 18, \"y\": 19,"
        " \"b\": 20, \"p\": 21, \"w\": 22, \"c\": 23, \"v\": 24, \"j\": 25, \"z\": 26,"
        " \"f\": 27, \"'\": 28, \"q\": 29, \"x\": 4}");
    expect_rejected(bundle);
}

void test_missing_vocab_id() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, kValidConfig,
        "{\"<blank>\": 0, \"<pad>\": 1, \"</s>\": 2, \"<unk>\": 3, \"a\": 4, \"i\": 5,"
        " \"e\": 6, \"n\": 7, \"o\": 8, \"u\": 9, \"t\": 10, \"s\": 11, \"r\": 12,"
        " \"m\": 13, \"k\": 14, \"l\": 15, \"d\": 16, \"g\": 17, \"h\": 18, \"y\": 19,"
        " \"b\": 20, \"p\": 21, \"w\": 22, \"c\": 23, \"v\": 24, \"j\": 25, \"z\": 26,"
        " \"f\": 27, \"'\": 28, \"q\": 29}");
    expect_rejected(bundle);
}

void test_blank_not_zero() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, kValidConfig,
        "{\"<blank>\": 1, \"<pad>\": 0, \"</s>\": 2, \"<unk>\": 3, \"a\": 4, \"i\": 5,"
        " \"e\": 6, \"n\": 7, \"o\": 8, \"u\": 9, \"t\": 10, \"s\": 11, \"r\": 12,"
        " \"m\": 13, \"k\": 14, \"l\": 15, \"d\": 16, \"g\": 17, \"h\": 18, \"y\": 19,"
        " \"b\": 20, \"p\": 21, \"w\": 22, \"c\": 23, \"v\": 24, \"j\": 25, \"z\": 26,"
        " \"f\": 27, \"'\": 28, \"q\": 29, \"x\": 30}");
    expect_rejected(bundle);
}

void test_swapped_vocab_ids_rejected() {
    // Same 31 tokens, same contiguity, but "a"/"i" swapped: shape-valid yet
    // a permuted mapping would silently send letters to the wrong CTC classes.
    TempDir dir;
    const auto bundle = make_bundle(
        dir, kValidConfig,
        "{\"<blank>\": 0, \"<pad>\": 1, \"</s>\": 2, \"<unk>\": 3, \"a\": 5, \"i\": 4, \"e\": 6,"
        " \"n\": 7, \"o\": 8, \"u\": 9, \"t\": 10, \"s\": 11, \"r\": 12, \"m\": 13,"
        " \"k\": 14, \"l\": 15, \"d\": 16, \"g\": 17, \"h\": 18, \"y\": 19, \"b\": 20,"
        " \"p\": 21, \"w\": 22, \"c\": 23, \"v\": 24, \"j\": 25, \"z\": 26, \"f\": 27,"
        " \"'\": 28, \"q\": 29, \"x\": 30}");
    expect_rejected(bundle);
}

void test_out_of_range_vocab_id() {
    TempDir dir;
    const auto bundle = make_bundle(
        dir, kValidConfig,
        "{\"<blank>\": 0, \"<pad>\": 1, \"</s>\": 2, \"<unk>\": 3, \"a\": 4, \"i\": 5,"
        " \"e\": 6, \"n\": 7, \"o\": 8, \"u\": 9, \"t\": 10, \"s\": 11, \"r\": 12,"
        " \"m\": 13, \"k\": 14, \"l\": 15, \"d\": 16, \"g\": 17, \"h\": 18, \"y\": 19,"
        " \"b\": 20, \"p\": 21, \"w\": 22, \"c\": 23, \"v\": 24, \"j\": 25, \"z\": 26,"
        " \"f\": 27, \"'\": 28, \"q\": 29, \"x\": 31}");
    expect_rejected(bundle);
}

}  // namespace

int main() {
    try {
        test_valid_config();
        test_wrong_model_type();
        test_missing_ctc_arch();
        test_unstable_layer_norm();
        test_wrong_vocab_size();
        test_wrong_sampling_rate();
        test_bad_stride_product();
        test_duplicate_vocab_id();
        test_missing_vocab_id();
        test_blank_not_zero();
        test_swapped_vocab_ids_rejected();
        test_out_of_range_vocab_id();
        std::cout << "mms_forced_aligner_assets_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_forced_aligner_assets_test: %s\n", error.what());
        return 1;
    }
}
