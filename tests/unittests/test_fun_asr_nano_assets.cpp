#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/safetensors.h"
#include "engine/models/fun_asr_nano/assets.h"
#include "test_assert.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ResourceOptions {
  bool include_tokenizer = true;
  bool include_stem = true;
  bool include_embedding = true;
};

std::vector<unsigned char> float_bytes(float value) {
  std::vector<unsigned char> bytes(sizeof(value));
  std::memcpy(bytes.data(), &value, sizeof(value));
  return bytes;
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary);
  output << text;
}

std::string processor_config() {
  return R"json({
          "audio_token":"<|object_ref_start|>",
          "default_transcription_prompt":"Transcribe the audio:",
          "feature_extractor":{
            "feature_size":80,
            "sampling_rate":16000,
            "frame_length":25,
            "frame_shift":10,
            "lfr_m":7,
            "lfr_n":6,
            "preemphasis":0.97,
            "return_attention_mask":true
          }
        })json";
}

engine::assets::ResourceBundle make_resources(const std::filesystem::path &root,
                                              const std::string &config_json,
                                              ResourceOptions options = {}) {
  std::filesystem::create_directories(root);
  write_text(root / "config.json", config_json);
  write_text(root / "generation_config.json",
             R"json({"eos_token_id":151645,"bos_token_id":151643})json");
  write_text(root / "processor_config.json", processor_config());
  if (options.include_tokenizer) {
    write_text(root / "tokenizer.json",
               R"json({"version":"1.0","added_tokens":[]})json");
  }
  std::vector<engine::io::SafeTensorWriteEntry> tensors;
  if (options.include_stem) {
    tensors.push_back({
        "model.audio_tower.stem.self_attn.q_proj.weight",
        "F32",
        {1},
        float_bytes(1.0F),
    });
  }
  if (options.include_embedding) {
    tensors.push_back({
        "model.language_model.embed_tokens.weight",
        "F32",
        {1},
        float_bytes(1.0F),
    });
  }
  engine::io::write_safetensors_file(root / "model.safetensors", tensors);

  engine::assets::ResourceBundle resources(root);
  resources.add_file("config", root / "config.json");
  resources.add_file("generation_config", root / "generation_config.json");
  resources.add_file("processor_config", root / "processor_config.json");
  if (options.include_tokenizer) {
    resources.add_file("tokenizer_json", root / "tokenizer.json");
  }
  resources.add_tensor_source("weights", root / "model.safetensors");
  return resources;
}

std::string current_config() {
  return R"json({
      "model_type":"fun_asr_nano",
      "audio_token_id":151646,
      "projector_hidden_size":2048,
      "tie_word_embeddings":true,
      "encoder_config":{
        "model_type":"fun_asr_nano_encoder",
        "num_mel_bins":80,
        "num_stacked_frames":7,
        "d_model":512,
        "encoder_attention_heads":4,
        "encoder_ffn_dim":2048,
        "encoder_layers":50,
        "num_timestamp_prediction_blocks":20,
        "kernel_size":11,
        "activation_function":"relu",
        "max_position_embeddings":2049
      },
      "adaptor_config":{
        "d_model":1024,
        "encoder_attention_heads":8,
        "encoder_ffn_dim":256,
        "encoder_layers":2,
        "activation_function":"relu"
      },
      "text_config":{
        "model_type":"qwen3",
        "vocab_size":151936,
        "hidden_size":1024,
        "intermediate_size":3072,
        "num_hidden_layers":28,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        "head_dim":128,
        "max_position_embeddings":40960,
        "rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":1000000},
        "tie_word_embeddings":true
      }
    })json";
}

std::string legacy_config() {
  return R"json({
      "model_type":"fun_asr_nano",
      "audio_token_id":151646,
      "activation_function":"relu",
      "adaptor_intermediate_size":2048,
      "adaptor_num_attention_heads":8,
      "adaptor_num_hidden_layers":2,
      "tie_word_embeddings":true,
      "encoder_config":{
        "model_type":"fun_asr_nano_encoder",
        "num_mel_bins":80,
        "num_stacked_frames":7,
        "d_model":512,
        "encoder_attention_heads":4,
        "encoder_ffn_dim":2048,
        "encoder_layers":50,
        "num_timestamp_prediction_blocks":20,
        "kernel_size":11,
        "activation_function":"relu"
      },
      "text_config":{
        "model_type":"qwen3",
        "vocab_size":151936,
        "hidden_size":1024,
        "intermediate_size":3072,
        "num_hidden_layers":28,
        "num_attention_heads":16,
        "num_key_value_heads":8,
        "head_dim":128,
        "max_position_embeddings":40960,
        "rms_norm_eps":0.000001,
        "rope_parameters":{"rope_theta":1000000},
        "tie_word_embeddings":true
      }
    })json";
}

std::string replace_once(std::string value, const std::string &from,
                         const std::string &to) {
  const auto position = value.find(from);
  if (position == std::string::npos) {
    throw std::runtime_error("test fixture marker not found: " + from);
  }
  value.replace(position, from.size(), to);
  return value;
}

void require_rejected(const std::filesystem::path &root,
                      const std::string &expected_message,
                      const std::function<void()> &action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error &error) {
    rejected =
        std::string(error.what()).find(expected_message) != std::string::npos;
  }
  engine::test::require(rejected,
                        "Fun-ASR did not reject invalid assets with: " +
                            expected_message);
  std::filesystem::remove_all(root);
}

void require_published_dimensions(
    const engine::models::fun_asr_nano::FunAsrNanoConfig &config) {
  engine::test::require_eq(config.model_type, std::string("fun_asr_nano"),
                           "Fun-ASR model type");
  engine::test::require_eq(config.encoder.input_size, int64_t{560},
                           "Fun-ASR encoder input size");
  engine::test::require_eq(config.encoder.d_model, int64_t{512},
                           "Fun-ASR encoder width");
  engine::test::require_eq(config.encoder.layers, int64_t{50},
                           "Fun-ASR main blocks");
  engine::test::require_eq(config.encoder.timestamp_prediction_layers,
                           int64_t{20}, "Fun-ASR timestamp blocks");
  engine::test::require_eq(config.encoder.attention_heads, int64_t{4},
                           "Fun-ASR encoder heads");
  engine::test::require_eq(config.encoder.ffn_dim, int64_t{2048},
                           "Fun-ASR encoder FFN");
  engine::test::require_eq(config.encoder.kernel_size, int64_t{11},
                           "Fun-ASR FSMN kernel");
  engine::test::require_eq(config.encoder.max_position_embeddings,
                           int64_t{2049}, "Fun-ASR encoder position limit");
  engine::test::require_eq(config.projector_hidden_size, int64_t{2048},
                           "Fun-ASR projector width");
  engine::test::require_eq(config.adaptor.d_model, int64_t{1024},
                           "Fun-ASR adaptor width");
  engine::test::require_eq(config.adaptor.attention_heads, int64_t{8},
                           "Fun-ASR adaptor heads");
  engine::test::require_eq(config.adaptor.ffn_dim, int64_t{256},
                           "Fun-ASR adaptor FFN");
  engine::test::require_eq(config.adaptor.layers, int64_t{2},
                           "Fun-ASR adaptor layers");
  engine::test::require_eq(config.text.hidden_size, int64_t{1024},
                           "Fun-ASR text width");
  engine::test::require_eq(config.text.intermediate_size, int64_t{3072},
                           "Fun-ASR text FFN");
  engine::test::require_eq(config.text.layers, int64_t{28},
                           "Fun-ASR text layers");
  engine::test::require_eq(config.text.attention_heads, int64_t{16},
                           "Fun-ASR text heads");
  engine::test::require_eq(config.text.key_value_heads, int64_t{8},
                           "Fun-ASR KV heads");
  engine::test::require_eq(config.text.head_dim, int64_t{128},
                           "Fun-ASR text head dimension");
  engine::test::require_eq(config.text.vocab_size, int64_t{151936},
                           "Fun-ASR vocabulary");
  engine::test::require_close(config.text.rope_theta, 1000000.0F, 0.0F,
                              "Fun-ASR RoPE theta");
  engine::test::require_eq(config.frontend.sample_rate, 16000,
                           "Fun-ASR sample rate");
  engine::test::require_eq(config.frontend.feature_size, int64_t{80},
                           "Fun-ASR frontend feature size");
  engine::test::require_eq(config.frontend.frame_length_ms, int64_t{25},
                           "Fun-ASR frontend frame length");
  engine::test::require_eq(config.frontend.frame_shift_ms, int64_t{10},
                           "Fun-ASR frontend frame shift");
  engine::test::require_eq(config.frontend.lfr_m, int64_t{7},
                           "Fun-ASR LFR stack");
  engine::test::require_eq(config.frontend.lfr_n, int64_t{6},
                           "Fun-ASR LFR stride");
  engine::test::require_close(config.frontend.preemphasis, 0.97F, 0.0F,
                              "Fun-ASR frontend preemphasis");
}

void test_current_nested_config() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_nested_assets_test";
  std::filesystem::remove_all(root);
  auto assets = engine::models::fun_asr_nano::load_fun_asr_nano_assets(
      make_resources(root, current_config()));
  require_published_dimensions(assets->config);
  std::filesystem::remove_all(root);
}

void test_legacy_flat_adaptor_config() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_legacy_assets_test";
  std::filesystem::remove_all(root);
  auto assets = engine::models::fun_asr_nano::load_fun_asr_nano_assets(
      make_resources(root, legacy_config()));
  require_published_dimensions(assets->config);
  std::filesystem::remove_all(root);
}

void test_loads_model_directory_through_family_spec() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_model_spec_test";
  std::filesystem::remove_all(root);
  (void)make_resources(root, current_config());
  auto assets = engine::models::fun_asr_nano::load_fun_asr_nano_assets(root);
  require_published_dimensions(assets->config);
  // Compare canonical forms: the loader stores the resolved root, while the
  // path we passed in may differ lexically yet denote the same directory
  // (macOS /var symlink to /private/var, Windows TEMP 8.3 short names).
  engine::test::require_eq(
      std::filesystem::weakly_canonical(assets->resources.model_root()),
      std::filesystem::weakly_canonical(root),
      "Fun-ASR model root");
  std::filesystem::remove_all(root);
}

void test_rejects_incompatible_adaptor_width() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_invalid_assets_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"d_model\":1024", "\"d_model\":768");
  require_rejected(root, "adaptor", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_adaptor_shape() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_adaptor_shape_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"encoder_attention_heads\":8",
                   "\"encoder_attention_heads\":16");
  require_rejected(root, "published adaptor architecture", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_encoder_shape() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_encoder_shape_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"d_model\":512", "\"d_model\":768");
  require_rejected(root, "published encoder architecture", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_encoder_position_limit() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_encoder_position_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"max_position_embeddings\":2049",
                   "\"max_position_embeddings\":1024");
  require_rejected(root, "published encoder architecture", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_text_shape() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_text_shape_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"intermediate_size\":3072",
                   "\"intermediate_size\":4096");
  require_rejected(root, "published Qwen architecture", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_zero_text_attention_heads() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_zero_text_heads_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"num_attention_heads\":16",
                   "\"num_attention_heads\":0");
  require_rejected(root, "text dimensions", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_projector_shape() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_projector_shape_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"projector_hidden_size\":2048",
                   "\"projector_hidden_size\":4096");
  require_rejected(root, "projector hidden size", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_nonpublished_frontend_parameters() {
  const std::vector<std::pair<std::string, std::string>> mutations = {
      {"\"frame_length\":25", "\"frame_length\":30"},
      {"\"frame_shift\":10", "\"frame_shift\":20"},
      {"\"lfr_n\":6", "\"lfr_n\":5"},
      {"\"preemphasis\":0.97", "\"preemphasis\":0.50"},
  };
  for (size_t index = 0; index < mutations.size(); ++index) {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("audiocpp_fun_asr_frontend_shape_test_" + std::to_string(index));
    std::filesystem::remove_all(root);
    auto resources = make_resources(root, current_config());
    write_text(root / "processor_config.json",
               replace_once(processor_config(), mutations[index].first,
                            mutations[index].second));
    require_rejected(root, "published frontend", [&] {
      (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
          std::move(resources));
    });
  }
}

void test_rejects_unsupported_activation() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_activation_assets_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"activation_function\":\"relu\"",
                   "\"activation_function\":\"gelu\"");
  require_rejected(root, "ReLU encoder", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_untied_embeddings() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_untied_assets_test";
  std::filesystem::remove_all(root);
  const auto invalid =
      replace_once(current_config(), "\"tie_word_embeddings\":true",
                   "\"tie_word_embeddings\":false");
  require_rejected(root, "tied text embeddings", [&] {
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, invalid));
  });
}

void test_rejects_missing_tokenizer() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_tokenizer_assets_test";
  std::filesystem::remove_all(root);
  require_rejected(root, "tokenizer_json", [&] {
    ResourceOptions options;
    options.include_tokenizer = false;
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, current_config(), options));
  });
}

void test_rejects_missing_stem_tensor() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_stem_assets_test";
  std::filesystem::remove_all(root);
  require_rejected(root, "audio_tower.stem", [&] {
    ResourceOptions options;
    options.include_stem = false;
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, current_config(), options));
  });
}

void test_rejects_missing_token_embedding() {
  const auto root = std::filesystem::temp_directory_path() /
                    "audiocpp_fun_asr_embedding_assets_test";
  std::filesystem::remove_all(root);
  require_rejected(root, "embed_tokens", [&] {
    ResourceOptions options;
    options.include_embedding = false;
    (void)engine::models::fun_asr_nano::load_fun_asr_nano_assets(
        make_resources(root, current_config(), options));
  });
}

} // namespace

int main() {
  try {
    test_current_nested_config();
    test_legacy_flat_adaptor_config();
    test_loads_model_directory_through_family_spec();
    test_rejects_incompatible_adaptor_width();
    test_rejects_nonpublished_adaptor_shape();
    test_rejects_nonpublished_encoder_shape();
    test_rejects_nonpublished_encoder_position_limit();
    test_rejects_nonpublished_text_shape();
    test_rejects_zero_text_attention_heads();
    test_rejects_nonpublished_projector_shape();
    test_rejects_nonpublished_frontend_parameters();
    test_rejects_unsupported_activation();
    test_rejects_untied_embeddings();
    test_rejects_missing_tokenizer();
    test_rejects_missing_stem_tensor();
    test_rejects_missing_token_embedding();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "fun_asr_nano_assets_test passed\n";
  return 0;
}
