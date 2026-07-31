#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/session.h"
#include "engine/models/qwen3_tts/talker.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::qwen3_tts::Qwen3TTSConfig;

Qwen3TTSConfig valid_config() {
  Qwen3TTSConfig config;
  config.variant = engine::models::qwen3_tts::Qwen3TTSVariant::Base;
  config.tokenizer_type = "qwen3_tts_tokenizer_12hz";
  config.max_new_tokens = 64;

  config.talker.max_position_embeddings = 4096;
  config.talker.hidden_size = 128;
  config.talker.text_hidden_size = 128;
  config.talker.text_vocab_size = 1024;
  config.talker.intermediate_size = 256;
  config.talker.num_hidden_layers = 2;
  config.talker.num_attention_heads = 4;
  config.talker.num_key_value_heads = 2;
  config.talker.head_dim = 32;
  config.talker.num_code_groups = 4;
  config.talker.vocab_size = 2048;

  config.code_predictor.hidden_size = 128;
  config.code_predictor.intermediate_size = 256;
  config.code_predictor.num_hidden_layers = 2;
  config.code_predictor.num_attention_heads = 4;
  config.code_predictor.num_key_value_heads = 2;
  config.code_predictor.head_dim = 32;
  config.code_predictor.vocab_size = 2048;

  config.speech_tokenizer.model_type = "qwen3_tts_tokenizer_12hz";
  config.speech_tokenizer.input_sample_rate = 24000;
  config.speech_tokenizer.output_sample_rate = 24000;
  config.speech_tokenizer.num_quantizers = 4;
  config.speech_tokenizer.codebook_size = 2048;
  config.speech_tokenizer.semantic_codebook_size = 2048;

  config.has_speaker_encoder = true;
  config.speaker_encoder.embedding_dim = 128;
  config.speaker_encoder.sample_rate = 24000;
  return config;
}

void expect_throw(const std::function<void()> &fn, const std::string &label) {
  try {
    fn();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(label + " did not throw");
}

void test_config_validation() {
  auto config = valid_config();
  engine::models::qwen3_tts::validate_qwen3_tts_config(config);

  auto invalid = config;
  invalid.talker.num_attention_heads = 0;
  expect_throw(
      [&] { engine::models::qwen3_tts::validate_qwen3_tts_config(invalid); },
      "zero talker attention heads");

  invalid = config;
  invalid.code_predictor.head_dim = 31;
  engine::models::qwen3_tts::validate_qwen3_tts_config(invalid);

  invalid = config;
  invalid.code_predictor.head_dim = 0;
  expect_throw(
      [&] { engine::models::qwen3_tts::validate_qwen3_tts_config(invalid); },
      "non-positive code predictor head dimension");

  invalid = config;
  invalid.speaker_encoder.embedding_dim = 64;
  expect_throw(
      [&] { engine::models::qwen3_tts::validate_qwen3_tts_config(invalid); },
      "speaker encoder and talker dimension mismatch");

  invalid = config;
  invalid.speech_tokenizer.num_quantizers = 3;
  expect_throw(
      [&] { engine::models::qwen3_tts::validate_qwen3_tts_config(invalid); },
      "mismatched codec group count");
}

void test_generation_options() {
  const auto config = valid_config();
  engine::runtime::TaskRequest request;
  request.options["max_tokens"] = "8";
  request.options["seed"] = "123";
  request.options["do_sample"] = "false";
  request.options["subtalker_do_sample"] = "false";
  request.options["temperature"] = "0";
  request.options["subtalker_temperature"] = "0";
  request.options["top_k"] = "0";
  request.options["subtalker_top_k"] = "0";
  request.options["top_p"] = "0";
  request.options["subtalker_top_p"] = "0";
  request.options["repetition_penalty"] = "1.1";

  const auto options =
      engine::models::qwen3_tts::qwen3_tts_generation_options_from_request(
          request, config);
  if (options.max_new_tokens != 8 || options.seed != 123 || options.do_sample ||
      options.subtalker_do_sample || options.top_p != 0.0F ||
      options.subtalker_top_p != 0.0F) {
    throw std::runtime_error(
        "canonical Qwen3 TTS generation options were not parsed");
  }

  request.options["do_sample"] = "true";
  expect_throw(
      [&] {
        (void)engine::models::qwen3_tts::
            qwen3_tts_generation_options_from_request(request, config);
      },
      "zero sampling temperature");

  request.options["do_sample"] = "false";
  request.options["top_p"] = "nan";
  expect_throw(
      [&] {
        (void)engine::models::qwen3_tts::
            qwen3_tts_generation_options_from_request(request, config);
      },
      "non-finite top_p");

  request.options["top_p"] = "-0.1";
  expect_throw(
      [&] {
        (void)engine::models::qwen3_tts::
            qwen3_tts_generation_options_from_request(request, config);
      },
      "negative top_p");

  request.options["top_p"] = "1";
  request.options["subtalker_top_p"] = "-0.1";
  expect_throw(
      [&] {
        (void)engine::models::qwen3_tts::
            qwen3_tts_generation_options_from_request(request, config);
      },
      "negative subtalker_top_p");

  request.options["subtalker_top_p"] = "1";
  request.options["subtalker_top_k"] = "-1";
  expect_throw(
      [&] {
        (void)engine::models::qwen3_tts::
            qwen3_tts_generation_options_from_request(request, config);
      },
      "negative subtalker_top_k");
}

void test_voice_clone_prefill_mode_validation() {
  engine::models::qwen3_tts::Qwen3TalkerPrefill prefill;
  prefill.speaker_embedding = engine::models::qwen3_tts::Qwen3SpeakerEmbedding{
      std::vector<float>(128, 0.0F), 128};
  prefill.x_vector_only_mode = true;
  engine::models::qwen3_tts::validate_qwen3_talker_voice_clone_prefill(
      prefill, 128);

  prefill.x_vector_only_mode = false;
  prefill.icl_mode = true;
  expect_throw(
      [&] {
        engine::models::qwen3_tts::validate_qwen3_talker_voice_clone_prefill(
            prefill, 128);
      },
      "ICL prefill without reference state");

  prefill.reference_ids = {1};
  prefill.reference_codes = engine::models::qwen3_tts::Qwen3SpeechCodes{};
  engine::models::qwen3_tts::validate_qwen3_talker_voice_clone_prefill(
      prefill, 128);
}

} // namespace

int main() {
  try {
    test_config_validation();
    test_generation_options();
    test_voice_clone_prefill_mode_validation();
    std::cout << "qwen3_tts_options_config_test: ok\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "qwen3_tts_options_config_test: failed: " << ex.what() << '\n';
    return 1;
  }
}
