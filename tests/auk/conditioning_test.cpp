#include "engine/community_models/auk/conditioning.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/model_spec/package.h"

#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <sstream>

int main(int argc, char ** argv) {
    if ((argc != 5 && argc != 6 && argc != 7 && argc != 9) || std::string(argv[3]) != "--log" ||
        (argc == 6 && std::string(argv[5]) != "--audio") ||
        (argc >= 7 && std::string(argv[5]) != "--encode" && !(argc == 7 && std::string(argv[5]) == "--audio-file")) ||
        (argc == 9 && std::string(argv[7]) != "--alternate-fixture" && std::string(argv[7]) != "--gguf" &&
         std::string(argv[7]) != "--native-audio")) {
        std::cerr << "Usage: auk_conditioning_test <qwen-dir> <fixture-dir> --log <log-file> [--audio|--audio-file WAV|--encode <auk-dir> [--alternate-fixture DIR|--gguf FILE|--native-audio WAV]]\n";
        return 2;
    }
    engine::debug::configure_logging({true, std::string(argv[4])});
    try {
        const std::filesystem::path qwen_dir = argv[1];
        const std::filesystem::path fixture_dir = argv[2];
        if (argc == 6 || (argc == 7 && std::string(argv[5]) == "--audio-file")) {
            const auto reference = engine::assets::open_tensor_source(fixture_dir / "reference.safetensors");
            const auto length_data = reference->require_tensor_data("qwen.audio.feature_lens");
            if (length_data.metadata.dtype != "I64" || length_data.bytes.size() != sizeof(int64_t)) {
                throw std::runtime_error("expected one audio feature length");
            }
            int64_t frames;
            std::memcpy(&frames, length_data.bytes.data(), sizeof(frames));
            const auto padded_shape = reference->require_metadata("condition.input_features").shape;
            if (padded_shape.size() != 3 || padded_shape[0] != 1 || padded_shape[1] != 128 || frames > padded_shape[2]) {
                throw std::runtime_error("unexpected audio feature shape");
            }
            const auto padded = reference->require_f32("condition.input_features");
            std::vector<float> features(128 * frames);
            for (int channel = 0; channel < 128; ++channel) {
                std::copy_n(padded.begin() + channel * padded_shape[2], frames, features.begin() + channel * frames);
            }
            if (argc == 7) {
                const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[6]));
                const auto extracted = engine::models::auk::extract_audio_features(wav.samples, wav.sample_rate, wav.channels, 8);
                if (extracted.frames != frames || extracted.values.size() != features.size()) {
                    throw std::runtime_error("native audio feature frame count differs from Python");
                }
                double squared_error = 0, maximum_error = 0;
                for (size_t i = 0; i < features.size(); ++i) {
                    const double delta = double(extracted.values[i]) - features[i];
                    if (!std::isfinite(delta)) throw std::runtime_error("non-finite audio features");
                    squared_error += delta * delta;
                    maximum_error = std::max(maximum_error, std::abs(delta));
                }
                const double rmse = std::sqrt(squared_error / features.size());
                engine::io::SafeTensorWriteEntry entry;
                entry.name = "features";
                entry.dtype = "F32";
                entry.shape = {128, frames};
                entry.data.resize(extracted.values.size() * sizeof(float));
                std::memcpy(entry.data.data(), extracted.values.data(), entry.data.size());
                engine::io::write_safetensors_file(std::string(argv[4]) + ".features.safetensors", {entry});
                std::ostringstream message;
                message << std::setprecision(12) << "AuK native mel: RMSE=" << rmse << " max_abs=" << maximum_error;
                engine::debug::log_message(message.str());
                std::cout << message.str() << '\n';
                if (!(rmse < 1.0e-5 && maximum_error < 1.0e-3)) throw std::runtime_error("audio feature parity gate failed");
                features = extracted.values;
            }
            const auto qwen = engine::assets::open_tensor_source(qwen_dir / "model.safetensors.index.json");
            engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
            engine::models::auk::AudioConditioningRuntime runtime(execution, *qwen, frames);
            const auto actual = runtime.encode(features);
            const auto expected = reference->require_f32("qwen.audio_output");
            if (actual.size() != expected.size()) throw std::runtime_error("audio tower output shape mismatch");
            double ab = 0, aa = 0, bb = 0, error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("non-finite audio tower output");
                ab += double(actual[i]) * expected[i];
                aa += double(actual[i]) * actual[i];
                bb += double(expected[i]) * expected[i];
                error += std::pow(double(actual[i]) - expected[i], 2);
            }
            const double cosine = ab / std::sqrt(aa * bb);
            std::ostringstream message;
            message << std::setprecision(12) << "AuK Qwen audio tower: cosine=" << cosine
                    << " RMSE=" << std::sqrt(error / actual.size());
            engine::debug::log_message(message.str());
            std::cout << message.str() << '\n';
            if (!(cosine > 0.99999)) throw std::runtime_error("audio tower parity gate failed");
            runtime.prepare(frames);
            if (runtime.encode(features) != actual) throw std::runtime_error("audio tower reused output differs");
            return 0;
        }
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_config_path = qwen_dir / "tokenizer_config.json";
        spec.tokenizer_json_path = qwen_dir / "tokenizer.json";
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        const auto tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
        const auto manifest = engine::io::json::parse_file(fixture_dir / "manifest.json");
        const auto reference = engine::assets::open_tensor_source(fixture_dir / "reference.safetensors");
        const int64_t audio_tokens = reference->has_tensor("qwen.audio_output")
            ? reference->require_metadata("qwen.audio_output").shape.at(0) : 0;
        const auto input = engine::models::auk::prepare_conditioning(
            *tokenizer, manifest.require("instruction").as_string(), audio_tokens);
        const auto read_i64 = [&reference](const std::string & name) {
            const auto raw = reference->require_tensor_data(name);
            if (raw.metadata.dtype != "I64" || raw.bytes.size() % sizeof(int64_t) != 0) {
                throw std::runtime_error("expected I64 fixture tensor: " + name);
            }
            std::vector<int64_t> values(raw.bytes.size() / sizeof(int64_t));
            std::memcpy(values.data(), raw.bytes.data(), raw.bytes.size());
            return values;
        };
        const auto ids = read_i64("condition.input_ids");
        const auto mask = read_i64("condition.attention_mask");
        const auto positions = read_i64("qwen.position_ids");
        if (ids.size() != input.token_ids.size() || mask.size() != ids.size() || positions.size() != 3 * ids.size()) {
            throw std::runtime_error("conditioning input lengths do not match Python");
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != input.token_ids[i] || mask[i] != input.attention_mask[i]) {
                throw std::runtime_error("token or mask mismatch at " + std::to_string(i));
            }
            for (size_t axis = 0; axis < 3; ++axis) {
                if (positions[axis * ids.size() + i] != input.positions[i]) {
                    throw std::runtime_error("position mismatch at " + std::to_string(i));
                }
            }
        }
        if (audio_tokens == 0) {
            const auto marked = engine::models::auk::prepare_conditioning(
                *tokenizer, manifest.require("instruction").as_string() + "|<no_prompt_audio>|");
            if (marked.token_ids != input.token_ids) throw std::runtime_error("no-reference marker was duplicated");
        }
        engine::debug::trace_log_i32("auk.condition.input_ids", {1, static_cast<int64_t>(ids.size())}, input.token_ids);
        const auto message = "AuK conditioning tokens/mask/positions match Python: " + std::to_string(ids.size()) + " tokens";
        engine::debug::log_message(message);
        std::cout << message << '\n';
        if (argc >= 7) {
            auto qwen = engine::assets::open_tensor_source(qwen_dir / "model.safetensors.index.json");
            auto auk = engine::assets::open_tensor_source(std::filesystem::path(argv[6]) / "auk_base.safetensors");
            if (argc == 9 && std::string(argv[7]) == "--gguf") {
                const auto embedded = engine::assets::read_gguf_embedded_model_spec(argv[8]);
                const auto current_spec = engine::model_spec::load_contract_spec(engine::model_spec::default_spec_path("auk"));
                if (!embedded || embedded->family != "auk") throw std::runtime_error("GGUF has no AuK spec");
                const auto packaged_spec = engine::io::json::parse(embedded->json);
                using JsonValue = engine::io::json::Value;
                std::vector<std::pair<const JsonValue *, const JsonValue *>> pending{{&packaged_spec, &current_spec}};
                while (!pending.empty()) {
                    const auto [packaged, current] = pending.back();
                    pending.pop_back();
                    if (packaged->kind() != current->kind()) throw std::runtime_error("GGUF spec value type differs");
                    if (current->is_object()) {
                        if (packaged->as_object().size() != current->as_object().size()) throw std::runtime_error("GGUF spec keys differ");
                        for (const auto & [key, value] : current->as_object()) {
                            pending.emplace_back(&packaged->require(key), &value);
                        }
                    } else if (current->is_array()) {
                        if (packaged->as_array().size() != current->as_array().size()) throw std::runtime_error("GGUF spec array sizes differ");
                        for (size_t i = 0; i < current->as_array().size(); ++i) {
                            pending.emplace_back(&packaged->as_array()[i], &current->as_array()[i]);
                        }
                    } else if (engine::io::json::stringify(*packaged) != engine::io::json::stringify(*current)) {
                        throw std::runtime_error("GGUF embedded AuK spec differs from the current local spec");
                    }
                }
                engine::debug::log_message("AuK GGUF embedded spec matches the current local spec");
                const auto gguf = engine::assets::open_tensor_source(argv[8]);
                const auto vae = engine::assets::open_tensor_source(std::filesystem::path(argv[6]) / "vae.safetensors");
                const std::vector<std::pair<std::string, std::shared_ptr<const engine::assets::TensorSource>>> originals{
                    {"model", auk}, {"qwen", qwen}, {"vae", vae}};
                size_t total = 0;
                for (const auto & component : originals) {
                    const auto converted = engine::assets::make_prefixed_tensor_source(gguf, component.first);
                    size_t retained = 0;
                    for (const auto & metadata : component.second->tensors()) {
                        if (component.first == "qwen" &&
                            (metadata.name.rfind("talker.", 0) == 0 || metadata.name.rfind("token2wav.", 0) == 0 ||
                             metadata.name.rfind("thinker.visual.", 0) == 0 || metadata.name.rfind("thinker.lm_head.", 0) == 0)) continue;
                        const auto original = component.second->require_tensor_data(metadata.name);
                        const auto actual = converted->require_tensor_data(metadata.name);
                        if (original.metadata.shape != actual.metadata.shape ||
                            engine::assets::ggml_type_for_tensor_dtype(original.metadata.dtype) != engine::assets::ggml_type_for_tensor_dtype(actual.metadata.dtype) ||
                            original.bytes != actual.bytes) throw std::runtime_error("GGUF tensor differs from original: " + component.first + "/" + metadata.name);
                        ++retained;
                    }
                    if (converted->tensors().size() != retained) throw std::runtime_error("GGUF has unexpected tensors in " + component.first);
                    total += retained;
                    engine::debug::log_message("GGUF exact tensor match: " + component.first + " count=" + std::to_string(retained));
                }
                if (gguf->tensors().size() != total) throw std::runtime_error("GGUF has unexpected tensor namespaces");
                engine::debug::log_message("GGUF exact original-dtype byte comparison passed: " + std::to_string(total) + " tensors");
                qwen = engine::assets::make_prefixed_tensor_source(gguf, "qwen");
                auk = engine::assets::make_prefixed_tensor_source(gguf, "model");
            }
            engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
            const bool bf16 = manifest.require("dtype").as_string() == "bf16";
            const bool native_audio = argc == 9 && std::string(argv[7]) == "--native-audio";
            auto audio_embeddings = audio_tokens > 0 ? reference->require_f32("qwen.audio_output") : std::vector<float>{};
            std::unique_ptr<engine::models::auk::AudioConditioningRuntime> audio_runtime;
            if (native_audio) {
                if (audio_tokens == 0 || bf16) throw std::runtime_error("native audio test requires an FP32 audio fixture");
                const auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[8]));
                const auto features = engine::models::auk::extract_audio_features(wav.samples, wav.sample_rate, wav.channels, 8);
                audio_runtime = std::make_unique<engine::models::auk::AudioConditioningRuntime>(execution, *qwen, features.frames);
                audio_embeddings = audio_runtime->encode(features.values);
                if (audio_embeddings.size() != static_cast<size_t>(audio_tokens * 2048)) {
                    throw std::runtime_error("native audio token count differs from Python");
                }
                const auto native_input = engine::models::auk::prepare_conditioning(*tokenizer,
                    manifest.require("instruction").as_string(), audio_embeddings.size() / 2048);
                if (native_input.token_ids != input.token_ids) throw std::runtime_error("native audio prompt differs from Python");
            }
            engine::models::auk::ConditioningRuntime runtime(execution, *qwen, *auk, ids.size(),
                reference->has_tensor("qwen.layer.0") || audio_tokens > 0, bf16, audio_tokens);
            const auto output = runtime.encode(input, audio_embeddings);
            const auto layers = runtime.captured_layers();
            bool merged_embedding_pass = true;
            if (audio_tokens > 0) {
                const auto expected_embeddings = reference->require_f32("qwen.inputs_embeds");
                const auto & merged = layers.at(36);
                if (merged.size() != expected_embeddings.size()) throw std::runtime_error("merged embedding shape differs");
                if (!native_audio && merged != expected_embeddings) {
                    throw std::runtime_error("merged embeddings differ from Python with captured audio features");
                }
                if (native_audio) {
                    double error = 0, dot = 0, norm = 0, expected_norm = 0;
                    size_t audio_values = 0;
                    for (size_t i = 0; i < merged.size(); ++i) {
                        const double delta = double(merged[i]) - expected_embeddings[i];
                        if (!std::isfinite(delta)) throw std::runtime_error("non-finite merged embedding");
                        if (input.token_ids[i / 2048] != 151646) {
                            if (delta != 0.0) throw std::runtime_error("audio merge changed a text embedding");
                        } else {
                            error += delta * delta;
                            dot += double(merged[i]) * expected_embeddings[i];
                            norm += double(merged[i]) * merged[i];
                            expected_norm += double(expected_embeddings[i]) * expected_embeddings[i];
                            ++audio_values;
                        }
                    }
                    const double rmse = std::sqrt(error / audio_values);
                    const double relative_l2 = std::sqrt(error / expected_norm);
                    const double audio_cosine = dot / std::sqrt(norm * expected_norm);
                    engine::debug::trace_log_scalar("auk.native_audio.merged_embedding_rmse", rmse);
                    engine::debug::trace_log_scalar("auk.native_audio.embedding_relative_l2", relative_l2);
                    engine::debug::trace_log_scalar("auk.native_audio.embedding_cosine", audio_cosine);
                    // Audio activations vary in scale; measure only audio rows, without dilution by exact text rows.
                    merged_embedding_pass = relative_l2 < 1.0e-4 && audio_cosine > 0.99999999;
                } else {
                    engine::debug::log_message("AuK merged audio/text embeddings are bit-identical to Python");
                }
            }
            std::vector<engine::io::SafeTensorWriteEntry> attention_fixture;
            for (size_t layer = 0; reference->has_tensor("qwen.layer.0") && layer < layers.size(); ++layer) {
                const char * projections[] = {"q_proj", "k_proj", "v_proj"};
                const bool is_projection = layer > 36 && layer < 40;
                if (is_projection || layer == 40) {
                    auto values = layers[layer];
                    if (is_projection && bf16) {
                        for (auto & value : values) value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
                    }
                    engine::io::SafeTensorWriteEntry entry;
                    entry.name = layer == 40 ? "attention_output" : projections[layer - 37];
                    entry.dtype = "F32";
                    entry.shape = {1, static_cast<int64_t>(ids.size()),
                                   static_cast<int64_t>(values.size() / ids.size())};
                    entry.data.resize(values.size() * sizeof(float));
                    std::memcpy(entry.data.data(), values.data(), entry.data.size());
                    attention_fixture.push_back(std::move(entry));
                }
                const auto expected_layer = reference->require_f32(
                    layer == 40 ? "qwen.attention_output" : layer == 36 ? "qwen.inputs_embeds" : is_projection ?
                    "qwen.layer." + std::string(projections[layer - 37]) : "qwen.layer." + std::to_string(layer));
                if (layer == 36 && layers[layer] != expected_layer) {
                    throw std::runtime_error("input embeddings do not exactly match Python");
                }
                double dot = 0, norm = 0, expected_norm = 0, error = 0;
                double double_round_error = 0;
                const auto bias = is_projection ? qwen->require_f32(
                    "thinker.model.layers.0.self_attn." + std::string(projections[layer - 37]) + ".bias") : std::vector<float>{};
                for (size_t i = 0; i < expected_layer.size(); ++i) {
                    const double x = is_projection && bf16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(layers[layer][i])) : layers[layer][i];
                    const double y = expected_layer[i];
                    dot += x * y;
                    norm += x * x;
                    expected_norm += y * y;
                    error += (x - y) * (x - y);
                    if (is_projection) {
                        const float b = bias[i % bias.size()];
                        const float product = ggml_bf16_to_fp32(ggml_fp32_to_bf16(layers[layer][i] - b));
                        const float alternate = ggml_bf16_to_fp32(ggml_fp32_to_bf16(product + b));
                        double_round_error += (alternate - y) * (alternate - y);
                    }
                }
                engine::debug::log_message("layer=" + std::to_string(layer) + " cosine=" +
                    std::to_string(dot / std::sqrt(norm * expected_norm)) + " rmse=" +
                    std::to_string(std::sqrt(error / expected_layer.size())));
                if (is_projection) {
                    engine::debug::log_message("projection=" + std::string(projections[layer - 37]) +
                        " round_before_bias_rmse=" + std::to_string(std::sqrt(double_round_error / expected_layer.size())));
                }
            }
            if (!attention_fixture.empty()) {
                engine::io::write_safetensors_file(std::string(argv[4]) + ".attention.safetensors", attention_fixture);
            }
            const auto expected = reference->require_f32("fm.first.text", {1, static_cast<int64_t>(ids.size()), 2048});
            double dot = 0, norm = 0, expected_norm = 0, squared_error = 0;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (!std::isfinite(output[i])) {
                    throw std::runtime_error("non-finite conditioning output");
                }
                dot += double(output[i]) * expected[i];
                norm += double(output[i]) * output[i];
                expected_norm += double(expected[i]) * expected[i];
                squared_error += std::pow(double(output[i]) - expected[i], 2);
            }
            const double cosine = dot / std::sqrt(norm * expected_norm);
            engine::debug::trace_log_scalar("auk.conditioning.cosine", cosine);
            engine::debug::trace_log_scalar("auk.conditioning.rmse", std::sqrt(squared_error / expected.size()));
            engine::debug::log_message("conditioning cosine=" + std::to_string(cosine) +
                                      " rmse=" + std::to_string(std::sqrt(squared_error / expected.size())));
            std::cout << "conditioning cosine=" << cosine << '\n';
            if (runtime.encode(input, audio_embeddings) != output) {
                throw std::runtime_error("reused conditioning graph changed output");
            }
            if (!(cosine > 0.99999)) {
                throw std::runtime_error("conditioning parity gate failed");
            }
            if (!merged_embedding_pass) throw std::runtime_error("native merged embedding parity gate failed");
            if (argc == 9 && std::string(argv[7]) == "--alternate-fixture") {
                const std::filesystem::path alternate_directory = argv[8];
                const auto alternate_manifest = engine::io::json::parse_file(alternate_directory / "manifest.json");
                if (alternate_manifest.require("dtype").as_string() != manifest.require("dtype").as_string()) {
                    throw std::runtime_error("alternate conditioning fixture dtype differs");
                }
                const auto alternate_input = engine::models::auk::prepare_conditioning(
                    *tokenizer, alternate_manifest.require("instruction").as_string());
                if (alternate_input.token_ids.size() == ids.size()) {
                    throw std::runtime_error("alternate conditioning fixture must change token count");
                }
                const auto alternate = engine::assets::open_tensor_source(alternate_directory / "reference.safetensors");
                const auto expected_alternate = alternate->require_f32("fm.first.text",
                    {1, static_cast<int64_t>(alternate_input.token_ids.size()), 2048});
                for (int cycle = 0; cycle < 3; ++cycle) {
                    runtime.prepare(alternate_input.token_ids.size());
                    const auto actual = runtime.encode(alternate_input);
                    const double ab = std::inner_product(actual.begin(), actual.end(), expected_alternate.begin(), 0.0,
                        std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                    const double aa = std::inner_product(actual.begin(), actual.end(), actual.begin(), 0.0,
                        std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                    const double bb = std::inner_product(expected_alternate.begin(), expected_alternate.end(), expected_alternate.begin(), 0.0,
                        std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                    const double similarity = ab / std::sqrt(aa * bb);
                    engine::debug::trace_log_scalar("auk.conditioning.alternate_cosine", similarity);
                    if (!(similarity > 0.99999)) throw std::runtime_error("alternate conditioning parity gate failed");
                    runtime.prepare(alternate_input.token_ids.size());
                    if (runtime.encode(alternate_input) != actual) throw std::runtime_error("same-shape prepare changed conditioning");
                    runtime.prepare(ids.size());
                    if (runtime.encode(input) != output) throw std::runtime_error("restored conditioning shape changed output");
                    const auto memory = execution.memory_snapshot();
                    if (memory.available) engine::debug::trace_log_scalar("auk.conditioning.cycle_device_used_bytes", memory.used_bytes);
                }
                engine::debug::log_message("AuK conditioning shape switching passed three round trips");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        engine::debug::log_message(engine::debug::LogLevel::Error, "auk", error.what());
        std::cerr << error.what() << '\n';
        return 1;
    }
}
