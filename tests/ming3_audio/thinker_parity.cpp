#include "engine/models/ming3_audio/backbone.h"
#include "engine/models/ming3_audio/audio_encoder.h"
#include "engine/models/ming3_audio/frontend.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    try {
        std::filesystem::path weights_path, config_path, fixture_path, audio_path, request_path;
        bool log = false;
        bool disk_experts = false;
        int64_t capacity = 0;
        int64_t prefill_tokens = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--log") log = true;
            else if (arg == "--disk-experts") disk_experts = true;
            else if ((arg == "--capacity" || arg == "--prefill-tokens") && i + 1 < argc) {
                const std::string value = argv[++i];
                size_t consumed = 0;
                const auto number = std::stoll(value, &consumed);
                if (number <= 0 || consumed != value.size()) throw std::runtime_error(arg + " requires a positive integer");
                if (arg == "--capacity") capacity = number;
                else prefill_tokens = number;
            }
            else if ((arg == "--weights" || arg == "--config" || arg == "--reference" || arg == "--audio" || arg == "--request") && i + 1 < argc) {
                const std::filesystem::path value = argv[++i];
                if (arg == "--weights") weights_path = value;
                else if (arg == "--config") config_path = value;
                else if (arg == "--reference") fixture_path = value;
                else if (arg == "--request") request_path = value;
                else audio_path = value;
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (!log || weights_path.empty() || config_path.empty() || fixture_path.empty()) {
            throw std::runtime_error("usage: ming3_audio_thinker_parity --weights <gguf> --config <json> --reference <safetensors> [--disk-experts] [--capacity N] [--prefill-tokens N] --log");
        }
        engine::debug::configure_logging({true, std::nullopt});
        const auto config = engine::models::ming3_audio::parse_ming_audio_config(engine::io::json::parse_file(config_path));
        const auto fixture = engine::assets::open_tensor_source(fixture_path);
        std::array<std::vector<int32_t>, 2> token_arrays;
        const std::array<std::string, 2> names = {"input_ids", "generated_ids"};
        for (size_t i = 0; i < names.size(); ++i) {
            const auto raw = fixture->require_tensor_data(names[i]);
            if (raw.metadata.dtype != "I64") throw std::runtime_error("Expected I64 reference token IDs");
            for (size_t offset = 0; offset < raw.bytes.size(); offset += sizeof(int64_t)) {
                int64_t value;
                std::memcpy(&value, raw.bytes.data() + offset, sizeof(value));
                token_arrays[i].push_back(static_cast<int32_t>(value));
            }
        }
        const auto & ids = token_arrays[0];
        const auto & expected = token_arrays[1];
        std::vector<float> audio;
        if (fixture->has_tensor("audio_embeddings")) audio = fixture->require_f32("audio_embeddings");
        engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
        const auto source = engine::assets::open_tensor_source(weights_path, "thinker");
        if (!audio_path.empty()) {
            const auto assets = engine::models::ming3_audio::load_ming3_audio_assets(weights_path);
            engine::models::ming3_audio::Frontend frontend(assets);
            const auto wav = engine::audio::read_wav_f32(audio_path);
            const auto features = frontend.extract({wav.sample_rate, wav.channels, wav.samples}, 8);
            std::string instruction = "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.";
            std::string system;
            bool thinking = false;
            if (!request_path.empty()) {
                const auto request = engine::io::json::parse_file(request_path);
                const auto & opts = request.require("requests").as_array().at(0).require("options");
                instruction = engine::io::json::optional_string(opts, "prompt", instruction);
                system = engine::io::json::optional_string(opts, "system_prompt", "");
                thinking = engine::io::json::optional_bool(opts, "enable_thinking", false);
            }
            const auto prompt = frontend.prompt(instruction, features.frames, system, thinking);
            if (ids != prompt) throw std::runtime_error("C++ frontend token IDs differ from the official processor");
            std::cout << "prompt_ids_identical=true\n";
            engine::models::ming3_audio::AudioEncoder encoder(execution, source, config.audio, config.thinker.hidden_size);
            const auto encoded = encoder.encode(features.log_mel, features.frames);
            if (encoded.size() != audio.size()) throw std::runtime_error("C++ audio embedding count differs from Python");
            double dot = 0, lhs = 0, rhs = 0;
            for (size_t i = 0; i < audio.size(); ++i) {
                dot += static_cast<double>(encoded[i]) * audio[i];
                lhs += static_cast<double>(encoded[i]) * encoded[i];
                rhs += static_cast<double>(audio[i]) * audio[i];
            }
            std::cout << "audio_embedding_cosine=" << dot / std::sqrt(lhs * rhs) << '\n';
            audio = encoded;
        }
        const auto required_capacity = static_cast<int64_t>(ids.size() + expected.size() + 1);
        if (capacity == 0) capacity = required_capacity;
        if (capacity < required_capacity) throw std::runtime_error("--capacity cannot hold the input and expected output");
        engine::models::ming3_audio::ThinkerOptions options;
        options.disk_experts = disk_experts;
        engine::models::ming3_audio::Thinker thinker(execution, source, config.thinker, capacity, options);
        engine::debug::trace_log_scalar("ming3_audio.test.capacity", capacity);
        engine::debug::trace_log_scalar("ming3_audio.test.disk_experts", disk_experts);
        engine::debug::trace_log_scalar("ming3_audio.test.prefill_tokens", prefill_tokens);
        const auto start = std::chrono::steady_clock::now();
        const auto audio_id = static_cast<int32_t>(config.audio.token_id);
        if (audio.size() != static_cast<size_t>(std::count(ids.begin(), ids.end(), audio_id)) * config.thinker.hidden_size) {
            throw std::runtime_error("Audio embedding size does not match the input token placeholders");
        }
        std::vector<float> logits;
        if (prefill_tokens == 0) {
            logits = thinker.prefill(ids, audio, audio_id);
        } else {
            size_t audio_offset = 0;
            for (size_t offset = 0; offset < ids.size(); offset += static_cast<size_t>(prefill_tokens)) {
                const std::vector<int32_t> chunk(ids.begin() + offset,
                    ids.begin() + std::min(ids.size(), offset + static_cast<size_t>(prefill_tokens)));
                const auto count = static_cast<size_t>(std::count(chunk.begin(), chunk.end(), audio_id)) * config.thinker.hidden_size;
                const std::vector<float> projected(audio.begin() + audio_offset, audio.begin() + audio_offset + count);
                logits = thinker.prefill(chunk, projected, audio_id);
                audio_offset += count;
            }
        }
        std::vector<int32_t> generated;
        for (size_t i = 0; i < expected.size(); ++i) {
            const auto token = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            generated.push_back(token);
            if (token == 156892 || token == 156895) break;
            logits = thinker.step(token);
        }
        const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
        size_t matches = 0;
        while (matches < generated.size() && matches < expected.size() && generated[matches] == expected[matches]) ++matches;
        std::cout << "generated_ids=";
        for (auto token : generated) std::cout << token << ',';
        std::cout << "\nmatching_prefix=" << matches << "/" << expected.size() << '\n';
        return generated == expected ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "ming3_audio_thinker_parity failed: " << error.what() << '\n';
        return 2;
    }
}
