#include "engine/community_models/auk/flow.h"
#include "engine/community_models/auk/vae.h"
#include "engine/community_models/auk/conditioning.h"
#include "engine/community_models/auk/session.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/io/json.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>

int main(int argc, char ** argv) {
    if ((argc != 5 && argc != 6 && argc != 7 && argc != 9) || std::string(argv[3]) != "--log" ||
        (argc >= 7 && std::string(argv[5]) != "--instruct" && std::string(argv[5]) != "--session") ||
        (argc == 9 && std::string(argv[7]) != "--benchmark-runs" && std::string(argv[7]) != "--alternate-fixture" && std::string(argv[7]) != "--task" && std::string(argv[7]) != "--session-lifetimes") ||
        (argc == 6 && std::string(argv[5]) != "--cfg" && std::string(argv[5]) != "--cfg-explicit" && std::string(argv[5]) != "--sample" && std::string(argv[5]) != "--sample-explicit")) {
        std::cerr << "Usage: auk_flow_test <auk_base.safetensors> <fixture-dir> --log <log-file> [--cfg|--cfg-explicit|--sample|--sample-explicit|--instruct <qwen-dir> [--benchmark-runs N|--alternate-fixture DIR]|--session <qwen-dir> [--benchmark-runs N|--alternate-fixture DIR|--task gen|--session-lifetimes N]]\n";
        return 2;
    }
    engine::debug::configure_logging({true, std::string(argv[4])});
    try {
        const bool instruct = argc >= 7;
        const bool alternate_fixture = argc == 9 && std::string(argv[7]) == "--alternate-fixture";
        const bool explicit_task = argc == 9 && std::string(argv[7]) == "--task";
        const bool lifetime_test = argc == 9 && std::string(argv[7]) == "--session-lifetimes";
        if (explicit_task && std::string(argv[5]) != "--session") throw std::runtime_error("--task requires --session");
        if (lifetime_test && std::string(argv[5]) != "--session") throw std::runtime_error("--session-lifetimes requires --session");
        const int lifetimes = lifetime_test ? std::stoi(argv[8]) : 1;
        if (lifetimes <= 0) throw std::runtime_error("session lifetimes must be positive");
        const bool benchmark = argc == 9 && std::string(argv[7]) == "--benchmark-runs";
        const int benchmark_runs = benchmark ? std::stoi(argv[8]) : 0;
        if (benchmark && benchmark_runs <= 0) throw std::runtime_error("benchmark runs must be positive");
        const bool cfg = argc >= 6;
        const bool explicit_attention = instruct || (cfg && (std::string(argv[5]) == "--sample-explicit" || std::string(argv[5]) == "--cfg-explicit"));
        const bool sample = instruct || (cfg && (std::string(argv[5]) == "--sample" || std::string(argv[5]) == "--sample-explicit"));
        const auto source = engine::assets::open_tensor_source(argv[1]);
        const std::filesystem::path directory = argv[2];
        const auto inputs = engine::assets::open_tensor_source(directory / "reference.safetensors");
        const auto expected = engine::assets::open_tensor_source(directory / (sample ? "reference.safetensors" : cfg ? "flow-cfg.safetensors" : "flow-embeddings.safetensors"));
        if (argc >= 7 && std::string(argv[5]) == "--session") {
            const std::filesystem::path qwen_dir = argv[6];
            auto assets = std::make_shared<engine::models::auk::AukAssets>();
            assets->model = source;
            assets->qwen = engine::assets::open_tensor_source(qwen_dir / "model.safetensors.index.json");
            assets->vae = engine::assets::open_tensor_source(std::filesystem::path(argv[1]).parent_path() / "vae.safetensors");
            engine::tokenizers::LlamaBpeTokenizerSpec spec;
            spec.tokenizer_config_path = qwen_dir / "tokenizer_config.json";
            spec.tokenizer_json_path = qwen_dir / "tokenizer.json";
            spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
            assets->tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
            engine::runtime::SessionOptions options;
            options.backend = {engine::core::BackendType::Cuda, 0, 8};
            options.options["auk.dtype"] = "fp32";
            engine::core::ExecutionContext memory_observer(options.backend);
            const auto before_session = memory_observer.memory_snapshot();
            if (!before_session.available) throw std::runtime_error("CUDA memory snapshot unavailable");
            engine::debug::trace_log_scalar("auk.session.device_used_before_bytes", before_session.used_bytes);
            auto session = std::make_unique<engine::models::auk::AukSession>(
                engine::runtime::TaskSpec{explicit_task ? engine::runtime::parse_voice_task_kind(argv[8]) : engine::runtime::VoiceTaskKind::Tts,
                 engine::runtime::RunMode::Offline}, options, assets, engine::runtime::require_model_contract("auk"));
            session->prepare({});
            std::vector<std::filesystem::path> fixtures{directory};
            if (alternate_fixture) {
                fixtures.emplace_back(argv[8]);
                fixtures.push_back(directory);
            }
            std::vector<float> first_output;
            engine::runtime::TaskRequest lifetime_request;
            for (size_t index = 0; index < fixtures.size(); ++index) {
                const auto manifest = engine::io::json::parse_file(fixtures[index] / "manifest.json");
                if (manifest.require("dtype").as_string() != "fp32") throw std::runtime_error("session fixture must be FP32");
                engine::runtime::TaskRequest request;
                request.text_input = engine::runtime::Transcript{manifest.require("instruction").as_string(), ""};
                for (const auto & message : manifest.require("messages").as_array()) {
                    for (const auto & content : message.require("content").as_array()) {
                        if (content.require("type").as_string() == "audio") {
                            if (request.audio_input) throw std::runtime_error("session fixture requires a single reference audio");
                            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(content.require("audio").as_string()));
                            request.audio_input = engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
                        }
                    }
                }
                request.options = {
                    {"duration", engine::io::json::stringify_number(manifest.require("duration").as_number())},
                    {"seed", std::to_string(manifest.require("seed").as_i64())},
                    {"num_inference_steps", std::to_string(manifest.require("steps").as_i64())},
                    {"cfg_scale", engine::io::json::stringify_number(manifest.require("cfg").as_number())},
                };
                if (const auto * sway = manifest.find("sway")) {
                    request.options["sway_sampling_coef"] = engine::io::json::stringify_number(sway->as_number());
                }
                const auto result = session->run(request);
                if (!result.audio_output || result.audio_output->sample_rate != 24000 || result.audio_output->channels != 1) {
                    throw std::runtime_error("session audio output contract failed");
                }
                const auto & actual = result.audio_output->samples;
                const auto fixture = engine::assets::open_tensor_source(fixtures[index] / "reference.safetensors");
                const auto reference = fixture->require_f32("waveform");
                if (actual.size() != reference.size()) throw std::runtime_error("session waveform length mismatch");
                double dot = 0, norm = 0, reference_norm = 0, error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    dot += double(actual[i]) * reference[i];
                    norm += double(actual[i]) * actual[i];
                    reference_norm += double(reference[i]) * reference[i];
                    error += std::pow(double(actual[i]) - reference[i], 2);
                }
                const double cosine = dot / std::sqrt(norm * reference_norm);
                engine::debug::trace_log_scalar("auk.session.waveform_cosine", cosine);
                engine::debug::trace_log_scalar("auk.session.waveform_rmse", std::sqrt(error / actual.size()));
                if (!(cosine > 0.99999)) throw std::runtime_error("session waveform parity gate failed");
                if (session->run(request).audio_output->samples != actual) throw std::runtime_error("session repeat output changed");
                if (request.audio_input && !explicit_task) {
                    auto voice_request = request;
                    voice_request.voice.emplace();
                    voice_request.voice->speaker.emplace();
                    voice_request.voice->speaker->audio = voice_request.audio_input;
                    voice_request.audio_input.reset();
                    if (session->run(voice_request).audio_output->samples != actual) {
                        throw std::runtime_error("speaker reference differs from audio_input reference");
                    }
                    engine::debug::log_message("AuK speaker-audio and audio_input references produce identical samples");
                }
                if (benchmark_runs) {
                    std::vector<double> milliseconds;
                    for (int run = 0; run < benchmark_runs + 2; ++run) {
                        const auto start = std::chrono::steady_clock::now();
                        const auto repeated = session->run(request);
                        const double elapsed = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
                        if (!repeated.audio_output || repeated.audio_output->samples != actual) {
                            throw std::runtime_error("session benchmark waveform changed");
                        }
                        if (run >= 2) {
                            milliseconds.push_back(elapsed);
                            engine::debug::trace_log_scalar("auk.session.inference_ms", elapsed);
                        }
                    }
                    std::sort(milliseconds.begin(), milliseconds.end());
                    const size_t middle = milliseconds.size() / 2;
                    const double median = (milliseconds[middle] + milliseconds[(milliseconds.size() - 1) / 2]) / 2;
                    engine::debug::trace_log_scalar("auk.session.inference_median_ms", median);
                    engine::debug::trace_log_scalar("auk.session.rtf", median * 24.0 / actual.size());
                }
                if (index == 0) {
                    first_output = actual;
                    if (lifetime_test) lifetime_request = request;
                }
                if (index == 2 && actual != first_output) throw std::runtime_error("session restored shape output changed");
                engine::audio::write_pcm16_wav(fixtures[index] / "native-session.wav", 24000, 1, actual);
                if (index + 1 == fixtures.size()) {
                    auto short_request = request;
                    short_request.options["duration"] = "0.1";
                    short_request.options["num_inference_steps"] = "1";
                    const auto short_result = session->run(short_request);
                    if (!short_result.audio_output || short_result.audio_output->samples.size() != 2400) {
                        throw std::runtime_error("0.1-second session duration must produce exactly 2400 samples");
                    }
                    engine::debug::log_message("AuK fractional duration contract passed: 0.1 seconds -> 2400 samples");
                    if (request.audio_input) {
                        auto default_request = short_request;
                        default_request.options.erase("duration");
                        const auto default_result = session->run(default_request);
                        // The fixture's valid prefix may be shorter than its padded latent storage.
                        const auto mask = fixture->require_tensor_data("fm.first.ref_mask");
                        const auto valid_frames = std::count(mask.bytes.begin(), mask.bytes.end(), std::byte{1});
                        if (!default_result.audio_output || default_result.audio_output->samples.size() !=
                            static_cast<size_t>(std::max<int64_t>(1, valid_frames) * 480)) {
                            throw std::runtime_error("default duration must match the valid reference prefix");
                        }
                        // Avoid a decimal boundary that Python's ceil can round into the next frame.
                        default_request.options["duration"] = engine::io::json::stringify_number(
                            (double(std::max<int64_t>(1, valid_frames)) - 0.5) * 480.0 / 24000.0);
                        if (session->run(default_request).audio_output->samples != default_result.audio_output->samples) {
                            throw std::runtime_error("default reference duration differs from explicit duration");
                        }
                        engine::debug::log_message("AuK default reference duration matches explicit duration");
                    }
                    if (session->run(request).audio_output->samples != actual) {
                        throw std::runtime_error("session output changed after restoring the original step count");
                    }
                    engine::debug::log_message("AuK original sampling schedule restored bit-identically after one-step request");
                }
            }
            engine::debug::trace_log_scalar("auk.session.device_used_loaded_bytes", memory_observer.memory_snapshot().used_bytes);
            session.reset();
            engine::debug::trace_log_scalar("auk.session.device_used_after_destroy_bytes", memory_observer.memory_snapshot().used_bytes);
            for (int lifetime = 1; lifetime < lifetimes; ++lifetime) {
                engine::debug::trace_log_scalar("auk.session.lifetime", lifetime + 1);
                session = std::make_unique<engine::models::auk::AukSession>(
                    engine::runtime::TaskSpec{engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
                    options, assets, engine::runtime::require_model_contract("auk"));
                session->prepare({});
                const auto result = session->run(lifetime_request);
                if (!result.audio_output || result.audio_output->sample_rate != 24000 ||
                    result.audio_output->channels != 1 || result.audio_output->samples != first_output) {
                    throw std::runtime_error("session recreation changed the seeded output");
                }
                engine::debug::trace_log_scalar("auk.session.device_used_loaded_bytes", memory_observer.memory_snapshot().used_bytes);
                session.reset();
                engine::debug::trace_log_scalar("auk.session.device_used_after_destroy_bytes", memory_observer.memory_snapshot().used_bytes);
            }
            engine::debug::log_message("AuK native offline session parity and repeat checks passed");
            return 0;
        }
        const auto audio_shape = inputs->require_metadata("fm.first.x").shape;
        const auto text_shape = inputs->require_metadata("fm.first.text").shape;
        if (audio_shape.size() != 3 || text_shape.size() != 3 || audio_shape[0] != 1 || text_shape[0] != 1 ||
            audio_shape[2] != 64 || text_shape[2] != 2048) {
            throw std::runtime_error("unsupported flow fixture shapes");
        }
        auto audio = inputs->require_f32("fm.first.x");
        auto text = inputs->require_f32("fm.first.text");
        const int64_t reference_frames = inputs->require_metadata("fm.first.ref").shape.at(1);
        const auto reference = reference_frames ? inputs->require_f32("fm.first.ref") : std::vector<float>{};
        int64_t valid_reference_frames = 0;
        if (reference_frames) {
            const auto mask = inputs->require_tensor_data("fm.first.ref_mask");
            if (mask.bytes.size() != static_cast<size_t>(reference_frames)) throw std::runtime_error("reference mask size mismatch");
            valid_reference_frames = std::count(mask.bytes.begin(), mask.bytes.end(), std::byte{1});
            for (int64_t i = 0; i < reference_frames; ++i) {
                if (std::to_integer<int>(mask.bytes[i]) != (i < valid_reference_frames)) throw std::runtime_error("reference mask must be a valid prefix");
            }
            if (instruct) throw std::runtime_error("reference input is not yet wired into the integrated session test");
        }
        engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
        bool parity_passed = true;
        std::unique_ptr<engine::models::auk::ConditioningRuntime> encoder;
        std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
        std::string instruction;
        uint64_t seed = 0;
        engine::sampling::TorchCudaSamplingPolicy policy;
        const auto compare = [&](const std::string & name, const std::vector<float> & actual,
                                 const engine::assets::TensorSource * comparison_source = nullptr) {
            const auto reference = (comparison_source ? comparison_source : expected.get())->require_f32(name);
            if (actual.size() != reference.size()) throw std::runtime_error(name + " size mismatch");
            double dot = 0, norm = 0, reference_norm = 0, error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error(name + " is non-finite");
                dot += double(actual[i]) * reference[i];
                norm += double(actual[i]) * actual[i];
                reference_norm += double(reference[i]) * reference[i];
                const double difference = double(actual[i]) - reference[i];
                error += difference * difference;
            }
            const double cosine = dot / std::sqrt(norm * reference_norm);
            std::ostringstream message;
            message << std::setprecision(10) << name << " cosine=" << cosine << " RMSE=" << std::sqrt(error / actual.size());
            engine::debug::log_message(message.str());
            std::cout << message.str() << '\n';
            if (!(cosine > 0.99999)) {
                parity_passed = false;
                engine::debug::log_message(name + " parity gate failed");
                if (!sample) throw std::runtime_error(name + " parity gate failed");
            }
        };
        if (instruct) {
            const std::filesystem::path qwen_dir = argv[6];
            const auto manifest = engine::io::json::parse_file(directory / "manifest.json");
            if (manifest.require("dtype").as_string() != "fp32") {
                throw std::runtime_error("native instruct integration currently requires an FP32 fixture");
            }
            instruction = manifest.require("instruction").as_string();
            seed = static_cast<uint64_t>(manifest.require("seed").as_i64());
            policy = engine::sampling::resolve_torch_cuda_sampling_policy(
                engine::core::BackendType::Cuda, 0, "auk", "AuK");
            const auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                audio.size(), static_cast<uint64_t>(manifest.require("seed").as_i64()), 0, policy);
            compare("fm.first.x", noise);
            if (!parity_passed) throw std::runtime_error("native initial noise parity gate failed");
            engine::debug::log_message(noise == audio
                ? "Native initial noise is bit-identical to Python"
                : "Native initial noise passes cosine gate but is not bit-identical to Python");
            audio = noise;
            engine::tokenizers::LlamaBpeTokenizerSpec spec;
            spec.tokenizer_config_path = qwen_dir / "tokenizer_config.json";
            spec.tokenizer_json_path = qwen_dir / "tokenizer.json";
            spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
            tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
            const auto condition = engine::models::auk::prepare_conditioning(
                *tokenizer, manifest.require("instruction").as_string());
            const auto qwen = engine::assets::open_tensor_source(qwen_dir / "model.safetensors.index.json");
            encoder = std::make_unique<engine::models::auk::ConditioningRuntime>(execution, *qwen, *source, condition.token_ids.size(), false, false);
            text = encoder->encode(condition);
            compare("fm.first.text", text);
            if (!parity_passed) throw std::runtime_error("conditioning gate failed before sampling");
        }
        engine::models::auk::FlowRuntime runtime(execution, *source, audio_shape[1], text_shape[1], cfg, !explicit_attention, !sample,
            reference_frames, valid_reference_frames);
        if (sample) {
            const auto vae = engine::assets::open_tensor_source(std::filesystem::path(argv[1]).parent_path() / "vae.safetensors");
            engine::models::auk::VaeDecoderRuntime decoder(execution, *vae, audio_shape[1]);
            engine::debug::log_message("Teacher-forced VAE before native sampling");
            const std::string generated_key = reference_frames ? "vae.normalized_input" : "fm.output";
            compare("waveform", decoder.decode(inputs->require_f32(generated_key)));
            if (!parity_passed) throw std::runtime_error("teacher-forced VAE gate failed before sampling");
            engine::debug::log_message("Native sampling with saved initial noise");
            const auto latents = runtime.sample(audio, text, 32, -1.0F, 2.0F, reference);
            compare(generated_key, latents);
            if (runtime.sample(audio, text, 32, -1.0F, 2.0F, reference) != latents) throw std::runtime_error("repeated sampling changed output");
            const auto waveform = decoder.decode(latents);
            engine::audio::write_pcm16_wav(directory / (instruct ? "native-instruct.wav" : explicit_attention ? "native-flow-sampled-explicit.wav" : "native-flow-sampled.wav"), 24000, 1, waveform);
            compare("waveform", waveform);
            if (alternate_fixture) {
                if (!parity_passed) throw std::runtime_error("primary parity gate failed before shape switching");
                const std::filesystem::path alternate_directory = argv[8];
                const auto alternate = engine::assets::open_tensor_source(alternate_directory / "reference.safetensors");
                const auto manifest = engine::io::json::parse_file(alternate_directory / "manifest.json");
                if (manifest.require("dtype").as_string() != "fp32") throw std::runtime_error("alternate fixture must be FP32");
                const auto alternate_shape = alternate->require_metadata("fm.first.x").shape;
                if (alternate_shape.size() != 3 || alternate_shape[0] != 1 || alternate_shape[2] != 64 ||
                    alternate_shape[1] == audio_shape[1]) throw std::runtime_error("alternate fixture must change valid audio length");
                const auto condition = engine::models::auk::prepare_conditioning(
                    *tokenizer, manifest.require("instruction").as_string());
                const auto alternate_noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                    alternate_shape[1] * 64, static_cast<uint64_t>(manifest.require("seed").as_i64()), 0, policy);
                compare("fm.first.x", alternate_noise, alternate.get());
                if (!parity_passed) throw std::runtime_error("alternate noise parity gate failed");
                for (int cycle = 0; cycle < 3; ++cycle) {
                    encoder->prepare(condition.token_ids.size());
                    const auto conditioning = encoder->encode(condition);
                    compare("fm.first.text", conditioning, alternate.get());
                    decoder.prepare(alternate_shape[1]);
                    compare("waveform", decoder.decode(alternate->require_f32("fm.output")), alternate.get());
                    if (!parity_passed) throw std::runtime_error("alternate deterministic component gate failed");
                    runtime.prepare(alternate_shape[1], condition.token_ids.size());
                    const auto generated = runtime.sample(alternate_noise, conditioning);
                    compare("fm.output", generated, alternate.get());
                    const auto alternate_wave = decoder.decode(generated);
                    compare("waveform", alternate_wave, alternate.get());
                    if (!parity_passed) throw std::runtime_error("alternate sampling parity gate failed");
                    runtime.prepare(alternate_shape[1], condition.token_ids.size());
                    if (runtime.sample(alternate_noise, conditioning) != generated) throw std::runtime_error("same-shape flow prepare changed output");
                    encoder->prepare(text_shape[1]);
                    const auto primary_condition = engine::models::auk::prepare_conditioning(*tokenizer, instruction);
                    if (encoder->encode(primary_condition) != text) throw std::runtime_error("restored conditioning changed output");
                    runtime.prepare(audio_shape[1], text_shape[1]);
                    if (runtime.sample(audio, text) != latents) throw std::runtime_error("restored flow changed output");
                    decoder.prepare(audio_shape[1]);
                    if (decoder.decode(latents) != waveform) throw std::runtime_error("restored waveform changed output");
                    const auto memory = execution.memory_snapshot();
                    if (memory.available) engine::debug::trace_log_scalar("auk.instruct.cycle_device_used_bytes", memory.used_bytes);
                }
                engine::debug::log_message("AuK full instruct shape switching passed three round trips");
            }
            if (benchmark_runs && !parity_passed) throw std::runtime_error("parity gate failed before benchmark");
            std::vector<double> timings;
            for (int run = 0; run < benchmark_runs + (benchmark_runs ? 2 : 0); ++run) {
                const auto start = std::chrono::steady_clock::now();
                const auto condition = engine::models::auk::prepare_conditioning(*tokenizer, instruction);
                const auto conditioning = encoder->encode(condition);
                const auto encoded = std::chrono::steady_clock::now();
                const auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(audio.size(), seed, 0, policy);
                const auto generated = runtime.sample(noise, conditioning);
                const auto sampled = std::chrono::steady_clock::now();
                const auto repeated = decoder.decode(generated);
                const auto done = std::chrono::steady_clock::now();
                if (repeated != waveform) throw std::runtime_error("benchmark waveform changed");
                if (run < 2) continue;
                const double total = std::chrono::duration<double, std::milli>(done - start).count();
                timings.push_back(total);
                engine::debug::log_message("AuK inference_ms=" + std::to_string(total) +
                    " conditioning_ms=" + std::to_string(std::chrono::duration<double, std::milli>(encoded - start).count()) +
                    " sampling_ms=" + std::to_string(std::chrono::duration<double, std::milli>(sampled - encoded).count()) +
                    " vae_ms=" + std::to_string(std::chrono::duration<double, std::milli>(done - sampled).count()));
            }
            if (!timings.empty()) {
                std::sort(timings.begin(), timings.end());
                const double median = (timings[(timings.size() - 1) / 2] + timings[timings.size() / 2]) * 0.5;
                engine::debug::log_message("AuK inference_median_ms=" + std::to_string(median) +
                    " RTF=" + std::to_string(median * 24.0 / waveform.size()));
            }
            return parity_passed ? 0 : 1;
        }
        const float times[] = {0.0F, 0.25F, 0.75F};
        for (int step = 0; step < 3; ++step) {
            const auto result = runtime.embed(audio, text, times[step], 2.0F, reference);
            compare("time." + std::to_string(step), result.time);
            compare("text", result.text);
            compare("audio", result.audio);
            for (const auto & entry : result.first_block_boundaries) {
                compare("first." + entry.first + "." + std::to_string(step), entry.second);
            }
            compare("block.audio." + std::to_string(step), result.first_block_audio);
            compare("block.text." + std::to_string(step), result.first_block_text);
            compare("double.audio." + std::to_string(step), result.double_stream_audio);
            compare("double.text." + std::to_string(step), result.double_stream_text);
            compare("single." + std::to_string(step), result.single_stream);
            compare("velocity." + std::to_string(step), result.velocity);
            if (cfg) {
                const auto middle = result.velocity.begin() + result.velocity.size() / 2;
                compare("conditional." + std::to_string(step), {result.velocity.begin(), middle});
                compare("unconditional." + std::to_string(step), {middle, result.velocity.end()});
                compare("guided." + std::to_string(step), result.guided_velocity);
            } else if (result.guided_velocity != result.velocity) {
                throw std::runtime_error("unguided velocity changed");
            }
            const auto repeated = runtime.embed(audio, text, times[step], 2.0F, reference);
            if (repeated.time != result.time || repeated.text != result.text || repeated.audio != result.audio ||
                repeated.first_block_audio != result.first_block_audio || repeated.first_block_text != result.first_block_text ||
                repeated.double_stream_audio != result.double_stream_audio || repeated.double_stream_text != result.double_stream_text ||
                repeated.single_stream != result.single_stream || repeated.velocity != result.velocity ||
                repeated.guided_velocity != result.guided_velocity) {
                throw std::runtime_error("flow embedding reuse changed output");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        engine::debug::log_message(error.what());
        std::cerr << error.what() << '\n';
        return 1;
    }
}
