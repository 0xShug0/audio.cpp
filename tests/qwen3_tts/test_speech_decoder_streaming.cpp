#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/prompt_tts_custom_voice.h"
#include "engine/models/qwen3_tts/tokenizer_speech_decoder.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Integration test with real decoder weights and identical input codes. Request
// sampling is deliberately outside this test so decoder errors cannot be hidden
// by different autoregressive trajectories between offline and streaming modes.
int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        std::string model, text;
        bool native = false;
        auto conv_storage = engine::assets::TensorStorageType::F32;
        engine::core::BackendConfig backend;
        backend.threads = 4;
        engine::debug::LoggingConfig logging;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--model" && i + 1 < argc) model = argv[++i];
            else if (arg == "--text" && i + 1 < argc) text = argv[++i];
            else if (arg == "--native") native = true;
            else if (arg == "--conv-type" && i + 1 < argc) {
                const std::string value(argv[++i]);
                if (value == "f32") conv_storage = engine::assets::TensorStorageType::F32;
                else if (value == "f16") conv_storage = engine::assets::TensorStorageType::F16;
                else throw std::runtime_error("test conv type must be f32 or f16");
            }
            else if (arg == "--backend" && i + 1 < argc) {
                const std::string value(argv[++i]);
                if (value == "cuda") backend.type = engine::core::BackendType::Cuda;
                else if (value == "cpu") backend.type = engine::core::BackendType::Cpu;
                else throw std::runtime_error("test backend must be cpu or cuda");
            } else if (arg == "--log" && i + 1 < argc) {
                logging.enabled = true;
                logging.file_path = argv[++i];
            } else throw std::runtime_error("unknown test argument: " + arg);
        }
        if (model.empty()) throw std::runtime_error("--model GGUF is required");
        engine::debug::configure_logging(logging);
        namespace qwen = engine::models::qwen3_tts;
        auto assets = qwen::load_qwen3_tts_assets(model);
        engine::core::ExecutionContext execution(backend);
        const auto storage = native ? engine::assets::TensorStorageType::Native : engine::assets::TensorStorageType::F32;
        constexpr size_t arena = 64 * 1024 * 1024;
        qwen::Qwen3SpeechCodes codes;
        codes.frames = 103;
        codes.code_groups = assets->config.speech_tokenizer.num_quantizers;
        const auto vocabulary = std::min(assets->config.speech_tokenizer.codebook_size,
                                         assets->config.speech_tokenizer.semantic_codebook_size);
        if (vocabulary <= 0) throw std::runtime_error("decoder has no codebook vocabulary");
        for (int64_t frame = 0; frame < codes.frames; ++frame) {
            for (int64_t group = 0; group < codes.code_groups; ++group) {
                codes.codes.push_back(static_cast<int32_t>((frame * 37 + group * 113) % vocabulary));
            }
        }
        if (!text.empty()) {
            if (assets->config.variant != qwen::Qwen3TTSVariant::CustomVoice) {
                throw std::runtime_error("--text requires CustomVoice weights");
            }
            qwen::Qwen3TextTokenizer tokenizer(assets);
            qwen::Qwen3TTSCustomVoicePromptBuilder prompt_builder(
                tokenizer, assets->config.talker.max_position_embeddings, assets->config.talker.max_position_embeddings);
            qwen::Qwen3TTSRequest request;
            request.text = text;
            request.language = "English";
            request.custom_voice = qwen::Qwen3CustomVoiceInput{"Aiden", ""};
            request.generation.seed = 42;
            request.generation.max_new_tokens = 256;
            qwen::Qwen3Talker talker(assets->config.talker);
            auto weights = talker.create_weights_runtime(
                assets, backend.type, backend.device, backend.threads, arena, arena, arena,
                engine::assets::TensorStorageType::Native, qwen::Qwen3TTSPerfMode::Standard);
            auto step = talker.create_step_runtime(weights, assets->config.talker.max_position_embeddings, 256);
            codes = step->generate(prompt_builder.build_prefill(request), request.generation).generated_codes;
            if (codes.frames <= 23) throw std::runtime_error("test text generated too few speech frames");
        }
        qwen::Qwen3SpeechTokenizerDecoderRuntime decoder(
            assets, execution, arena, arena, storage, conv_storage,
            qwen::Qwen3TTSPerfMode::Standard, {}, true);
        const auto reference = decoder.decode(codes);
        const size_t samples_per_frame = reference.samples.size() / static_cast<size_t>(codes.frames);
        if (samples_per_frame * codes.frames != reference.samples.size()) {
            throw std::runtime_error("offline decoder output is not frame aligned");
        }
        for (const bool with_reference : {false, true}) {
            for (const int64_t cap : {1, 4, 12}) {
                for (const bool grow : {false, true}) {
                    decoder.reset_stream();
                    std::vector<float> output;
                    const int64_t prefix = with_reference ? 23 : 0;
                    int64_t target = grow ? 1 : cap;
                    for (int64_t offset = prefix; offset < codes.frames;) {
                        const int64_t count = std::min(target, codes.frames - offset);
                        const int64_t context = std::min(offset, decoder.kLeftContextFrames);
                        qwen::Qwen3SpeechCodes window;
                        window.frames = context + count;
                        window.code_groups = codes.code_groups;
                        window.codes.assign(codes.codes.begin() + (offset - context) * codes.code_groups,
                                            codes.codes.begin() + (offset + count) * codes.code_groups);
                        auto audio = decoder.decode_stream_chunk(window, context, cap);
                        output.insert(output.end(), audio.samples.begin(), audio.samples.end());
                        offset += count;
                        if (grow) target = std::min(cap, target * 2);
                    }
                    const size_t skip = static_cast<size_t>(prefix) * samples_per_frame;
                    if (output.size() != reference.samples.size() - skip) {
                        throw std::runtime_error("stateful decoder output length differs from offline");
                    }
                    double squared = 0.0, reference_squared = 0.0, maximum = 0.0;
                    for (size_t i = 0; i < output.size(); ++i) {
                        if (!std::isfinite(output[i])) throw std::runtime_error("nonfinite stateful audio");
                        const double expected = reference.samples[i + skip];
                        const double error = output[i] - expected;
                        maximum = std::max(maximum, std::abs(error));
                        squared += error * error;
                        reference_squared += expected * expected;
                    }
                    const double relative = std::sqrt(squared / reference_squared);
                    engine::debug::trace_log_scalar("qwen3_tts.decoder_test.max_abs", maximum);
                    engine::debug::trace_log_scalar("qwen3_tts.decoder_test.relative_rmse", relative);
                    std::cout << "prefix=" << prefix << " cap=" << cap << " grow=" << grow
                              << " max_abs=" << maximum << " relative_rmse=" << relative << '\n';
                    // Batch geometry changes floating point reduction order;
                    // this bound catches state/reset/padding errors, not ULP drift.
                    const double tolerance = backend.type == engine::core::BackendType::Cpu && !native && conv_storage == engine::assets::TensorStorageType::F32 ? 1.0e-4 : 0.02;
                    if (!std::isfinite(relative) || maximum > tolerance || relative > tolerance) {
                        throw std::runtime_error("stateful decoder differs from offline beyond tolerance");
                    }
                }
            }
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cout << "qwen3_tts_decoder_streaming_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
