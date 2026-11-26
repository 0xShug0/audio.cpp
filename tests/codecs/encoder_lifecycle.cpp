#include "app/cli/args.h"
#include "lifecycle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/codecs/qwen3_audio_tokenizer_runtime.h"
#include "engine/framework/conditioners/musicgen_style_conditioner_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/speech_encoders/s3_tokenizer.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        using minitts::cli::find_arg;
        engine::debug::configure_logging({true, find_arg(argc, argv, "--log-file").value()});
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log required");
        const auto component = find_arg(argc, argv, "--component").value();
        const auto source = engine::assets::open_tensor_source(
            std::filesystem::path(find_arg(argc, argv, "--model").value()),
            find_arg(argc, argv, "--prefix").value_or(""));
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(find_arg(argc, argv, "--audio").value()));
        if (wav.channels != 1 || wav.samples.size() < 2) throw std::runtime_error("supply a nonempty mono recording");
        engine::runtime::AudioBuffer audio{wav.sample_rate, wav.channels, wav.samples};
        if (component == "wav2vec2" || component == "musicgen_style") {
            if (audio.sample_rate != 24000) throw std::runtime_error("MERT input must be 24 kHz");
            audio.samples.resize(std::min(audio.samples.size(), size_t{4 * 24000}));
        }
        auto b = audio;
        std::reverse(b.samples.begin(), b.samples.end());
        auto c = audio;
        c.samples.resize(c.samples.size() / 2);
        engine::core::ExecutionContext execution(minitts::cli::parse_backend_config_from_cli(argc, argv, 8));
        if (component == "qwen3") {
            const auto attention = find_arg(argc, argv, "--attention").value_or("eager");
            if (attention != "eager" && attention != "flash") throw std::runtime_error("expected eager or flash attention");
            const auto make = [&] {
                return std::make_unique<engine::codecs::Qwen3SpeechTokenizerEncoderRuntime>(
                    source, execution, 16ull << 20, engine::assets::TensorStorageType::F32,
                    engine::assets::TensorStorageType::F32, attention == "flash");
            };
            check_runtime_lifecycle(make, [](auto & runtime, const auto & input) { return runtime.encode(input).codes; }, audio, b, c);
        } else if (component == "s3") {
            const auto construction = find_arg(argc, argv, "--construction").value_or("factory");
            if (construction != "factory" && construction != "weights") throw std::runtime_error("expected factory or weights construction");
            std::shared_ptr<const engine::modules::S3TokenizerWeights> weights;
            if (construction == "weights") {
                auto loaded = engine::modules::S3TokenizerComponent::load_from_source(source, execution);
                weights = loaded.weights();
                std::cout << "public_weights_constructor valid_loaded_weights=1" << std::endl;
            }
            const auto make = [&] {
                if (weights) return std::make_unique<engine::modules::S3TokenizerComponent>(weights, execution);
                return std::make_unique<engine::modules::S3TokenizerComponent>(
                    engine::modules::S3TokenizerComponent::load_from_source(source, execution));
            };
            const auto encode = [](auto & runtime, const auto & input) { return runtime.tokenize(input, std::nullopt).tokens; };
            check_runtime_lifecycle(make, encode, audio, b, c, [](auto & runtime) { runtime.release_runtime_cache(); });
        } else if (component == "wav2vec2") {
            auto config = engine::conditioners::MusicGenStyleConfig::mert_default().mert;
            config.release_graph_after_encode = false;
            engine::modules::Wav2Vec2EncoderWeightBinding binding;
            binding.layer.post_attention_layer_norm = "layer_norm";
            const auto make = [&] {
                return std::make_unique<engine::modules::Wav2Vec2EncoderRuntime>(
                    engine::modules::Wav2Vec2EncoderRuntime::load_from_tensor_source(source, execution.config(), config, binding));
            };
            const auto encode = [](auto & runtime, const auto & input) {
                return runtime.encode(input.samples, 1, static_cast<int64_t>(input.samples.size())).hidden_states;
            };
            check_runtime_lifecycle(make, encode, audio, b, c, [](auto & runtime) { runtime.release_runtime_graph(); });
        } else if (component == "musicgen_style") {
            const auto auxiliary = engine::assets::open_tensor_source(
                std::filesystem::path(find_arg(argc, argv, "--model").value()),
                find_arg(argc, argv, "--aux-prefix").value());
            const auto make = [&] {
                return std::make_unique<engine::conditioners::MusicGenStyleConditionerRuntime>(source, auxiliary, execution);
            };
            const auto encode = [](auto & runtime, const auto & input) {
                return runtime.encode_audio(input.samples, 1, static_cast<int64_t>(input.samples.size())).values;
            };
            check_runtime_lifecycle(make, encode, audio, b, c, [](auto & runtime) { runtime.release_runtime_graphs(); });
        } else {
            throw std::runtime_error("expected component qwen3, s3, wav2vec2 or musicgen_style");
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
