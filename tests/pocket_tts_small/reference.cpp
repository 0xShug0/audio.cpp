#include "app/cli/args.h"
#include "engine/models/pocket_tts_small/assets.h"
#include "engine/models/pocket_tts_small/mimi_decoder.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/io/json.h"
#include "engine/models/pocket_tts_small/conditioning.h"
#include <cstring>
#include <iomanip>
#include <sstream>

#include <algorithm>
#include <cmath>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto reference = minitts::cli::find_arg(argc, argv, "--reference").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto mode = minitts::cli::find_arg(argc, argv, "--mode").value_or("offline");
        if (mode != "offline" && mode != "streaming") throw std::runtime_error("Expected offline or streaming mode");
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log is required");
        const auto chunk_frames = minitts::cli::parse_int_arg(argc, argv, "--chunk-frames", 4);
        const bool full_decoder = minitts::cli::has_arg(argc, argv, "--full-decoder");
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        const auto index = engine::io::load_safetensors_index(reference);
        const auto source = engine::assets::open_tensor_source(reference);
        const auto latents = source->require_f32("latents", index.tensors.at("latents").shape);
        const auto expected = source->require_f32("waveform");
        auto assets = engine::models::pocket_tts_small::load_assets(model);
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        const auto weights = engine::models::pocket_tts_small::load_pocket_tts_small_backend_weights(
            *assets, execution.backend(), execution.backend_type(), engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::Native, 8 << 20, 64 << 20, 16 << 20);
        const auto texts = engine::io::json::parse(index.metadata.at("prepared_text")).as_array();
        engine::models::pocket_tts_small::TextConditioner conditioner({assets->model_config.flow_dim});
        const auto chunks = conditioner.split_into_sentence_chunks(*assets, index.metadata.at("text"),
            std::stoll(index.metadata.at("max_tokens")));
        if (chunks.size() != texts.size()) throw std::runtime_error("Official text chunk count differs");
        for (size_t i = 0; i < texts.size(); ++i) {
            std::ostringstream name;
            name << "text." << std::setw(4) << std::setfill('0') << i << ".token_ids";
            const auto tokens = source->require_tensor_data(name.str());
            const auto prepared = conditioner.prepare(*assets, weights->host, chunks[i], execution.backend(), 8);
            if (prepared.prepared_text != texts[i].as_string())
                throw std::runtime_error("Official prepared text differs");
            if (tokens.metadata.dtype != "I64" || tokens.bytes.size() != prepared.tokens.size() * sizeof(int64_t))
                throw std::runtime_error("Official text token count or type differs");
            for (size_t j = 0; j < prepared.tokens.size(); ++j) {
                int64_t token;
                std::memcpy(&token, tokens.bytes.data() + j * sizeof(token), sizeof(token));
                if (token != prepared.tokens[j]) throw std::runtime_error("Official text tokens differ");
            }
        }
        const auto voice = engine::assets::open_tensor_source(assets->default_voice_path.value());
        for (const auto & [name, info] : index.tensors) {
            (void) info;
            if (name.rfind("voice.", 0) != 0) continue;
            const auto expected_voice = source->require_tensor_data(name);
            const auto actual_voice = voice->require_tensor_data(name.substr(6));
            if (expected_voice.metadata.shape != actual_voice.metadata.shape ||
                expected_voice.metadata.dtype != actual_voice.metadata.dtype || expected_voice.bytes != actual_voice.bytes)
                throw std::runtime_error("Official voice input differs: " + name);
        }
        const auto config = assets->model_config.decoder;
        const auto frames = static_cast<int64_t>(latents.size()) / config.latent_size;
        engine::models::pocket_tts_small::MimiDecoder decoder(config);
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> actual;
        const auto fade = engine::audio::make_linear_fade_window(240, 120);
        if (mode == "offline") {
            if (texts.size() != 1) throw std::runtime_error("Offline replay expects one text chunk");
            actual = decoder.decode(execution.backend(), 8, *assets, *weights, latents,
                frames, 32 << 20, 96 << 20, 64 << 20, chunk_frames, 900, full_decoder);
            for (size_t i = 0; i < std::min<size_t>(120, actual.size()); ++i) actual[i] *= fade[i];
        } else {
            const auto calls = engine::io::json::parse(index.metadata.at("decoder_calls")).as_array();
            for (const auto & call : calls) {
                if (call.require("reset").as_bool()) decoder.reset_streaming_state();
                std::ostringstream name;
                name << "decoder." << std::setw(4) << std::setfill('0') << call.require("index").as_i64() << ".latent";
                const auto input = source->require_f32(name.str());
                std::vector<float> samples;
                for (size_t offset = 0; offset < input.size(); offset += config.latent_size) {
                    const std::vector<float> frame(input.begin() + offset, input.begin() + offset + config.latent_size);
                    auto output = decoder.decode_streaming(execution.backend(), 8, *assets, *weights,
                        frame, 32 << 20, 96 << 20, 64 << 20);
                    samples.insert(samples.end(), output.begin(), output.end());
                }
                if (samples.size() != static_cast<size_t>(call.require("samples").as_i64()))
                    throw std::runtime_error("Streaming decoder call length differs");
                if (call.require("reset").as_bool())
                    for (size_t i = 0; i < std::min<size_t>(120, samples.size()); ++i) samples[i] *= fade[i];
                actual.insert(actual.end(), samples.begin(), samples.end());
            }
        }
        if (actual.size() != expected.size()) throw std::runtime_error("Teacher-forced waveform length differs");
        double aa = 0, bb = 0, ab = 0, max_abs = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite teacher-forced waveform");
            aa += double(actual[i]) * actual[i]; bb += double(expected[i]) * expected[i];
            ab += double(actual[i]) * expected[i];
            max_abs = std::max(max_abs, std::abs(double(actual[i]) - expected[i]));
        }
        double first_aa = 0, first_bb = 0, first_ab = 0;
        for (size_t i = 0; i < std::min<size_t>(1920, actual.size()); ++i) {
            first_aa += double(actual[i]) * actual[i];
            first_bb += double(expected[i]) * expected[i];
            first_ab += double(actual[i]) * expected[i];
        }
        engine::debug::trace_log_scalar("pocket_tts_small.reference.first_frame_waveform_cosine",
            first_ab / std::sqrt(first_aa * first_bb));
        const double cosine = ab / std::sqrt(aa * bb);
        engine::debug::trace_log_scalar("pocket_tts_small.reference.waveform_cosine", cosine);
        engine::debug::trace_log_scalar("pocket_tts_small.reference.waveform_max_abs", max_abs);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cout << "waveform_cosine=" << cosine << " samples=" << actual.size() << '\n';
        if (!(cosine > .99999)) throw std::runtime_error("Teacher-forced waveform parity failed");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
