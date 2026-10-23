#include "engine/models/ming3_audio/assets.h"
#include "engine/models/ming3_audio/talker.h"
#include "engine/models/ming3_audio/talker_frontend.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        std::filesystem::path weights, fixture, output, generated_output, voice_ref;
        bool log = false;
        bool eager = false;
        bool reference_voice = false;
        bool official_inputs = false;
        std::string text = "The sun warms the earth, and a soft breeze moves through the garden.";
        std::string voice = "US_James", language = "en", reference_text;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--log") log = true;
            else if (arg == "--eager") eager = true;
            else if (arg == "--reference-voice") reference_voice = true;
            else if (arg == "--official-inputs") official_inputs = true;
            else if ((arg == "--text" || arg == "--voice-id" || arg == "--language" || arg == "--reference-text") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--text") text = value;
                else if (arg == "--voice-id") voice = value;
                else if (arg == "--language") language = value;
                else reference_text = value;
            }
            else if ((arg == "--weights" || arg == "--reference" || arg == "--output" || arg == "--generated-output" || arg == "--voice-ref") && i + 1 < argc) {
                const std::filesystem::path value = argv[++i];
                if (arg == "--weights") weights = value;
                else if (arg == "--reference") fixture = value;
                else if (arg == "--output") output = value;
                else if (arg == "--generated-output") generated_output = value;
                else { voice_ref = value; reference_voice = true; }
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        if (!log || weights.empty() || fixture.empty() || output.empty()) {
            throw std::runtime_error("usage: ming3_audio_talker_parity --weights <gguf> --reference <safetensors> --output <wav> --log");
        }
        engine::debug::configure_logging({true, std::nullopt});
        const auto assets = engine::models::ming3_audio::load_ming3_audio_assets(weights);
        const auto reference = engine::assets::open_tensor_source(fixture);
        engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
        engine::models::ming3_audio::TalkerFrontend frontend(*assets);
        engine::models::ming3_audio::Talker talker(execution, assets->talker_weights, assets->talker);
        std::vector<float> speaker, reference_latents;
        if (reference_voice) {
            speaker = reference->require_f32("reference.speaker_features");
            reference_latents = reference->require_f32("chunk.0.reference_latents");
        }
        if (!voice_ref.empty()) {
            const auto wav = engine::audio::read_wav_f32(voice_ref);
            engine::modules::CampplusEncoderConfig config;
            config.weight_layout = engine::modules::CampplusEncoderWeightLayout::Fused;
            config.normalize_partial_segment_by_full_length = true;
            auto speaker_encoder = engine::modules::CampplusEncoderComponent::load_from_tensor_source(
                assets->speaker_weights, {engine::core::BackendType::Cuda, 0, 8}, config);
            const auto features = engine::audio::ReferenceAudioFrontend::campplus_fbank_16k(
                engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                    wav.samples, wav.sample_rate, wav.channels, 16000));
            speaker = speaker_encoder.embed_from_features(features.values, features.frames, features.feature_dim).embedding;
            auto waveform = engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                wav.samples, wav.sample_rate, wav.channels, assets->talker.codec.sample_rate);
            const auto alignment = assets->talker.codec.encoder.hop_size * assets->talker.codec.patch_size;
            waveform.insert(waveform.begin(), (alignment - waveform.size() % alignment) % alignment, 0.0f);
            engine::modules::MingAudioEncoder encoder(assets->vae_weights, assets->talker.codec, execution);
            const auto count = encoder.latent_frames(waveform.size()) * assets->talker.codec.encoder.latent_dim;
            const auto policy = engine::sampling::resolve_torch_cuda_sampling_policy(
                engine::core::BackendType::Cuda, 0, "ming3_audio.test", "Ming3 Audio reference");
            const auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(count, 42, 0, policy,
                engine::sampling::TorchRandnPrecision::BFloat16);
            reference_latents = encoder.encode(waveform, noise);
            if (engine::sampling::torch_cuda_tensor_iterator_offset_blocks(count, policy) !=
                static_cast<uint64_t>(reference->require_i64_scalar("chunk.0.initial_rng_offset_blocks")))
                throw std::runtime_error("Native reference RNG consumption differs from the official implementation");
            std::cout << "native_reference_preparation=true\n";
        }
        const auto prompt = frontend.build(text, voice, language, reference_voice, reference_text,
            reference_latents.size() / assets->talker.flow.latent_dim);
        if (reference_voice) {
            const auto raw = reference->require_tensor_data("chunk.0.prompt_ids");
            if (raw.metadata.dtype != "I32") throw std::runtime_error("Expected I32 prompt IDs");
            std::vector<int32_t> ids(raw.bytes.size() / sizeof(int32_t));
            std::memcpy(ids.data(), raw.bytes.data(), raw.bytes.size());
            if (prompt.ids != ids) throw std::runtime_error("Talker clone prompt tokens differ from the official implementation");
            std::cout << "ar_prompt_ids_identical=true\n";
        }
        const auto inputs = talker.prepare_inputs(prompt, speaker, reference_latents);
        const auto expected_inputs = reference->require_f32("chunk.0.prompt_embeddings");
        if (inputs.size() != expected_inputs.size()) throw std::runtime_error("Talker AR prompt embedding lengths differ");
        double input_dot = 0, input_lhs = 0, input_rhs = 0;
        double speaker_error = 0, reference_error = 0, text_error = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            input_dot += static_cast<double>(inputs[i]) * expected_inputs[i];
            input_lhs += static_cast<double>(inputs[i]) * inputs[i];
            input_rhs += static_cast<double>(expected_inputs[i]) * expected_inputs[i];
            const auto row = static_cast<int64_t>(i / assets->talker.backbone.hidden_size);
            const double error = std::abs(static_cast<double>(inputs[i]) - expected_inputs[i]);
            if (row == prompt.speaker_position) speaker_error = std::max(speaker_error, error);
            else if (prompt.latent_position >= 0 && row >= prompt.latent_position) reference_error = std::max(reference_error, error);
            else text_error = std::max(text_error, error);
        }
        engine::debug::trace_log_scalar("ming3_audio.test.ar_inputs_cosine", input_dot / std::sqrt(input_lhs * input_rhs));
        engine::debug::trace_log_scalar("ming3_audio.test.speaker_max_abs", speaker_error);
        engine::debug::trace_log_scalar("ming3_audio.test.reference_max_abs", reference_error);
        engine::debug::trace_log_scalar("ming3_audio.test.text_max_abs", text_error);
        const bool inputs_identical = inputs == expected_inputs;
        std::cout << "ar_inputs_identical=" << (inputs_identical ? "true" : "false") << "\n";
        engine::modules::MingAudioDecoder decoder(assets->vae_weights, assets->talker.codec, execution,
            eager ? engine::core::AttentionPreference::Eager : engine::core::AttentionPreference::Auto);
        const auto latents = reference->require_f32("chunk.0.generated_latents");
        const auto expected = reference->require_f32("chunk.0.waveform");
        const auto start = std::chrono::steady_clock::now();
        const auto actual = decoder.decode(latents);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        if (actual.size() != expected.size()) throw std::runtime_error("Teacher-forced waveform lengths differ");
        double dot = 0, lhs = 0, rhs = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            dot += static_cast<double>(actual[i]) * expected[i];
            lhs += static_cast<double>(actual[i]) * actual[i];
            rhs += static_cast<double>(expected[i]) * expected[i];
        }
        const double cosine = dot / std::sqrt(lhs * rhs);
        engine::audio::WavWriteOptions options;
        options.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(output, assets->talker.codec.sample_rate, 1, actual, options);
        std::cout << std::setprecision(12) << "waveform_cosine=" << cosine << "\n";
        if (!generated_output.empty()) {
            uint64_t offset = static_cast<uint64_t>(reference->require_i64_scalar("chunk.0.initial_rng_offset_blocks"));
            std::vector<float> history;
            if (!reference_latents.empty()) history.assign(
                reference_latents.end() - assets->talker.flow.latent_dim, reference_latents.end());
            const auto generation_start = std::chrono::steady_clock::now();
            const auto generated = talker.generate(official_inputs ? expected_inputs : inputs, {}, offset, history);
            const auto waveform = decoder.decode(generated);
            engine::debug::timing_log_scalar("ming3_audio.test.normal_generation_ms", engine::debug::elapsed_ms(generation_start));
            engine::audio::write_wav(generated_output, assets->talker.codec.sample_rate, 1, waveform, options);
            std::cout << "normal_generation_inputs=" << (official_inputs ? "official" : "native") << "\n";
        }
        return inputs_identical && cosine > 0.99999 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "ming3_audio_talker_parity failed: " << error.what() << '\n';
        return 2;
    }
}
