#include "engine/models/plapre/kanade.h"
#include "engine/models/plapre/assets.h"
#include "engine/models/plapre/prompt.h"
#include "engine/models/plapre/frontend.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/sampling/torch_random.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) try {
    std::filesystem::path weights, reference, output;
    engine::core::BackendConfig backend;
    backend.type = engine::core::BackendType::Cuda;
    backend.threads = 8;
    bool log = false;
    bool waveform = false;
    bool inputs_only = false;
    bool generate = false;
    bool synthesize = false;
    bool encode = false;
    bool ssl_only = false;
    bool vocoder_only = false;
    bool f32_control = false;
    bool native_noise = false;
    bool repeat = false;
    float temperature = 0;
    std::string text;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--log") log = true;
        else if (arg == "--waveform") waveform = true;
        else if (arg == "--inputs-only") inputs_only = true;
        else if (arg == "--generate") generate = true;
        else if (arg == "--synthesize") { generate = true; synthesize = true; }
        else if (arg == "--encode") encode = true;
        else if (arg == "--ssl-only") { encode = true; ssl_only = true; }
        else if (arg == "--vocoder-only") vocoder_only = true;
        else if (arg == "--f32-control") f32_control = true;
        else if (arg == "--native-noise") native_noise = true;
        else if (arg == "--repeat") repeat = true;
        else if (arg == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
        else if (arg == "--text" && i + 1 < argc) text = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights = argv[++i];
        else if (arg == "--reference" && i + 1 < argc) reference = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else if (arg == "--backend" && i + 1 < argc) {
            const std::string name = argv[++i];
            if (name == "cuda") backend.type = engine::core::BackendType::Cuda;
            else if (name == "vulkan") backend.type = engine::core::BackendType::Vulkan;
            else if (name == "cpu") backend.type = engine::core::BackendType::Cpu;
            else throw std::runtime_error("unsupported backend: " + name);
        } else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (weights.empty() || reference.empty() || !log) {
        throw std::runtime_error("usage: plapre_codec_parity --weights <safetensors> --reference <safetensors> --backend cuda|vulkan|cpu --log");
    }
    engine::debug::configure_logging({true, std::nullopt});
    std::shared_ptr<const engine::models::plapre::PlapreAssets> assets;
    if (weights.extension() == ".gguf") assets = engine::models::plapre::load_plapre_assets(weights);
    const auto source = assets ? assets->resources.open_tensor_source("kanade")
                               : engine::assets::open_tensor_source(weights);
    const auto fixture = engine::assets::open_tensor_source(reference);
    if (encode) {
        if (!assets) throw std::runtime_error("encoder parity requires the complete Plapre GGUF");
        engine::core::ExecutionContext execution(backend);
        const auto start = std::chrono::steady_clock::now();
        const auto audio = fixture->has_tensor("audio_original")
            ? engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                fixture->require_f32("audio_original"),
                static_cast<int>(fixture->require_f32("original_sample_rate").at(0)), 1, 24000,
                engine::audio::torchaudio_sinc_hann_float32_options())
            : fixture->require_f32("audio_24khz");
        if (ssl_only) {
            std::vector<float> padded(audio.size() + 600, 0.0f);
            std::copy(audio.begin(), audio.end(), padded.begin() + 300);
            engine::audio::TorchaudioSincHannResampleOptions resampling;
            resampling.accumulation = engine::audio::TorchaudioSincHannAccumulation::Float32;
            const auto samples = engine::audio::resample_mono_torchaudio_sinc_hann(padded, 24000, 16000, resampling);
            auto wavlm = engine::modules::WavlmEncoderComponent::load_from_tensor_source(
                *assets->resources.open_tensor_source("wavlm"), backend, assets->encoder_config.wavlm);
            const std::vector<int64_t> layers{1, 2, 3, 4, 6, 9};
            const auto features = wavlm.encode_layers(samples, 1, static_cast<int64_t>(samples.size()), layers);
            std::vector<engine::io::SafeTensorWriteEntry> outputs;
            if (!output.empty()) {
                engine::io::SafeTensorWriteEntry entry{"ssl_resampled", "F32", {1, static_cast<int64_t>(samples.size())}, {}};
                entry.data.resize(samples.size() * sizeof(float));
                std::memcpy(entry.data.data(), samples.data(), entry.data.size());
                outputs.push_back(std::move(entry));
            }
            for (size_t layer = 0; layer < layers.size(); ++layer) {
                const auto name = "ssl_layer." + std::to_string(layers[layer]);
                const auto & actual = features.hidden_states[layer];
                const auto expected = fixture->require_f32(name);
                if (actual.size() != expected.size()) throw std::runtime_error("WavLM layer shape mismatch");
                double dot = 0, norm_a = 0, norm_b = 0, error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const double a = actual[i], b = expected[i];
                    dot += a * b; norm_a += a * a; norm_b += b * b; error += (a - b) * (a - b);
                }
                engine::debug::trace_log_scalar("plapre.parity." + name + ".cosine", dot / std::sqrt(norm_a * norm_b));
                engine::debug::trace_log_scalar("plapre.parity." + name + ".rmse", std::sqrt(error / actual.size()));
                if (!output.empty()) {
                    engine::io::SafeTensorWriteEntry entry{name, "F32", {1, features.tokens, features.hidden_size}, {}};
                    entry.data.resize(actual.size() * sizeof(float));
                    std::memcpy(entry.data.data(), actual.data(), entry.data.size());
                    outputs.push_back(std::move(entry));
                }
            }
            if (!output.empty()) engine::io::write_safetensors_file(output, outputs);
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
            return 0;
        }
        engine::models::plapre::KanadeAudioEncoder encoder(source,
            assets->resources.open_tensor_source("wavlm"), assets->encoder_config, execution);
        const auto features = encoder.encode(audio);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        const auto expected_codes = fixture->require_f32("codes");
        const auto expected_speaker = fixture->require_f32("speaker", {128});
        size_t mismatches = 0;
        for (size_t i = 0; i < std::min(features.codes.size(), expected_codes.size()); ++i) {
            if (features.codes[i] != expected_codes[i]) ++mismatches;
        }
        double dot = 0, norm_a = 0, norm_b = 0, error = 0;
        for (size_t i = 0; i < features.speaker.size(); ++i) {
            const double a = features.speaker[i], b = expected_speaker.at(i);
            dot += a * b; norm_a += a * a; norm_b += b * b; error += (a - b) * (a - b);
        }
        const auto cosine = dot / std::sqrt(norm_a * norm_b);
        engine::debug::trace_log_scalar("plapre.parity.encoder.code_mismatches", mismatches);
        engine::debug::trace_log_scalar("plapre.parity.encoder.speaker_cosine", cosine);
        engine::debug::trace_log_scalar("plapre.parity.encoder.speaker_rmse", std::sqrt(error / features.speaker.size()));
        std::cout << "encoder codes cpp=" << features.codes.size() << " python=" << expected_codes.size()
                  << " mismatches=" << mismatches << " speaker cosine=" << cosine << '\n';
        if (!output.empty()) {
            engine::io::SafeTensorWriteEntry codes{"codes", "I32", {static_cast<int64_t>(features.codes.size())}, {}};
            codes.data.resize(features.codes.size() * sizeof(int32_t));
            std::memcpy(codes.data.data(), features.codes.data(), codes.data.size());
            engine::io::SafeTensorWriteEntry speaker{"speaker", "F32", {128}, {}};
            speaker.data.resize(features.speaker.size() * sizeof(float));
            std::memcpy(speaker.data.data(), features.speaker.data(), speaker.data.size());
            engine::io::SafeTensorWriteEntry resampled{"audio_24khz", "F32", {static_cast<int64_t>(audio.size())}, {}};
            resampled.data.resize(audio.size() * sizeof(float));
            std::memcpy(resampled.data.data(), audio.data(), resampled.data.size());
            engine::io::write_safetensors_file(output, {codes, speaker, resampled});
        }
        return mismatches == 0 && features.codes.size() == expected_codes.size() && cosine > 0.99999 ? 0 : 1;
    }
    if (inputs_only || generate) {
        if (!assets || text.empty()) throw std::runtime_error("AR input parity requires a GGUF and --text");
        engine::models::plapre::PlapreTextFrontend frontend(assets->resources);
        const engine::models::plapre::PlaprePromptBuilder builder(frontend.tokenizer());
        const auto ids = [&](const std::string & name) {
            std::vector<int32_t> out;
            for (const auto value : fixture->require_f32(name)) {
                if (!std::isfinite(value) || std::floor(value) != value) throw std::runtime_error("noninteger prompt fixture: " + name);
                out.push_back(static_cast<int32_t>(value));
            }
            return out;
        };
        engine::models::plapre::PlaprePromptControls controls;
        if (fixture->has_tensor("reference_codes")) {
            controls.voice_reference = engine::models::plapre::PlapreAudioReference{
                ids("reference_text_ids"), ids("reference_codes")};
        }
        if (fixture->has_tensor("duration_frames")) {
            const auto values = ids("duration_frames");
            controls.duration_frames = std::vector<int64_t>(values.begin(), values.end());
        }
        if (fixture->has_tensor("context_text_ids")) {
            controls.context_text = ids("context_text_ids");
            controls.same_speaker = fixture->require_f32("same_speaker").at(0) != 0;
            if (fixture->has_tensor("context_codes")) controls.context_codes = ids("context_codes");
        }
        if (fixture->has_tensor("lex_codes")) controls.pronunciations.push_back({ids("lex_text_ids"), ids("lex_codes")});
        const bool editing = fixture->has_tensor("edit_range");
        const auto prompt = editing ? builder.edit(frontend.encode(text, false), ids("original_codes"),
            ids("edit_range").at(0), ids("edit_range").at(1), controls.pronunciations)
            : builder.generate(frontend.encode(text, true), controls);
        const auto expected_ids = fixture->require_f32("prompt_ids");
        if (prompt.size() != expected_ids.size()) throw std::runtime_error("AR prompt length mismatch");
        for (size_t i = 0; i < prompt.size(); ++i) {
            if (prompt[i] != expected_ids[i]) throw std::runtime_error("AR prompt token mismatch at " + std::to_string(i));
        }
        engine::core::ExecutionContext execution(backend);
        engine::models::plapre::PlapreLlamaGenerator generator(assets->resources.open_tensor_source("ar"),
            assets->resources.open_tensor_source("speaker_projection"), assets->ar_config, execution);
        const auto start = std::chrono::steady_clock::now();
        const auto actual = generator.prepare_inputs(prompt, fixture->require_f32("speaker", {128}));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        const auto expected = fixture->require_f32("prompt_embeddings");
        if (actual.size() != expected.size()) throw std::runtime_error("AR input embedding shape mismatch");
        size_t mismatches = 0;
        double max_abs = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != expected[i]) ++mismatches;
            max_abs = std::max(max_abs, std::abs(static_cast<double>(actual[i]) - expected[i]));
        }
        engine::debug::trace_log_scalar("plapre.parity.input_embedding_mismatches", mismatches);
        engine::debug::trace_log_scalar("plapre.parity.input_embedding_max_abs", max_abs);
        std::cout << "AR prompt ids match; embedding mismatches=" << mismatches << " max_abs=" << max_abs << '\n';
        if (mismatches != 0 || !generate) return mismatches == 0 ? 0 : 1;
        engine::models::plapre::PlapreSamplingOptions options;
        options.temperature = temperature;
        options.max_tokens = 500;
        options.seed = 42;
        options.stop_ids = builder.stop_ids(false);
        const auto generated = generator.generate(actual, options);
        if (repeat) {
            const auto repeated = generator.generate(actual, options);
            engine::debug::trace_log_scalar("plapre.parity.ar.repeat_equal", generated == repeated);
            if (generated != repeated) {
                size_t first = 0;
                while (first < std::min(generated.size(), repeated.size()) && generated[first] == repeated[first]) ++first;
                std::cout << "AR repeat lengths=" << generated.size() << ',' << repeated.size() << " first divergence=" << first << '\n';
                throw std::runtime_error("Plapre AR repeated tokens differ");
            }
        }
        const auto expected_tokens = temperature == 0 ? fixture->require_f32("generated_ids") : std::vector<float>{};
        size_t token_mismatches = 0;
        for (size_t i = 0; i < std::min(generated.size(), expected_tokens.size()); ++i) {
            if (generated[i] != expected_tokens[i]) {
                if (token_mismatches == 0) {
                    std::cout << "first greedy divergence at " << i << ": cpp=" << generated[i]
                              << " python=" << expected_tokens[i] << '\n';
                }
                ++token_mismatches;
            }
        }
        if (temperature == 0) {
            std::cout << "greedy tokens cpp=" << generated.size() << " python=" << expected_tokens.size()
                      << " mismatches=" << token_mismatches << '\n';
        } else {
            std::cout << "sampled tokens cpp=" << generated.size() << '\n';
        }
        if (!output.empty()) {
            engine::io::SafeTensorWriteEntry entry{"generated_ids", "I32",
                {static_cast<int64_t>(generated.size())}, {}};
            entry.data.resize(generated.size() * sizeof(int32_t));
            std::memcpy(entry.data.data(), generated.data(), entry.data.size());
            engine::io::write_safetensors_file(output, {entry});
        }
        if (synthesize) {
            if (output.empty()) throw std::runtime_error("synthesis validation requires --output");
            const auto codes = builder.audio_codes(generated);
            auto config = assets->mel_config;
            if (f32_control) config.bf16_autocast = false;
            engine::models::plapre::KanadeMelDecoder decoder(
                source, config, execution);
            const auto mel = decoder.decode(codes, fixture->require_f32("speaker", {128}));
            auto vocoder = engine::modules::HiftVocoderComponent::load_from_tensor_source(
                assets->resources.open_tensor_source("hift"), backend, assets->hift_config);
            const auto random = fixture->has_tensor("source_random")
                ? fixture->require_f32("source_random") : std::vector<float>{};
            const auto audio = vocoder.synthesize(mel, static_cast<int64_t>(codes.size()) * 2,
                42, 0, random.empty() || native_noise ? nullptr : &random);
            auto wav = output;
            wav.replace_extension(".wav");
            engine::audio::write_wav(wav, 24000, 1, audio.waveform);
            std::cout << "unforced waveform=" << wav << " duration=" << audio.waveform.size() / 24000.0 << '\n';
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return temperature > 0 || (token_mismatches == 0 && generated.size() == expected_tokens.size()) ? 0 : 1;
    }
    const auto raw_codes = fixture->require_f32("codes");
    const auto speaker = fixture->require_f32("speaker", {128});
    const auto expected = fixture->require_f32("mel", {80, static_cast<int64_t>(raw_codes.size()) * 2});
    std::vector<int32_t> codes;
    for (const auto value : raw_codes) {
        if (!std::isfinite(value) || value < 0 || value >= 12800 || std::floor(value) != value) {
            throw std::runtime_error("invalid reference audio code");
        }
        codes.push_back(static_cast<int32_t>(value));
    }
    engine::core::ExecutionContext execution(backend);
    auto mel_config = assets ? assets->mel_config : engine::models::plapre::KanadeMelConfig{};
    if (f32_control) mel_config.bf16_autocast = false;
    engine::models::plapre::KanadeMelDecoder decoder(source, mel_config, execution);
    const auto start = std::chrono::steady_clock::now();
    const auto actual = decoder.decode(codes, speaker);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
    if (actual.size() != expected.size()) throw std::runtime_error("Kanade mel shape mismatch");
    double error = 0, dot = 0, norm_a = 0, norm_b = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) throw std::runtime_error("Kanade produced a nonfinite mel value");
        const double a = actual[i], b = expected[i], difference = a - b;
        error += difference * difference;
        maximum = std::max(maximum, std::abs(difference));
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    const double rmse = std::sqrt(error / actual.size());
    const double cosine = dot / std::sqrt(norm_a * norm_b);
    engine::debug::trace_log_scalar("plapre.parity.mel_rmse", rmse);
    engine::debug::trace_log_scalar("plapre.parity.mel_cosine", cosine);
    engine::debug::trace_log_scalar("plapre.parity.mel_max_abs", maximum);
    std::cout << "mel RMSE=" << rmse << " cosine=" << cosine << " max_abs=" << maximum << '\n';
    bool pass = rmse <= 0.05 && cosine >= 0.999;
    std::vector<engine::io::SafeTensorWriteEntry> outputs;
    const auto record = [&](std::string name, const std::vector<float> & values, std::vector<int64_t> shape) {
        engine::io::SafeTensorWriteEntry entry{std::move(name), "F32", std::move(shape), {}};
        entry.data.resize(values.size() * sizeof(float));
        std::memcpy(entry.data.data(), values.data(), entry.data.size());
        outputs.push_back(std::move(entry));
    };
    if (!output.empty()) record("mel", actual, {80, static_cast<int64_t>(codes.size()) * 2});
    if (waveform) {
        if (!assets) throw std::runtime_error("waveform parity requires the complete Plapre GGUF");
        auto vocoder = engine::modules::HiftVocoderComponent::load_from_tensor_source(
            assets->resources.open_tensor_source("hift"), backend, assets->hift_config);
        const auto random = fixture->require_f32("source_random");
        if (native_noise) {
            const auto phase_count = static_cast<size_t>(assets->hift_config.nb_harmonics + 1);
            const auto count = random.size() - phase_count;
            const auto policy = engine::sampling::resolve_torch_cuda_sampling_policy(
                backend.type, backend.device, "plapre.parity", "Plapre codec");
            const auto candidate = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                count, 42, engine::sampling::torch_cuda_tensor_iterator_offset_blocks(phase_count, policy), policy);
            const auto previous = engine::sampling::generate_torch_cuda_randn(
                count, 42, engine::sampling::TorchRandnPrecision::Float32, phase_count);
            double error = 0, previous_error = 0;
            for (size_t i = 0; i < count; ++i) {
                const double expected = random[phase_count + i];
                const auto index = (i % phase_count) * (count / phase_count) + i / phase_count;
                error += std::pow(candidate[index] - expected, 2);
                previous_error += std::pow(previous[i] - expected, 2);
            }
            engine::debug::trace_log_scalar("plapre.parity.native_noise_rmse", std::sqrt(error / count));
            engine::debug::trace_log_scalar("plapre.parity.previous_noise_rmse", std::sqrt(previous_error / count));
        }
        const auto expected_f0 = fixture->require_f32("f0");
        const auto & vocoder_mel = vocoder_only ? expected : actual;
        const auto f0 = vocoder.predict_f0(vocoder_mel, static_cast<int64_t>(codes.size()) * 2);
        if (!output.empty()) record("f0", f0, {static_cast<int64_t>(f0.size())});
        double f0_error = 0, f0_max = 0;
        for (size_t i = 0; i < f0.size(); ++i) {
            const double difference = f0[i] - expected_f0.at(i);
            f0_error += difference * difference;
            f0_max = std::max(f0_max, std::abs(difference));
        }
        engine::debug::trace_log_scalar("plapre.parity.f0_rmse", std::sqrt(f0_error / f0.size()));
        engine::debug::trace_log_scalar("plapre.parity.f0_max_abs", f0_max);
        const auto expected_audio = fixture->require_f32("waveform");
        const auto audio = vocoder.synthesize(vocoder_mel, static_cast<int64_t>(codes.size()) * 2, 42, 0,
            native_noise ? nullptr : &random);
        if (repeat) {
            const auto repeated_mel = decoder.decode(codes, speaker);
            if (repeated_mel.size() != actual.size()) throw std::runtime_error("Plapre repeated mel shape changed");
            double mel_max = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(repeated_mel[i])) throw std::runtime_error("Plapre repeated mel is nonfinite");
                mel_max = std::max(mel_max, std::abs(double(actual[i]) - repeated_mel[i]));
            }
            engine::debug::trace_log_scalar("plapre.parity.mel.repeat_max_abs", mel_max);
            const auto repeated = vocoder.synthesize(vocoder_only ? expected : repeated_mel, static_cast<int64_t>(codes.size()) * 2, 42, 0,
                native_noise ? nullptr : &random);
            double dot = 0, norm_a = 0, norm_b = 0;
            for (size_t i = 0; i < audio.waveform.size(); ++i) {
                const double a = audio.waveform[i], b = repeated.waveform.at(i);
                if (!std::isfinite(b)) throw std::runtime_error("Plapre repeated waveform is nonfinite");
                dot += a * b; norm_a += a * a; norm_b += b * b;
            }
            const auto cosine = dot / std::sqrt(norm_a * norm_b);
            engine::debug::trace_log_scalar("plapre.parity.vocoder.repeat_cosine", cosine);
            if (mel_max > 1e-5 || cosine < 0.99999) throw std::runtime_error("Plapre codec repeated output changed");
        }
        if (!output.empty()) record("waveform", audio.waveform, {static_cast<int64_t>(audio.waveform.size())});
        if (audio.waveform.size() != expected_audio.size()) throw std::runtime_error("HiFT waveform length mismatch");
        double wave_error = 0, wave_dot = 0, wave_a = 0, wave_b = 0;
        for (size_t i = 0; i < expected_audio.size(); ++i) {
            const double a = audio.waveform[i], b = expected_audio[i];
            if (!std::isfinite(a)) throw std::runtime_error("HiFT waveform is nonfinite");
            wave_error += (a - b) * (a - b);
            wave_dot += a * b;
            wave_a += a * a;
            wave_b += b * b;
        }
        const auto wave_rmse = std::sqrt(wave_error / expected_audio.size());
        const auto wave_cosine = wave_dot / std::sqrt(wave_a * wave_b);
        engine::debug::trace_log_scalar("plapre.parity.waveform_rmse", wave_rmse);
        engine::debug::trace_log_scalar("plapre.parity.waveform_cosine", wave_cosine);
        std::cout << "HiFT RMSE=" << wave_rmse << " cosine=" << wave_cosine << '\n';
        pass = pass && wave_rmse <= 0.001 && wave_cosine > 0.99999;
    }
    if (!output.empty()) engine::io::write_safetensors_file(output, outputs);
    return pass ? 0 : 1;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
