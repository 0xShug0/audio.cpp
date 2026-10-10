#include "engine/framework/codecs/s3gen_runtime.h"
#include "engine/community_models/kitten_tts2/speaker.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        if (argc < 5 || argc > 7) throw std::runtime_error("usage: kitten_tts2_components_probe S3GEN VOICES_JSON CODES_JSON OUTPUT_DIR [cpu|cuda [REFERENCE_MODEL_DIR]]");
        namespace s3 = engine::codecs::s3gen;
        namespace json = engine::io::json;
        engine::core::BackendConfig config;
        config.type = engine::core::BackendType::Cpu;
        if (argc >= 6) {
            const std::string backend = argv[5];
            if (backend == "cuda") config.type = engine::core::BackendType::Cuda;
            else if (backend != "cpu") throw std::runtime_error("probe backend must be cpu or cuda");
        }
        config.threads = 8;
        engine::core::ExecutionContext execution(config);
        std::filesystem::create_directories(argv[4]);
        if (argc == 7) {
            // Test-only speaker comparison through its public component API.
            const auto root = std::filesystem::path(argv[6]);
            auto speaker_source = engine::assets::open_tensor_source(root / "speaker/model.safetensors");
            auto lm_source = engine::assets::open_tensor_source(root / "lm/model.safetensors");
            engine::community_models::kitten_tts2::SpeakerEncoder speaker(*speaker_source, *lm_source, execution);
            const auto voices = json::parse_file(root / "voices/voices.json");
            const auto wav = engine::audio::read_wav_f32(root / "voices" / voices.require("Bruno").require("reference").as_string());
            const auto mono = engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
            const auto input = engine::audio::resample_mono_torchaudio_sinc_hann(mono, wav.sample_rate, 16000,
                engine::audio::torchaudio_sinc_hann_float32_options());
            const auto embedding = speaker.embed(input);
            const auto projected = speaker.project(embedding);
            const auto array = [](const std::vector<float> & values) {
                json::Value::Array result;
                for (float value : values) result.push_back(json::Value::make_number(value));
                return json::Value::make_array(std::move(result));
            };
            json::Value::Object result;
            result.emplace("embedding", array(embedding));
            result.emplace("projection", array(projected));
            std::ofstream(std::filesystem::path(argv[4]) / "speaker.json") << json::stringify(json::Value::make_object(std::move(result)));
            engine::audio::WavWriteOptions options;
            options.format = engine::audio::WavSampleFormat::Float32;
            engine::audio::write_wav(std::filesystem::path(argv[4]) / "speaker_input.wav", 16000, 1, input, options);
        }
        auto source = engine::assets::open_tensor_source(argv[1]);
        s3::S3GenConfig s3gen_config;
        s3gen_config.decoder.variant = s3::S3FlowVariant::MeanFlow;
        s3::S3GenRuntime runtime(source, execution, s3gen_config, {engine::assets::TensorStorageType::F32});
        const auto all = json::parse_file(argv[2]);
        const auto & voice = all.require("Bruno");
        s3::S3GenConditioning ref;
        for (const auto & x : voice.require("prompt_token").as_array()[0].as_array()) ref.prompt_tokens.push_back(x.as_number());
        for (const auto & row : voice.require("prompt_feat").as_array()[0].as_array())
            for (const auto & x : row.as_array()) ref.prompt_feat.push_back(x.as_number());
        for (const auto & x : voice.require("embedding").as_array()[0].as_array()) ref.embedding.push_back(x.as_number());
        ref.prompt_token_count = ref.prompt_tokens.size();
        ref.prompt_feat_frames = ref.prompt_feat.size()/80;
        ref.prompt_feat_dims = 80;
        ref.embedding_size = 192;
        std::vector<int32_t> tokens;
        const auto codes = json::parse_file(argv[3]);
        for (const auto & x : codes.as_array()) tokens.push_back(x.as_number());
        tokens.insert(tokens.end(), 3, 4299);
        const int64_t frames = 2 * (ref.prompt_token_count + tokens.size());
        // Isolate the encoder through the runtime's stage API.
        auto table = source->require_f32("flow.input_embedding.weight", {6561,512});
        auto all_tokens = ref.prompt_tokens;
        all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
        std::vector<float> embedded;
        for (auto token : all_tokens) for (int i = 0; i < 512; ++i)
            embedded.push_back(table[token*512+i]);
        auto encoded = runtime.encode(embedded, all_tokens.size(), all_tokens.size(), 512);
        std::ofstream hidden(std::filesystem::path(argv[4])/"encoder.f32", std::ios::binary);
        hidden.write(reinterpret_cast<const char *>(encoded.hidden.data()), encoded.hidden.size()*sizeof(float));
        auto mel = runtime.token_to_mel(ref, tokens, tokens.size(),
            2, 0, false, std::vector<float>(80*frames, 0), 0);
        std::filesystem::create_directories(argv[4]);
        std::ofstream out(std::filesystem::path(argv[4])/"zero_noise_mel.f32", std::ios::binary);
        out.write(reinterpret_cast<const char *>(mel.mel.data()), mel.mel.size()*sizeof(float));
        const auto & vocoder = runtime.vocoder();
        auto f0 = vocoder.predict_f0(mel.mel, mel.frames);
        std::vector<float> random(9 + 9*480*mel.frames, 0);
        auto audio = vocoder.synthesize(mel.mel, mel.frames, 0, 0, &random);
        std::ofstream pitch(std::filesystem::path(argv[4])/"f0.f32", std::ios::binary);
        pitch.write(reinterpret_cast<const char *>(f0.data()), f0.size()*sizeof(float));
        std::ofstream wave(std::filesystem::path(argv[4])/"vocoder.f32", std::ios::binary);
        wave.write(reinterpret_cast<const char *>(audio.waveform.data()), audio.waveform.size()*sizeof(float));
        std::cout << "mel_frames=" << mel.frames << std::endl;
    } catch (const std::exception & e) { std::cerr << e.what() << std::endl; return 1; }
}
