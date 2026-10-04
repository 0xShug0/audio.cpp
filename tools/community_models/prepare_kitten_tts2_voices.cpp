// Offline preset conditioning. All neural inference uses audio.cpp / GGML.
#include "engine/community_models/kitten_tts2/speaker.h"
#include "engine/community_models/kitten_tts2/reference.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace json = engine::io::json;

template <typename T>
json::Value array(const std::vector<T> & values) {
    json::Value::Array out;
    out.reserve(values.size());
    for (const auto value : values) {
        if (!std::isfinite(static_cast<double>(value))) throw std::runtime_error("non-finite voice conditioning");
        out.push_back(json::Value::make_number(value));
    }
    return json::Value::make_array(std::move(out));
}

json::Value batch(json::Value values) {
    return json::Value::make_array({std::move(values)});
}

int main(int argc, char ** argv) {
    try {
        if (argc < 5 || argc > 7)
            throw std::runtime_error("usage: audiocpp_kitten_tts2_prepare_voices MODEL S3GEN MANIFEST OUTPUT [cpu|cuda] [THREADS]");
        namespace kitten = engine::community_models::kitten_tts2;
        namespace s3 = engine::models::chatterbox;
        namespace assets = engine::assets;
        namespace audio = engine::audio;
        engine::core::BackendConfig config;
        config.type = engine::core::BackendType::Cpu;
        if (argc > 5) {
            const std::string backend = argv[5];
            if (backend == "cuda") config.type = engine::core::BackendType::Cuda;
            else if (backend != "cpu") throw std::runtime_error("preparer backend must be cpu or cuda");
        }
        config.threads = argc > 6 ? std::stoi(argv[6]) : 8;
        engine::core::ExecutionContext execution(config);
        const auto model = std::filesystem::path(argv[1]);
        auto lm = assets::open_tensor_source(model / "lm/model.safetensors");
        auto speaker_source = assets::open_tensor_source(model / "speaker/model.safetensors");
        auto source = assets::open_tensor_source(argv[2]);
        kitten::SpeakerEncoder speaker(*speaker_source, *lm, execution);
        auto tokenizer = s3::S3TokenizerComponent::load_from_source(*source, execution, assets::TensorStorageType::F32);
        auto camp = s3::CAMPPlusEncoderComponent::load_from_source(source, execution, assets::TensorStorageType::F32);
        json::Value::Object voices;
        const auto manifest = json::parse_file(argv[3]);
        for (const auto & [name, item] : manifest.as_object()) {
            const auto wav = audio::read_wav_f32(std::filesystem::u8path(json::require_string(item, "reference")));
            if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty())
                throw std::runtime_error("invalid preset reference: " + name);
            for (float sample : wav.samples)
                if (!std::isfinite(sample)) throw std::runtime_error("non-finite reference audio: " + name);
            auto mono = audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
            engine::runtime::AudioBuffer wav24{24000, 1, audio::resample_mono_torchaudio_sinc_hann(
                mono, wav.sample_rate, 24000, audio::torchaudio_sinc_hann_float32_options())};
            const auto ref = kitten::prepare_decoder_reference(std::move(wav24), tokenizer, camp);
            auto identity = json::number_array_as<float>(item.require("identity"));
            if (identity.size() != 512) throw std::runtime_error("invalid preset identity: " + name);
            const auto projected = speaker.project(identity);
            const auto tokens = json::number_array_as<int32_t>(item.require("reference_tokens"));
            if (tokens.empty() || ref.prompt_tokens.empty() || ref.prompt_feat.empty() ||
                ref.prompt_feat.size() % 80 || ref.embedding.size() != 192)
                throw std::runtime_error("invalid preset conditioning: " + name);
            for (auto token : tokens)
                if (token < 0 || token >= 6561) throw std::runtime_error("invalid preset codec token: " + name);
            json::Value::Array mel;
            for (size_t i = 0; i < ref.prompt_feat.size(); i += 80)
                mel.push_back(array(std::vector<float>(ref.prompt_feat.begin() + i, ref.prompt_feat.begin() + i + 80)));
            voices.emplace(name, json::Value::make_object({
                {"transcript", item.require("transcript")}, {"reference_tokens", array(tokens)},
                {"speaker", array(projected)}, {"prompt_token", batch(array(ref.prompt_tokens))},
                {"prompt_feat", batch(json::Value::make_array(std::move(mel)))},
                {"embedding", batch(array(ref.embedding))},
            }));
            std::cout << "Prepared " << name << std::endl;
        }
        std::ofstream output(argv[4], std::ios::binary);
        output << json::stringify(json::Value::make_object(std::move(voices))) << '\n';
        if (!output) throw std::runtime_error("could not write prepared voices");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
