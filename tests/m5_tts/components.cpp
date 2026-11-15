#include "app/cli/args.h"
#include "engine/models/m5_tts/ar.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto reference = minitts::cli::find_arg(argc, argv, "--reference").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cpu");
        const auto mode = minitts::cli::find_arg(argc, argv, "--mode").value_or("offline");
        if (mode != "offline" && mode != "streaming") throw std::runtime_error("Expected offline or streaming mode");
        const int codec_chunk = mode == "streaming" ? 4 : 0;
        const auto check_encoder = minitts::cli::has_arg(argc, argv, "--check-encoder");
        const auto waveform_min_cosine = minitts::cli::parse_optional_float_arg(
            argc, argv, "--min-waveform-cosine").value_or(0.99999F);
        if (!std::isfinite(waveform_min_cosine) || waveform_min_cosine <= 0 || waveform_min_cosine > 1)
            throw std::runtime_error("--min-waveform-cosine must be in (0, 1]");
        const auto out = minitts::cli::find_arg(argc, argv, "--out").value();
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log required");
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        const auto started = std::chrono::steady_clock::now();
        const auto fixtures = engine::assets::open_tensor_source(reference);
        const auto integers = [&](const std::string & name) {
            const auto data = fixtures->require_tensor_data(name);
            if (data.metadata.dtype != "I32") throw std::runtime_error("Expected I32: " + name);
            std::vector<int32_t> result(data.bytes.size() / sizeof(int32_t));
            std::memcpy(result.data(), data.bytes.data(), data.bytes.size());
            return result;
        };
        const auto compare = [&](const std::string & name, const std::vector<float> & actual,
                                 const std::vector<float> & expected) {
            if (actual.size() != expected.size()) throw std::runtime_error(name + " size differs");
            double aa = 0, bb = 0, ab = 0, error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error(name + " nonfinite");
                aa += double(actual[i]) * actual[i]; bb += double(expected[i]) * expected[i];
                ab += double(actual[i]) * expected[i];
                error = std::max(error, std::abs(double(actual[i]) - expected[i]));
            }
            const double cosine = ab / std::sqrt(aa * bb);
            engine::debug::trace_log_scalar("m5_tts.reference." + name + ".cosine", cosine);
            engine::debug::trace_log_scalar("m5_tts.reference." + name + ".max_abs", error);
            return cosine;
        };
        const auto assets = engine::models::m5_tts::load_assets(model);
        const engine::assets::GgufMetadata metadata(model);
        if (metadata.require_string_array("audiocpp.tensor_sources.paths") !=
            std::vector<std::string>{"model.safetensors", "codec.safetensors"})
            throw std::runtime_error("Package source paths must be portable");
        for (const auto & tensor : assets->codec->tensors()) {
            if (engine::assets::tensor_storage_type_for_dtype(tensor.dtype) != engine::assets::TensorStorageType::F32)
                throw std::runtime_error("Expected preserved F32 codec tensor: " + tensor.name);
        }
        for (const auto * name : {"backbone.rope_factors", "decoder.rope_factors"}) {
            if (engine::assets::tensor_storage_type_for_dtype(assets->model->require_metadata(name).dtype) !=
                engine::assets::TensorStorageType::F32)
                throw std::runtime_error("Expected preserved F32 RoPE factors");
        }
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        engine::models::m5_tts::M5LlamaRuntime ar(*assets, execution,
            engine::assets::TensorStorageType::F32, engine::core::AttentionPreference::Eager);
        const auto tokens = integers("prompt.tokens");
        const auto mask = integers("prompt.mask");
        if (tokens.size() != mask.size() || tokens.size() % 17) throw std::runtime_error("Invalid prompt");
        size_t text_rows = 0;
        while (text_rows < tokens.size() / 17 && mask[text_rows * 17 + 16]) ++text_rows;
        std::vector<int32_t> text;
        for (size_t row = 0; row < text_rows; ++row) text.push_back(tokens[row * 17 + 16]);
        const auto index = engine::io::load_safetensors_index(reference);
        engine::tokenizers::LlamaBpeTokenizer tokenizer({{}, {}, assets->resources.require_file("tokenizer_config"),
            assets->resources.require_file("tokenizer_json"), engine::tokenizers::LlamaBpePreTokenizer::Llama3, {}, {}, true});
        auto tokenized = tokenizer.encode(index.metadata.at("conditioned_text"), true);
        tokenized.insert(tokenized.begin(), assets->bos_token_id);
        tokenized.push_back(assets->eos_token_id);
        if (tokenized != text) {
            engine::debug::trace_log_i32("m5_tts.reference.actual_token_ids", {static_cast<int64_t>(tokenized.size())}, tokenized);
            engine::debug::trace_log_i32("m5_tts.reference.expected_token_ids", {static_cast<int64_t>(text.size())}, text);
            throw std::runtime_error("Conditioning text token IDs differ");
        }
        auto prompt = ar.text_embeddings(text);
        for (size_t row = text_rows; row < tokens.size() / 17; ++row) {
            engine::models::m5_tts::AudioFrame frame;
            std::copy_n(tokens.begin() + row * 17, 16, frame.begin());
            const auto embedding = ar.audio_embedding(frame);
            prompt.insert(prompt.end(), embedding.begin(), embedding.end());
        }
        if (compare("prompt", prompt, fixtures->require_f32("backbone.input.0")) < 0.99999)
            throw std::runtime_error("AR input embedding parity failed");
        const auto forced = integers("sampled.tokens");
        ar.begin(prompt, forced.size() / 16);
        size_t selections = 0;
        double minimum_logit_cosine = 1.0;
        const auto selector = [&](int64_t frame, int64_t book, const std::vector<float> & logits, bool) {
            minimum_logit_cosine = std::min(minimum_logit_cosine,
                compare("logits." + std::to_string(selections), logits,
                    fixtures->require_f32("logits." + std::to_string(selections))));
            ++selections;
            return forced.at(static_cast<size_t>(frame * 16 + book));
        };
        engine::codecs::Qwen3SpeechCodes codes;
        codes.code_groups = 16;
        std::vector<int32_t> reference_codes;
        if (assets->variant == "voice_clone") {
            reference_codes = integers("reference.codes");
        }
        if (check_encoder) {
            if (assets->variant != "voice_clone") throw std::runtime_error("Encoder check requires VoiceClone");
            engine::codecs::Qwen3SpeechTokenizerEncoderRuntime encoder(assets->codec, execution,
                16 << 20, engine::assets::TensorStorageType::F32, engine::assets::TensorStorageType::F32, false);
            const auto encoded = encoder.encode({24000, 1, fixtures->require_f32("reference.waveform")});
            engine::io::SafeTensorWriteEntry codes_entry{"reference.codes", "I32",
                {encoded.frames, encoded.code_groups}, std::vector<unsigned char>(encoded.codes.size() * sizeof(int32_t))};
            std::memcpy(codes_entry.data.data(), encoded.codes.data(), codes_entry.data.size());
            auto codes_path = std::filesystem::path(out);
            codes_path.replace_extension(".safetensors");
            engine::io::write_safetensors_file(codes_path, {codes_entry});
            for (int64_t book = 0; book < encoded.code_groups; ++book) {
                int64_t equal = 0;
                for (int64_t frame = 0; frame < encoded.frames; ++frame) {
                    const size_t offset = static_cast<size_t>(frame * encoded.code_groups + book);
                    if (encoded.codes.at(offset) == reference_codes.at(offset)) ++equal;
                }
                engine::debug::trace_log_scalar("m5_tts.reference.encoder.codebook_" + std::to_string(book) + ".agreement",
                    static_cast<double>(equal) / encoded.frames);
            }
            if (encoded.codes != reference_codes) {
                engine::debug::trace_log_i32("m5_tts.reference.actual_reference_codes", {encoded.frames, encoded.code_groups}, encoded.codes);
                engine::debug::trace_log_i32("m5_tts.reference.expected_reference_codes", {encoded.frames, encoded.code_groups}, reference_codes);
                throw std::runtime_error("Reference encoder codes differ");
            }
        }
        codes.codes = reference_codes;
        for (size_t frame = 0; frame < forced.size() / 16; ++frame) {
            const auto generated = ar.next_frame(selector);
            if (std::all_of(generated.begin(), generated.end(), [](int32_t x) { return x == 0; })) break;
            for (const auto token : generated) codes.codes.push_back(std::clamp(token - 1, 0, 2047));
        }
        if (codes.codes != integers("codec.codes")) throw std::runtime_error("Teacher-forced codec codes differ");
        if (minimum_logit_cosine < 0.99999) throw std::runtime_error("AR logit parity failed");
        codes.frames = codes.codes.size() / 16;
        engine::codecs::Qwen3SpeechTokenizerDecoderRuntime decoder(assets->codec, assets->codec_config,
            execution, 16 << 20, 4 << 20, engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F32, false, codec_chunk > 0);
        engine::runtime::AudioBuffer audio{24000, 1, {}};
        const auto decode_started = std::chrono::steady_clock::now();
        if (codec_chunk > 0) {
            for (int64_t start = 0; start < codes.frames; start += codec_chunk) {
                engine::codecs::Qwen3SpeechCodes chunk;
                chunk.code_groups = 16;
                chunk.frames = std::min<int64_t>(codec_chunk, codes.frames - start);
                chunk.codes.assign(codes.codes.begin() + start * 16,
                    codes.codes.begin() + (start + chunk.frames) * 16);
                auto decoded = decoder.decode_stream_chunk(chunk, 0, codec_chunk);
                audio.samples.insert(audio.samples.end(), decoded.samples.begin(), decoded.samples.end());
            }
        } else audio = decoder.decode(codes);
        engine::debug::timing_log_scalar("m5_tts.reference.codec_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_started).count());
        const size_t skip = reference_codes.size() / 16 * 1920;
        if (audio.samples.size() <= skip + 2400) throw std::runtime_error("Output too short");
        std::vector<float> waveform(audio.samples.begin() + skip, audio.samples.end() - 2400);
        const auto cosine = compare("waveform", waveform, fixtures->require_f32("waveform"));
        engine::audio::WavWriteOptions options;
        options.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(out, 24000, 1, waveform, options);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
        if (cosine < waveform_min_cosine) throw std::runtime_error("Teacher-forced waveform parity failed");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
