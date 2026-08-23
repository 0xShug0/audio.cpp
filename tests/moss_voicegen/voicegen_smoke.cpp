// End-to-end smoke test for MOSS-VoiceGenerator: designs a voice from a written
// instruction, speaks the given text in it, and writes a WAV.
//
//   moss_voicegen_smoke --model <dir> --instruct "<voice description>" --text "<sentence>" \
//       --language English --output out.wav [--weight-type bf16] [--seed 0] [--threads N]

#include "engine/community_models/moss_voicegen/assets.h"
#include "engine/community_models/moss_voicegen/backbone.h"
#include "engine/community_models/moss_voicegen/delay_decoder.h"
#include "engine/community_models/moss_voicegen/heads.h"
#include "engine/community_models/moss_voicegen/tokenizer_text.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"
#include "engine/models/moss/shared/token_rows.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    if (value == "f32") {
        return engine::assets::TensorStorageType::F32;
    }
    if (value == "bf16") {
        return engine::assets::TensorStorageType::BF16;
    }
    if (value == "q8_0") {
        return engine::assets::TensorStorageType::Q8_0;
    }
    throw std::runtime_error("unsupported --weight-type: " + value);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_dir = arg_value(argc, argv, "--model", "");
        const std::string instruction = arg_value(argc, argv, "--instruct", "");
        const std::string text = arg_value(argc, argv, "--text", "");
        const std::string language = arg_value(argc, argv, "--language", "English");
        const std::string output = arg_value(argc, argv, "--output", "moss_voicegen.wav");
        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "bf16"));
        const auto seed = static_cast<uint32_t>(std::stoul(arg_value(argc, argv, "--seed", "0")));
        const int threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        if (model_dir.empty() || text.empty()) {
            std::cerr << "usage: moss_voicegen_smoke --model <dir> --text <sentence>"
                         " [--instruct <voice>] [--output out.wav]\n";
            return 2;
        }

        const auto assets = engine::models::moss_voicegen::load_moss_voicegen_assets(model_dir);
        const auto & config = assets->config;
        const int64_t n_vq = config.num_codebooks;
        const int64_t hidden_size = config.backbone.hidden_size;

        const engine::models::moss_voicegen::MossVoiceGenTextProcessor processor(assets);
        const auto prompt = processor.build_generation_prefix(text, instruction, language);
        const auto prompt_rows = static_cast<int64_t>(prompt.text_tokens.size());

        // With no reference recording there is nothing anchoring the duration, so bound it
        // from the text. Roughly 12.5 frames a second at this codec's hop; the floor keeps
        // the model from collapsing on the first frame, the ceiling from rambling.
        const auto characters = static_cast<int64_t>(text.size());
        const int64_t min_frames = std::max<int64_t>(8, characters / 6);
        const int64_t max_frames = std::max<int64_t>(min_frames * 4, characters / 2 + 50);
        const int64_t max_steps = max_frames + n_vq + 4;
        std::cout << "prompt rows=" << prompt_rows << " frame bounds=[" << min_frames << ", "
                  << max_frames << "] max steps=" << max_steps << "\n";

        engine::models::moss::AudioCodebookSpec codebook_spec;
        codebook_spec.hidden_size = hidden_size;
        codebook_spec.num_codebooks = n_vq;
        codebook_spec.audio_vocab_size = config.audio_vocab_size + 1;
        codebook_spec.audio_pad_token_id = config.audio_pad_code;
        codebook_spec.tensor_prefix = "emb_ext";
        const engine::models::moss::AudioCodebookEmbeddings codebooks(*assets->model_weights, codebook_spec);

        std::vector<float> prompt_bias(static_cast<size_t>(prompt_rows * hidden_size), 0.0F);
        for (int64_t row = 0; row < prompt_rows; ++row) {
            codebooks.add_bias(
                prompt.audio_codes.data() + static_cast<size_t>(row * n_vq),
                prompt_bias.data() + static_cast<size_t>(row * hidden_size));
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = threads;
        engine::core::ExecutionContext execution_context(backend_config);

        const engine::models::moss_voicegen::MossVoiceGenBackboneRuntime backbone(
            assets, execution_context, 512ull * 1024ull * 1024ull, 8192ull * 1024ull * 1024ull, weight_type);
        const engine::models::moss_voicegen::MossVoiceGenHeadsRuntime heads(
            assets, execution_context, 256ull * 1024ull * 1024ull, 4096ull * 1024ull * 1024ull, weight_type);

        // Checkpoint defaults from the model card; this family degenerates at a generic preset.
        engine::models::moss_voicegen::MossVoiceGenSamplingOptions sampling;
        sampling.text_temperature = std::stof(arg_value(argc, argv, "--text-temperature", "1.5"));
        sampling.audio_temperature = std::stof(arg_value(argc, argv, "--audio-temperature", "1.5"));
        sampling.audio_top_p = std::stof(arg_value(argc, argv, "--audio-top-p", "0.6"));
        sampling.audio_top_k = std::stoi(arg_value(argc, argv, "--audio-top-k", "50"));
        sampling.audio_repetition_penalty =
            std::stof(arg_value(argc, argv, "--audio-repetition-penalty", "1.1"));
        engine::models::moss_voicegen::MossVoiceGenLengthBounds bounds;
        bounds.min_frames = min_frames;
        bounds.max_frames = max_frames;
        engine::models::moss_voicegen::MossVoiceGenDelayDecoder decoder(config, sampling, seed, bounds);

        const auto generation_start = std::chrono::steady_clock::now();
        backbone.begin_generation(prompt_rows + max_steps + 8);
        auto hidden = backbone.prefill(prompt.text_tokens, prompt_bias);

        engine::models::moss_voicegen::MossVoiceGenStepLogits logits;
        std::vector<float> row_bias(static_cast<size_t>(hidden_size), 0.0F);
        int64_t steps = 0;
        std::vector<int32_t> text_trace;
        for (; steps < max_steps; ++steps) {
            heads.evaluate(hidden, logits);
            const auto row = decoder.step(logits);
            text_trace.push_back(row.text_token);
            if (decoder.stopped()) {
                break;
            }
            std::fill(row_bias.begin(), row_bias.end(), 0.0F);
            codebooks.add_bias(row.codes.data(), row_bias.data());
            hidden = backbone.step(row.text_token, row_bias);
        }
        const double generation_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - generation_start).count();

        int64_t codebooks_out = 0;
        int64_t frames = 0;
        const auto flat_codes = decoder.extract_audio_codes(codebooks_out, frames);
        std::cout << "generated steps=" << steps << " frames=" << frames
                  << " in " << generation_seconds << " s\n";
        std::cout << "text token trace (start=" << config.audio_start_token_id
                  << " gen=" << config.audio_assistant_gen_slot_token_id
                  << " delay=" << config.audio_assistant_delay_slot_token_id
                  << " end=" << config.audio_end_token_id
                  << " im_end=" << config.im_end_token_id << "):\n ";
        for (const int32_t token : text_trace) {
            std::cout << " " << token;
        }
        std::cout << "\n";
        if (frames <= 0) {
            std::cerr << "FAIL: no audio frames were produced\n";
            return 1;
        }

        std::vector<std::vector<int32_t>> codes(static_cast<size_t>(codebooks_out));
        for (int64_t codebook = 0; codebook < codebooks_out; ++codebook) {
            codes[static_cast<size_t>(codebook)].assign(
                flat_codes.begin() + static_cast<int64_t>(codebook * frames),
                flat_codes.begin() + static_cast<int64_t>((codebook + 1) * frames));
        }

        const std::string codes_dump = arg_value(argc, argv, "--dump-codes", "");
        if (!codes_dump.empty()) {
            // Plain JSON so the reference codec can decode exactly these codes and the two
            // waveforms can be compared.
            std::ofstream out(codes_dump);
            out << "{\"codebooks\": " << codebooks_out << ", \"frames\": " << frames << ", \"codes\": [";
            for (int64_t codebook = 0; codebook < codebooks_out; ++codebook) {
                out << (codebook == 0 ? "[" : ", [");
                for (int64_t frame = 0; frame < frames; ++frame) {
                    out << (frame == 0 ? "" : ", ") << codes[static_cast<size_t>(codebook)][static_cast<size_t>(frame)];
                }
                out << "]";
            }
            out << "]}\n";
            std::cout << "wrote codes to " << codes_dump << "\n";
        }

        const auto decode_start = std::chrono::steady_clock::now();
        const engine::models::moss::MossAudioTokenizerDecoder codec(
            *assets->audio_tokenizer_weights,
            execution_context,
            codebooks_out,
            4096ull * 1024ull * 1024ull,
            2048ull * 1024ull * 1024ull,
            engine::models::moss::moss_audio_tokenizer_v1_config());
        const auto channels = codec.decode(codes);
        const double decode_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_start).count();

        const auto & waveform = channels.at(0);
        double peak = 0.0;
        double energy = 0.0;
        for (const float sample : waveform) {
            peak = std::max(peak, static_cast<double>(std::fabs(sample)));
            energy += static_cast<double>(sample) * sample;
        }
        const double rms = waveform.empty() ? 0.0 : std::sqrt(energy / static_cast<double>(waveform.size()));
        const double duration =
            static_cast<double>(waveform.size()) / static_cast<double>(codec.sampling_rate());

        engine::audio::write_pcm16_wav(output, static_cast<int>(codec.sampling_rate()), 1, waveform);
        std::cout << "decoded " << waveform.size() << " samples (" << duration << " s at "
                  << codec.sampling_rate() << " Hz) in " << decode_seconds << " s\n";
        std::cout << "peak=" << peak << " rms=" << rms << "\n";
        std::cout << "real-time factor: " << ((generation_seconds + decode_seconds) / duration) << "\n";
        std::cout << "wrote " << output << "\n";
        return peak > 0.0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
