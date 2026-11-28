#include "app/cli/args.h"
#include "engine/models/voxmel_tts/ar.h"
#include "engine/models/voxmel_tts/runtime.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/vocoders/vocos_vocoder.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

int main(int argc, char ** argv) {
    try {
        using namespace engine;
        using namespace engine::models::voxmel_tts;
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto reference = minitts::cli::find_arg(argc, argv, "--reference").value();
        const auto audio_path = minitts::cli::find_arg(argc, argv, "--reference-audio").value();
        const auto output = minitts::cli::find_arg(argc, argv, "--out").value();
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto preference = core::parse_attention_preference(
            minitts::cli::find_arg(argc, argv, "--attention").value_or("eager"), "attention");
        if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log required");
        minitts::cli::require_known_args(argc, argv);
        debug::configure_logging({true, log});
        const auto started = std::chrono::steady_clock::now();
        auto fixture = assets::open_tensor_source(reference);
        auto a = load_assets(model);
        core::ExecutionContext execution({backend == "cuda" ? core::BackendType::Cuda :
            backend == "vulkan" ? core::BackendType::Vulkan : core::BackendType::Cpu, 0, 8});
        bool passed = true;
        const auto compare = [&](const std::string & name, const std::vector<float> & got, const std::vector<float> & expected) {
            if (got.size() != expected.size()) throw std::runtime_error(name + " size mismatch");
            double aa = 0, bb = 0, ab = 0, maximum = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (!std::isfinite(got[i])) throw std::runtime_error(name + " nonfinite");
                aa += double(got[i]) * got[i]; bb += double(expected[i]) * expected[i]; ab += double(got[i]) * expected[i];
                maximum = std::max(maximum, std::abs(double(got[i]) - expected[i]));
            }
            const double cosine = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : maximum == 0 ? 1 : 0;
            debug::timing_log_scalar("voxmel_tts.parity." + name + ".cosine", cosine);
            debug::timing_log_scalar("voxmel_tts.parity." + name + ".max_abs", maximum);
            if (cosine < 0.99999) passed = false;
            return cosine;
        };
        const auto integers = [&](const std::string & name) {
            const auto data = fixture->require_tensor_data(name);
            if (data.metadata.dtype != "I32") throw std::runtime_error("Expected I32: " + name);
            std::vector<int32_t> out(data.bytes.size() / sizeof(int32_t));
            std::memcpy(out.data(), data.bytes.data(), data.bytes.size()); return out;
        };
        auto recording = audio::read_wav_f32(std::filesystem::path(audio_path));
        auto mono = audio::mixdown_interleaved_to_mono_average(recording.samples, recording.channels);
        mono = audio::resample_mono_torchaudio_sinc_hann(mono, recording.sample_rate, 24000,
            audio::torchaudio_sinc_hann_float32_options());
        compare("reference_mel", reference_mel(*a, mono, 8), fixture->require_f32("reference.mel"));
        const auto metadata = io::json::parse_file(std::filesystem::path(reference).replace_extension(".json"));
        tokenizers::LlamaBpeTokenizer tokenizer({{}, {}, a->resources.require_file("tokenizer_config"),
            a->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Qwen2});
        auto target = metadata.require("text").as_string();
        if (!target.empty() && static_cast<unsigned char>(target.front()) < 128) target = " " + target;
        const auto target_ids = tokenizer.encode(target, true);
        const auto reference_ids = tokenizer.encode(metadata.require("reference_text").as_string(), true);
        if (target_ids != integers("target.ids") || reference_ids != integers("reference.ids"))
            throw std::runtime_error("VoxMel token IDs differ from official tokenizer");
        std::vector<float> coarse;
        {
            VoxMelQwenRuntime qwen(*a, execution, assets::TensorStorageType::F32, preference);
            std::vector<int32_t> all_ids = reference_ids;
            all_ids.insert(all_ids.end(), target_ids.begin(), target_ids.end());
            const auto words = qwen.text_embeddings(all_ids);
            const auto first = fixture->require_f32("ar.0.input");
            compare("text_embeddings", words, std::vector<float>(first.begin() + a->hidden,
                first.begin() + (1 + static_cast<int64_t>(all_ids.size())) * a->hidden));
            const auto shape = fixture->require_metadata("ar.0.input").shape;
            const int64_t branches = shape.at(0), prefix_size = shape.at(1) * shape.at(2);
            std::vector<std::vector<float>> prefixes;
            for (int64_t b = 0; b < branches; ++b)
                prefixes.emplace_back(first.begin() + b * prefix_size, first.begin() + (b + 1) * prefix_size);
            auto state = qwen.begin(prefixes, metadata.require("ar_steps").as_i64());
            for (int64_t i = 0; i < metadata.require("ar_steps").as_i64(); ++i) {
                compare("ar_hidden_" + std::to_string(i), state.hidden, fixture->require_f32("ar." + std::to_string(i) + ".hidden"));
                const auto expected_stop = fixture->require_f32("ar." + std::to_string(i) + ".stop").front();
                compare("ar_stop_" + std::to_string(i), {state.stop_logit}, {expected_stop});
                if ((state.stop_logit > 0) != (expected_stop > 0))
                    throw std::runtime_error("AR stop decision differs at step " + std::to_string(i));
                const auto name = "coarse." + std::to_string(i) + ".output";
                if (fixture->has_tensor(name)) {
                    const auto frame = fixture->require_f32(name);
                    coarse.insert(coarse.end(), frame.begin(), frame.end());
                    if (i + 1 < metadata.require("ar_steps").as_i64()) state = qwen.advance(frame);
                }
            }
        }
        {
            VoxMelFlowRuntime flow(*a, execution, assets::TensorStorageType::F32, "coarse_flow", a->hidden);
            for (int64_t i = 0; fixture->has_tensor("coarse." + std::to_string(i) + ".output"); ++i) {
                const auto prefix = "coarse." + std::to_string(i);
                auto cond = fixture->require_f32(prefix + ".condition");
                const auto drop = fixture->require_f32(prefix + ".drop_text");
                const auto both = fixture->require_f32(prefix + ".drop_both");
                cond.insert(cond.end(), drop.begin(), drop.end()); cond.insert(cond.end(), both.begin(), both.end());
                compare("coarse_flow_" + std::to_string(i), flow.sample(cond, 1, 3,
                    fixture->require_f32(prefix + ".noise"), 32, 3.5f, 3.0f), fixture->require_f32(prefix + ".output"));
            }
        }
        auto x = coarse;
        const int64_t channels = a->mels / 2;
        for (size_t tier = 0; tier < 2; ++tier) {
            const auto name = "tier." + std::to_string(tier);
            const auto expected_known = fixture->require_f32(name + ".known");
            const size_t prefix_size = expected_known.size() - x.size();
            std::vector<float> known(expected_known.begin(), expected_known.begin() + prefix_size);
            known.insert(known.end(), x.begin(), x.end());
            VoxMelTierRuntime context(*a, execution, assets::TensorStorageType::F32, preference, tier);
            const auto conditioning = context.condition(known);
            compare(name + "_context", context.condition(expected_known), fixture->require_f32(name + ".context"));
            VoxMelFlowRuntime flow(*a, execution, assets::TensorStorageType::F32,
                "up_tiers." + std::to_string(tier) + ".flow", a->tier_hidden);
            const auto key = "residual" + std::to_string(tier) + ".0";
            compare(name + "_flow", flow.sample(fixture->require_f32(key + ".condition"),
                static_cast<int64_t>(expected_known.size()) / channels, 1,
                fixture->require_f32(key + ".noise"), 32), fixture->require_f32(key + ".output"));
            const auto detail = context.denormalize(flow.sample(conditioning,
                static_cast<int64_t>(known.size()) / channels, 1, fixture->require_f32(key + ".noise"), 32));
            std::vector<float> next(x.size() * 2);
            for (size_t i = 0; i < x.size(); ++i) {
                const size_t j = tier == 0 ? (2 * (i / channels)) * channels + i % channels : 2 * i;
                const size_t stride = tier == 0 ? static_cast<size_t>(channels) : size_t{1};
                next[j] = x[i] + detail[prefix_size + i]; next[j + stride] = x[i] - detail[prefix_size + i];
            }
            x = std::move(next);
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] * a->mel_std[i % a->mels] + a->mel_mean[i % a->mels];
        compare("teacher_forced_mel", x, fixture->require_f32("mel"));
        modules::VocosVocoder vocos(a->vocos, execution.backend(), 8, {a->mels, 1024, a->hop, true});
        const auto independent = vocos.decode(fixture->require_f32("mel"));
        compare("vocos_same_mel", independent, fixture->require_f32("waveform"));
        const auto wave = vocos.decode(x);
        compare("teacher_forced_waveform", wave, fixture->require_f32("waveform"));
        audio::WavWriteOptions wav_options;
        wav_options.format = audio::WavSampleFormat::Float32;
        audio::write_wav(output, 24000, 1, wave, wav_options);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << (passed ? "PASS" : "FAIL") << " VoxMel component parity\n";
        return passed ? 0 : 1;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
