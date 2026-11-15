#include "app/cli/args.h"
#include "engine/models/soulx_singer/runtime.h"
#include "engine/models/soulx_singer/routing.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/audio/wav_writer.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cstring>

int main(int argc, char **argv) {
    const auto session_started = std::chrono::steady_clock::now();
    try {
        const auto model = minitts::cli::find_arg(argc, argv, "--model").value();
        const auto reference = minitts::cli::find_arg(argc, argv, "--reference").value();
        const auto backend = minitts::cli::find_arg(argc, argv, "--backend").value_or("cuda");
        const auto family = minitts::cli::find_arg(argc, argv, "--family").value_or("soulx_singer");
        const auto control = minitts::cli::find_arg(argc, argv, "--control").value_or("score");
        const auto mode = minitts::cli::find_arg(argc, argv, "--mode").value_or("denoise");
        const auto out = minitts::cli::find_arg(argc, argv, "--out").value();
        const auto log = minitts::cli::find_arg(argc, argv, "--log-file").value();
        const double waveform_minimum = std::stod(
            minitts::cli::find_arg(argc, argv, "--min-waveform-cosine").value_or("0.99999"));
        if (!std::isfinite(waveform_minimum) || waveform_minimum <= 0 || waveform_minimum > 1)
            throw std::invalid_argument("--min-waveform-cosine must be in (0,1]");
        if (!minitts::cli::has_arg(argc, argv, "--log"))
            throw std::invalid_argument("--log required");
        minitts::cli::require_known_args(argc, argv);
        engine::debug::configure_logging({true, log});
        auto source = engine::assets::open_tensor_source(reference);
        const auto read = [&](const std::string &name) {
            return source->require_f32(name, source->require_metadata(name).shape);
        };
        const auto compare = [&](const std::string &name, const std::vector<float> &a,
                                 const std::vector<float> &b, double minimum = 0.99999) {
            if (a.size() != b.size())
                throw std::runtime_error(name + " size mismatch: " + std::to_string(a.size()) + " vs " +
                                         std::to_string(b.size()));
            double aa = 0, bb = 0, ab = 0, err = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (!std::isfinite(a[i]))
                    throw std::runtime_error(name + " nonfinite");
                aa += double(a[i]) * a[i];
                bb += double(b[i]) * b[i];
                ab += double(a[i]) * b[i];
                err = std::max(err, std::abs(double(a[i]) - b[i]));
            }
            const double cosine = ab / std::sqrt(aa * bb);
            engine::debug::trace_log_scalar(family + ".reference." + name + ".cosine", cosine);
            engine::debug::trace_log_scalar(family + ".reference." + name + ".max_abs", err);
            if (!std::isfinite(cosine) || cosine < minimum)
                throw std::runtime_error(name + " parity failed");
        };
        engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend), 0, 8});
        auto assets = engine::models::soulx_singer::load_assets(model);
        engine::models::soulx_singer::SoulXSingerRuntime runtime(assets, execution,
                                                                 engine::assets::TensorStorageType::F32,
                                                                 engine::core::AttentionPreference::Eager);
        using engine::models::soulx_singer::EncodedAudio;
        EncodedAudio prompt, target;
        prompt.waveform = read("reference_waveform");
        target.waveform = read("source_waveform");
        prompt.mel = read("reference_mel");
        prompt.frames = static_cast<int64_t>(prompt.mel.size() / 128);
        prompt.f0 = read("reference_f0");
        prompt.f0.resize(static_cast<size_t>(prompt.frames));
        prompt.content = read("reference_content");
        prompt.content.resize(static_cast<size_t>(prompt.frames * 512));
        target.f0 = read("source_f0");
        target.frames = static_cast<int64_t>(target.f0.size());
        target.content = read("source_content");
        const auto score = [&](const std::string &side) {
            const auto meta = engine::io::json::parse_file(
                std::filesystem::path(reference).parent_path() /
                (side == "reference" ? "reference_metadata.json" : "target_metadata.json"));
            auto value = engine::models::soulx_singer::parse_score(meta.as_array().front(), assets->phones);
            for (const auto &item :
                 {std::pair<std::string, const std::vector<int32_t> *>{"phoneme", &value.phonemes},
                  {"note_pitch", &value.pitches},
                  {"note_type", &value.types},
                  {"mel2note", &value.frame_notes}}) {
                const auto data = source->require_tensor_data(side + "_" + item.first);
                if (data.metadata.dtype != "I32" ||
                    data.bytes.size() != item.second->size() * sizeof(int32_t) ||
                    std::memcmp(data.bytes.data(), item.second->data(), data.bytes.size()))
                    throw std::runtime_error(side + "." + item.first + " score alignment differs");
            }
            return value;
        };
        if (mode == "encode" && assets->impl->task == engine::runtime::VoiceTaskKind::AudioGeneration) {
            const auto prompt_score = score("reference"), target_score = score("source");
            const auto median = [](const auto &input) {
                auto values = input;
                values.erase(std::remove_if(values.begin(), values.end(), [](auto f) { return f <= 0; }),
                             values.end());
                std::sort(values.begin(), values.end());
                return values[(values.size() - 1) / 2];
            };
            const int shift = static_cast<int>(std::nearbyint(
                control == "score" ? float(median(prompt_score.pitches) - median(target_score.pitches))
                                   : 12 * std::log2(median(prompt_score.f0) / median(target_score.f0))));
            engine::runtime::AudioBuffer audio{24000, 1, prompt.waveform};
            const auto actual =
                runtime.encode_scores(prompt_score, target_score, shift, control == "melody", audio);
            compare("reference_content", actual.first.content, prompt.content);
            compare("source_content", actual.second.content, target.content);
            compare("reference_mel", actual.first.mel, prompt.mel);
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(session_started));
            return 0;
        }
        if (mode == "encode") {
            const auto actual = runtime.encode({24000, 1, prompt.waveform}, true);
            compare("reference_mel", actual.mel, prompt.mel);
            compare("reference_content", actual.content, prompt.content);
            compare("reference_f0", actual.f0, prompt.f0);
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(session_started));
            return 0;
        }
        auto mel = read("mel");
        if (mode == "denoise") {
            std::vector<float> voiced_ref, voiced_source;
            for (auto x : prompt.f0)
                if (x > 0)
                    voiced_ref.push_back(x);
            for (auto x : target.f0)
                if (x > 0)
                    voiced_source.push_back(x);
            std::sort(voiced_ref.begin(), voiced_ref.end());
            std::sort(voiced_source.begin(), voiced_source.end());
            int shift = 0;
            if (assets->impl->task == engine::runtime::VoiceTaskKind::AudioGeneration && control == "score") {
                auto p = score("reference").pitches, t = score("source").pitches;
                p.erase(std::remove(p.begin(), p.end(), 0), p.end());
                t.erase(std::remove(t.begin(), t.end(), 0), t.end());
                std::sort(p.begin(), p.end());
                std::sort(t.begin(), t.end());
                shift = p[(p.size() - 1) / 2] - t[(t.size() - 1) / 2];
            } else
                shift = static_cast<int>(
                    std::nearbyint(12 * std::log2(voiced_ref[(voiced_ref.size() - 1) / 2] /
                                                  voiced_source[(voiced_source.size() - 1) / 2])));
            runtime.prepare_conditioning(prompt, target, shift);
            engine::models::soulx_singer::GenerationOptions options;
            const auto start = std::chrono::steady_clock::now();
            mel = runtime.denoise(read("noise"), options);
            engine::debug::timing_log_scalar(family + ".reference.denoise_ms",
                                             engine::debug::elapsed_ms(start));
            compare("mel", mel, read("mel"));
        } else if (mode != "decode")
            throw std::invalid_argument("Expected encode, denoise or decode");
        auto waveform = runtime.decode(mel);
        const auto expected = read("waveform");
        engine::audio::write_wav(out, 24000, 1, waveform, {engine::audio::WavSampleFormat::Float32});
        compare("waveform", waveform, expected, waveform_minimum);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(session_started));
        return 0;
    } catch (const std::exception &error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(session_started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
