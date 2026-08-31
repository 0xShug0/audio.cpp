// Stage probe for the sopro_tts port.
//
// The offline pipeline has five stages and a broken one is hard to localise
// from the final waveform. This binary exercises them in isolation:
//
//   mel      analysis mel -> vocoder -> waveform. Vocos is trained to invert
//            its own analysis mel, so a clean round trip proves the mel front
//            end and the whole vocoder graph at once.
//   semantic reference waveform -> FSQ token ids (histogram + first ids).
//   speaker  reference waveform -> id/style/style-ctrl embedding statistics.
//
// Usage: sopro_probe <model-dir> <reference.wav> [out-dir]

#include "engine/community_models/sopro_tts/acoustic.h"
#include "engine/community_models/sopro_tts/assets.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/semantic_encoder.h"
#include "engine/community_models/sopro_tts/speaker_encoder.h"
#include "engine/community_models/sopro_tts/vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace sopro = engine::community_models::sopro_tts;

namespace {

struct Stats {
    float min = 0.0F;
    float max = 0.0F;
    double mean = 0.0;
    double rms = 0.0;
    size_t nonfinite = 0;
};

Stats describe(const std::vector<float> & values) {
    Stats out;
    if (values.empty()) {
        return out;
    }
    out.min = out.max = values.front();
    double sum = 0.0;
    double square_sum = 0.0;
    for (const float value : values) {
        if (!std::isfinite(value)) {
            ++out.nonfinite;
            continue;
        }
        out.min = std::min(out.min, value);
        out.max = std::max(out.max, value);
        sum += value;
        square_sum += static_cast<double>(value) * value;
    }
    const auto count = static_cast<double>(values.size());
    out.mean = sum / count;
    out.rms = std::sqrt(square_sum / count);
    return out;
}

void print_stats(const char * label, const std::vector<float> & values) {
    const auto stats = describe(values);
    std::printf(
        "  %-22s n=%-8zu min=%+.4f max=%+.4f mean=%+.5f rms=%.5f nonfinite=%zu\n",
        label, values.size(), stats.min, stats.max, stats.mean, stats.rms, stats.nonfinite);
}

// Segmental SNR between two aligned signals, in dB.
double snr_db(const std::vector<float> & reference, const std::vector<float> & test) {
    const size_t n = std::min(reference.size(), test.size());
    double signal = 0.0;
    double noise = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double r = reference[i];
        const double d = r - test[i];
        signal += r * r;
        noise += d * d;
    }
    if (noise <= 0.0) {
        return 999.0;
    }
    return 10.0 * std::log10(std::max(signal, 1e-20) / noise);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model-dir> <reference.wav> [out-dir]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path model_path = argv[1];
    const std::filesystem::path reference_path = argv[2];
    const std::filesystem::path out_dir = argc > 3 ? std::filesystem::path(argv[3])
                                                   : std::filesystem::path(".");
    std::filesystem::create_directories(out_dir);

    auto assets = sopro::load_sopro_tts_assets(model_path);
    const auto & config = assets->config;
    const auto sample_rate = static_cast<int>(config.sample_rate);

    engine::core::BackendConfig backend_config;
    backend_config.type = engine::core::BackendType::Cpu;
    backend_config.threads = 8;
    engine::core::ExecutionContext execution(backend_config);

    constexpr size_t kWeightBytes = 512ull * 1024ull * 1024ull;
    constexpr size_t kGraphBytes = 1024ull * 1024ull * 1024ull;
    const auto storage = engine::assets::TensorStorageType::F32;

    // ---- input ----
    const auto wav = engine::audio::read_wav_f32(reference_path);
    std::vector<float> mono = wav.channels == 1
        ? wav.samples
        : engine::audio::mixdown_interleaved_to_mono_average(wav.samples, wav.channels);
    if (wav.sample_rate != sample_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(mono, wav.sample_rate, sample_rate);
    }
    for (auto & value : mono) {
        value = std::min(1.0F, std::max(-1.0F, value));
    }
    std::printf("reference: %s\n", reference_path.string().c_str());
    std::printf("  source_rate=%d channels=%d samples=%zu -> %.3f s at %d Hz\n",
                wav.sample_rate, wav.channels, mono.size(),
                static_cast<double>(mono.size()) / sample_rate, sample_rate);

    // ---- crop / level normalisation, as prepare_reference does ----
    std::mt19937_64 rng(1234);
    auto cropped = sopro::audio_ops::crop_on_pause(
        mono, config.generation.ref_seconds, sample_rate, rng);
    const auto level = sopro::audio_ops::speech_level_db(mono, sample_rate);
    std::printf("  speech_level=%.2f dB active=%.3f s\n", level.level_db, level.active_seconds);
    std::printf("  crop_on_pause(%.1f s): %zu -> %zu samples (%.3f s)\n",
                config.generation.ref_seconds, mono.size(), cropped.size(),
                static_cast<double>(cropped.size()) / sample_rate);
    const auto crop_level = sopro::audio_ops::speech_level_db(cropped, sample_rate);
    float crop_peak = 0.0F;
    for (const float value : cropped) {
        crop_peak = std::max(crop_peak, std::fabs(value));
    }
    auto normalisation = sopro::audio_ops::normalize_reference(cropped, sample_rate);
    const auto normalised = std::move(normalisation.wav);
    std::printf("  normalize_reference: level %.2f -> %.2f dB (gain %+.2f dB, crop peak %.3f)\n",
                crop_level.level_db, normalisation.level_db,
                normalisation.level_db - crop_level.level_db, crop_peak);

    // ---- stage: vocoder round trip ----
    {
        std::printf("\n[mel] analysis mel -> vocoder round trip\n");
        sopro::SoproVocoderRuntime vocoder(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        const auto mel = vocoder.log_mel(normalised);
        const int64_t n_mels = vocoder.n_mels();
        const int64_t frames = static_cast<int64_t>(mel.size()) / n_mels;
        std::printf("  frames=%lld (expected %lld)\n",
                    static_cast<long long>(frames),
                    static_cast<long long>(vocoder.mel_frames(static_cast<int64_t>(normalised.size()))));
        print_stats("log_mel", mel);
        // Dump for an independent numpy check of the front end.
        {
            std::FILE * fh = std::fopen((out_dir / "probe_input_logmel.f32").string().c_str(), "wb");
            if (fh != nullptr) {
                std::fwrite(mel.data(), sizeof(float), mel.size(), fh);
                std::fclose(fh);
            }
            std::FILE * wh = std::fopen((out_dir / "probe_input_audio.f32").string().c_str(), "wb");
            if (wh != nullptr) {
                std::fwrite(normalised.data(), sizeof(float), normalised.size(), wh);
                std::fclose(wh);
            }
        }

        const auto audio = vocoder.decode(mel, frames);
        print_stats("round_trip_audio", audio);
        engine::audio::write_pcm16_wav(out_dir / "probe_vocoder_roundtrip.wav", sample_rate, 1, audio);
        {
            std::FILE * fh = std::fopen((out_dir / "probe_roundtrip_audio.f32").string().c_str(), "wb");
            if (fh != nullptr) {
                std::fwrite(audio.data(), sizeof(float), audio.size(), fh);
                std::fclose(fh);
            }
        }

        // Waveform SNR is phase-sensitive and Vocos predicts phase, so the
        // meaningful check is whether re-analysing the output reproduces the
        // mel the vocoder was asked to render.
        std::printf("  waveform SNR vs input: %.2f dB (phase-sensitive, informative only)\n",
                    snr_db(normalised, audio));
        const auto remel = vocoder.log_mel(audio);
        const int64_t reframes = static_cast<int64_t>(remel.size()) / n_mels;
        const int64_t common = std::min(frames, reframes);
        double abs_sum = 0.0;
        double abs_max = 0.0;
        size_t counted = 0;
        for (int64_t c = 0; c < n_mels; ++c) {
            for (int64_t t = 0; t < common; ++t) {
                const double a = mel[static_cast<size_t>(c * frames + t)];
                const double b = remel[static_cast<size_t>(c * reframes + t)];
                // Ignore bins pinned at the log floor; they carry no signal.
                if (a <= -16.0) {
                    continue;
                }
                const double d = std::fabs(a - b);
                abs_sum += d;
                abs_max = std::max(abs_max, d);
                ++counted;
            }
        }
        const double mae = counted > 0 ? abs_sum / static_cast<double>(counted) : 0.0;
        std::printf("  mel round-trip MAE=%.4f max=%.4f over %zu bins (log units)\n",
                    mae, abs_max, counted);
        std::printf("  -> %s\n",
                    mae < 0.35 ? "vocoder + mel front end look CORRECT"
                               : "vocoder or mel front end is WRONG");
    }

    // ---- stage: semantic encoder ----
    {
        std::printf("\n[semantic] FSQ tokeniser\n");
        sopro::SoproSemanticEncoderRuntime encoder(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        const auto tokens = encoder.encode(normalised);
        const int64_t expected = (static_cast<int64_t>(normalised.size()) +
                                  config.semantic_encoder.token_samples_24k - 1) /
                                 config.semantic_encoder.token_samples_24k;
        std::printf("  tokens=%zu (expected %lld)\n", tokens.size(),
                    static_cast<long long>(expected));
        std::map<int32_t, int> histogram;
        for (const int32_t token : tokens) {
            ++histogram[token];
        }
        int32_t lo = tokens.empty() ? 0 : *std::min_element(tokens.begin(), tokens.end());
        int32_t hi = tokens.empty() ? 0 : *std::max_element(tokens.begin(), tokens.end());
        std::printf("  distinct=%zu range=[%d, %d] of [0, %lld)\n",
                    histogram.size(), lo, hi,
                    static_cast<long long>(config.model.semantic_vocab_size));
        std::printf("  first ids:");
        for (size_t i = 0; i < std::min<size_t>(16, tokens.size()); ++i) {
            std::printf(" %d", tokens[i]);
        }
        std::printf("\n");
        // A collapsed tokeniser (one id repeated) means the encoder is broken.
        int most = 0;
        for (const auto & [id, count] : histogram) {
            (void) id;
            most = std::max(most, count);
        }
        const double share = tokens.empty() ? 0.0 : static_cast<double>(most) / tokens.size();
        std::printf("  most common id share: %.1f%%  %s\n", share * 100.0,
                    share > 0.5 ? "(COLLAPSED - encoder likely wrong)" : "(looks healthy)");
    }

    // ---- stage: speaker encoder ----
    {
        std::printf("\n[speaker] identity / style embeddings\n");
        sopro::SoproSpeakerEncoderRuntime encoder(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        const auto wav16 = engine::audio::resample_mono_torchaudio_sinc_hann(
            normalised, sample_rate, static_cast<int>(encoder.sample_rate()));
        const auto embeddings = encoder.encode(wav16);
        print_stats("id_emb", embeddings.id_emb);
        print_stats("style_emb", embeddings.style_emb);
        print_stats("style_ctrl", embeddings.style_ctrl);
        double norm = 0.0;
        for (const float value : embeddings.id_emb) {
            norm += static_cast<double>(value) * value;
        }
        std::printf("  |id_emb| = %.6f (must be 1.0)\n", std::sqrt(norm));
    }

    // ---- stage: acoustic head self-reconstruction ----
    // Re-render the second half of the reference from its own semantic tokens,
    // conditioned on the first half as the prompt. The acoustic head is lossy
    // (it only sees FSQ ids) but a correct one tracks the real mel closely;
    // a broken one produces something uncorrelated with it.
    {
        std::printf("\n[acoustic] self-reconstruction from the reference's own tokens\n");
        sopro::SoproVocoderRuntime vocoder(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        sopro::SoproSpeakerEncoderRuntime speaker(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        sopro::SoproSemanticEncoderRuntime semantic(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        sopro::SoproReferenceBuilder builder(*assets, speaker, semantic, vocoder);
        std::mt19937_64 build_rng(1234);
        const auto voice = builder.build(mono, config.generation.ref_seconds, build_rng);

        const int64_t n_mels = config.model.acoustic_mel_n_mels;
        const int64_t hop_ratio = config.hop_ratio();
        const auto total_tokens = static_cast<int64_t>(voice.semantic_tokens.size());
        const int64_t split = total_tokens / 2;
        const int64_t prompt_frames = std::min(voice.mel_frames, split * hop_ratio);
        const int64_t gen_tokens = total_tokens - split;
        const int64_t total_frames = prompt_frames + gen_tokens * hop_ratio;
        std::printf("  reference: %lld tokens, %lld mel frames\n",
                    static_cast<long long>(total_tokens), static_cast<long long>(voice.mel_frames));
        std::printf("  prompt: %lld frames, regenerating %lld tokens -> %lld frames\n",
                    static_cast<long long>(prompt_frames), static_cast<long long>(gen_tokens),
                    static_cast<long long>(total_frames));

        sopro::SoproAcousticRequest request;
        request.semantic_tokens = voice.semantic_tokens;
        request.cond_vec = voice.cond_vec;
        request.prompt_mel.assign(static_cast<size_t>(n_mels * prompt_frames), 0.0F);
        for (int64_t c = 0; c < n_mels; ++c) {
            for (int64_t t = 0; t < prompt_frames; ++t) {
                request.prompt_mel[static_cast<size_t>(c * prompt_frames + t)] =
                    voice.mel[static_cast<size_t>(c * voice.mel_frames + t)];
            }
        }
        request.prompt_frames = prompt_frames;
        request.total_frames = total_frames;
        request.steps = 32;   // many steps: isolate the field from the schedule
        request.seed = 1234;

        sopro::SoproAcousticRuntime acoustic(
            *assets, execution, kWeightBytes, kGraphBytes, storage, storage);
        const auto solved = acoustic.solve(request);
        print_stats("solved_mel(normalised)", solved);

        // Compare only the regenerated span against the true reference mel.
        const int64_t compare_end = std::min(total_frames, voice.mel_frames);
        double abs_sum = 0.0;
        double ref_sq = 0.0;
        double err_sq = 0.0;
        size_t counted = 0;
        for (int64_t c = 0; c < n_mels; ++c) {
            for (int64_t t = prompt_frames; t < compare_end; ++t) {
                const double truth = voice.mel[static_cast<size_t>(c * voice.mel_frames + t)];
                const double got = solved[static_cast<size_t>(c * total_frames + t)];
                abs_sum += std::fabs(truth - got);
                ref_sq += truth * truth;
                err_sq += (truth - got) * (truth - got);
                ++counted;
            }
        }
        if (counted > 0) {
            const double mae = abs_sum / static_cast<double>(counted);
            const double nmse = err_sq / std::max(ref_sq, 1e-12);
            std::printf("  regenerated span vs true mel: MAE=%.4f  NMSE=%.4f (%zu bins)\n",
                        mae, nmse, counted);
            std::printf("  -> %s\n",
                        nmse < 0.6 ? "acoustic head tracks the reference (looks CORRECT)"
                                   : "acoustic head output is uncorrelated with the reference (WRONG)");
        }
        std::FILE * fh = std::fopen((out_dir / "probe_acoustic_solved.f32").string().c_str(), "wb");
        if (fh != nullptr) {
            std::fwrite(solved.data(), sizeof(float), solved.size(), fh);
            std::fclose(fh);
        }
        std::FILE * th = std::fopen((out_dir / "probe_ref_tokens.i32").string().c_str(), "wb");
        if (th != nullptr) {
            std::fwrite(voice.semantic_tokens.data(), sizeof(int32_t), voice.semantic_tokens.size(), th);
            std::fclose(th);
        }
        std::FILE * ch = std::fopen((out_dir / "probe_cond_vec.f32").string().c_str(), "wb");
        if (ch != nullptr) {
            std::fwrite(voice.cond_vec.data(), sizeof(float), voice.cond_vec.size(), ch);
            std::fclose(ch);
        }
        std::FILE * mh = std::fopen((out_dir / "probe_ref_mel.f32").string().c_str(), "wb");
        if (mh != nullptr) {
            std::fwrite(voice.mel.data(), sizeof(float), voice.mel.size(), mh);
            std::fclose(mh);
        }
    }

    std::printf("\nwrote %s\n", (out_dir / "probe_vocoder_roundtrip.wav").string().c_str());
    return 0;
}
