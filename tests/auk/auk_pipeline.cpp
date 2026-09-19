// AuK end to end: conditioning -> CFM sampling -> VAE decode -> a WAV you can play.
//
// This is the first point in the port where the output is meant to be JUDGED BY EAR
// rather than diffed. Every numeric stage has its own parity test; this one exists to
// answer the question those cannot: does it sound like speech.
//
//   auk_pipeline --thinker thinker.gguf --dit auk_dit.gguf --vae auk_vae_raw.gguf
//                --tokens ids.i32 --fusion fusion.f32 --out speech.wav
//                [--ref-audio prompt.wav] [--seconds 3.0] [--steps 32] [--cfg 2.0]
//
// --ref-audio is the voice prompt: it is VAE-encoded and prepended as the reference
// latent, which is how AuK clones a voice.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/output.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"
#include "engine/models/auk/audio_tower.h"
#include "engine/models/auk/conditioner.h"
#include "engine/models/auk/message.h"
#include "engine/models/auk/task.h"
#include "engine/models/auk/dit.h"
#include "engine/models/auk/sampler.h"
#include "engine/models/auk/vae_encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

double double_arg(int argc, char ** argv, const std::string & name, double fallback) {
    return std::stod(arg_value(argc, argv, name, std::to_string(fallback)));
}

template <typename T>
std::vector<T> read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

double seconds_since(const std::chrono::steady_clock::time_point & start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void describe(const char * label, const std::vector<float> & values) {
    double lo = 0.0, hi = 0.0, sq = 0.0;
    for (float value : values) {
        lo = std::min(lo, double(value));
        hi = std::max(hi, double(value));
        sq += double(value) * double(value);
    }
    std::cout << "  " << std::left << std::setw(12) << label << " n=" << values.size()
              << std::fixed << std::setprecision(4)
              << " min=" << lo << " max=" << hi
              << " rms=" << (values.empty() ? 0.0 : std::sqrt(sq / double(values.size()))) << "\n";
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path thinker_path = arg_value(argc, argv, "--thinker", "");
    const std::filesystem::path dit_path = arg_value(argc, argv, "--dit", "");
    const std::filesystem::path vae_path = arg_value(argc, argv, "--vae", "");
    const std::filesystem::path tokens_path = arg_value(argc, argv, "--tokens", "");
    const std::filesystem::path fusion_path = arg_value(argc, argv, "--fusion", "");
    const std::filesystem::path out_path = arg_value(argc, argv, "--out", "auk_speech.wav");
    const std::filesystem::path ref_audio = arg_value(argc, argv, "--ref-audio", "");
    // The SAME clip usually plays both roles: 16 kHz into the Thinker so the model can
    // hear what it is being asked about, 24 kHz into the VAE as the latent to continue
    // from. --source-audio is the first, --ref-audio the second.
    const std::filesystem::path source_audio = arg_value(argc, argv, "--source-audio", "");
    const bool has_prompt = !tokens_path.empty() ||
                            !arg_value(argc, argv, "--instruction", "").empty() ||
                            !arg_value(argc, argv, "--task", "").empty();
    if (thinker_path.empty() || dit_path.empty() || vae_path.empty() || fusion_path.empty() || !has_prompt) {
        std::cerr <<
            "usage: auk_pipeline --thinker <gguf> --dit <gguf> --vae <gguf> --fusion <f32>\n"
            "                    (--task <name> [--variant v] [--param k=v]... [--language en|zh]\n"
            "                     | --instruction \"text\" | --tokens <i32>)\n"
            "                    [--omni <dir>] [--source-audio <wav>] [--ref-audio <wav>]\n"
            "                    [--out <wav>] [--seconds 4] [--steps 32] [--cfg 2.0]\n"
            "                    [--seed 0] [--backend cpu|cuda] [--threads N]\n"
            "\n"
            "  --task addresses AuK's own task types directly, e.g.\n"
            "      --task pitch_edit --variant increase --param semitones=4\n"
            "      --task emotion_edit --param emotion=happy\n"
            "      --task content_edit --variant replace --param orig=quiet --param new=busy\n"
            "  --source-audio is 16 kHz for the Thinker; --ref-audio is 24 kHz for the VAE.\n";
        return 2;
    }

    const double seconds = double_arg(argc, argv, "--seconds", 3.0);
    const int threads = int_arg(argc, argv, "--threads", 1);

    engine::core::BackendConfig backend;
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    backend.type = backend_name == "cuda" ? engine::core::BackendType::Cuda
                 : backend_name == "best" ? engine::core::BackendType::BestAvailable
                                          : engine::core::BackendType::Cpu;
    backend.threads = threads;
    engine::core::ExecutionContext execution(backend);

    // ---- conditioning ------------------------------------------------------------
    auto started = std::chrono::steady_clock::now();
    engine::models::auk::AukConditionerConfig conditioner_config;
    auto fusion_values = read_file<float>(fusion_path);
    engine::models::auk::AukFusionParameters fusion;
    fusion.layer_weights.assign(fusion_values.begin(), fusion_values.end() - 1);
    fusion.layer_scale = fusion_values.back();
    auto thinker_source = engine::assets::open_tensor_source(thinker_path);
    engine::models::auk::AukConditioner conditioner(conditioner_config, fusion, *thinker_source, execution);
    // Either a fixture of ids, or an instruction tokenized here. The second is what a
    // user actually has; the first is what a parity test needs.
    std::vector<int32_t> tokens;
    std::vector<float> audio_embeddings;
    std::vector<int64_t> audio_positions;
    // --task addresses the model's own task types directly, with parameters, instead of
    // hoping a phrasing lands: --task pitch_edit --variant increase --param semitones=4.
    // --instruction remains for arbitrary text and for reproducing a fixture.
    std::string instruction = arg_value(argc, argv, "--instruction", "");
    const std::string task_name = arg_value(argc, argv, "--task", "");
    if (!task_name.empty()) {
        engine::models::auk::TaskRequest request;
        request.task = task_name;
        request.variant = arg_value(argc, argv, "--variant", "");
        request.language = arg_value(argc, argv, "--language", "en");
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) != "--param") continue;
            const std::string pair = argv[i + 1];
            const auto split = pair.find('=');
            if (split == std::string::npos) throw std::runtime_error("--param wants name=value, got " + pair);
            request.params[pair.substr(0, split)] = pair.substr(split + 1);
        }
        const auto rendered = engine::models::auk::render_task(request);
        instruction = rendered.instruction;
        std::cout << "task: " << task_name
                  << (request.variant.empty() ? "" : "/" + request.variant)
                  << " [" << rendered.language << "]";
        if (rendered.language != request.language) {
            // Said out loud, because it changes what the model was asked in a way the
            // caller did not choose.
            std::cout << " (no " << request.language << " template; this task is "
                      << rendered.language << "-only)";
        }
        std::cout << "\n  \"" << instruction << "\"\n";
    }
    if (!instruction.empty()) {
        const std::filesystem::path omni_dir = arg_value(argc, argv, "--omni", "");
        if (omni_dir.empty()) throw std::runtime_error("--instruction needs --omni <dir with tokenizer.json>");
        engine::assets::ResourceBundle bundle(omni_dir);
        bundle.add_model_file("tokenizer_config", "tokenizer_config.json");
        bundle.add_model_file("tokenizer_json", "tokenizer.json");
        bundle.add_optional_model_file("vocab", "vocab.json");
        bundle.add_optional_model_file("merges", "merges.txt");
        const auto tokenizer = engine::tokenizers::load_qwen_bpe_tokenizer(bundle);

        int64_t audio_tokens = 0;
        std::vector<float> source16k;
        if (!source_audio.empty()) {
            const auto clip = engine::audio::read_wav_f32(source_audio);
            source16k = clip.sample_rate == 16000
                ? clip.samples
                : engine::audio::resample_mono_soxr_or_linear(
                      clip.samples, clip.sample_rate, 16000, engine::audio::SoxrResampleOptions{});
            audio_tokens = engine::models::auk::audio_token_count(static_cast<int64_t>(source16k.size()));
        }
        const auto text = engine::models::auk::build_user_message(instruction, audio_tokens);
        tokens = tokenizer->encode(text, true);
        std::cout << "message: " << tokens.size() << " tokens (" << audio_tokens << " audio)\n";

        if (audio_tokens > 0) {
            engine::models::auk::AukAudioTowerConfig tower_config;
            engine::models::auk::AukAudioTower tower(tower_config, *thinker_source, execution);
            int64_t mel_frames = 0;
            const auto mel = engine::models::auk::AukAudioTower::log_mel(source16k, mel_frames);
            const auto embedded = tower.encode(mel, mel_frames);
            audio_embeddings = embedded.values;
            const int32_t placeholder = static_cast<int32_t>(int_arg(argc, argv, "--audio-token", 151646));
            for (size_t index = 0; index < tokens.size(); ++index) {
                if (tokens[index] == placeholder) audio_positions.push_back(static_cast<int64_t>(index));
            }
            if (static_cast<int64_t>(audio_positions.size()) != embedded.tokens) {
                throw std::runtime_error(
                    "the template placed " + std::to_string(audio_positions.size()) +
                    " audio tokens but the tower produced " + std::to_string(embedded.tokens));
            }
            std::cout << "audio tower: " << mel_frames << " mel frames -> " << embedded.tokens << " tokens\n";
        }
    } else {
        tokens = read_file<int32_t>(tokens_path);
    }
    const auto conditioning = conditioner.encode(tokens, audio_embeddings, audio_positions);
    std::cout << "conditioner: " << conditioning.tokens << " tokens -> "
              << conditioning.hidden_size << " dims in " << std::fixed << std::setprecision(2)
              << seconds_since(started) << "s\n";

    // ---- reference voice, when one is given ---------------------------------------
    const engine::modules::BigVganVocoderConfig vae_config = [] {
        engine::modules::BigVganVocoderConfig config;
        config.sampling_rate = 24000;
        config.num_mels = 64;
        config.hop_size = 480;
        config.upsample_initial_channel = 1536;
        config.snake_logscale = true;
        config.upsample_rates = {5, 4, 3, 2, 2, 2};
        config.upsample_kernel_sizes = {10, 8, 6, 4, 4, 4};
        config.resblock_kernel_sizes = {3, 7, 11};
        config.n_fft = 1920;      // validation-only, never read by the decode graph
        config.win_size = 1920;
        return config;
    }();
    auto vae_source = engine::assets::make_prefixed_tensor_source(
        engine::assets::open_tensor_source(vae_path), "vae/");

    std::vector<float> reference_latent;
    int64_t ref_frames = 0;
    if (!ref_audio.empty()) {
        started = std::chrono::steady_clock::now();
        const auto prompt = engine::audio::read_wav_f32(ref_audio);
        if (prompt.sample_rate != vae_config.sampling_rate) {
            throw std::runtime_error(
                "reference audio must be 24 kHz; got " + std::to_string(prompt.sample_rate));
        }
        engine::models::auk::AukVaeEncoderConfig encoder_config;
        engine::models::auk::AukVaeEncoder encoder(encoder_config, *vae_source, execution);
        const auto stats = encoder.encode(prompt.samples);
        // The distribution's MEAN, not a draw from it: a sampled reference would make
        // the same prompt give a different voice every run, which is not what a voice
        // prompt is for.
        const auto global_mean = vae_source->require_f32("global_mean", {encoder_config.latent_dim});
        const auto global_log_std = vae_source->require_f32("global_log_std", {encoder_config.latent_dim});
        ref_frames = stats.frames;
        reference_latent.resize(static_cast<size_t>(ref_frames * encoder_config.latent_dim));
        for (int64_t frame = 0; frame < ref_frames; ++frame) {
            for (int64_t channel = 0; channel < encoder_config.latent_dim; ++channel) {
                const float mean = stats.values[static_cast<size_t>(channel * stats.frames + frame)];
                reference_latent[static_cast<size_t>(frame * encoder_config.latent_dim + channel)] =
                    (mean - global_mean[static_cast<size_t>(channel)]) /
                    std::sqrt(global_log_std[static_cast<size_t>(channel)]);
            }
        }
        std::cout << "reference: " << prompt.samples.size() << " samples -> " << ref_frames
                  << " latent frames in " << seconds_since(started) << "s\n";
    }

    // ---- sampling ------------------------------------------------------------------
    engine::models::auk::AukDitConfig dit_config;
    auto dit_source = engine::assets::open_tensor_source(dit_path);
    engine::models::auk::AukDit dit(dit_config, *dit_source, execution);

    engine::models::auk::AukDitInputs inputs;
    inputs.text = conditioning.values;
    inputs.text_tokens = conditioning.tokens;
    inputs.reference = reference_latent;
    inputs.ref_frames = ref_frames;
    inputs.gen_frames = static_cast<int64_t>(std::llround(seconds * vae_config.sampling_rate / vae_config.hop_size));
    if (inputs.gen_frames <= 0) throw std::runtime_error("--seconds must produce at least one frame");

    // ⚠ The caller sets the duration. AuK cannot estimate its own output length --
    // upstream issue #503 records this as its known flaw -- so --seconds is a required
    // input rather than something inferred.
    // --noise injects a draw from elsewhere, which is what makes an end-to-end
    // comparison against the reference possible at all: the two RNGs cannot agree.
    std::vector<float> noise;
    const std::filesystem::path noise_path = arg_value(argc, argv, "--noise", "");
    if (!noise_path.empty()) {
        noise = read_file<float>(noise_path);
        const int64_t supplied = static_cast<int64_t>(noise.size()) / dit_config.latent_dim;
        if (supplied != inputs.gen_frames) {
            std::cout << "noise fixture carries " << supplied << " frames; using that instead of --seconds\n";
            inputs.gen_frames = supplied;
        }
    } else {
        std::mt19937 rng(static_cast<unsigned>(int_arg(argc, argv, "--seed", 0)));
        std::normal_distribution<float> gaussian(0.0F, 1.0F);
        noise.resize(static_cast<size_t>(inputs.gen_frames * dit_config.latent_dim));
        for (auto & value : noise) value = gaussian(rng);
    }

    engine::models::auk::AukSamplerOptions options;
    options.steps = int_arg(argc, argv, "--steps", 32);
    options.cfg_strength = static_cast<float>(double_arg(argc, argv, "--cfg", 2.0));

    started = std::chrono::steady_clock::now();
    const auto sampled = engine::models::auk::sample_latents(dit, inputs, noise, options);
    const double sampling_seconds = seconds_since(started);
    std::cout << "sampler: " << options.steps << " steps, cfg " << options.cfg_strength
              << ", " << inputs.gen_frames << " frames in " << sampling_seconds << "s ("
              << sampling_seconds / double(options.steps) << "s/step)\n";

    // ---- decode ---------------------------------------------------------------------
    started = std::chrono::steady_clock::now();
    engine::modules::BigVganGraphOptions graph_options;
    graph_options.causal = true;
    auto vocoder = engine::modules::BigVganVocoderComponent::load_from_tensor_source(
        vae_source, backend, vae_config, graph_options);

    const auto global_mean = vae_source->require_f32("global_mean", {vae_config.num_mels});
    const auto global_log_std = vae_source->require_f32("global_log_std", {vae_config.num_mels});
    std::vector<float> decoder_input(sampled.latents.size());
    for (int64_t frame = 0; frame < sampled.frames; ++frame) {
        for (int64_t channel = 0; channel < vae_config.num_mels; ++channel) {
            const float value = sampled.latents[static_cast<size_t>(frame * vae_config.num_mels + channel)];
            decoder_input[static_cast<size_t>(channel * sampled.frames + frame)] =
                value * std::sqrt(global_log_std[static_cast<size_t>(channel)]) +
                global_mean[static_cast<size_t>(channel)];
        }
    }
    const auto decoded = vocoder.synthesize(decoder_input, sampled.frames);
    std::cout << "vae: " << sampled.frames << " frames -> " << decoded.samples
              << " samples in " << seconds_since(started) << "s\n";

    // The reference prompt is decoded along with the generation; drop it so the file
    // holds only what the model produced.
    const int64_t prompt_samples = ref_frames * vae_config.hop_size;
    std::vector<float> generated(
        decoded.waveform.begin() + std::min<int64_t>(prompt_samples, decoded.waveform.size()),
        decoded.waveform.end());
    describe("generated", generated);

    engine::audio::WavPcm16Sink().write(
        out_path, engine::audio::AudioBuffer{static_cast<int>(vae_config.sampling_rate), 1, generated});
    std::cout << "wrote " << out_path.string() << " ("
              << double(generated.size()) / double(vae_config.sampling_rate) << "s at 24 kHz)\n";

    // ⚠ THE MODEL CANNOT REPORT FAILURE. It is a flow-matching generator: noise and
    // conditioning in, a velocity field out. There is no refusal token and no
    // confidence, so an instruction it cannot act on produces audio that resembles the
    // input -- which is indistinguishable from "the edit was subtle" unless something
    // measures it. That measurement belongs here, not in the model.
    //
    // This detects ONE of the three ways an edit goes wrong, and only names that one:
    //
    //   1. the output is unchanged            <- detected here, by correlation
    //   2. the instruction named something not in the audio (a target or anchor the
    //      speaker never clearly says) -- needs the source transcribed, and word-level
    //      ASR regularizes a mumbled word into a plausible one, so it is not reliable
    //   3. the output changed, but not in the way asked -- needs the OUTPUT transcribed
    //      and compared against the intent
    //
    // Reporting (1) as "unchanged" rather than "failed" is deliberate: an edit that
    // legitimately changes little looks the same from here.
    //
    // ⚠ AND IT IS TASK-DEPENDENT. AuK REGENERATES rather than splices, so a content edit
    // comes back near zero correlation even when ASR shows the words are nearly
    // identical -- this check is meaningless there. Paralinguistic edits stay
    // waveform-aligned (0.70-0.99 measured), which is where it discriminates.
    if (!ref_audio.empty()) {
        const auto source = engine::audio::read_wav_f32(ref_audio);
        const size_t count = std::min(source.samples.size(), generated.size());
        if (count > 0) {
            double dot = 0.0, lhs = 0.0, rhs = 0.0;
            for (size_t index = 0; index < count; ++index) {
                dot += double(source.samples[index]) * double(generated[index]);
                lhs += double(source.samples[index]) * double(source.samples[index]);
                rhs += double(generated[index]) * double(generated[index]);
            }
            const double correlation = (lhs == 0.0 || rhs == 0.0) ? 0.0 : dot / std::sqrt(lhs * rhs);
            const double gate = double_arg(argc, argv, "--unchanged-above", 0.98);
            std::cout << std::fixed << std::setprecision(4)
                      << "change: correlation with the source is " << correlation;
            if (correlation >= gate) {
                std::cout << " -- UNCHANGED (at or above " << gate << ")\n"
                          << "  the instruction may name something the audio does not contain,\n"
                          << "  or the model did not act on it. The audio was still written.\n";
                return 3;
            }
            std::cout << " (below " << gate << ", so something changed)\n";
        }
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
}
