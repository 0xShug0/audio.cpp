// Reverb-TTS (software-mansion/Reverb-TTS): zero-shot voice-cloning TTS in seven languages, a pruned
// and retrained OmniVoice. It reuses OmniVoice's Qwen2 BPE text tokenizer and Higgs Audio v2 codec
// and brings its own generator (prefix-cached masked diffusion, see generator.cpp) and the text,
// length and trimming rules of reverb_tts/tts.py.

#include "engine/community_models/reverb/session.h"

#include "engine/community_models/reverb/frontend.h"
#include "engine/community_models/reverb/generator.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/omnivoice/assets.h"
#include "engine/models/omnivoice/audio_tokenizer.h"
#include "engine/models/omnivoice/tokenizer_text.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::reverb {
namespace {

namespace json = engine::io::json;
namespace omnivoice = engine::models::omnivoice;
constexpr const char * family_name = "reverb";
constexpr const char * kDefaultVoice = "en_female_narrator";

struct Voice {
    std::vector<int32_t> codes;  // [8, frames], codebook-major
    int64_t frames = 0;
    std::string transcript;      // prepared
    std::string language;
    int64_t speech_frames = 0;   // 0 when unknown

    int64_t pace_frames() const noexcept {
        return speech_frames <= 0 ? frames : std::max<int64_t>(1, std::min(speech_frames, frames));
    }
};

struct Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const omnivoice::OmniVoiceAssets> omni;  // tokenizer, codec config and weights
    ReverbArchitecture architecture;
    ReverbSamplerConfig sampler;
    std::map<std::string, Voice> voices;
};

Voice parse_voice(const std::string & name, const json::Value & item) {
    Voice voice;
    voice.transcript = json::require_string(item, "transcript");
    voice.language = json::require_string(item, "language");
    if (!is_supported_language(voice.language)) {
        throw std::runtime_error("Reverb preset voice " + name + " has an unsupported language");
    }
    for (const auto & value : item.require("codes").as_array()) {
        const double number = value.as_number();
        if (!(number >= 0.0 && number < 1024.0) || std::floor(number) != number) {
            throw std::runtime_error("Reverb preset voice " + name + " has an invalid code");
        }
        voice.codes.push_back(static_cast<int32_t>(number));
    }
    if (voice.codes.empty() || voice.codes.size() % 8 != 0) {
        throw std::runtime_error("Reverb preset voice " + name + " needs 8 x frames codes");
    }
    voice.frames = static_cast<int64_t>(voice.codes.size() / 8);
    voice.speech_frames = json::optional_i64(item, "speech_frames", 0);
    return voice;
}

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<Assets>();
    out->omni = omnivoice::load_omnivoice_assets(path, model_spec::default_spec_path(family_name));
    out->resources = out->omni->resources;
    const auto config = out->omni->resources.parse_json("config");
    const auto * reverb = config.find("reverb");
    if (reverb == nullptr || !reverb->is_object()) {
        throw std::runtime_error("not a Reverb-TTS checkpoint: config.json has no reverb section");
    }
    const auto & llm = out->omni->config.llm;
    auto & a = out->architecture;
    a.vocab_size = llm.vocab_size;
    a.hidden_size = llm.hidden_size;
    a.intermediate_size = llm.intermediate_size;
    a.num_hidden_layers = llm.num_hidden_layers;
    a.num_attention_heads = llm.num_attention_heads;
    a.num_key_value_heads = llm.num_key_value_heads;
    a.head_dim = llm.head_dim;
    a.rms_norm_eps = llm.rms_norm_eps;
    a.rope_theta = llm.rope_theta;
    a.codebooks = out->omni->config.num_audio_codebook;
    a.codes = out->omni->config.audio_vocab_size;
    a.mask_id = static_cast<int32_t>(out->omni->config.audio_mask_id);
    if (const auto * architecture = reverb->find("architecture"); architecture != nullptr) {
        a.embedding_rank = json::optional_i64(*architecture, "embedding_rank", 0);
    }
    if (a.embedding_rank == 0 && out->omni->model_weights->has_tensor("llm.embed_tokens.table.weight")) {
        a.embedding_rank = out->omni->model_weights->require_metadata("llm.embed_tokens.table.weight").shape.back();
    }
    const auto voices = json::parse_file(out->omni->resources.require_file("voices"));
    for (const auto & [name, item] : voices.as_object()) {
        out->voices.emplace(name, parse_voice(name, item));
    }
    if (out->voices.empty()) {
        throw std::runtime_error("Reverb package has no preset voices");
    }
    return out;
}

class Session final : public runtime::IOfflineVoiceTaskSession, public runtime::RuntimeSessionBase {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(std::move(task)), assets_(std::move(assets)), contract_(std::move(contract)) {
        if ((task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
            task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Reverb supports offline TTS and voice cloning sessions");
        }
        runtime::validate_spec_backed_session_options(options, *contract_, family_name, "Reverb");
        const auto weight_type = runtime::parse_tensor_storage_option(options.options, "reverb.weight_type",
            assets::TensorStorageType::Native,
            {assets::TensorStorageType::Native, assets::TensorStorageType::F32, assets::TensorStorageType::F16,
             assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0});
        codec_weight_type_ = runtime::parse_tensor_storage_option(options.options, "reverb.codec_weight_type",
            assets::TensorStorageType::Native,
            {assets::TensorStorageType::Native, assets::TensorStorageType::F32, assets::TensorStorageType::F16,
             assets::TensorStorageType::BF16});
        tokenizer_ = std::make_unique<omnivoice::OmniVoiceTextTokenizer>(assets_->omni);
        generator_ = std::make_unique<ReverbGenerator>(
            assets_->omni->model_weights, assets_->architecture, execution_context(), weight_type);
    }

    std::string family() const override { return family_name; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        const auto start = std::chrono::steady_clock::now();
        require_prepared("Reverb run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Reverb");
        if (!request.option_arrays.empty()) {
            throw std::runtime_error("Reverb does not accept list-valued options");
        }
        if (!request.text_input || request.text_input->text.find_first_not_of(" \t\r\n") == std::string::npos) {
            throw std::runtime_error("Reverb requires non-empty text_input");
        }
        const auto & options = request.options;

        const runtime::AudioBuffer * reference = request.audio_input ? &*request.audio_input : nullptr;
        std::string voice_name = kDefaultVoice;
        if (request.voice && request.voice->speaker) {
            if (request.voice->speaker->audio) {
                reference = &*request.voice->speaker->audio;
            }
            if (request.voice->speaker->cached_voice_id) {
                voice_name = *request.voice->speaker->cached_voice_id;
            }
        }
        if (const auto id = runtime::find_option(options, {"voice_id", "voice"})) {
            voice_name = *id;
        }
        std::optional<std::string> language = runtime::find_option(options, {"language"});
        if (!language && !request.text_input->language.empty()) {
            language = request.text_input->language;
        }
        if (language && !is_supported_language(*language)) {
            throw std::runtime_error("Reverb speaks de, en, es, fr, it, pl and pt; unsupported language: " + *language);
        }

        const Voice * voice = nullptr;
        if (reference != nullptr) {
            const auto transcript = runtime::find_option(options, {"reference_text"});
            if (!transcript || transcript->find_first_not_of(" \t\r\n") == std::string::npos) {
                throw std::runtime_error("Reverb voice cloning requires reference_text, the exact transcript of the reference audio");
            }
            voice = &clone_voice(*reference, *transcript, language.value_or("en"));
        } else {
            if (task_.task == runtime::VoiceTaskKind::VoiceCloning) {
                throw std::runtime_error("Reverb voice cloning requires reference audio");
            }
            const auto found = assets_->voices.find(voice_name);
            if (found == assets_->voices.end()) {
                std::string names;
                for (const auto & [name, _] : assets_->voices) {
                    names += (names.empty() ? "" : ", ") + name;
                }
                throw std::runtime_error("unknown Reverb voice: " + voice_name + " (expected one of: " + names + ")");
            }
            voice = &found->second;
        }
        const std::string text_language = language.value_or(voice->language);

        const auto config = sampler_config(options);
        const float speed = runtime::parse_float_option(options, {"speed"}).value_or(1.0F);
        if (!std::isfinite(speed) || speed <= 0.0F) {
            throw std::runtime_error("Reverb speed must be positive");
        }
        const auto duration = runtime::parse_float_option(options, {"duration_sec"});
        if (duration && !(std::isfinite(*duration) && *duration > 0.0F)) {
            throw std::runtime_error("Reverb duration_sec must be positive");
        }
        const std::string pace = runtime::find_option(options, {"pace"}).value_or("speech");
        if (pace != "speech" && pace != "recording") {
            throw std::runtime_error("Reverb pace must be speech or recording");
        }
        const bool trim = runtime::find_option(options, {"trim"})
            ? runtime::parse_bool_option(*runtime::find_option(options, {"trim"}), "trim") : true;
        // "burst" (default) or a fixed number of seconds; see cut_end_burst.
        const std::string end_cut = runtime::find_option(options, {"end_cut"}).value_or("burst");
        std::optional<double> fixed_end_cut;
        if (end_cut != "burst") {
            char * end = nullptr;
            const double seconds = std::strtod(end_cut.c_str(), &end);
            if (end == end_cut.c_str() || *end != '\0' || !std::isfinite(seconds) || seconds < 0.0) {
                throw std::runtime_error("Reverb end_cut must be burst or a number of seconds of at least 0");
            }
            fixed_end_cut = seconds;
        }
        const auto seed_option = runtime::parse_i64_option(options, {"seed"});
        if (seed_option && (*seed_option < 0 || *seed_option > std::numeric_limits<uint32_t>::max())) {
            throw std::runtime_error("Reverb seed must be between 0 and 4294967295");
        }
        const auto base_seed = static_cast<uint32_t>(
            seed_option.value_or(static_cast<int64_t>(runtime::random_u64_seed() & 0x7fffffffu)));

        const std::string words = prepare_text(request.text_input->text);
        if (words.empty()) {
            throw std::runtime_error("Reverb text is empty after preparation");
        }
        const int64_t reference_frames = pace == "speech" ? voice->pace_frames() : voice->frames;
        std::vector<std::pair<std::string, int64_t>> pieces;
        if (duration) {
            pieces.emplace_back(words, std::max<int64_t>(1, round_half_even(*duration * static_cast<double>(kFrameRate))));
        } else {
            for (auto & part : split_text(words, voice->transcript, reference_frames)) {
                const auto estimate = estimate_frames(part, voice->transcript, reference_frames);
                pieces.emplace_back(std::move(part),
                    std::max<int64_t>(1, round_half_even(static_cast<double>(estimate) / static_cast<double>(speed))));
            }
        }

        runtime::AudioBuffer audio{24000, 1, {}};
        const int sample_rate = assets_->omni->config.audio_tokenizer.sample_rate;
        audio.sample_rate = sample_rate;
        double generate_ms = 0.0;
        double decode_ms = 0.0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            const auto & [part, frames] = pieces[i];
            const std::string prompt = "<|denoise|><|lang_start|>" + text_language +
                "<|lang_end|><|instruct_start|>None<|instruct_end|><|text_start|>" + voice->transcript + " " + part +
                "<|text_end|>";
            const auto ids = tokenizer_->encode(prompt);
            const auto generate_start = std::chrono::steady_clock::now();
            const auto codes = generator_->generate(ids, voice->codes, voice->frames, frames,
                base_seed + static_cast<uint32_t>(i), config);
            generate_ms += debug::elapsed_ms(generate_start);
            const auto decode_start = std::chrono::steady_clock::now();
            auto samples = decode(codes, frames);
            decode_ms += debug::elapsed_ms(decode_start);
            if (fixed_end_cut) {
                const auto cut = static_cast<size_t>(*fixed_end_cut * sample_rate);
                samples.resize(samples.size() > cut ? samples.size() - cut : 0);
            } else {
                samples = cut_end_burst(std::move(samples), sample_rate);
            }
            if (trim) {
                samples = trim_silence(samples, sample_rate);
            }
            audio.samples.insert(audio.samples.end(), samples.begin(), samples.end());
        }
        debug::timing_log_scalar("reverb.generate_ms", generate_ms);
        debug::timing_log_scalar("reverb.decode_ms", decode_ms);
        debug::timing_log_scalar("reverb.prefix_ms", generator_->last_stats().prefix_ms);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        runtime::TaskResult result;
        result.audio_output = std::move(audio);
        return result;
    }

private:
    ReverbSamplerConfig sampler_config(const std::unordered_map<std::string, std::string> & options) const {
        ReverbSamplerConfig c = assets_->sampler;
        if (const auto v = runtime::parse_i64_option(options, {"num_inference_steps", "steps"})) {
            c.steps = *v;
        }
        if (const auto v = runtime::find_option(options, {"remasking"})) {
            c.remasking = runtime::parse_bool_option(*v, "remasking");
        }
        const auto f = [&](std::string_view key, float & target) {
            if (const auto v = runtime::parse_float_option(options, {key})) {
                target = *v;
            }
        };
        f("remask_ratio", c.remask_ratio);
        f("shift", c.t_shift);
        f("codebook_penalty", c.codebook_penalty);
        f("position_temperature", c.position_temperature);
        f("polish_bias", c.polish_bias);
        f("polish_guidance", c.polish_guidance);
        if (c.steps < 1 || c.steps > 64) {
            throw std::runtime_error("Reverb num_inference_steps must be between 1 and 64");
        }
        return c;
    }

    omnivoice::OmniVoiceHiggsAudioV2TokenizerRuntime & codec() {
        if (!codec_) {
            codec_ = std::make_unique<omnivoice::OmniVoiceHiggsAudioV2TokenizerRuntime>(
                assets_->omni, execution_context(), 128ull * 1024ull * 1024ull, 128ull * 1024ull * 1024ull,
                codec_weight_type_);
        }
        return *codec_;
    }

    std::vector<float> decode(const std::vector<int32_t> & codes, int64_t frames) {
        omnivoice::OmniVoiceGeneratedAudioTokens tokens;
        tokens.frames = frames;
        tokens.codebooks = 8;
        tokens.token_ids.resize(codes.size());
        for (int64_t c = 0; c < 8; ++c) {
            for (int64_t f = 0; f < frames; ++f) {
                tokens.token_ids[static_cast<size_t>(f * 8 + c)] = codes[static_cast<size_t>(c * frames + f)];
            }
        }
        auto audio = codec().decode_audio_tokens(tokens);
        return std::move(audio.samples);
    }

    const Voice & clone_voice(const runtime::AudioBuffer & input, const std::string & transcript,
        const std::string & language) {
        if (input.sample_rate <= 0 || input.channels <= 0 || input.samples.empty() ||
            input.samples.size() % static_cast<size_t>(input.channels) != 0) {
            throw std::runtime_error("invalid Reverb reference audio format");
        }
        if (clone_ && clone_transcript_ == transcript && clone_->language == language &&
            reference_cache_.sample_rate == input.sample_rate && reference_cache_.channels == input.channels &&
            reference_cache_.samples == input.samples) {
            return *clone_;
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<float> mono(input.samples.size() / static_cast<size_t>(input.channels));
        for (size_t i = 0; i < mono.size(); ++i) {
            double sum = 0.0;
            for (int c = 0; c < input.channels; ++c) {
                sum += input.samples[i * static_cast<size_t>(input.channels) + static_cast<size_t>(c)];
            }
            mono[i] = static_cast<float>(sum / input.channels);
        }
        const auto & codec_config = assets_->omni->config.audio_tokenizer;
        mono = raise_quiet(resample_kaiser_best(mono, input.sample_rate, codec_config.sample_rate));
        // Whole codec frames, as the training references were: pad up to a hop multiple.
        const auto hop = static_cast<size_t>(codec_config.hop_length);
        if (hop > 0 && mono.size() % hop != 0) {
            mono.resize(mono.size() + hop - mono.size() % hop, 0.0F);
        }
        const double seconds = static_cast<double>(mono.size()) / codec_config.sample_rate;
        if (seconds < 1.0 || seconds > 30.0) {
            throw std::runtime_error("Reverb reference audio must be 1 to 30 seconds long (3 to 10 s of clean speech works best)");
        }
        omnivoice::OmniVoiceReferenceAudioOptions encode_options;
        encode_options.preprocess_prompt = false;
        encode_options.has_reference_text = true;
        encode_options.normalize_rms = false;
        const auto tokens = codec().encode_reference_audio(
            runtime::AudioBuffer{codec_config.sample_rate, 1, mono}, encode_options);
        auto voice = std::make_unique<Voice>();
        voice->frames = tokens.frames;
        voice->codes.resize(static_cast<size_t>(8 * tokens.frames));
        if (tokens.codebooks != 8 || tokens.frames <= 0) {
            throw std::runtime_error("Reverb reference audio produced no codec frames");
        }
        for (int64_t c = 0; c < 8; ++c) {
            for (int64_t f = 0; f < tokens.frames; ++f) {
                voice->codes[static_cast<size_t>(c * tokens.frames + f)] =
                    tokens.token_ids[static_cast<size_t>(f * 8 + c)];
            }
        }
        voice->transcript = prepare_text(transcript);
        voice->language = language;
        voice->speech_frames = round_half_even(speech_length(mono, codec_config.sample_rate) * kFrameRate);
        clone_ = std::move(voice);
        clone_transcript_ = transcript;
        reference_cache_ = input;
        debug::timing_log_scalar("reverb.reference_ms", debug::elapsed_ms(started));
        return *clone_;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    assets::TensorStorageType codec_weight_type_ = assets::TensorStorageType::Native;
    std::unique_ptr<omnivoice::OmniVoiceTextTokenizer> tokenizer_;
    std::unique_ptr<ReverbGenerator> generator_;
    std::unique_ptr<omnivoice::OmniVoiceHiggsAudioV2TokenizerRuntime> codec_;
    std::unique_ptr<Voice> clone_;
    std::string clone_transcript_;
    runtime::AudioBuffer reference_cache_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_reverb_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = family_name;
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                                std::shared_ptr<const Assets> assets,
                                std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::reverb
