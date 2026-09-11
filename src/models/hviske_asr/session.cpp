#include "engine/models/hviske_asr/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultWeightContextBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kDefaultEncoderGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderPrefillGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderDecodeGraphArenaBytes = 512ull * 1024ull * 1024ull;

constexpr const char * kFamily = "hviske_asr";

std::shared_ptr<const HviskeASRAssets> require_assets(std::shared_ptr<const HviskeASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Hviske ASR session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Hviske ASR session requires a model contract");
    }
    return contract;
}

runtime::SessionOptions require_supported_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    const auto checked_contract = require_contract(contract);
    runtime::validate_spec_backed_session_options(options, *checked_contract, kFamily, "Hviske ASR");
    return options;
}

std::unordered_map<std::string, std::string> normalize_request_options(
    std::unordered_map<std::string, std::string> options) {
    return runtime::apply_option_v1_compatibility(
        std::move(options),
        {
            {"audio_chunk_seconds", "audio_chunk_duration_sec"},
            {"audio_chunk_duration_seconds", "audio_chunk_duration_sec"},
            {"audio_chunk_duration", "audio_chunk_duration_sec"},
        },
        "Hviske ASR",
        "request");
}

void ensure_supported_language(const HviskeASRAssets & assets, const std::string & language) {
    if (language.empty()) {
        throw std::runtime_error("Hviske ASR requires a language");
    }
    if (std::find(assets.config.supported_languages.begin(), assets.config.supported_languages.end(), language) ==
        assets.config.supported_languages.end()) {
        throw std::runtime_error("Hviske ASR unsupported language: " + language);
    }
}

std::string chunk_separator(const std::string & language) {
    static const std::set<std::string> no_space = {"ja", "zh"};
    return no_space.count(language) != 0 ? "" : " ";
}

runtime::AudioBuffer make_segment_audio(const std::vector<float> & waveform, int sample_rate, int64_t start, int64_t end) {
    runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    audio.samples.assign(
        waveform.begin() + static_cast<std::ptrdiff_t>(start),
        waveform.begin() + static_cast<std::ptrdiff_t>(end));
    return audio;
}

int64_t frontend_frames_for_samples(int64_t source_frames, int64_t source_sample_rate, const HviskeFrontendConfig & config) {
    if (source_sample_rate <= 0 || source_frames <= 0) {
        return 0;
    }
    const double resampled =
        static_cast<double>(source_frames) * static_cast<double>(config.sample_rate) / static_cast<double>(source_sample_rate);
    int64_t frames = static_cast<int64_t>(std::ceil(resampled)) / config.hop_length + 1;
    if (config.pad_to > 0 && frames % config.pad_to != 0) {
        frames += config.pad_to - (frames % config.pad_to);
    }
    return frames;
}

int64_t prepared_frontend_frames(const runtime::AudioPreparationContract & audio, const HviskeConfig & config) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.max_input_samples <= 0) {
        return 0;
    }
    const int64_t request_frames = frontend_frames_for_samples(
        audio.max_input_samples / audio.channels,
        audio.sample_rate,
        config.frontend);
    const int64_t max_clip_frames = frontend_frames_for_samples(
        config.max_audio_clip_seconds * config.frontend.sample_rate,
        config.frontend.sample_rate,
        config.frontend);
    return std::max(request_frames, max_clip_frames);
}

std::unique_ptr<runtime::IVoiceTaskSession> create_hviske_asr_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const HviskeASRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<HviskeASRSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

HviskeASRSession::HviskeASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HviskeASRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(require_supported_session_options(std::move(options), contract)),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      weight_context_bytes_(runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"hviske_asr.weight_context_mb"}, kDefaultWeightContextBytes)),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"hviske_asr.encoder_graph_arena_mb"}, kDefaultEncoderGraphArenaBytes)),
      decoder_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"hviske_asr.decoder_prefill_graph_arena_mb"}, kDefaultDecoderPrefillGraphArenaBytes)),
      decoder_decode_graph_arena_bytes_(runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"hviske_asr.decoder_decode_graph_arena_mb"}, kDefaultDecoderDecodeGraphArenaBytes)),
      matmul_weight_storage_type_(runtime::parse_tensor_storage_option(
          RuntimeSessionBase::options().options,
          "hviske_asr.weight_type",
          engine::assets::TensorStorageType::Native,
          {
              engine::assets::TensorStorageType::Native,
              engine::assets::TensorStorageType::F32,
              engine::assets::TensorStorageType::F16,
              engine::assets::TensorStorageType::BF16,
              engine::assets::TensorStorageType::Q8_0,
          })),
      conv_weight_storage_type_(runtime::parse_tensor_storage_option(
          RuntimeSessionBase::options().options,
          "hviske_asr.conv_weight_type",
          "hviske_asr.weight_type",
          matmul_weight_storage_type_,
          {
              engine::assets::TensorStorageType::Native,
              engine::assets::TensorStorageType::F32,
              engine::assets::TensorStorageType::F16,
          })),
      frontend_(assets_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Hviske ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Hviske ASR currently supports offline sessions");
    }
    weights_ = load_hviske_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    encoder_ = std::make_unique<HviskeEncoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        encoder_graph_arena_bytes_);
    decoder_ = std::make_unique<HviskeDecoderRuntime>(
        assets_,
        weights_,
        execution_context(),
        decoder_prefill_graph_arena_bytes_,
        decoder_decode_graph_arena_bytes_);
}

HviskeASRSession::~HviskeASRSession() = default;

std::string HviskeASRSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind HviskeASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HviskeASRSession::run_mode() const {
    return task_.mode;
}

void HviskeASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    const auto prepare_start = Clock::now();
    if (!request.audio.has_value()) {
        throw std::runtime_error("Hviske ASR prepare() requires an audio contract");
    }
    const auto & frontend_config = assets_->config.frontend;
    const int64_t frames = prepared_frontend_frames(*request.audio, assets_->config);
    if (frames > 0) {
        encoder_->prepare_capacity(frames, frontend_config.features);
    }
    mark_prepared();
    debug::timing_log_scalar("hviske_asr.prepare_ms", engine::debug::elapsed_ms(prepare_start, Clock::now()));
    debug::trace_log_scalar("hviske_asr.prepare.max_input_samples", request.audio->max_input_samples);
}

std::string HviskeASRSession::language_for_request(
    const runtime::TaskRequest & request,
    const std::unordered_map<std::string, std::string> & options) const {
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        return request.text_input->language;
    }
    if (const auto language = runtime::find_option(options, {"language"})) {
        return *language;
    }
    return "da";
}

bool HviskeASRSession::punctuation_for_request(const std::unordered_map<std::string, std::string> & options) const {
    if (const auto value = runtime::find_option(options, {"punctuation"})) {
        return runtime::parse_bool_option(*value, "punctuation");
    }
    return true;
}

HviskeDecodingOptions HviskeASRSession::decoding_options_for_request(
    const std::unordered_map<std::string, std::string> & request_options) const {
    HviskeDecodingOptions options;
    options.max_new_tokens = assets_->config.decoder.max_new_tokens;
    if (const auto value = runtime::parse_i64_option(request_options, {"max_tokens"})) {
        if (*value <= 0) {
            throw std::runtime_error("Hviske ASR max_tokens must be positive");
        }
        options.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_i64_option(request_options, {"num_beams"})) {
        if (*value <= 0) {
            throw std::runtime_error("Hviske ASR num_beams must be positive");
        }
        options.num_beams = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request_options, {"length_penalty"})) {
        if (*value <= 0.0f) {
            throw std::runtime_error("Hviske ASR length_penalty must be positive");
        }
        options.length_penalty = *value;
    }
    if (const auto value = runtime::find_option(request_options, {"do_sample"})) {
        options.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (const auto value = runtime::parse_finite_float_option(request_options, {"temperature"})) {
        if (*value <= 0.0f) {
            throw std::runtime_error("Hviske ASR temperature must be positive");
        }
        options.temperature = *value;
    }
    if (const auto value = runtime::parse_i64_option(request_options, {"top_k"})) {
        if (*value < 0) {
            throw std::runtime_error("Hviske ASR top_k must be non-negative");
        }
        options.top_k = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request_options, {"top_p"})) {
        if (*value <= 0.0f || *value > 1.0f) {
            throw std::runtime_error("Hviske ASR top_p must be within (0, 1]");
        }
        options.top_p = *value;
    }
    if (const auto value = runtime::parse_u32_option(request_options, {"seed"})) {
        options.seed = *value;
    }
    return options;
}

std::vector<HviskeASRSession::Segment> HviskeASRSession::prepare_segments(
    const runtime::AudioBuffer & audio,
    const std::unordered_map<std::string, std::string> & options) const {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("Hviske ASR requires non-empty audio input");
    }
    const int sample_rate = static_cast<int>(assets_->config.frontend.sample_rate);
    const auto waveform = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        sample_rate);
    std::vector<Segment> segments;
    const auto append_whole_audio = [&]() {
        segments.push_back({runtime::AudioBuffer{sample_rate, 1, waveform}});
    };

    const auto mode = engine::audio::parse_audio_chunk_mode(options);
    if (mode == engine::audio::AudioChunkMode::None) {
        append_whole_audio();
        return segments;
    }

    if (mode == engine::audio::AudioChunkMode::Vad) {
        throw std::runtime_error("Hviske ASR does not support audio_chunk_mode=vad");
    }

    const double chunk_seconds = static_cast<double>(
        engine::audio::parse_audio_chunk_seconds_override(options).value_or(
            static_cast<float>(assets_->config.max_audio_clip_seconds)));
    if (!(chunk_seconds > 0.0)) {
        throw std::runtime_error("Hviske ASR audio_chunk_duration_sec must be positive");
    }
    const int64_t chunk_size = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(chunk_seconds * static_cast<double>(sample_rate))));

    if (mode == engine::audio::AudioChunkMode::Auto) {
        const double duration = static_cast<double>(waveform.size()) / static_cast<double>(sample_rate);
        const double fast_path_threshold =
            chunk_seconds - static_cast<double>(assets_->config.overlap_chunk_seconds);
        if (duration <= fast_path_threshold) {
            append_whole_audio();
            return segments;
        }
    }

    std::vector<runtime::TimeSpan> spans;
    if (mode == engine::audio::AudioChunkMode::Fixed) {
        const auto fixed_spans = engine::audio::plan_audio_chunks(
            static_cast<int64_t>(waveform.size()),
            engine::audio::AudioChunkSpec{
                chunk_size,
                chunk_size,
                engine::audio::AudioChunkPadMode::Zero,
                engine::audio::AudioChunkTailAlignment::Start,
                0,
            });
        spans.reserve(fixed_spans.size());
        for (const auto & span : fixed_spans) {
            spans.push_back({
                span.output_start_sample,
                span.output_start_sample + span.valid_samples,
            });
        }
    } else {
        const int64_t boundary_context = std::max<int64_t>(1, assets_->config.overlap_chunk_seconds * sample_rate);
        spans = engine::audio::plan_quiet_energy_audio_chunks(
            waveform,
            {chunk_size, boundary_context, assets_->config.min_energy_window_samples});
    }
    segments.reserve(spans.size());
    for (const auto & span : spans) {
        segments.push_back({make_segment_audio(waveform, sample_rate, span.start_sample, span.end_sample)});
    }
    return segments;
}

runtime::TaskResult HviskeASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Hviske ASR run()");
    auto request_options = normalize_request_options(request.options);
    runtime::validate_spec_backed_request_options(request_options, *contract_, "Hviske ASR");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Hviske ASR run() requires audio_input");
    }
    const auto wall_start = Clock::now();
    const auto config_start = Clock::now();
    const std::string language = language_for_request(request, request_options);
    ensure_supported_language(*assets_, language);
    const bool punctuation = punctuation_for_request(request_options);
    const auto decoding_options = decoding_options_for_request(request_options);
    debug::timing_log_scalar("hviske_asr.request_config_ms", engine::debug::elapsed_ms(config_start, Clock::now()));

    const auto prompt_start = Clock::now();
    const auto prompt_ids = tokenize_hviske_prompt(*assets_, language, punctuation);
    debug::timing_log_scalar("hviske_asr.prompt_tokenize_ms", engine::debug::elapsed_ms(prompt_start, Clock::now()));

    const auto segments_start = Clock::now();
    const auto segments = prepare_segments(*request.audio_input, request_options);
    debug::timing_log_scalar("hviske_asr.prepare_segments_ms", engine::debug::elapsed_ms(segments_start, Clock::now()));

    std::vector<std::string> texts;
    texts.reserve(segments.size());
    for (const auto & segment : segments) {
        const auto frontend_start = Clock::now();
        const auto features = frontend_.extract(segment.audio);
        const auto frontend_end = Clock::now();
        const auto encoder_start = Clock::now();
        const auto encoded = encoder_->encode(features);
        const auto encoder_end = Clock::now();
        const auto decoder_start = Clock::now();
        const auto decoded = decoder_->generate(prompt_ids, encoded, decoding_options);
        const auto decoder_end = Clock::now();
        texts.push_back(engine::io::trim_ascii_whitespace(decode_hviske_tokens(*assets_, decoded.token_ids)));
        debug::timing_log_scalar("hviske_asr.segment.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
        debug::timing_log_scalar("hviske_asr.segment.encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
        debug::timing_log_scalar("hviske_asr.segment.decoder_ms", engine::debug::elapsed_ms(decoder_start, decoder_end));
    }

    const std::string separator = chunk_separator(language);
    const auto postprocess_start = Clock::now();
    std::string joined;
    for (const auto & text : texts) {
        if (text.empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined += separator;
        }
        joined += text;
    }
    debug::timing_log_scalar("hviske_asr.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, Clock::now()));

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{joined, language};
    debug::trace_log_scalar("hviske_asr.language", language);
    debug::trace_log_scalar("hviske_asr.segment_count", segments.size());
    debug::trace_log_scalar("hviske_asr.prompt_tokens", prompt_ids.size());
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_hviske_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<HviskeASRAssets> config;
    config.family = kFamily;
    config.load_assets = load_hviske_asr_assets;
    config.create_session = create_hviske_asr_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::hviske_asr
