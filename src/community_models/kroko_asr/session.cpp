#include "engine/community_models/kroko_asr/session.h"

#include "engine/community_models/kroko_asr/frontend.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::kroko_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::string_view kFamily = "kroko_asr";

std::shared_ptr<const KrokoASRAssets> require_assets(
    std::shared_ptr<const KrokoASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error(
            "Kroko ASR session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error(
            "Kroko ASR session requires a model contract");
    }
    return contract;
}

std::string normalized_language(std::string value) {
    const size_t separator = value.find_first_of("-_");
    if (separator != std::string::npos) {
        value.resize(separator);
    }
    std::string result = value;
    std::transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (result == "iw") {
        return "he";
    }
    if (result == "eng") {
        return "en";
    }
    if (result == "deu" || result == "ger") {
        return "de";
    }
    if (result == "spa") {
        return "es";
    }
    if (result == "fra" || result == "fre") {
        return "fr";
    }
    if (result == "ita") {
        return "it";
    }
    if (result == "heb") {
        return "he";
    }
    if (result == "nld" || result == "dut") {
        return "nl";
    }
    if (result == "por") {
        return "pt";
    }
    if (result == "swe") {
        return "sv";
    }
    if (result == "tur") {
        return "tr";
    }
    return result;
}

bool starts_word(const std::string & piece) {
    return piece.rfind("\xE2\x96\x81", 0) == 0 ||
        (!piece.empty() && piece.front() == ' ');
}

int64_t normalized_audio_samples(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0;
    }
    const double frames = static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.channels);
    return static_cast<int64_t>(std::llround(
        frames * 16000.0 / static_cast<double>(audio.sample_rate)));
}

runtime::AudioBuffer with_tail_padding(
    const runtime::AudioBuffer & audio) {
    runtime::AudioBuffer result = audio;
    if (result.sample_rate <= 0 || result.channels <= 0) {
        return result;
    }
    const int64_t padding_frames = static_cast<int64_t>(
        std::llround(0.66 * static_cast<double>(result.sample_rate)));
    result.samples.resize(
        result.samples.size() +
            static_cast<size_t>(
                padding_frames * result.channels),
        0.0F);
    return result;
}

std::vector<runtime::WordTimestamp> build_word_timestamps(
    const KrokoTokenizer & tokenizer,
    const KrokoDecodedTokens & decoded,
    int64_t audio_samples) {
    struct PendingWord {
        std::vector<int32_t> ids;
        int64_t frame = 0;
    };
    std::vector<PendingWord> pending;
    const size_t count = std::min(
        decoded.ids.size(), decoded.frame_indices.size());
    for (size_t index = 0; index < count; ++index) {
        const int32_t id = decoded.ids[index];
        const bool boundary = starts_word(tokenizer.piece(id));
        if (pending.empty() || boundary) {
            pending.push_back(PendingWord{
                {},
                decoded.frame_indices[index],
            });
        }
        pending.back().ids.push_back(id);
    }

    constexpr int64_t kSamplesPerEncoderFrame = 160 * 4;
    std::vector<runtime::WordTimestamp> result;
    result.reserve(pending.size());
    for (auto & word : pending) {
        std::string text = tokenizer.decode(word.ids);
        if (text.empty()) {
            continue;
        }
        runtime::WordTimestamp timestamp;
        timestamp.span.start_sample =
            word.frame * kSamplesPerEncoderFrame;
        timestamp.span.end_sample =
            timestamp.span.start_sample + kSamplesPerEncoderFrame;
        timestamp.word = std::move(text);
        result.push_back(std::move(timestamp));
    }
    for (size_t index = 0; index + 1 < result.size(); ++index) {
        result[index].span.end_sample = std::max(
            result[index].span.start_sample + 1,
            result[index + 1].span.start_sample);
    }
    if (!result.empty()) {
        result.back().span.end_sample = std::max(
            result.back().span.start_sample + kSamplesPerEncoderFrame,
            audio_samples);
    }
    return result;
}

std::vector<std::string> split_hotwords(
    const std::string & value) {
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end = value.find_first_of("/\r\n", start);
        std::string phrase = value.substr(
            start,
            end == std::string::npos
                ? std::string::npos
                : end - start);
        while (!phrase.empty() &&
               std::isspace(
                   static_cast<unsigned char>(phrase.front())) != 0) {
            phrase.erase(phrase.begin());
        }
        while (!phrase.empty() &&
               std::isspace(
                   static_cast<unsigned char>(phrase.back())) != 0) {
            phrase.pop_back();
        }
        if (!phrase.empty()) {
            result.push_back(std::move(phrase));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
        while (start < value.size() &&
               (value[start] == '\r' || value[start] == '\n')) {
            ++start;
        }
    }
    return result;
}

float non_negative_finite_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    float fallback) {
    const float value =
        runtime::parse_finite_float_option(options, keys)
            .value_or(fallback);
    if (value < 0.0F) {
        const auto match =
            runtime::find_option_match(options, keys);
        throw std::runtime_error(
            (match.has_value() ? match->key : "Kroko option") +
            " must be non-negative");
    }
    return value;
}

}  // namespace

KrokoASRSession::KrokoASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const KrokoASRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(
          assets_->tokens,
          static_cast<int32_t>(assets_->config.blank_id),
          static_cast<int32_t>(assets_->config.unk_id)),
      decoder_(assets_),
      subsampling_(assets_, execution_context()),
      zipformer_(assets_, execution_context()),
      chunk_scratch_(
          static_cast<size_t>(
              assets_->config.chunk_size *
              assets_->config.feature_dim),
          0.0F) {
    if (task_.task != runtime::VoiceTaskKind::Asr ||
        (task_.mode != runtime::RunMode::Offline &&
         task_.mode != runtime::RunMode::Streaming)) {
        throw std::runtime_error(
            "Kroko ASR supports offline and streaming ASR sessions");
    }
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("kroko_asr.", 0) == 0 &&
            contract_->session_option_keys.find(key) ==
                contract_->session_option_keys.end()) {
            throw std::runtime_error(
                "unknown Kroko ASR session option: " + key);
        }
    }
}

KrokoASRSession::~KrokoASRSession() = default;

std::string KrokoASRSession::family() const {
    return std::string(kFamily);
}

runtime::VoiceTaskKind KrokoASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode KrokoASRSession::run_mode() const {
    return task_.mode;
}

void KrokoASRSession::prepare(
    const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error(
            "Kroko ASR prepare() requires an audio contract");
    }
    mark_prepared();
}

void KrokoASRSession::configure_request(
    const runtime::TaskRequest & request) {
    KrokoDecoderOptions decoder_options;
    const std::string method = runtime::find_option(
        request.options, {"decoding_method"})
        .value_or("greedy_search");
    if (method == "greedy_search") {
        decoder_options.method =
            KrokoDecodingMethod::GreedySearch;
    } else if (method == "modified_beam_search") {
        decoder_options.method =
            KrokoDecodingMethod::ModifiedBeamSearch;
    } else {
        throw std::runtime_error(
            "Kroko decoding_method must be greedy_search or modified_beam_search");
    }
    const int64_t max_active_paths =
        runtime::parse_i64_option(
            request.options, {"num_beams"})
            .value_or(4);
    if (max_active_paths < 1 ||
        max_active_paths > 64) {
        throw std::runtime_error(
            "Kroko num_beams must be between 1 and 64");
    }
    decoder_options.max_active_paths =
        static_cast<int32_t>(max_active_paths);
    decoder_options.blank_penalty =
        non_negative_finite_option(
            request.options, {"blank_penalty"}, 0.0F);
    decoder_options.hotwords_score =
        non_negative_finite_option(
            request.options, {"hotwords_score"}, 1.5F);
    if (const auto hotwords = runtime::find_option(
            request.options, {"hotwords"})) {
        for (const auto & phrase :
             split_hotwords(*hotwords)) {
            decoder_options.hotwords.push_back(
                tokenizer_.encode_hotword(phrase));
        }
    }
    engine::debug::trace_log_scalar(
        "kroko_asr.decoding_method",
        std::string_view(
            decoder_options.method ==
                KrokoDecodingMethod::GreedySearch
            ? "greedy_search"
            : "modified_beam_search"));
    engine::debug::trace_log_scalar(
        "kroko_asr.max_active_paths",
        decoder_options.max_active_paths);
    engine::debug::trace_log_scalar(
        "kroko_asr.blank_penalty",
        static_cast<double>(decoder_options.blank_penalty));
    engine::debug::trace_log_scalar(
        "kroko_asr.hotword_phrases",
        decoder_options.hotwords.size());
    decoder_.configure(std::move(decoder_options));

    endpoint_enabled_ = false;
    if (const auto enabled = runtime::find_option_match(
            request.options, {"enable_endpoint"})) {
        endpoint_enabled_ = runtime::parse_bool_option(
            enabled->value, enabled->key);
    }
    endpoint_rule1_silence_ =
        non_negative_finite_option(
            request.options,
            {"rule1_min_trailing_silence_sec"},
            2.4F);
    endpoint_rule2_silence_ =
        non_negative_finite_option(
            request.options,
            {"rule2_min_trailing_silence_sec"},
            1.2F);
    endpoint_rule3_utterance_ =
        non_negative_finite_option(
            request.options,
            {"rule3_min_utterance_length_sec"},
            20.0F);
    engine::debug::trace_log_scalar(
        "kroko_asr.endpoint_enabled",
        endpoint_enabled_);
    completed_decoded_ = KrokoDecodedTokens{};
    endpoint_segments_.clear();
    endpoint_frame_offset_ = 0;
    endpoint_segment_start_sample_ = 0;
}

std::string KrokoASRSession::request_language(
    const runtime::TaskRequest & request) const {
    std::string requested;
    if (request.text_input.has_value()) {
        requested = request.text_input->language;
    }
    if (const auto it = request.options.find("language");
        it != request.options.end()) {
        requested = it->second;
    }
    const std::string package =
        normalized_language(assets_->config.language);
    const std::string normalized_requested =
        normalized_language(requested);
    const std::string normalized =
        normalized_requested.empty() ||
            normalized_requested == "auto"
        ? package
        : normalized_requested;
    if (normalized != package) {
        throw std::runtime_error(
            "Kroko ASR package language is " + package +
            ", but the request selected " + normalized);
    }
    return package;
}

KrokoDecodedTokens KrokoASRSession::combined_decoded()
    const {
    KrokoDecodedTokens result = completed_decoded_;
    const auto & current = decoder_.decoded();
    result.ids.insert(
        result.ids.end(),
        current.ids.begin(),
        current.ids.end());
    result.frame_indices.insert(
        result.frame_indices.end(),
        current.frame_indices.begin(),
        current.frame_indices.end());
    return result;
}

bool KrokoASRSession::endpoint_detected() const {
    if (!endpoint_enabled_) {
        return false;
    }
    constexpr float kFrameSeconds = 0.04F;
    const int64_t segment_frames = std::max<int64_t>(
        0,
        decoder_.decoded_frames() -
            endpoint_frame_offset_);
    const int64_t trailing_frames = std::min(
        segment_frames,
        decoder_.trailing_blank_frames());
    const float utterance =
        static_cast<float>(segment_frames) *
        kFrameSeconds;
    const float trailing =
        static_cast<float>(trailing_frames) *
        kFrameSeconds;
    const bool contains_non_silence =
        utterance > trailing;
    const bool rule1 =
        trailing >= endpoint_rule1_silence_;
    const bool rule2 =
        contains_non_silence &&
        trailing >= endpoint_rule2_silence_;
    const bool rule3 =
        utterance >= endpoint_rule3_utterance_;
    return rule1 || rule2 || rule3;
}

std::optional<runtime::SpeechSegment>
KrokoASRSession::close_endpoint_segment(
    int64_t audio_samples,
    bool final) {
    if (!endpoint_enabled_) {
        return std::nullopt;
    }
    constexpr int64_t kSamplesPerEncoderFrame = 640;
    const int64_t frame_boundary =
        decoder_.decoded_frames() *
        kSamplesPerEncoderFrame;
    const int64_t end = final
        ? audio_samples
        : std::min(audio_samples, frame_boundary);
    if (end <= endpoint_segment_start_sample_) {
        return std::nullopt;
    }
    const bool contains_speech =
        !decoder_.decoded().ids.empty();
    const int64_t start =
        endpoint_segment_start_sample_;
    endpoint_segment_start_sample_ = end;
    if (!contains_speech) {
        return std::nullopt;
    }
    runtime::SpeechSegment segment;
    segment.span.start_sample = start;
    segment.span.end_sample = end;
    endpoint_segments_.push_back(segment);
    return segment;
}

void KrokoASRSession::archive_decoder_and_reset(
    int64_t frame_offset) {
    const auto & current = decoder_.decoded();
    completed_decoded_.ids.insert(
        completed_decoded_.ids.end(),
        current.ids.begin(),
        current.ids.end());
    completed_decoded_.frame_indices.insert(
        completed_decoded_.frame_indices.end(),
        current.frame_indices.begin(),
        current.frame_indices.end());
    decoder_.reset_segment(frame_offset);
    endpoint_frame_offset_ = frame_offset;
}

runtime::TaskResult KrokoASRSession::make_result(
    const KrokoDecodedTokens & decoded,
    int64_t audio_samples,
    const std::string & language) const {
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{
        tokenizer_.decode(decoded.ids),
        language};
    result.word_timestamps = build_word_timestamps(
        tokenizer_, decoded, audio_samples);
    result.speech_segments = endpoint_segments_;
    return result;
}

runtime::TaskResult KrokoASRSession::run(
    const runtime::TaskRequest & request) {
    require_prepared("Kroko ASR run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error(
            "Kroko ASR run() requires an offline session");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error(
            "Kroko ASR requires --audio");
    }
    const std::string language = request_language(request);
    configure_request(request);
    subsampling_.reset();
    zipformer_.reset();
    const auto start = Clock::now();
    const auto padded_audio =
        with_tail_padding(*request.audio_input);
    const auto features =
        compute_kroko_fbank(padded_audio);
    if (features.frames <= 0) {
        throw std::runtime_error(
            "Kroko ASR frontend produced no frames");
    }
    const int64_t chunk_size = assets_->config.chunk_size;
    const int64_t chunk_shift = assets_->config.chunk_shift;
    const int64_t dimension = assets_->config.feature_dim;
    int64_t encoder_frames = 0;
    for (int64_t offset = 0;
         offset < features.frames;
         offset += chunk_shift) {
        std::fill(
            chunk_scratch_.begin(),
            chunk_scratch_.end(),
            0.0F);
        const int64_t available = std::min(
            chunk_size, features.frames - offset);
        std::copy_n(
            features.values.data() + offset * dimension,
            available * dimension,
            chunk_scratch_.data());
        const auto embedded =
            subsampling_.encode_subsampled_chunk(chunk_scratch_);
        const auto encoded =
            zipformer_.encode_chunk(embedded.values);
        const int64_t consumed = std::min(
            chunk_shift, features.frames - offset);
        const int64_t valid_frames = std::min<int64_t>(
            encoded.frames, (consumed + 3) / 4);
        decoder_.append(
            encoded.values, valid_frames, encoded.channels);
        encoder_frames += valid_frames;
        if (endpoint_detected()) {
            (void)close_endpoint_segment(
                normalized_audio_samples(
                    *request.audio_input),
                false);
            archive_decoder_and_reset(
                decoder_.decoded_frames());
        }
    }
    (void)close_endpoint_segment(
        normalized_audio_samples(*request.audio_input),
        true);
    const auto decoded = combined_decoded();
    const auto result = make_result(
        decoded,
        normalized_audio_samples(*request.audio_input),
        language);
    engine::debug::trace_log_scalar(
        "kroko_asr.frontend_frames", features.frames);
    engine::debug::trace_log_scalar(
        "kroko_asr.encoder_frames", encoder_frames);
    engine::debug::timing_log_scalar(
        "kroko_asr.session_ms",
        engine::debug::elapsed_ms(start, Clock::now()));
    return result;
}

runtime::StreamingPolicy KrokoASRSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = 16000;
    policy.preferred_audio_chunk_seconds = 1.0;
    return policy;
}

void KrokoASRSession::start_stream(
    const runtime::TaskRequest & request) {
    reset();
    streaming_language_ = request_language(request);
    configure_request(request);
    stream_start_ = Clock::now();
    stream_started_ = true;
}

void KrokoASRSession::set_stream_event_sink(
    runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void KrokoASRSession::reset() {
    require_prepared("Kroko ASR reset");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error(
            "Kroko ASR reset() requires a streaming session");
    }
    streaming_audio_ = runtime::AudioBuffer{
        16000, 1, {}};
    streaming_language_.clear();
    streaming_resampler_source_.clear();
    completed_decoded_ = KrokoDecodedTokens{};
    endpoint_segments_.clear();
    processed_feature_offset_ = 0;
    streaming_total_samples_ = 0;
    streaming_source_offset_ = 0;
    streaming_source_frames_ = 0;
    streaming_next_output_sample_ = 0;
    streaming_encoder_chunks_ = 0;
    endpoint_frame_offset_ = 0;
    endpoint_segment_start_sample_ = 0;
    streaming_peak_buffer_values_ = 0;
    streaming_peak_source_values_ = 0;
    streaming_source_sample_rate_ = 0;
    streaming_source_channels_ = 0;
    streaming_received_audio_ = false;
    stream_started_ = false;
    subsampling_.reset();
    zipformer_.reset();
    decoder_.reset();
}

void KrokoASRSession::append_streaming_chunk(
    const runtime::AudioChunk & chunk) {
    if (chunk.sample_rate <= 0 || chunk.channels <= 0 ||
        chunk.samples.empty() ||
        chunk.samples.size() %
                static_cast<size_t>(chunk.channels) !=
            0) {
        throw std::runtime_error(
            "Kroko streaming audio chunk has an invalid format");
    }
    if (streaming_source_sample_rate_ == 0) {
        streaming_source_sample_rate_ = chunk.sample_rate;
        streaming_source_channels_ = chunk.channels;
    } else if (
        streaming_source_sample_rate_ != chunk.sample_rate ||
        streaming_source_channels_ != chunk.channels) {
        throw std::runtime_error(
            "Kroko streaming audio format cannot change within a stream");
    }

    const auto mono =
        engine::audio::mixdown_interleaved_to_mono_average(
            chunk.samples,
            chunk.channels);
    streaming_received_audio_ = true;
    streaming_source_frames_ +=
        static_cast<int64_t>(mono.size());
    if (chunk.sample_rate == 16000) {
        streaming_audio_.samples.insert(
            streaming_audio_.samples.end(),
            mono.begin(),
            mono.end());
        streaming_next_output_sample_ +=
            static_cast<int64_t>(mono.size());
        streaming_total_samples_ =
            streaming_next_output_sample_;
        return;
    }

    streaming_resampler_source_.insert(
        streaming_resampler_source_.end(),
        mono.begin(),
        mono.end());
    streaming_peak_source_values_ = std::max(
        streaming_peak_source_values_,
        streaming_resampler_source_.size());
    const double scale =
        16000.0 /
        static_cast<double>(streaming_source_sample_rate_);
    const int64_t available_end =
        streaming_source_offset_ +
        static_cast<int64_t>(
            streaming_resampler_source_.size());
    while (true) {
        const double source_position =
            static_cast<double>(
                streaming_next_output_sample_) /
            scale;
        const int64_t left = static_cast<int64_t>(
            std::floor(source_position));
        const int64_t right = left + 1;
        if (right >= available_end) {
            break;
        }
        if (left < streaming_source_offset_) {
            throw std::runtime_error(
                "Kroko streaming resampler lost source history");
        }
        const size_t local = static_cast<size_t>(
            left - streaming_source_offset_);
        const float fraction = static_cast<float>(
            source_position -
            static_cast<double>(left));
        streaming_audio_.samples.push_back(
            streaming_resampler_source_[local] *
                    (1.0F - fraction) +
                streaming_resampler_source_[local + 1] *
                    fraction);
        ++streaming_next_output_sample_;
    }
    streaming_total_samples_ =
        streaming_next_output_sample_;

    const int64_t next_left = static_cast<int64_t>(
        std::floor(
            static_cast<double>(
                streaming_next_output_sample_) /
            scale));
    const int64_t discard = std::clamp(
        next_left - streaming_source_offset_,
        int64_t{0},
        static_cast<int64_t>(
            streaming_resampler_source_.size()));
    if (discard > 0) {
        streaming_resampler_source_.erase(
            streaming_resampler_source_.begin(),
            streaming_resampler_source_.begin() +
                static_cast<std::ptrdiff_t>(discard));
        streaming_source_offset_ += discard;
    }
}

void KrokoASRSession::flush_streaming_resampler() {
    if (streaming_source_sample_rate_ <= 0 ||
        streaming_source_sample_rate_ == 16000) {
        return;
    }
    const double scale =
        16000.0 /
        static_cast<double>(streaming_source_sample_rate_);
    const int64_t target_samples =
        static_cast<int64_t>(std::llround(
            static_cast<double>(
                streaming_source_frames_) *
            scale));
    if (streaming_next_output_sample_ >= target_samples) {
        streaming_total_samples_ = target_samples;
        return;
    }
    if (streaming_resampler_source_.empty()) {
        throw std::runtime_error(
            "Kroko streaming resampler has no final source sample");
    }
    const int64_t available_end =
        streaming_source_offset_ +
        static_cast<int64_t>(
            streaming_resampler_source_.size());
    while (streaming_next_output_sample_ <
           target_samples) {
        const double source_position =
            static_cast<double>(
                streaming_next_output_sample_) /
            scale;
        const int64_t left = static_cast<int64_t>(
            std::floor(source_position));
        const int64_t right = std::min(
            left + 1,
            streaming_source_frames_ - 1);
        if (left < streaming_source_offset_ ||
            right >= available_end) {
            throw std::runtime_error(
                "Kroko streaming resampler final history is invalid");
        }
        const size_t local_left = static_cast<size_t>(
            left - streaming_source_offset_);
        const size_t local_right = static_cast<size_t>(
            right - streaming_source_offset_);
        const float fraction = static_cast<float>(
            source_position -
            static_cast<double>(left));
        streaming_audio_.samples.push_back(
            streaming_resampler_source_[local_left] *
                    (1.0F - fraction) +
                streaming_resampler_source_[local_right] *
                    fraction);
        ++streaming_next_output_sample_;
    }
    streaming_total_samples_ = target_samples;
}

runtime::StreamEvent KrokoASRSession::process_streaming_audio(
    bool final) {
    runtime::StreamEvent event;
    event.is_final = final;
    if (streaming_audio_.samples.empty()) {
        return event;
    }
    const auto frontend_audio =
        final ? with_tail_padding(streaming_audio_) : streaming_audio_;
    const auto features = compute_kroko_fbank(frontend_audio);
    const int64_t chunk_size = assets_->config.chunk_size;
    const int64_t chunk_shift = assets_->config.chunk_shift;
    const int64_t dimension = assets_->config.feature_dim;
    bool processed = false;
    while (processed_feature_offset_ < features.frames) {
        if (!final &&
            processed_feature_offset_ + chunk_size + 1 >
                features.frames) {
            break;
        }
        std::fill(
            chunk_scratch_.begin(),
            chunk_scratch_.end(),
            0.0F);
        const int64_t available = std::min(
            chunk_size,
            features.frames - processed_feature_offset_);
        std::copy_n(
            features.values.data() +
                processed_feature_offset_ * dimension,
            available * dimension,
            chunk_scratch_.data());
        const auto embedded =
            subsampling_.encode_subsampled_chunk(chunk_scratch_);
        const auto encoded =
            zipformer_.encode_chunk(embedded.values);
        const int64_t consumed = std::min(
            chunk_shift,
            features.frames - processed_feature_offset_);
        const int64_t valid_frames = std::min<int64_t>(
            encoded.frames, (consumed + 3) / 4);
        decoder_.append(
            encoded.values, valid_frames, encoded.channels);
        processed_feature_offset_ += chunk_shift;
        ++streaming_encoder_chunks_;
        processed = true;
        if (endpoint_detected()) {
            if (const auto segment =
                    close_endpoint_segment(
                        streaming_total_samples_,
                        false)) {
                runtime::VoiceActivityEvent activity;
                activity.kind =
                    runtime::VoiceActivityEvent::Kind::
                        SpeechSegment;
                activity.sample =
                    segment->span.end_sample;
                activity.segment = *segment;
                event.voice_activity.push_back(
                    std::move(activity));
            }
            archive_decoder_and_reset(
                decoder_.decoded_frames());
        }
    }
    if (!final && processed &&
        processed_feature_offset_ > 1) {
        const size_t discard = static_cast<size_t>(
            (processed_feature_offset_ - 1) *
            160 * streaming_audio_.channels);
        if (discard <= streaming_audio_.samples.size()) {
            streaming_audio_.samples.erase(
                streaming_audio_.samples.begin(),
                streaming_audio_.samples.begin() +
                    static_cast<std::ptrdiff_t>(discard));
            processed_feature_offset_ = 1;
        }
    }
    if (final) {
        if (const auto segment =
                close_endpoint_segment(
                    streaming_total_samples_,
                    true)) {
            runtime::VoiceActivityEvent activity;
            activity.kind =
                runtime::VoiceActivityEvent::Kind::
                    SpeechSegment;
            activity.sample = segment->span.end_sample;
            activity.segment = *segment;
            event.voice_activity.push_back(
                std::move(activity));
        }
    }
    if (processed || final) {
        const auto decoded = combined_decoded();
        const auto result = make_result(
            decoded,
            streaming_total_samples_,
            streaming_language_);
        event.partial_text = result.text_output;
        event.word_timestamps = result.word_timestamps;
    }
    return event;
}

runtime::StreamEvent KrokoASRSession::process_audio_chunk(
    const runtime::AudioChunk & chunk) {
    require_prepared("Kroko ASR process_audio_chunk");
    if (task_.mode != runtime::RunMode::Streaming ||
        !stream_started_) {
        throw std::runtime_error(
            "Kroko ASR streaming has not been started");
    }
    append_streaming_chunk(chunk);
    streaming_peak_buffer_values_ = std::max(
        streaming_peak_buffer_values_,
        streaming_audio_.samples.size());
    return process_streaming_audio(false);
}

runtime::TaskResult KrokoASRSession::finalize() {
    require_prepared("Kroko ASR finalize");
    if (task_.mode != runtime::RunMode::Streaming ||
        !stream_started_) {
        throw std::runtime_error(
            "Kroko ASR streaming has not been started");
    }
    if (!streaming_received_audio_) {
        throw std::runtime_error(
            "Kroko ASR finalize() requires streamed audio");
    }
    flush_streaming_resampler();
    auto event = process_streaming_audio(true);
    if (stream_event_sink_) {
        stream_event_sink_(event);
    }
    runtime::TaskResult result;
    result.text_output = event.partial_text;
    result.speech_segments = endpoint_segments_;
    result.word_timestamps = std::move(event.word_timestamps);
    engine::debug::timing_log_scalar(
        "kroko_asr.session_ms",
        engine::debug::elapsed_ms(stream_start_, Clock::now()));
    engine::debug::trace_log_scalar(
        "kroko_asr.streaming.encoder_chunks",
        streaming_encoder_chunks_);
    engine::debug::trace_log_scalar(
        "kroko_asr.streaming.peak_buffer_values",
        streaming_peak_buffer_values_);
    engine::debug::trace_log_scalar(
        "kroko_asr.streaming.peak_resampler_source_values",
        streaming_peak_source_values_);
    engine::debug::trace_log_scalar(
        "kroko_asr.streaming.total_samples",
        streaming_total_samples_);
    stream_started_ = false;
    return result;
}

runtime::TaskResult KrokoASRSession::finish_stream() {
    return finalize();
}

std::shared_ptr<runtime::IVoiceModelLoader> make_kroko_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<KrokoASRAssets> config;
    config.family = std::string(kFamily);
    config.load_assets = load_kroko_asr_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const KrokoASRAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<KrokoASRSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::kroko_asr
