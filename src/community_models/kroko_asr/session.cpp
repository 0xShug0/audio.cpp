#include "engine/community_models/kroko_asr/session.h"

#include "engine/community_models/kroko_asr/frontend.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::kroko_asr {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const KrokoASRAssets> require_assets(
    std::shared_ptr<const KrokoASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error(
            "Kroko ASR session requires assets");
    }
    return assets;
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

}  // namespace

KrokoASRSession::KrokoASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const KrokoASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_(
          assets_->tokens,
          static_cast<int32_t>(assets_->config.blank_id),
          static_cast<int32_t>(assets_->config.unk_id)),
      decoder_(assets_),
      subsampling_(assets_, execution_context()),
      zipformer_(assets_, execution_context()) {
    if (task_.task != runtime::VoiceTaskKind::Asr ||
        (task_.mode != runtime::RunMode::Offline &&
         task_.mode != runtime::RunMode::Streaming)) {
        throw std::runtime_error(
            "Kroko ASR supports offline and streaming ASR sessions");
    }
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("kroko_asr.", 0) == 0) {
            throw std::runtime_error(
                "unknown Kroko ASR session option: " + key);
        }
    }
}

KrokoASRSession::~KrokoASRSession() = default;

std::string KrokoASRSession::family() const {
    return "kroko_asr";
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
    subsampling_.reset();
    zipformer_.reset();
    decoder_.reset();
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
        std::vector<float> chunk(
            static_cast<size_t>(chunk_size * dimension), 0.0F);
        const int64_t available = std::min(
            chunk_size, features.frames - offset);
        std::copy_n(
            features.values.data() + offset * dimension,
            available * dimension,
            chunk.data());
        const auto embedded =
            subsampling_.encode_subsampled_chunk(chunk);
        const auto encoded =
            zipformer_.encode_chunk(embedded.values);
        const int64_t consumed = std::min(
            chunk_shift, features.frames - offset);
        const int64_t valid_frames = std::min<int64_t>(
            encoded.frames, (consumed + 3) / 4);
        std::vector<float> valid_values(
            encoded.values.begin(),
            encoded.values.begin() +
                valid_frames * encoded.channels);
        decoder_.append(
            valid_values, valid_frames, encoded.channels);
        encoder_frames += valid_frames;
    }
    const auto result = make_result(
        decoder_.decoded(),
        normalized_audio_samples(*request.audio_input),
        request_language(request));
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
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_language_.clear();
    processed_feature_offset_ = 0;
    streaming_total_samples_ = 0;
    streaming_encoder_chunks_ = 0;
    streaming_peak_buffer_values_ = 0;
    stream_started_ = false;
    subsampling_.reset();
    zipformer_.reset();
    decoder_.reset();
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
        std::vector<float> chunk(
            static_cast<size_t>(chunk_size * dimension), 0.0F);
        const int64_t available = std::min(
            chunk_size,
            features.frames - processed_feature_offset_);
        std::copy_n(
            features.values.data() +
                processed_feature_offset_ * dimension,
            available * dimension,
            chunk.data());
        const auto embedded =
            subsampling_.encode_subsampled_chunk(chunk);
        const auto encoded =
            zipformer_.encode_chunk(embedded.values);
        const int64_t consumed = std::min(
            chunk_shift,
            features.frames - processed_feature_offset_);
        const int64_t valid_frames = std::min<int64_t>(
            encoded.frames, (consumed + 3) / 4);
        std::vector<float> valid_values(
            encoded.values.begin(),
            encoded.values.begin() +
                valid_frames * encoded.channels);
        decoder_.append(
            valid_values, valid_frames, encoded.channels);
        processed_feature_offset_ += chunk_shift;
        ++streaming_encoder_chunks_;
        processed = true;
    }
    if (!final && processed &&
        streaming_audio_.sample_rate == 16000 &&
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
    if (processed || final) {
        const auto result = make_result(
            decoder_.decoded(),
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
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;
    runtime::append_audio_buffer(streaming_audio_, audio);
    streaming_total_samples_ += normalized_audio_samples(audio);
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
    if (streaming_audio_.samples.empty()) {
        throw std::runtime_error(
            "Kroko ASR finalize() requires streamed audio");
    }
    auto event = process_streaming_audio(true);
    if (stream_event_sink_) {
        stream_event_sink_(event);
    }
    runtime::TaskResult result;
    result.text_output = event.partial_text;
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
        "kroko_asr.streaming.total_samples",
        streaming_total_samples_);
    stream_started_ = false;
    return result;
}

runtime::TaskResult KrokoASRSession::finish_stream() {
    return finalize();
}

}  // namespace engine::models::kroko_asr
