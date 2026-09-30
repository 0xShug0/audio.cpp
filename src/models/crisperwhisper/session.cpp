#include "engine/models/crisperwhisper/model.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::crisperwhisper {
namespace {

class CrisperWhisperSession final : public runtime::RuntimeSessionBase,
    public runtime::IOfflineVoiceTaskSession, public runtime::IStreamingVoiceTaskSession {
public:
    CrisperWhisperSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CrisperWhisperAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)),
          mode_(task.mode), task_(task.task) {
        runtime::validate_spec_backed_session_options(options, *contract_, "crisperwhisper", "CrisperWhisper");
        if (task_ != runtime::VoiceTaskKind::Asr && task_ != runtime::VoiceTaskKind::Alignment) {
            throw std::runtime_error("CrisperWhisper requires an ASR or alignment session");
        }
        if (task_ == runtime::VoiceTaskKind::Alignment && mode_ != runtime::RunMode::Offline) {
            throw std::runtime_error("CrisperWhisper alignment requires offline mode");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"crisperwhisper.weight_type"}).value_or("native"));
        weights_ = load_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<CrisperWhisperEncoderDecoderRuntime>(*assets_, *weights_, execution_context());
    }
    std::string family() const override { return "crisperwhisper"; }
    runtime::VoiceTaskKind task_kind() const override { return task_; }
    runtime::RunMode run_mode() const override { return mode_; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "CrisperWhisper");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("CrisperWhisper run");
        begin(request);
        while (next_chunk_ < chunks_.size()) {
            decode_chunk();
        }
        return finalize();
    }
    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::PullEvents;
        return policy;
    }
    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("CrisperWhisper start_stream");
        begin(request);
    }
    std::optional<runtime::StreamEvent> next_stream_event() override {
        if (!started_) {
            throw std::runtime_error("CrisperWhisper stream has not started");
        }
        if (next_chunk_ == chunks_.size()) {
            return std::nullopt;
        }
        decode_chunk();
        runtime::StreamEvent event;
        auto delta = publisher_.publish(result_.text_output->text);
        if (!delta.empty()) {
            event.partial_text = runtime::Transcript{std::move(delta), language_};
        }
        event.word_timestamps = result_.word_timestamps;
        return event;
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("CrisperWhisper streaming requires complete audio; output is committed per continuation window");
    }
    runtime::TaskResult finalize() override {
        if (!started_ || next_chunk_ != chunks_.size()) {
            throw std::runtime_error("CrisperWhisper finalize requires all continuation windows to finish");
        }
        if (task_ == runtime::VoiceTaskKind::Alignment) {
            std::vector<CrisperWhisperWord> hypothesis;
            for (const auto & word : result_.word_timestamps) {
                hypothesis.push_back({word.word, static_cast<double>(word.span.start_sample) / sample_rate_,
                    static_cast<double>(word.span.end_sample) / sample_rate_});
            }
            const auto aligned = align_transcript(transcript_, hypothesis, static_cast<double>(audio_.size()) / 16000);
            result_.word_timestamps.clear();
            result_.text_output->text.clear();
            for (const auto & word : aligned) {
                runtime::WordTimestamp timed;
                timed.word = word.text;
                timed.span = {std::llround(word.start * sample_rate_), std::llround(word.end * sample_rate_)};
                result_.word_timestamps.push_back(std::move(timed));
                if (!result_.text_output->text.empty()) {
                    result_.text_output->text += ' ';
                }
                result_.text_output->text += word.text;
            }
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start_time_));
        auto result = std::move(result_);
        reset();
        return result;
    }
    void reset() override {
        started_ = false;
        next_chunk_ = 0;
        audio_.clear();
        chunks_.clear();
        words_.clear();
        publisher_.reset();
        result_ = {};
    }

private:
    void begin(const runtime::TaskRequest & request) {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "CrisperWhisper");
        if (!request.audio_input) {
            throw std::runtime_error("CrisperWhisper requires audio input");
        }
        reset();
        start_time_ = std::chrono::steady_clock::now();
        language_ = runtime::find_option(request.options, {"language"}).value_or("en");
        transcription_mode_ = runtime::find_option(request.options, {"transcription_mode"}).value_or("verbatim");
        transcript_ = request.text_input ? io::trim_ascii_whitespace(request.text_input->text) : "";
        if ((transcription_mode_ == "verbatimize" || task_ == runtime::VoiceTaskKind::Alignment) && transcript_.empty()) {
            throw std::runtime_error("CrisperWhisper alignment and verbatimize require transcript text input");
        }
        if (task_ == runtime::VoiceTaskKind::Alignment && transcription_mode_ == "verbatimize") {
            throw std::runtime_error("CrisperWhisper alignment requires transcription_mode=verbatim or intended");
        }
        max_tokens_ = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(256);
        return_timestamps_ = runtime::parse_bool_option(
            runtime::find_option(request.options, {"return_timestamps"}).value_or("false"), "return_timestamps");
        return_timestamps_ = return_timestamps_ || task_ == runtime::VoiceTaskKind::Alignment;
        const auto & audio = *request.audio_input;
        sample_rate_ = audio.sample_rate;
        audio_ = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples, audio.sample_rate, audio.channels, audio.sample_rate);
        if (sample_rate_ != 16000) {
            const auto resampled = audio::try_resample_mono_soxr(audio_, sample_rate_, 16000, {});
            if (!resampled) {
                throw std::runtime_error("CrisperWhisper requires SOXR for resampling; provide 16 kHz audio");
            }
            audio_ = *resampled;
        }
        if (transcription_mode_ == "verbatimize" && audio_.size() > 480000) {
            throw std::runtime_error("CrisperWhisper verbatimize requires audio no longer than 30 seconds");
        }
        const auto chunk_mode = transcription_mode_ == "verbatimize" ? audio::AudioChunkMode::None :
            audio::parse_audio_chunk_mode(request.options);
        const float duration = chunk_mode == audio::AudioChunkMode::None ? 30.0f :
            audio::parse_audio_chunk_seconds_override(request.options).value_or(30.0f);
        const float overlap = chunk_mode == audio::AudioChunkMode::None ? 0.0f :
            runtime::parse_float_option(request.options, {"audio_chunk_overlap_sec"}).value_or(4.0f);
        if (!std::isfinite(duration) || duration <= 0 || duration > 30 ||
            !std::isfinite(overlap) || overlap < 0 || overlap >= duration) {
            throw std::runtime_error("CrisperWhisper requires 0 <= audio_chunk_overlap_sec < audio_chunk_duration_sec <= 30");
        }
        if (chunk_mode != audio::AudioChunkMode::Auto && chunk_mode != audio::AudioChunkMode::Fixed &&
            chunk_mode != audio::AudioChunkMode::None) {
            throw std::runtime_error("CrisperWhisper continuation supports auto, fixed or none audio chunking");
        }
        if (chunk_mode == audio::AudioChunkMode::None && audio_.size() > 480000) {
            throw std::runtime_error("CrisperWhisper audio exceeds 30 seconds; use audio_chunk_mode=auto");
        }
        hop_ = static_cast<int64_t>((duration - overlap) * 16000);
        const auto window = static_cast<int64_t>(duration * 16000);
        chunks_ = audio::plan_audio_chunks(audio_.size(), {window, hop_});
        // The shared planner emits every hop; continuation stops at the first
        // window that already covers the input end, as the reference does.
        const auto last = std::find_if(chunks_.begin(), chunks_.end(), [&](const auto & span) {
            return span.copy_start_sample + span.valid_samples >= static_cast<int64_t>(audio_.size());
        });
        if (last != chunks_.end()) {
            chunks_.erase(last + 1, chunks_.end());
        }
        result_.text_output = runtime::Transcript{"", language_};
        started_ = true;
        trace(debug::LogLevel::Info, "crisperwhisper", "language=" + language_ + " mode=" + transcription_mode_ +
            " windows=" + std::to_string(chunks_.size()) + " max_tokens=" + std::to_string(max_tokens_));
    }
    void decode_chunk() {
        const auto & span = chunks_.at(next_chunk_);
        std::string context;
        for (size_t i = words_.size() > 12 ? words_.size() - 12 : 0; i < words_.size(); ++i) {
            if (!context.empty()) {
                context += ' ';
            }
            context += words_[i];
        }
        const std::vector<float> samples(audio_.begin() + span.copy_start_sample,
            audio_.begin() + span.copy_start_sample + span.valid_samples);
        const auto decoded = runtime_->transcribe(samples,
            assets_->prompt(language_, transcription_mode_, context, transcript_), max_tokens_, language_,
            chunks_.size() == 1 ? 0.0 : next_chunk_ + 1 == chunks_.size() ? 2.0 : 4.0,
            transcription_mode_ == "verbatimize" ? std::vector<int32_t>{} :
                assets_->prompt(language_, transcription_mode_ == "verbatim" ? "intended" : "verbatim", context));
        const auto & words = decoded.words;
        size_t keep = words.size();
        if (next_chunk_ + 1 < chunks_.size()) {
            const auto overlap = std::find_if(words.begin(), words.end(), [&](const auto & word) {
                return word.start >= static_cast<double>(hop_) / 16000;
            });
            const bool placed = std::any_of(words.begin(), words.end(), [](const auto & word) { return word.start >= 0; });
            const size_t count_drop = words.size() > 2 ? words.size() - 2 : 0;
            keep = placed ? std::max(count_drop, static_cast<size_t>(overlap - words.begin())) : count_drop;
        }
        for (size_t i = 0; i < keep; ++i) {
            const auto & word = words[i];
            words_.push_back(word.text);
            if (!result_.text_output->text.empty()) {
                result_.text_output->text += ' ';
            }
            result_.text_output->text += word.text;
            if (return_timestamps_ && word.start >= 0) {
                const double offset = static_cast<double>(span.copy_start_sample) / 16000;
                runtime::WordTimestamp timed;
                timed.word = word.text;
                timed.span = {std::llround((word.start + offset) * sample_rate_),
                              std::llround((word.end + offset) * sample_rate_)};
                if (!result_.word_timestamps.empty()) {
                    timed.span.start_sample = std::max(timed.span.start_sample, result_.word_timestamps.back().span.end_sample);
                    timed.span.end_sample = std::max(timed.span.end_sample, timed.span.start_sample);
                }
                result_.word_timestamps.push_back(std::move(timed));
            }
        }
        if (chunks_.size() == 1) {
            result_.text_output->text = io::trim_ascii_whitespace(assets_->decode_text(decoded.tokens));
        }
        trace(debug::LogLevel::Info, "crisperwhisper", "window=" + std::to_string(next_chunk_) +
            " tokens=" + std::to_string(decoded.tokens.size()) + " committed_words=" + std::to_string(keep) +
            " eos=" + (decoded.reached_eos ? "true" : "false"));
        ++next_chunk_;
    }
    std::shared_ptr<const CrisperWhisperAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<CrisperWhisperWeights> weights_;
    std::unique_ptr<CrisperWhisperEncoderDecoderRuntime> runtime_;
    runtime::RunMode mode_;
    runtime::VoiceTaskKind task_;
    std::string language_, transcription_mode_, transcript_;
    int64_t max_tokens_ = 256, hop_ = 416000;
    int sample_rate_ = 16000;
    std::vector<float> audio_;
    std::vector<audio::AudioChunkSpan> chunks_;
    std::vector<std::string> words_;
    runtime::PartialTextPublisher publisher_;
    runtime::TaskResult result_;
    size_t next_chunk_ = 0;
    bool started_ = false;
    bool return_timestamps_ = false;
    std::chrono::steady_clock::time_point start_time_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_crisperwhisper_loader() {
    runtime::SpecBackedVoiceModelConfig<CrisperWhisperAssets> config;
    config.family = "crisperwhisper";
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CrisperWhisperAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<CrisperWhisperSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::crisperwhisper
