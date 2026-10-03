#include "engine/models/owsm/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::owsm {
namespace {

class OWSMV4Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession,
                            public runtime::IStreamingVoiceTaskSession {
public:
    OWSMV4Session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const OWSMV4Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)), mode_(task.mode) {
        runtime::validate_spec_backed_session_options(options, *contract_, "owsm", "OWSM v4");
        if (task.task != runtime::VoiceTaskKind::Asr) {
            throw std::runtime_error("OWSM v4 requires an ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"owsm.weight_type"}).value_or("native"));
        weights_ = load_owsm_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<OWSMV4Runtime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "owsm"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return mode_; }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::FinalResult;
        return policy;
    }
    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }
    void start_stream(const runtime::TaskRequest & request) override {
        reset();
        stream_result_ = transcribe(request, true);
    }
    void reset() override { stream_result_.reset(); }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("OWSM streaming requires complete audio, not live audio chunks");
    }
    runtime::TaskResult finalize() override {
        if (!stream_result_) {
            throw std::runtime_error("OWSM stream has not completed");
        }
        auto result = std::move(*stream_result_);
        reset();
        return result;
    }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "OWSM v4");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        return transcribe(request, false);
    }

private:
    runtime::TaskResult transcribe(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("OWSM v4 run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "OWSM v4");
        if (!request.audio_input) {
            throw std::runtime_error("OWSM v4 requires audio input");
        }
        const auto started = std::chrono::steady_clock::now();
        auto language = runtime::find_option(request.options, {"language"}).value_or("eng");
        const auto target = runtime::find_option(request.options, {"target_language"});
        auto language_token = language == "auto" ? -1 : assets_->token_id("<" + language + ">");
        const auto task_token = target.has_value()
            ? assets_->token_id("<st_" + *target + ">")
            : assets_->token_id("<asr>");
        const bool timestamps = runtime::parse_bool_option(
            runtime::find_option(request.options, {"return_timestamps"}).value_or("false"),
            "return_timestamps");
        const bool condition_previous = runtime::parse_bool_option(
            runtime::find_option(request.options, {"condition_on_previous_text"}).value_or("false"),
            "condition_on_previous_text");
        const auto max_tokens = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(0);
        const auto beam_size = runtime::parse_i64_option(request.options, {"beam_size"}).value_or(1);
        if (beam_size < 1 || beam_size > assets_->config.vocabulary_size) {
            throw std::runtime_error("OWSM v4 beam_size must be positive and no larger than the vocabulary");
        }
        if (max_tokens < 0 || max_tokens > assets_->config.max_decode_tokens) {
            throw std::runtime_error("OWSM v4 max_tokens must be between 0 and 374");
        }

        const auto & input = *request.audio_input;
        auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            input.samples, input.sample_rate, input.channels, input.sample_rate);
        if (input.sample_rate != 16000) {
            const auto resampled = audio::try_resample_mono_soxr(mono, input.sample_rate, 16000, {});
            if (!resampled) {
                throw std::runtime_error("OWSM v4 requires SOXR for resampling; provide 16 kHz audio");
            }
            mono = *resampled;
        }
        if (mono.empty()) {
            throw std::runtime_error("OWSM v4 audio input is empty");
        }

        const auto mode = audio::parse_audio_chunk_mode(request.options);
        if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed &&
            mode != audio::AudioChunkMode::None) {
            throw std::runtime_error("OWSM v4 supports auto, fixed, and none audio chunking");
        }
        const auto seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(30.0F);
        if (!std::isfinite(seconds) || seconds <= 0.0F || seconds > 30.0F) {
            throw std::runtime_error("OWSM v4 audio_chunk_duration_sec must be positive and at most 30");
        }
        if (mode == audio::AudioChunkMode::None &&
            mono.size() > static_cast<size_t>(assets_->config.max_audio_samples)) {
            throw std::runtime_error("OWSM v4 audio exceeds 30 seconds; use audio_chunk_mode=auto");
        }
        const auto chunk_samples = mode == audio::AudioChunkMode::None
            ? static_cast<int64_t>(mono.size())
            : static_cast<int64_t>(std::llround(seconds * 16000.0F));
        const auto chunks = audio::plan_audio_chunks(
            static_cast<int64_t>(mono.size()), {chunk_samples, chunk_samples});
        const bool dynamic_windows = mode == audio::AudioChunkMode::Auto && mono.size() > static_cast<size_t>(chunk_samples);
        trace(debug::LogLevel::Info, "owsm",
            "variant=" + assets_->config.variant + " chunks=" + std::to_string(chunks.size()) +
            " language=" + language + " task=" + (target ? "st_" + *target : "asr"));

        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", target.value_or(language)};
        runtime::PartialTextPublisher publisher;
        const auto publish = [&](const std::string & text) {
            if (streaming && sink_) {
                auto delta = publisher.publish(text);
                if (!delta.empty()) {
                    runtime::StreamEvent event;
                    event.partial_text = runtime::Transcript{std::move(delta), result.text_output->language};
                    sink_(event);
                }
            }
        };
        std::string previous_text;
        int64_t offset = 0;
        for (size_t index = 0; offset < static_cast<int64_t>(mono.size()); ++index) {
            const auto chunk = dynamic_windows
                ? audio::AudioChunkSpan{static_cast<int64_t>(index), offset,
                    std::min(chunk_samples, static_cast<int64_t>(mono.size()) - offset), offset, 0}
                : chunks.at(index);
            offset = chunk.copy_start_sample + chunk.valid_samples;
            if (chunk.valid_samples < 3200 && chunks.size() > 1) {
                trace(debug::LogLevel::Info, "owsm",
                    "skip_short_chunk=" + std::to_string(chunk.index) +
                    " samples=" + std::to_string(chunk.valid_samples));
                continue;
            }
            std::vector<int32_t> prompt;
            if (condition_previous && !previous_text.empty()) {
                prompt.push_back(assets_->config.sop_id);
                auto previous = assets_->tokenize_text(previous_text);
                prompt.insert(prompt.end(), previous.begin(), previous.end());
            }
            prompt.push_back(assets_->config.sos_id);
            prompt.push_back(language_token);
            prompt.push_back(task_token);
            if (!timestamps && !dynamic_windows) {
                prompt.push_back(assets_->config.notimestamps_id);
            }
            const auto begin = mono.begin() + chunk.copy_start_sample;
            const std::vector<float> samples(begin, begin + chunk.valid_samples);
            std::function<void(const std::vector<int32_t> &)> on_tokens;
            // Continuation can discard trailing text, and beam search can revise it.
            // Publish those paths only after the window has been committed.
            if (streaming && sink_ && !dynamic_windows && beam_size == 1) {
                on_tokens = [&](const std::vector<int32_t> & tokens) {
                    const auto partial = assets_->decode_visible(tokens);
                    publish(result.text_output->text +
                        (result.text_output->text.empty() || partial.empty() ? "" : " ") + partial);
                };
            }
            auto decoded = runtime_->decode(samples, prompt, timestamps || dynamic_windows, max_tokens, beam_size, on_tokens);
            if (!decoded.detected_language.empty()) {
                language = decoded.detected_language;
                language_token = assets_->token_id("<" + language + ">");
                result.text_output->language = target.value_or(language);
                trace(debug::LogLevel::Info, "owsm", "detected_language=" + language);
            }
            if (dynamic_windows) {
                const auto first = assets_->config.first_timestamp_id;
                const auto last = assets_->config.last_timestamp_id;
                std::vector<size_t> time_positions;
                for (size_t i = 0; i < decoded.tokens.size(); ++i) {
                    if (decoded.tokens[i] >= first && decoded.tokens[i] <= last) {
                        time_positions.push_back(i);
                    }
                }
                if (time_positions.empty()) {
                    throw std::runtime_error("OWSM v4 long-form decoding produced no timestamps");
                }
                if (time_positions.size() == 1) {
                    time_positions.push_back(decoded.tokens.size());
                    decoded.tokens.push_back(last);
                }
                int32_t next_time = decoded.tokens[time_positions.back()];
                if (time_positions.size() % 2 != 0) {
                    decoded.tokens.resize(time_positions.back());
                } else if (time_positions.size() > 2 &&
                    (next_time - first) * 320 > chunk_samples - 16000) {
                    next_time = decoded.tokens[time_positions[time_positions.size() - 2]];
                    decoded.tokens.resize(time_positions[time_positions.size() - 2]);
                }
                const int64_t advance = static_cast<int64_t>(next_time - first) * 320;
                if (advance <= 0) {
                    throw std::runtime_error("OWSM v4 long-form timestamps do not advance the audio window");
                }
                offset = chunk.copy_start_sample + advance;
            }
            const auto text = assets_->decode_visible(decoded.tokens);
            if (!result.text_output->text.empty() && !text.empty()) {
                result.text_output->text += ' ';
            }
            result.text_output->text += text;
            publish(result.text_output->text);
            previous_text = text;
            if (condition_previous && dynamic_windows) {
                std::vector<runtime::SpeechSegment> previous_segments;
                append_segments(decoded.tokens, 0, previous_segments);
                previous_text.clear();
                for (const auto & segment : previous_segments) {
                    previous_text += segment.text;
                }
            }
            if (timestamps) {
                append_segments(decoded.tokens, chunk.copy_start_sample, result.speech_segments);
            }
            trace(debug::LogLevel::Info, "owsm",
                "chunk=" + std::to_string(chunk.index) +
                " samples=" + std::to_string(chunk.valid_samples) +
                " tokens=" + std::to_string(decoded.tokens.size()));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    void append_segments(
        const std::vector<int32_t> & tokens,
        int64_t sample_offset,
        std::vector<runtime::SpeechSegment> & output) const {
        std::vector<size_t> timestamps;
        for (size_t index = 0; index < tokens.size(); ++index) {
            if (tokens[index] >= assets_->config.first_timestamp_id &&
                tokens[index] <= assets_->config.last_timestamp_id) {
                timestamps.push_back(index);
            }
        }
        for (size_t index = 0; index + 1 < timestamps.size(); index += 2) {
            const auto begin = timestamps[index];
            const auto end = timestamps[index + 1];
            if (end <= begin) {
                continue;
            }
            runtime::SpeechSegment segment;
            segment.span.start_sample = sample_offset +
                static_cast<int64_t>(tokens[begin] - assets_->config.first_timestamp_id) * 320;
            segment.span.end_sample = sample_offset +
                static_cast<int64_t>(tokens[end] - assets_->config.first_timestamp_id) * 320;
            segment.text = assets_->decode_visible(
                std::vector<int32_t>(tokens.begin() + static_cast<std::ptrdiff_t>(begin + 1),
                                     tokens.begin() + static_cast<std::ptrdiff_t>(end)));
            segment.confidence = 1.0F;
            output.push_back(std::move(segment));
        }
    }

    std::shared_ptr<const OWSMV4Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<OWSMV4Weights> weights_;
    std::unique_ptr<OWSMV4Runtime> runtime_;
    runtime::RunMode mode_;
    runtime::StreamEventCallback sink_;
    std::optional<runtime::TaskResult> stream_result_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_owsm_loader() {
    runtime::SpecBackedVoiceModelConfig<OWSMV4Assets> config;
    config.family = "owsm";
    config.load_assets = load_owsm_assets;
    config.create_session = [](
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const OWSMV4Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<OWSMV4Session>(
            task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::owsm
