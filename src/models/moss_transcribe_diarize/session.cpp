#include "engine/models/moss_transcribe_diarize/session.h"
#include "engine/models/moss_transcribe_diarize/runtime.h"

#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <cmath>
#include <regex>

namespace engine::models::moss_transcribe_diarize {
namespace {

struct TranscribeAssets {
    assets::ResourceBundle resources;
};

class TranscribeSession final : public runtime::RuntimeSessionBase,
                                public runtime::IOfflineVoiceTaskSession,
                                public runtime::IStreamingVoiceTaskSession {
public:
    TranscribeSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                      std::shared_ptr<const TranscribeAssets> resources,
                      std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)), resources_(std::move(resources)), mode_(task.mode) {
        runtime::validate_spec_backed_session_options(options, *contract_, "moss_transcribe_diarize", "MOSS-Transcribe-Diarize");
        if (task.task != runtime::VoiceTaskKind::Asr ||
            (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming)) {
            throw std::runtime_error("MOSS-Transcribe-Diarize supports offline and streaming ASR");
        }
        default_max_tokens_ = io::json::require_i64(resources_->resources.parse_json("generation_config"), "max_new_tokens");
        model_ = std::make_unique<TranscribeRuntime>(resources_->resources, execution_context());
    }

    std::string family() const override { return "moss_transcribe_diarize"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return mode_; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio || request.audio->sample_rate <= 0 || request.audio->channels <= 0) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires an audio preparation contract");
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-Transcribe-Diarize run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "MOSS-Transcribe-Diarize");
        if (!request.audio_input) {
            throw std::runtime_error("MOSS-Transcribe-Diarize requires audio_input");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, default_max_tokens_);
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or("");
        const auto text = io::trim_ascii_whitespace(model_->transcribe(*request.audio_input, instruction, max_tokens));
        auto result = parse_result(text, request.audio_input->sample_rate);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::PullEvents;
        return policy;
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("MOSS-Transcribe-Diarize start_stream");
        if (mode_ != runtime::RunMode::Streaming) {
            throw std::runtime_error("MOSS-Transcribe-Diarize start_stream requires streaming mode");
        }
        runtime::validate_spec_backed_request_options(request.options, *contract_, "MOSS-Transcribe-Diarize");
        if (!request.audio_input) {
            throw std::runtime_error("MOSS-Transcribe-Diarize streaming requires complete audio_input");
        }
        reset();
        const auto max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, default_max_tokens_);
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or("");
        model_->start(*request.audio_input, instruction, max_tokens);
        stream_sample_rate_ = request.audio_input->sample_rate;
        stream_started_ = true;
    }

    std::optional<runtime::StreamEvent> next_stream_event() override {
        if (!stream_started_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize stream has not been started");
        }
        auto delta = model_->next_text();
        if (!delta) {
            stream_complete_ = true;
            return std::nullopt;
        }
        stream_text_ += *delta;
        runtime::StreamEvent event;
        event.partial_text = runtime::Transcript{std::move(*delta), ""};
        return event;
    }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("MOSS-Transcribe-Diarize supports text-output streaming, not incremental audio input");
    }

    runtime::TaskResult finalize() override {
        if (!stream_started_ || !stream_complete_) {
            throw std::runtime_error("MOSS-Transcribe-Diarize finalize requires completed text generation");
        }
        auto result = parse_result(stream_text_, stream_sample_rate_);
        reset();
        return result;
    }

    void reset() override {
        model_->reset();
        stream_started_ = false;
        stream_complete_ = false;
        stream_text_.clear();
        stream_sample_rate_ = 0;
    }

private:
    static runtime::TaskResult parse_result(const std::string & text, int sample_rate) {
        static const std::regex pattern(
            R"(\[([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\]\s*\[(S[0-9]+)\]([\s\S]*?)\[([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\](?=\s*(?:\[|$)))");
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{text, ""};
        for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it) {
            const double start = std::stod((*it)[1].str());
            const double end = std::stod((*it)[4].str());
            const auto words = io::trim_ascii_whitespace((*it)[3].str());
            if (end < start || words.empty()) {
                continue;
            }
            const runtime::TimeSpan span{
                std::llround(start * sample_rate),
                std::llround(end * sample_rate)};
            result.speech_segments.push_back({span, 0.f, words});
            result.speaker_turns.push_back({span, (*it)[2].str(), 0.f, words});
        }
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::shared_ptr<const TranscribeAssets> resources_;
    std::unique_ptr<TranscribeRuntime> model_;
    int64_t default_max_tokens_ = 0;
    runtime::RunMode mode_;
    bool stream_started_ = false;
    bool stream_complete_ = false;
    int stream_sample_rate_ = 0;
    std::string stream_text_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_transcribe_diarize_loader() {
    runtime::SpecBackedVoiceModelConfig<TranscribeAssets> config;
    config.family = "moss_transcribe_diarize";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<TranscribeAssets>();
        assets->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("moss_transcribe_diarize"));
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const TranscribeAssets> resources,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<TranscribeSession>(task, options, std::move(resources), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::moss_transcribe_diarize
