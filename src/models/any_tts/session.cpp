#include "engine/models/any_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <chrono>
#include <stdexcept>

namespace engine::models::any_tts {
namespace {

using namespace runtime;

class Session final : public IOfflineVoiceTaskSession, public IStreamingVoiceTaskSession {
public:
    Session(TaskSpec task, const SessionOptions & options, std::shared_ptr<ILoadedVoiceModel> base,
            std::shared_ptr<ILoadedVoiceModel> converter)
        : task_(task), base_model_(std::move(base)), converter_model_(std::move(converter)) {
        base_session_ = base_model_->create_task_session({VoiceTaskKind::Tts, RunMode::Offline}, options);
        base_ = dynamic_cast<IOfflineVoiceTaskSession *>(base_session_.get());
        if (!base_) {
            throw std::runtime_error("Any TTS requires a base model with offline TTS support");
        }
        converter_session_ = converter_model_->create_task_session(
            {VoiceTaskKind::VoiceConversion, RunMode::Offline}, {options.backend, {}});
        converter_ = dynamic_cast<IOfflineVoiceTaskSession *>(converter_session_.get());
        if (!converter_) {
            throw std::runtime_error("Tone Color VC does not provide an offline conversion session");
        }
    }

    std::string family() const override { return "any_tts"; }
    VoiceTaskKind task_kind() const override { return task_.task; }
    RunMode run_mode() const override { return task_.mode; }

    void prepare(const SessionPreparationRequest & request) override {
        if (!request.text || request.text->text.empty()) {
            throw std::runtime_error("Any TTS requires text");
        }
        if (!request.voice || !request.voice->speaker || !request.voice->speaker->audio) {
            throw std::runtime_error("Any TTS requires reference speaker audio");
        }
    }

    TaskResult run(const TaskRequest & request) override {
        start_stream(request);
        while (next_stream_event()) {}
        return finish_stream();
    }

    StreamingPolicy streaming_policy() const override {
        return {StreamingInputKind::None, StreamingOutputKind::PullEvents, 0, 0};
    }

    void start_stream(const TaskRequest & request) override {
        prepare(build_preparation_request(request));
        reset();
        started_ = std::chrono::steady_clock::now();
        const auto size = text::parse_text_chunk_size_override(request.options).value_or(240);
        const auto mode = text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default);
        if (size <= 0) {
            throw std::runtime_error("Any TTS text_chunk_size must be positive");
        }
        chunks_ = text::split_text_chunks(request.text_input->text, size, mode);
        reference_ = request.voice;
        request_ = request;
        if (request_.voice->speaker->cached_voice_id) {
            request_.voice->speaker->audio.reset();
        } else {
            request_.voice->speaker.reset();
        }
    }

    std::optional<StreamEvent> next_stream_event() override {
        if (next_ == chunks_.size()) {
            return std::nullopt;
        }
        auto request = request_;
        request.text_input->text = std::move(chunks_[next_++]);
        TaskRequest conversion;
        conversion.voice = reference_;
        for (const auto & key : {"vc_temperature", "vc_seed"}) {
            const auto found = request.options.find(key);
            if (found != request.options.end()) {
                conversion.options.emplace(std::string(key).substr(3), found->second);
                request.options.erase(found);
            }
        }
        request.options.erase("text_chunk_size");
        request.options.erase("text_chunk_mode");
        base_->prepare(build_preparation_request(request));
        auto generated = base_->run(request);
        if (!generated.audio_output) {
            throw std::runtime_error("Any TTS base model returned no audio");
        }
        conversion.audio_input = std::move(generated.audio_output);
        converter_->prepare(build_preparation_request(conversion));
        auto converted = converter_->run(conversion);
        if (!converted.audio_output) {
            throw std::runtime_error("Tone Color VC returned no audio");
        }
        if (!result_.audio_output) {
            result_.audio_output.emplace();
        }
        append_audio_buffer(*result_.audio_output, *converted.audio_output);
        StreamEvent event;
        event.audio_output = std::move(converted.audio_output);
        event.is_final = next_ == chunks_.size();
        return event;
    }

    TaskResult finish_stream() override {
        if (next_ != chunks_.size()) {
            throw std::runtime_error("Any TTS stream must be consumed before finalization");
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started_));
        return std::move(result_);
    }

    void reset() override {
        chunks_.clear();
        request_ = {};
        reference_.reset();
        next_ = 0;
        result_ = {};
    }
    StreamEvent process_audio_chunk(const AudioChunk &) override {
        throw std::runtime_error("Any TTS accepts text, not streaming audio input");
    }
    TaskResult finalize() override { return finish_stream(); }

private:
    TaskSpec task_;
    std::shared_ptr<ILoadedVoiceModel> base_model_;
    std::shared_ptr<ILoadedVoiceModel> converter_model_;
    std::unique_ptr<IVoiceTaskSession> base_session_, converter_session_;
    IOfflineVoiceTaskSession * base_ = nullptr;
    IOfflineVoiceTaskSession * converter_ = nullptr;
    std::vector<std::string> chunks_;
    TaskRequest request_;
    std::optional<VoiceCondition> reference_;
    size_t next_ = 0;
    TaskResult result_;
    std::chrono::steady_clock::time_point started_;
};

class Model final : public ILoadedVoiceModel {
public:
    Model(std::unique_ptr<ILoadedVoiceModel> base, std::unique_ptr<ILoadedVoiceModel> converter)
        : base_(std::move(base)), converter_(std::move(converter)), contract_(require_model_contract("any_tts")) {}
    const ModelMetadata & metadata() const noexcept override { return contract_->metadata; }
    const CapabilitySet & capabilities() const noexcept override { return contract_->capabilities; }
    std::unique_ptr<IVoiceTaskSession> create_task_session(const TaskSpec & task, const SessionOptions & options) const override {
        if (task.task != VoiceTaskKind::VoiceCloning && task.task != VoiceTaskKind::Tts) {
            throw std::runtime_error("Any TTS supports tts and clon tasks");
        }
        validate_spec_backed_session_options(options, *contract_, "any_tts", "Any TTS");
        return std::make_unique<Session>(task, options, base_, converter_);
    }
private:
    std::shared_ptr<ILoadedVoiceModel> base_;
    std::shared_ptr<ILoadedVoiceModel> converter_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
};

class Loader final : public IVoiceModelLoader {
public:
    std::string family() const override { return "any_tts"; }
    bool can_load(const ModelLoadRequest & request) const override { return request.family_hint == "any_tts"; }
    CapabilitySet advertised_capabilities() const override { return require_model_contract("any_tts")->capabilities; }
    ModelInspection inspect(const ModelLoadRequest & request) const override {
        const auto manifest = io::json::parse_file(request.model_path);
        auto inspection = make_default_registry().inspect(component_request(request, manifest, false));
        const auto contract = require_model_contract("any_tts");
        inspection.metadata = contract->metadata;
        inspection.capabilities = contract->capabilities;
        inspection.cli = contract->cli;
        return inspection;
    }
    std::unique_ptr<ILoadedVoiceModel> load(const ModelLoadRequest & request) const override {
        const auto manifest = io::json::parse_file(request.model_path);
        auto registry = make_default_registry();
        auto base = registry.load(component_request(request, manifest, false));
        auto converter = registry.load(component_request(request, manifest, true));
        return std::make_unique<Model>(std::move(base), std::move(converter));
    }
private:
    static ModelLoadRequest component_request(const ModelLoadRequest & request,
                                              const io::json::Value & manifest, bool converter) {
        ModelLoadRequest component;
        component.model_path = manifest.require(converter ? "converter_model" : "base_model").as_string();
        if (component.model_path.is_relative()) {
            component.model_path = request.model_path.parent_path() / component.model_path;
        }
        component.family_hint = converter ? "tone_color_vc" : manifest.require("base_family").as_string();
        if (*component.family_hint == "any_tts") {
            throw std::runtime_error("Any TTS cannot use itself as its base model");
        }
        if (!converter) {
            component.options = request.options;
        }
        return component;
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_any_tts_loader() { return std::make_shared<Loader>(); }

}  // namespace engine::models::any_tts
