#include "engine/models/open_wake_word/session.h"
#ifdef ENGINE_HAVE_NEON
#include "engine/framework/neon/execution.h"
#endif
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/open_wake_word/runtime.h"

#include <algorithm>

namespace engine::models::open_wake_word {
namespace {
struct Assets {
    assets::ResourceBundle resources;
    io::json::Value config;
    std::shared_ptr<const Weights> weights;
};

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(const runtime::SessionOptions &options, runtime::RunMode mode, std::shared_ptr<const Assets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), mode_(mode), assets_(std::move(assets)),
          contract_(std::move(contract)),
          runtime_(*assets_->weights, execution_context(), [&] {
              const auto backend = execution_context().backend_type();
              const auto plans = runtime::parse_exec_plan(options.options, "open_wake_word.exec_plan",
                  {{"embedding", "ggml",
                    {{"ggml", {backend}, {}, {}}, {"neon", {core::BackendType::Cpu}, {}, {}}}}});
              if (plans.at("embedding").impl != "neon") return false;
#ifdef ENGINE_HAVE_NEON
              if (backend != core::BackendType::Cpu || !neon::available()) {
                  throw std::runtime_error("open_wake_word.exec_plan.embedding: impl neon requires CPU with AArch64 NEON");
              }
              return true;
#else
              throw std::runtime_error("open_wake_word.exec_plan.embedding: impl neon requires ENGINE_ENABLE_NEON=ON");
#endif
          }()) {}
    std::string family() const override {
        return "open_wake_word";
    }
    runtime::VoiceTaskKind task_kind() const override {
        return runtime::VoiceTaskKind::WakeWord;
    }
    runtime::RunMode run_mode() const override {
        return mode_;
    }

    void prepare(const runtime::SessionPreparationRequest &request) override {
        configure(request.options, request.option_arrays);
        if (!request.audio || request.audio->sample_rate != 16000 || request.audio->channels != 1) {
            throw std::invalid_argument("OpenWakeWord requires 16000 Hz mono audio");
        }
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest &request) override {
        require_prepared("OpenWakeWord run");
        configure(request.options, request.option_arrays);
        if (!request.audio_input || request.audio_input->sample_rate != 16000 || request.audio_input->channels != 1) {
            throw std::invalid_argument("OpenWakeWord requires 16000 Hz mono audio");
        }
        reset();
        observe(runtime_.push(request.audio_input->samples));
        observe(runtime_.finish());
        return result();
    }
    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::PullEvents, 1280, 0.0};
    }
    void start_stream(const runtime::TaskRequest &request) override {
        require_prepared("OpenWakeWord stream");
        configure(request.options, request.option_arrays);
        reset();
        streaming_ = true;
    }
    void reset() override {
        runtime_.reset();
        detections_.clear();
        received_ = 0;
        active_ = false;
        streaming_ = false;
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &chunk) override {
        if (!streaming_) {
            throw std::runtime_error("OpenWakeWord stream was not started");
        }
        if (chunk.sample_rate != 16000 || chunk.channels != 1 || chunk.start_sample != received_) {
            throw std::invalid_argument("OpenWakeWord requires contiguous 16000 Hz mono chunks");
        }
        const size_t before = detections_.size();
        observe(runtime_.push(chunk.samples));
        received_ += chunk.samples.size();
        runtime::StreamEvent event;
        for (size_t i = before; i < detections_.size(); ++i) {
            runtime::VoiceActivityEvent wake;
            wake.kind = runtime::VoiceActivityEvent::Kind::SpeechSegment;
            wake.sample = detections_[i].span.end_sample;
            wake.probability = detections_[i].confidence;
            wake.segment = detections_[i];
            event.voice_activity.push_back(std::move(wake));
        }
        return event;
    }
    runtime::TaskResult finish_stream() override {
        if (!streaming_) {
            throw std::runtime_error("OpenWakeWord stream was not started");
        }
        observe(runtime_.finish());
        streaming_ = false;
        return result();
    }
    runtime::TaskResult finalize() override {
        return finish_stream();
    }

private:
    void configure(const std::unordered_map<std::string, std::string> &options,
                   const std::unordered_map<std::string, std::vector<std::string>> &arrays) {
        runtime::validate_spec_backed_request_options(options, arrays, *contract_, "OpenWakeWord");
        threshold_ = runtime::parse_finite_float_option(options, {"threshold"}).value_or(0.5F);
        if (threshold_ < 0 || threshold_ > 1) {
            throw std::invalid_argument("OpenWakeWord threshold must be within [0,1]");
        }
    }
    void observe(const std::vector<Score> &scores) {
        for (const auto &score : scores) {
            if (score.probability > threshold_ && !active_) {
                runtime::SpeechSegment detection;
                detection.span = {std::max<int64_t>(0, score.end_sample - 1280), score.end_sample};
                detection.confidence = score.probability;
                detection.text = assets_->config.require("wake_phrase").as_string();
                detections_.push_back(std::move(detection));
                active_ = true;
            } else if (score.probability <= threshold_) {
                active_ = false;
            }
        }
    }
    runtime::TaskResult result() const {
        runtime::TaskResult output;
        output.speech_segments = detections_;
        return output;
    }
    runtime::RunMode mode_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    Runtime runtime_;
    std::vector<runtime::SpeechSegment> detections_;
    int64_t received_ = 0;
    float threshold_ = 0.5F;
    bool active_ = false, streaming_ = false;
};
} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_open_wake_word_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "open_wake_word";
    config.load_assets = [](const std::filesystem::path &path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "open_wake_word");
        assets->config = assets->resources.parse_json("config");
        assets->weights = load_weights(*assets->resources.open_tensor_source("weights"), assets->config);
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec &task, const runtime::SessionOptions &options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::WakeWord) {
            throw std::invalid_argument("OpenWakeWord supports the wake-word task");
        }
        auto validation_options = options;
        // Older embedded contracts omit exec_plan; the session parses it explicitly.
        if (!contract->session_option_keys.count("open_wake_word.exec_plan")) {
            validation_options.options.erase("open_wake_word.exec_plan");
        }
        runtime::validate_spec_backed_session_options(validation_options, *contract, "open_wake_word", "OpenWakeWord");
        return std::make_unique<Session>(options, task.mode, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::models::open_wake_word
