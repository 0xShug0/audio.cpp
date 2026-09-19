#include "engine/models/pulsevad/session.h"

#include "engine/models/pulsevad/runtime.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace engine::models::pulsevad {
namespace {

struct Assets {
    assets::ResourceBundle resources;
};

const std::initializer_list<runtime::OptionV1CompatibilityAlias> kRequestAliases = {
    {"pulsevad.threshold", "threshold"},
    {"pulsevad.hop_size_samples", "hop_size_samples"},
    {"pulsevad.min_speech_duration_ms", "min_speech_duration_ms"},
    {"pulsevad.min_silence_duration_ms", "min_silence_duration_ms"},
};

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)),
          runtime_(std::move(source), execution_context(),
              runtime::parse_tensor_storage_option(options.options, "pulsevad.weight_type",
                  assets::TensorStorageType::Native,
                  {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
                   assets::TensorStorageType::F16, assets::TensorStorageType::BF16})) {}

    std::string family() const override { return "pulsevad"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Vad; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        const auto options = runtime::apply_option_v1_compatibility(
            request.options, kRequestAliases, "PulseVAD", "request");
        runtime::validate_spec_backed_request_options(options, request.option_arrays, *contract_, "PulseVAD");
        if (!request.audio) {
            throw std::runtime_error("PulseVAD preparation requires an audio contract");
        }
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("PulseVAD run");
        const auto options = runtime::apply_option_v1_compatibility(
            request.options, kRequestAliases, "PulseVAD", "request");
        runtime::validate_spec_backed_request_options(options, request.option_arrays, *contract_, "PulseVAD");
        if (!request.audio_input) {
            throw std::runtime_error("PulseVAD requires audio_input");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto & audio = *request.audio_input;
        if (audio.sample_rate != 16000 || audio.channels != 1) {
            throw std::runtime_error("PulseVAD requires 16000 Hz mono audio");
        }
        const float threshold = runtime::parse_finite_float_option(options, {"threshold"}).value_or(0.5f);
        const int hop = runtime::parse_int_option(options, {"hop_size_samples"}).value_or(1600);
        const int min_speech_ms = runtime::parse_int_option(options, {"min_speech_duration_ms"}).value_or(100);
        const int min_silence_ms = runtime::parse_int_option(options, {"min_silence_duration_ms"}).value_or(100);
        if (threshold < 0 || threshold > 1 || hop <= 0 || min_speech_ms < 0 || min_silence_ms < 0) {
            throw std::runtime_error("Invalid PulseVAD threshold, hop, or minimum duration");
        }
        const int64_t length = static_cast<int64_t>(audio.samples.size());
        runtime::TaskResult result;
        std::vector<float> window(3200);
        int64_t start = -1;
        int64_t last_speech = 0;
        const auto append_segment = [&](int64_t begin, int64_t end) {
            runtime::SpeechSegment segment;
            segment.span.start_sample = begin;
            segment.span.end_sample = std::min(end, length);
            result.speech_segments.push_back(segment);
        };
        for (int64_t offset = 0; offset <= std::max<int64_t>(0, length - 3200); offset += hop) {
            std::fill(window.begin(), window.end(), 0.0f);
            std::copy_n(audio.samples.begin() + offset, std::min<int64_t>(3200, length - offset), window.begin());
            const auto logits = runtime_.infer_features(runtime_.extract_features(window));
            const float probability = 1.0f / (1.0f + std::exp(logits[0] - logits[1]));
            if (length < 3200) {
                if (probability >= threshold) {
                    append_segment(0, length);
                }
                break;
            }
            if (probability >= threshold) {
                if (start < 0) {
                    start = offset;
                }
                last_speech = offset + 3200;
            } else if (start >= 0 && offset - last_speech >= int64_t(min_silence_ms) * 16) {
                if (last_speech - start >= int64_t(min_speech_ms) * 16) {
                    append_segment(start, last_speech);
                }
                start = -1;
            }
        }
        if (start >= 0 && last_speech - start >= int64_t(min_speech_ms) * 16) {
            append_segment(start, last_speech);
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    PulseVADRuntime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_pulsevad_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "pulsevad";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "pulsevad");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Vad || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("PulseVAD supports offline VAD");
        }
        runtime::validate_spec_backed_session_options(options, *contract, "pulsevad", "PulseVAD");
        return std::make_unique<Session>(options, assets->resources.open_tensor_source("weights"), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::pulsevad
