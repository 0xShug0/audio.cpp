#include "engine/models/sidon/session.h"
#include "engine/models/sidon/frontend.h"
#include "engine/models/sidon/runtime.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace engine::models::sidon {
namespace {

struct Assets {
    assets::ResourceBundle resources;
};

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), contract_(std::move(contract)), runtime_(std::move(source), execution_context()) {}

    std::string family() const override { return "sidon"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::SpeechToSpeech; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Sidon");
        if (!request.audio) throw std::runtime_error("Sidon requires input audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Sidon run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Sidon");
        if (!request.audio_input) throw std::runtime_error("Sidon requires --audio");
        const auto started = std::chrono::steady_clock::now();
        auto samples = prepare_audio(*request.audio_input);
        runtime::AudioBuffer output;
        output.sample_rate = 48000;
        output.channels = 1;
        std::vector<float> previous_frame;
        constexpr size_t chunk_samples = 96 * 16000;
        for (size_t offset = 0; offset < samples.size(); offset += chunk_samples) {
            const size_t count = std::min(chunk_samples, samples.size() - offset);
            std::vector<float> padded(count + 320, 0.0f);
            std::copy_n(samples.begin() + offset, count, padded.begin() + 160);
            modules::Wav2Vec2BertEncoderInput features;
            const auto frontend_ms = debug::measure_ms([&] {
                features = extract_features(padded, execution_context().config().threads);
            });
            modules::Wav2Vec2BertEncoderOutput hidden;
            const auto encoder_ms = debug::measure_ms([&] { hidden = runtime_.encode(features); });
            if (!previous_frame.empty()) {
                hidden.values.insert(hidden.values.begin(), previous_frame.begin(), previous_frame.end());
                ++hidden.frames;
            }
            previous_frame.assign(hidden.values.end() - 1024, hidden.values.end());
            std::vector<float> audio;
            const auto decoder_ms = debug::measure_ms([&] { audio = runtime_.decode(hidden.values, hidden.frames); });
            if (audio.size() <= 960) throw std::runtime_error("Sidon decoder returned insufficient samples");
            output.samples.insert(output.samples.end(), audio.begin(), audio.end() - 960);
            debug::timing_log_scalar("sidon.frontend.ms", frontend_ms);
            debug::timing_log_scalar("sidon.encoder.ms", encoder_ms);
            debug::timing_log_scalar("sidon.decoder.ms", decoder_ms);
        }
        const size_t target_samples = request.audio_input->samples.size() / request.audio_input->channels *
            uint64_t(48000) / request.audio_input->sample_rate;
        if (output.samples.size() < target_samples) throw std::runtime_error("Sidon output is shorter than the input");
        output.samples.resize(target_samples);
        runtime::TaskResult result;
        result.audio_output = std::move(output);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const model_spec::ModelContract> contract_;
    SidonRuntime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_sidon_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "sidon";
    config.load_assets = [](const std::filesystem::path & path) {
        auto assets = std::make_shared<Assets>();
        assets->resources = model_spec::load_resource_bundle_for_family(path, "sidon");
        return assets;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::SpeechToSpeech || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("Sidon supports offline speech restoration");
        runtime::validate_spec_backed_session_options(options, *contract, "sidon", "Sidon");
        return std::make_unique<Session>(options, assets->resources.open_tensor_source("weights"), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::sidon
