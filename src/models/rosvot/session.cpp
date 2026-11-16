#include "engine/models/rosvot/session.h"
#include "engine/models/rosvot/runtime.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::rosvot {
namespace {
class RosvotSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    RosvotSession(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const RosvotAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)),
          runtime_(std::move(assets), execution_context()) {
        runtime::validate_spec_backed_session_options(options, *contract_, "rosvot", "ROSVOT");
        if (task.task != runtime::VoiceTaskKind::Midi || task.mode != runtime::RunMode::Offline)
            throw runtime::InvalidRequestError("ROSVOT supports offline MIDI transcription");
    }
    std::string family() const override { return "rosvot"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ROSVOT");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("ROSVOT run");
        const auto start = std::chrono::steady_clock::now();
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "ROSVOT");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.option_arrays.empty()) throw runtime::InvalidRequestError("ROSVOT options must be scalar values");
        if (!request.audio_input || request.audio_input->sample_rate <= 0 || request.audio_input->channels <= 0 ||
            request.audio_input->samples.empty() || request.audio_input->samples.size() % request.audio_input->channels)
            throw runtime::InvalidRequestError("ROSVOT requires valid input audio");
        const auto & a = *request.audio_input;
        for (float value : a.samples) if (!std::isfinite(value)) throw runtime::InvalidRequestError("ROSVOT audio contains non-finite samples");
        auto samples = audio::mixdown_interleaved_to_mono_average(a.samples, a.channels);
        if (a.sample_rate != 24000) {
            audio::SoxrResampleOptions resampling;
            resampling.reject_empty_output = true;
            const auto resampled = audio::try_resample_mono_soxr(samples, a.sample_rate, 24000, resampling);
            if (!resampled) throw runtime::InvalidRequestError("ROSVOT sample-rate conversion requires SOXR support");
            samples = *resampled;
        }
        const auto threshold = runtime::parse_finite_float_option(request.options, {"note_boundary_threshold"}).value_or(0.85f);
        const auto word_threshold = runtime::parse_finite_float_option(request.options, {"word_boundary_threshold"}).value_or(0.5f);
        if (threshold < 0 || threshold > 1 || word_threshold < 0 || word_threshold > 1)
            throw runtime::InvalidRequestError("ROSVOT boundary thresholds must be in [0, 1]");
        const auto format = runtime::find_option(request.options, {"output_format"}).value_or("midi");
        if (format != "json" && format != "midi") throw runtime::InvalidRequestError("ROSVOT output_format must be midi or json");
        std::vector<double> word_durations;
        if (const auto file = runtime::find_option(request.options, {"word_durations_file"})) {
            const auto v = io::json::parse_file(*file);
            for (const auto & item : v.as_array()) {
                const double d = item.as_number();
                if (!std::isfinite(d) || d < 0) throw runtime::InvalidRequestError("ROSVOT word durations must be non-negative finite seconds");
                word_durations.push_back(d);
            }
            if (word_durations.empty()) throw runtime::InvalidRequestError("ROSVOT word durations must not be empty");
        }
        auto result = runtime_.transcribe(samples, threshold, word_threshold, word_durations);
        io::json::Value::Array notes;
        for (size_t i = 0; i < result.notes.size(); ++i) {
            const auto & n = result.notes[i];
            notes.push_back(io::json::Value::make_object({
                {"pitch", io::json::Value::make_number(n.pitch)},
                {"start", io::json::Value::make_number(n.onset_seconds)},
                {"end", io::json::Value::make_number(n.offset_seconds)},
                {"word_index", io::json::Value::make_number(result.note_to_word[i])}}));
        }
        const auto json = io::json::Value::make_object({{"notes", io::json::Value::make_array(std::move(notes))}});
        const auto text = io::json::stringify(json);
        runtime::TaskResult out;
        out.text_output = runtime::Transcript{text, "midi-json"};
        out.custom_schema_output = runtime::CustomSchemaOutput{"rosvot-notes-v1", json};
        if (format == "json") out.artifact_output = runtime::make_text_artifact(runtime::ArtifactKind::Custom, "result", text,
            {{"mime", "application/json"}, {"extension", "json"}, {"format", "rosvot-notes-v1"}});
        else {
            result.notes.erase(std::remove_if(result.notes.begin(), result.notes.end(), [](const midi::MidiNote & n) {
                return n.pitch == 0;
            }), result.notes.end());
            out.artifact_output = runtime::make_voice_artifact(runtime::ArtifactKind::Midi, "result",
                midi::write_standard_midi_file(result.notes), {{"mime", "audio/midi"}, {"extension", "mid"}, {"format", "midi"}});
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return out;
    }
private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    RosvotRuntime runtime_;
};
}
std::shared_ptr<runtime::IVoiceModelLoader> make_rosvot_loader() {
    runtime::SpecBackedVoiceModelConfig<RosvotAssets> config;
    config.family = "rosvot";
    config.load_assets = load_rosvot_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const RosvotAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<RosvotSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::models::rosvot
