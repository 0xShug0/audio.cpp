#include "engine/community_models/dettato/session.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/midi/midi_file.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/community_models/dettato/assets.h"
#include "engine/community_models/dettato/pipeline.h"
#include <chrono>

namespace engine::community_models::dettato {
namespace {
class DettatoSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
  public:
    DettatoSession(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const DettatoAssets> assets,
                   std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)),
          pipeline_(std::move(assets), execution_context(),
                    runtime::parse_int_option(options.options, {"dettato.window_hop_sec"}).value_or(2)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "dettato", "Dettato");
        if (task.task != runtime::VoiceTaskKind::Midi || task.mode != runtime::RunMode::Offline)
            throw runtime::InvalidRequestError("Dettato supports offline MIDI transcription");
    }
    std::string family() const override {
        return "dettato";
    }
    runtime::VoiceTaskKind task_kind() const override {
        return task_.task;
    }
    runtime::RunMode run_mode() const override {
        return task_.mode;
    }
    void prepare(const runtime::SessionPreparationRequest &request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Dettato");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest &request) override {
        require_prepared("Dettato run");
        const auto start = std::chrono::steady_clock::now();
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Dettato");
        if (!request.option_arrays.empty())
            throw runtime::InvalidRequestError("Dettato options must be scalar values");
        if (!request.audio_input || request.audio_input->sample_rate <= 0 || request.audio_input->channels <= 0 ||
            request.audio_input->samples.empty() || request.audio_input->samples.size() % request.audio_input->channels)
            throw runtime::InvalidRequestError("Dettato requires valid input audio");
        const auto format = runtime::find_option(request.options, {"output_format"}).value_or("midi");
        if (format != "midi" && format != "json")
            throw runtime::InvalidRequestError("Dettato output_format must be midi or json");
        const auto &audio = *request.audio_input;
        const auto transcription = pipeline_.transcribe(
            audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels), audio.sample_rate);
        io::json::Value::Array tracks;
        std::vector<midi::MidiNote> midi_notes;
        for (const auto &track : transcription.tracks) {
            io::json::Value::Array notes;
            for (const auto &note : track.notes) {
                notes.push_back(
                    io::json::Value::make_object({{"onset", io::json::Value::make_number(note.onset)},
                                                  {"offset", io::json::Value::make_number(note.offset)},
                                                  {"pitch", io::json::Value::make_number(note.pitch)},
                                                  {"velocity", io::json::Value::make_number(note.velocity)}}));
                midi_notes.push_back(
                    {track.is_drum, track.program, note.onset, note.offset, note.pitch, track.name, note.velocity});
            }
            tracks.push_back(io::json::Value::make_object({{"name", io::json::Value::make_string(track.name)},
                                                           {"program", io::json::Value::make_number(track.program)},
                                                           {"is_drum", io::json::Value::make_bool(track.is_drum)},
                                                           {"notes", io::json::Value::make_array(std::move(notes))}}));
        }
        const auto json =
            io::json::Value::make_object({{"tracks", io::json::Value::make_array(std::move(tracks))},
                                          {"tuning_cents", io::json::Value::make_number(transcription.tuning_cents)},
                                          {"transcribed", io::json::Value::make_number(transcription.transcribed)},
                                          {"dropped", io::json::Value::make_number(transcription.dropped)},
                                          {"inserted", io::json::Value::make_number(transcription.inserted)},
                                          {"drum_router", io::json::Value::make_number(transcription.drum_router)},
                                          {"drum_model", io::json::Value::make_string(transcription.drum_model)}});
        const auto text = io::json::stringify(json);
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{text, "midi-json"};
        result.custom_schema_output = runtime::CustomSchemaOutput{"dettato-notes-v1", json};
        if (format == "json")
            result.artifact_output = runtime::make_text_artifact(
                runtime::ArtifactKind::Custom, "result", text,
                {{"mime", "application/json"}, {"extension", "json"}, {"format", "dettato-notes-v1"}});
        else {
            midi::MidiFileOptions options;
            options.ticks_per_beat = 220;
            options.preserve_note_events = true;
            result.artifact_output = runtime::make_voice_artifact(
                runtime::ArtifactKind::Midi, "result", midi::write_standard_midi_file(midi_notes, options),
                {{"mime", "audio/midi"}, {"extension", "mid"}, {"format", "midi"}});
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

  private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    DettatoPipeline pipeline_;
};
} // namespace
std::shared_ptr<runtime::IVoiceModelLoader> make_dettato_loader() {
    runtime::SpecBackedVoiceModelConfig<DettatoAssets> config;
    config.family = "dettato";
    config.load_assets = load_dettato_assets;
    config.create_session = [](const runtime::TaskSpec &task, const runtime::SessionOptions &options,
                               std::shared_ptr<const DettatoAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<DettatoSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::community_models::dettato
