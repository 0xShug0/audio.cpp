#include "engine/models/plapre/session.h"

#include "engine/models/plapre/assets.h"
#include "engine/models/plapre/frontend.h"
#include "engine/models/plapre/prompt.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace engine::models::plapre {
namespace {

class PlapreSession final : public runtime::RuntimeSessionBase,
    public runtime::IOfflineVoiceTaskSession, public runtime::StreamingTtsSessionBase {
public:
    PlapreSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
        std::shared_ptr<const PlapreAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          frontend_(assets_->resources), prompts_(frontend_.tokenizer()),
          generator_(assets_->resources.open_tensor_source("ar"),
              assets_->resources.open_tensor_source("speaker_projection"), assets_->ar_config, execution_context()),
          mel_(assets_->resources.open_tensor_source("kanade"), assets_->mel_config, execution_context()),
          vocoder_(modules::HiftVocoderComponent::load_from_tensor_source(
              assets_->resources.open_tensor_source("hift"), options.backend, assets_->hift_config)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "plapre", "Plapre 3");
    }

    std::string family() const override { return "plapre"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        return synthesize(request, false);
    }

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override {
        return synthesize(request, true);
    }

    void reset_stream_state() override {}

    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("Plapre 3 run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Plapre 3");
        if (!request.text_input || request.text_input->text.empty()) throw runtime::InvalidRequestError("Plapre 3 requires text input");
        const auto start = std::chrono::steady_clock::now();
        const auto & options = request.options;
        auto voice_id = runtime::find_option(options, {"voice_id"}).value_or("tor");
        std::optional<runtime::AudioBuffer> reference;
        if (request.voice && request.voice->speaker) {
            reference = request.voice->speaker->audio;
            if (request.voice->speaker->cached_voice_id && *request.voice->speaker->cached_voice_id != "default") {
                voice_id = *request.voice->speaker->cached_voice_id;
            }
        }
        const auto voice = assets_->speakers.find(voice_id);
        if (voice == assets_->speakers.end()) throw runtime::InvalidRequestError("unknown Plapre voice_id: " + voice_id);
        auto speaker = voice->second;
        PlaprePromptControls controls;
        std::optional<KanadeAudioFeatures> reference_features;
        if (reference) {
            reference_features = encode_reference(*reference, "speaker");
            speaker = reference_features->speaker;
            const auto x_vector_only = runtime::parse_bool_option(
                runtime::find_option(options, {"x_vector_only_mode"}).value_or("false"), "x_vector_only_mode");
            if (!x_vector_only) {
                controls.voice_reference = PlapreAudioReference{
                    frontend_.encode(runtime::find_option(options, {"reference_text"}).value_or(""), false), reference_features->codes};
            }
        }
        if (const auto durations = runtime::find_option(options, {"word_duration_frames"})) {
            controls.duration_frames.emplace();
            const auto parsed = io::json::parse(*durations);
            for (const auto & value : parsed.as_array()) {
                const auto number = value.as_number();
                if (!std::isfinite(number) || number < 0 || number > 399 || std::floor(number) != number) {
                    throw runtime::InvalidRequestError("word_duration_frames must be a JSON array of integers from 0 to 399");
                }
                controls.duration_frames->push_back(static_cast<int64_t>(number));
            }
        }
        if (const auto previous_text = runtime::find_option(options, {"context_text"})) {
            controls.context_text = frontend_.encode(*previous_text, false);
            controls.same_speaker = runtime::parse_bool_option(
                runtime::find_option(options, {"same_speaker"}).value_or("true"), "same_speaker");
            if (const auto previous_audio = runtime::find_option(options, {"context_audio_file"})) {
                const auto wav = audio::read_wav_f32(std::filesystem::path(*previous_audio));
                controls.context_codes = encode_reference({wav.sample_rate, wav.channels, wav.samples}, "context").codes;
            }
        }
        if (const auto pronunciations = runtime::find_option(options, {"pronunciations"})) {
            const auto parsed = io::json::parse(*pronunciations);
            size_t total_codes = 0;
            for (const auto & item : parsed.as_array()) {
                if (controls.pronunciations.size() == 10 || total_codes == 200) break;
                const auto wav = audio::read_wav_f32(std::filesystem::path(item.require("audio_file").as_string()));
                auto features = encode_reference({wav.sample_rate, wav.channels, wav.samples},
                    "pronunciation." + std::to_string(controls.pronunciations.size()));
                total_codes += std::min({size_t{60}, size_t{200} - total_codes, features.codes.size()});
                controls.pronunciations.push_back({frontend_.encode(item.require("word").as_string(), false), std::move(features.codes)});
            }
        }
        const auto edit_start = runtime::parse_i64_option(options, {"edit_start_frame"});
        const auto edit_end = runtime::parse_i64_option(options, {"edit_end_frame"});
        const bool editing = edit_start.has_value() || edit_end.has_value();
        if (editing && (!edit_start || !edit_end || !reference_features)) {
            throw runtime::InvalidRequestError("Plapre editing requires voice_ref, edit_start_frame, and edit_end_frame");
        }
        PlapreSamplingOptions sampling;
        sampling.seed = runtime::parse_u64_option(options, {"seed"}).value_or(0);
        sampling.temperature = runtime::parse_finite_float_option(options, {"temperature"}).value_or(0.8f);
        sampling.top_p = runtime::parse_finite_float_option(options, {"top_p"}).value_or(0.95f);
        sampling.top_k = runtime::parse_i64_option(options, {"top_k"}).value_or(50);
        sampling.max_tokens = runtime::parse_positive_i64_option(options, {"max_tokens"}, 500);
        sampling.min_tokens = runtime::parse_i64_option(options, {"min_tokens"}).value_or(0);
        sampling.stop_ids = prompts_.stop_ids(editing);
        const auto chunk_mode = text::parse_text_chunk_mode_override(options).value_or(text::TextChunkMode::Default);
        const auto normalized = frontend_.normalize(request.text_input->text, chunk_mode == text::TextChunkMode::Endline);
        auto chunks = editing || controls.duration_frames ? std::vector<std::string>{normalized}
            : text::split_text_chunks(normalized,
                text::parse_text_chunk_size_override(options).value_or(300),
                chunk_mode);
        const auto gap = runtime::parse_finite_float_option(options, {"silence_duration_sec"}).value_or(0.1f);
        if (gap < 0) throw runtime::InvalidRequestError("silence_duration_sec must be nonnegative");
        std::vector<float> waveform;
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto prompt = editing ? prompts_.edit(frontend_.encode(chunks[i], false), reference_features->codes,
                *edit_start, *edit_end, controls.pronunciations)
                : prompts_.generate(frontend_.encode(chunks[i], true), controls);
            const auto embeddings = generator_.prepare_inputs(prompt, speaker);
            const auto generated = generator_.generate(embeddings, sampling);
            auto codes = prompts_.audio_codes(generated);
            if (editing) {
                const auto & original = reference_features->codes;
                codes.insert(codes.begin(), original.begin(), original.begin() + *edit_start);
                codes.insert(codes.end(), original.begin() + *edit_end, original.end());
            }
            if (codes.empty()) throw runtime::InvalidRequestError("Plapre generated no audio codes; check input text, max_tokens, and sampling settings");
            const auto mel = mel_.decode(codes, speaker);
            const auto vocoder_start = std::chrono::steady_clock::now();
            const auto audio = vocoder_.synthesize(mel, static_cast<int64_t>(codes.size()) * 2, sampling.seed + i);
            debug::timing_log_scalar("plapre.vocoder.wall_ms", debug::elapsed_ms(vocoder_start));
            if (i > 0) waveform.insert(waveform.end(), static_cast<size_t>(gap * 24000), 0.0f);
            waveform.insert(waveform.end(), audio.waveform.begin(), audio.waveform.end());
            if (streaming && stream_event_sink()) {
                runtime::StreamEvent event;
                event.audio_output = runtime::AudioBuffer{24000, 1, {}};
                if (i > 0) event.audio_output->samples.resize(static_cast<size_t>(gap * 24000), 0.0f);
                event.audio_output->samples.insert(event.audio_output->samples.end(),
                    audio.waveform.begin(), audio.waveform.end());
                stream_event_sink()(event);
            }
        }
        if (waveform.empty()) throw runtime::InvalidRequestError("Plapre input is empty after text normalization");
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{24000, 1, std::move(waveform)};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

    KanadeAudioFeatures encode_reference(const runtime::AudioBuffer & audio, const std::string & role) {
        struct ReferenceCache {
            runtime::AudioBuffer audio;
            KanadeAudioFeatures features;
        };
        const auto key = "plapre.reference." + role;
        const auto * cached = cache().find<ReferenceCache>(key);
        if (cached && cached->audio.sample_rate == audio.sample_rate && cached->audio.channels == audio.channels &&
            cached->audio.samples == audio.samples) return cached->features;
        if (!encoder_) encoder_ = std::make_unique<KanadeAudioEncoder>(assets_->resources.open_tensor_source("kanade"),
            assets_->resources.open_tensor_source("wavlm"), assets_->encoder_config, execution_context());
        auto mono = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, 24000, audio::torchaudio_sinc_hann_float32_options());
        auto features = encoder_->encode(mono);
        cache().put(key, ReferenceCache{audio, features});
        return features;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const PlapreAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    PlapreTextFrontend frontend_;
    PlaprePromptBuilder prompts_;
    PlapreLlamaGenerator generator_;
    KanadeMelDecoder mel_;
    modules::HiftVocoderComponent vocoder_;
    std::unique_ptr<KanadeAudioEncoder> encoder_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_plapre_loader() {
    runtime::SpecBackedVoiceModelConfig<PlapreAssets> config;
    config.family = "plapre";
    config.load_assets = load_plapre_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const PlapreAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        if ((task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) ||
            (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming)) {
            throw runtime::InvalidRequestError("Plapre 3 supports offline/streaming tts and clon");
        }
        return std::make_unique<PlapreSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::plapre
